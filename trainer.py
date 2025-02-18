"""
Training functionality for Mini-GPT
"""
import os
import math
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import save_checkpoint, plot_training_progress

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(
    model, 
    train_loader, 
    val_loader=None,
    epochs=3, 
    learning_rate=3e-4,
    max_grad_norm=1.0,
    warmup_steps=1000,
    checkpoint_dir="./checkpoints",
    log_interval=100,
    resume_from=None
):
    """
    Train the Mini-GPT model.
    
    Args:
        model: The Mini-GPT model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        epochs: Number of training epochs
        learning_rate: Maximum learning rate after warmup
        max_grad_norm: Maximum gradient norm for clipping
        warmup_steps: Number of warmup steps for learning rate
        checkpoint_dir: Directory to save checkpoints
        log_interval: How often to log training stats
        resume_from: Path to checkpoint to resume training from
    
    Returns:
        train_losses: List of training loss records
        val_losses: List of validation loss records
    """
    # Create directory for checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        # Linear warmup followed by cosine decay
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine decay to 10% of max learning rate
            progress = (step - warmup_steps) / max(1, epochs * len(train_loader) - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training statistics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    global_step = 0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from epoch {start_epoch} with global step {global_step}")
    
    # Total training time tracking
    total_train_time = 0
    
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        epoch_loss = 0
        epoch_start_time = time.time()
        
        # Progress bar for the epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+epochs}")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            batch_start = time.time()
            
            # Move to device
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            
            # Reshape for loss calculation
            # From [batch, seq_len, vocab] to [batch*seq_len, vocab]
            logits = logits.reshape(-1, logits.size(-1))
            y = y.reshape(-1)
            
            # Calculate loss
            loss = F.cross_entropy(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update statistics
            global_step += 1
            epoch_loss += loss.item()
            batch_time = time.time() - batch_start
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.6f}",
                'time': f"{batch_time:.2f}s"
            })
            
            # Log training progress
            if global_step % log_interval == 0:
                train_losses.append({
                    'step': global_step,
                    'loss': loss.item(),
                    'lr': current_lr,
                    'epoch': epoch + (batch_idx / len(train_loader))
                })
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Average loss: {avg_epoch_loss:.4f}")
        
        # Validation
        if val_loader is not None:
            val_loss = evaluate_model(model, val_loader)
            print(f"Validation loss: {val_loss:.4f}")
            val_losses.append({
                'epoch': epoch + 1,
                'loss': val_loss
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, val_loss, 
                    f"{checkpoint_dir}/best_model.pt",
                    global_step=global_step,
                    best_val_loss=best_val_loss
                )
                print(f"New best model saved!")
            
        # Save checkpoint after each epoch
        save_checkpoint(
            model, optimizer, epoch, avg_epoch_loss,
            f"{checkpoint_dir}/epoch_{epoch+1}.pt",
            global_step=global_step,
            best_val_loss=best_val_loss
        )
        
        # Generate intermediate training curves
        if epoch % 2 == 0 or epoch == epochs - 1:
            try:
                plot_training_progress(train_losses, val_losses)
            except Exception as e:
                print(f"Warning: Could not plot training progress: {e}")
    
    print(f"Training completed in {total_train_time:.2f}s")
    return train_losses, val_losses

def evaluate_model(model, val_loader):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits = logits.reshape(-1, logits.size(-1))
            y = y.reshape(-1)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_with_config(config, train_loader, val_loader=None):
    """Train model with a configuration dictionary."""
    from model import MiniGPT
    
    # Create model from config
    model = MiniGPT(
        vocab_size=config.get('vocab_size', 50257),
        max_seq_length=config.get('max_seq_length', 128),
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 4),
        ff_dim=config.get('ff_dim', 1024),
        dropout=config.get('dropout', 0.1)
    ).to(device)
    
    # Train model
    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get('epochs', 3),
        learning_rate=config.get('learning_rate', 3e-4),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        warmup_steps=config.get('warmup_steps', 1000),
        checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
        log_interval=config.get('log_interval', 100),
        resume_from=config.get('resume_from', None)
    )