"""
Main script for training and using Mini-GPT
"""
import os
import argparse
import torch
from utils import initialize_tokenizer, get_training_data
from dataset import prepare_dataset_from_huggingface, create_dataloader
from model import MiniGPT, test_model
from trainer import train_model
from generator import generate_text, interactive_generation, visualize_generation_attention

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Mini-GPT: Train and generate text with a small language model")
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'generate', 'interactive', 'test'],
                        help='Operation mode: train, generate, interactive, or test')
    
    # Model configuration
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--ff_dim', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps for scheduler')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory for checkpoints')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--dataset', type=str, default='wikitext-2', help='Dataset to use for training')
    
    # Generation parameters
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model')
    parser.add_argument('--prompt', type=str, default='Once upon a time', help='Text prompt for generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling parameter')
    parser.add_argument('--do_sample', action='store_true', help='Use sampling (vs greedy)')
    parser.add_argument('--repetition_penalty', type=float, default=1.1, help='Penalty for repetition')
    parser.add_argument('--visualize', action='store_true', help='Visualize attention')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    print(f"Running Mini-GPT in {args.mode} mode on {device}")
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer()
    
    if args.mode == 'train':
        print(f"Training new model with {args.num_layers} layers, {args.embed_dim} dimensions")
        
        # Load dataset
        train_data = get_training_data(args.dataset, split="train")
        val_data = get_training_data(args.dataset, split="validation")
        
        # Prepare datasets
        train_dataset = prepare_dataset_from_huggingface(
            train_data, tokenizer, max_length=args.max_seq_length)
        val_dataset = prepare_dataset_from_huggingface(
            val_data, tokenizer, max_length=args.max_seq_length)
        
        # Create dataloaders
        train_loader = create_dataloader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = create_dataloader(
            val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        model = MiniGPT(
            vocab_size=args.vocab_size,
            max_seq_length=args.max_seq_length,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout
        ).to(device)
        
        # Train model
        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_steps=args.warmup_steps,
            checkpoint_dir=args.checkpoint_dir,
            log_interval=args.log_interval,
            resume_from=args.resume_from
        )
        
    elif args.mode == 'generate':
        if args.model_path is None:
            print("Error: model_path must be provided for generation mode")
            return
            
        # Load model
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model = MiniGPT(
            vocab_size=args.vocab_size,
            max_seq_length=args.max_seq_length,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Generate text
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty
        )
        
        print("\nGenerated text:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        
        # Visualize attention if requested
        if args.visualize:
            visualize_generation_attention(model, tokenizer, args.prompt, generated_text)
            
    elif args.mode == 'interactive':
        if args.model_path is None:
            print("Error: model_path must be provided for interactive mode")
            return
            
        # Load model
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model = MiniGPT(
            vocab_size=args.vocab_size,
            max_seq_length=args.max_seq_length,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Start interactive session
        interactive_generation(
            model=model,
            tokenizer=tokenizer,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty
        )
        
    elif args.mode == 'test':
        # Test model architecture
        model = test_model()
        
        # Test data loading
        print("\nTesting data loading...")
        test_data = get_training_data(args.dataset, split="test")
        test_dataset = prepare_dataset_from_huggingface(
            test_data, tokenizer, max_length=16)  # Small for testing
        test_loader = create_dataloader(test_dataset, batch_size=2)
        print(f"Successfully created test dataset with {len(test_dataset)} examples")

if __name__ == "__main__":
    main()