"""
Dataset implementations for Mini-GPT
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class MiniGPTDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        """
        Prepare texts for Mini-GPT training.
        
        Args:
            texts: List of text documents
            tokenizer: Our initialized tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all texts and join them
        print("Tokenizing dataset...")
        self.all_tokens = []
        for text in tqdm(texts, desc="Processing texts"):
            if text.strip():  # Skip empty texts
                tokens = tokenizer.encode(text)
                self.all_tokens.extend(tokens)
        
        # Create examples of length max_length with stride of max_length // 2
        stride = max_length // 2
        self.examples = []
        for i in range(0, len(self.all_tokens) - max_length, stride):
            self.examples.append(self.all_tokens[i:i + max_length])
        
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Get tokens for this example
        tokens = self.examples[idx]
        
        # Input: all tokens except last one
        # Target: all tokens except first one (shifted by 1)
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y

def prepare_dataset_from_huggingface(dataset, tokenizer, max_length=128, column_name='text'):
    """
    Convert a Hugging Face dataset into our custom MiniGPT format.
    
    Args:
        dataset: A Hugging Face dataset
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        column_name: Name of the text column in the dataset
    
    Returns:
        A MiniGPTDataset ready for training
    """
    # Extract all texts from the dataset
    all_texts = dataset[column_name]
    
    # Create our custom dataset
    return MiniGPTDataset(all_texts, tokenizer, max_length)

def create_dataloader(dataset, batch_size=16, shuffle=True, num_workers=2):
    """Create a DataLoader for efficient batch processing."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # This helps speed up data transfer to GPU
    )

def split_dataset(dataset, val_ratio=0.1):
    """Split a dataset into training and validation sets."""
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Split dataset into {train_size} training and {val_size} validation examples")
    return train_dataset, val_dataset

def create_dataloaders_from_huggingface(
    train_data,
    tokenizer,
    val_data=None,
    max_length=128,
    batch_size=16,
    num_workers=2,
    val_ratio=0.1,
    column_name='text'
):
    """Create training and validation dataloaders from Hugging Face datasets."""
    # Process training data
    train_dataset = prepare_dataset_from_huggingface(
        train_data, tokenizer, max_length, column_name)
    
    # Handle validation data
    if val_data is not None:
        # Use provided validation data
        val_dataset = prepare_dataset_from_huggingface(
            val_data, tokenizer, max_length, column_name)
    else:
        # Split training data
        train_dataset, val_dataset = split_dataset(train_dataset, val_ratio)
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = create_dataloader(
        val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, train_dataset, val_dataset

def visualize_dataset_examples(dataset, tokenizer, num_examples=3):
    """Visualize some examples from our dataset."""
    import random
    
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    
    for i, idx in enumerate(indices):
        x, y = dataset[idx]
        
        # Decode tokens back to text
        input_text = tokenizer.decode(x)
        target_text = tokenizer.decode(y)
        
        print(f"\n--- Example {i+1} ---")
        print(f"Input tokens shape: {x.shape}")
        print(f"Target tokens shape: {y.shape}")
        print(f"\nInput text snippet: \"{input_text[:100]}...\"")
        print(f"Target text snippet: \"{target_text[:100]}...\"")
        
        # Show how they overlap
        print("\nNotice how the target is shifted by one token:")
        for j in range(min(5, len(x))):
            print(f"Input token {j}: '{tokenizer.decode([x[j]])}'")
            print(f"Target token {j}: '{tokenizer.decode([y[j]])}'")