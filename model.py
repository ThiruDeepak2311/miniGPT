"""
Mini-GPT Model Implementation
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Convert token IDs to rich vector representations.
        
        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim: Dimension of embeddings
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, x):
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        return self.embedding(x) * math.sqrt(self.embed_dim)
        # Scaling by sqrt(embed_dim) helps with training stability

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length=1000):
        """
        Add position information to embeddings.
        
        Args:
            embed_dim: Dimension of embeddings
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        # Create a matrix of shape (max_seq_length, embed_dim)
        pe = torch.zeros(max_seq_length, embed_dim)
        
        # Create a vector of shape (max_seq_length, 1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create a vector of shape (1, embed_dim/2)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension (1, max_seq_length, embed_dim)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Add positional encoding to token embeddings
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Multi-head attention mechanism.
        
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input of shape (batch_size, seq_len, embed_dim)
            mask: Mask to apply on attention scores
        """
        batch_size, seq_len, _ = x.size()
        
        # Project inputs to queries, keys, and values
        # Shape: (batch_size, seq_len, embed_dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for multi-head attention
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask (if provided)
        if mask is not None:
            # Expand mask for batch size and num_heads
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
                mask = mask.expand(batch_size, self.num_heads, -1, -1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        self.attention_weights = attention_weights.detach()  # Store for visualization
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attention_weights, v)
        
        # Reshape back
        # Shape: (batch_size, seq_len, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.output_projection(context)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        """
        Position-wise feed-forward network.
        
        Args:
            embed_dim: Input/output dimension
            ff_dim: Hidden dimension (typically 4x embed_dim)
            dropout: Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),  # Modern activation function
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        A single transformer block.
        
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Attention block (with residual connection and layer norm)
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        
        # Feed-forward block (with residual connection and layer norm)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        
        return x

class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size=50257,  # Default GPT-2 vocabulary size
        max_seq_length=128,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        ff_dim=1024,
        dropout=0.1
    ):
        """
        Mini-GPT language model.
        
        Args:
            vocab_size: Size of vocabulary
            max_seq_length: Maximum sequence length
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        # Token embeddings
        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        
        # Positional encodings
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_length)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.output = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Save configurations
        self.config = {
            'vocab_size': vocab_size,
            'max_seq_length': max_seq_length,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'ff_dim': ff_dim,
            'dropout': dropout
        }
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_causal_mask(self, seq_len, device):
        """Create a causal mask for autoregressive generation."""
        # Lower triangular matrix
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)
        return mask
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tokens of shape (batch_size, seq_len)
            mask: Optional attention mask
        """
        # Get sequence length
        seq_len = x.size(1)
        
        # Create causal mask if not provided
        if mask is None:
            mask = self.get_causal_mask(seq_len, x.device)
        
        # Token embeddings + positional encodings
        x = self.token_embed(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer normalization
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.output(x)
        
        return logits
    
    def get_attention_weights(self):
        """Extract attention weights from all layers for visualization."""
        weights = []
        for block in self.blocks:
            if hasattr(block.attention, 'attention_weights'):
                weights.append(block.attention.attention_weights)
        return weights

def load_model_from_checkpoint(checkpoint_path, device=None):
    """Load model from checkpoint with state_dict only."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_config' in checkpoint:
        # Checkpoint has explicit config
        config = checkpoint['model_config']
        model = MiniGPT(**config).to(device)
    else:
        # Try to infer from state_dict
        state_dict = checkpoint['model_state_dict']
        
        # Infer vocab_size from output layer
        vocab_size = state_dict['output.weight'].size(0)
        
        # Infer embed_dim from various layers
        embed_dim = state_dict['token_embed.embedding.weight'].size(1)
        
        # Infer num_layers from number of block entries
        # Count unique block indices
        num_layers = len([k for k in state_dict.keys() if k.startswith('blocks.')]) // 5
        
        # Estimate num_heads from attention layer dimensions
        head_dim = state_dict['blocks.0.attention.query.weight'].size(0) // embed_dim
        num_heads = embed_dim // head_dim
        
        # Estimate ff_dim from feed-forward layer dimensions
        ff_dim = state_dict['blocks.0.feed_forward.net.0.weight'].size(0)
        
        # Create model with inferred params
        model = MiniGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim
        ).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    return model

def test_model(model=None, vocab_size=50257, seq_len=32, batch_size=2):
    """Test that model can process inputs and produce outputs."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a model if not provided
    if model is None:
        model = MiniGPT(
            vocab_size=vocab_size,
            max_seq_length=128,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            ff_dim=512
        ).to(device)
    
    # Generate random input
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output represents probabilities for {output.shape[-1]} tokens")
    
    # Check that output makes sense
    expected_shape = (batch_size, seq_len, vocab_size)
    assert output.shape == expected_shape, f"Output shape {output.shape} doesn't match expected {expected_shape}"
    
    print("Model test passed!")
    return model