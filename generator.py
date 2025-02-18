"""
Text generation functions for Mini-GPT
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    repetition_penalty=1.0,
    return_attention_weights=False
):
    """
    Generate text from a prompt.
    
    Args:
        model: The trained Mini-GPT model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Text prompt to start generation
        max_length: Maximum number of tokens to generate
        temperature: Controls randomness (lower = more deterministic)
        top_k: Only sample from the top k most likely tokens
        top_p: Only sample from tokens with cumulative probability < top_p
        do_sample: If False, use greedy decoding instead of sampling
        repetition_penalty: Penalize repeated tokens
        return_attention_weights: Whether to return attention weights
    
    Returns:
        Generated text string (and optionally attention weights)
    """
    model.eval()
    
    # Encode prompt
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    generated = input_ids.clone()
    attention_weights = []
    
    # Set up for generation
    with torch.no_grad():
        for _ in tqdm(range(max_length), desc="Generating"):
            # Get model output
            outputs = model(generated)
            
            # Store attention weights if requested
            if return_attention_weights:
                weights = model.get_attention_weights()
                if weights:
                    attention_weights.append(weights)
            
            # Get logits for the next token only
            next_token_logits = outputs[:, -1, :].squeeze()
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    next_token_logits[token_id] /= repetition_penalty
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Filter out unlikely tokens (top-k)
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Filter by nucleus sampling (top-p)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Add to generated sequence
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated[0].tolist())
    
    if return_attention_weights:
        return generated_text, attention_weights
    return generated_text

def batch_generate(
    model, 
    tokenizer, 
    prompts, 
    max_length=50,
    **kwargs
):
    """Generate text for multiple prompts."""
    results = []
    for prompt in prompts:
        result = generate_text(
            model,
            tokenizer,
            prompt,
            max_length=max_length,
            **kwargs
        )
        results.append(result)
    return results

def interactive_generation(model, tokenizer, **kwargs):
    """Interactive text generation loop."""
    print("-" * 50)
    print("Mini-GPT Interactive Generation")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        prompt = input("\nEnter a prompt: ")
        if prompt.lower() == 'exit':
            break
        
        try:
            generated = generate_text(model, tokenizer, prompt, **kwargs)
            print("\nGenerated text:")
            print("-" * 50)
            print(generated)
            print("-" * 50)
        except Exception as e:
            print(f"Error during generation: {e}")

def visualize_generation_attention(
    model, 
    tokenizer, 
    prompt, 
    generated_text=None,
    layer_idx=None,
    head_idx=None
):
    """
    Visualize attention patterns during text generation.
    
    Args:
        model: The Mini-GPT model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt text
        generated_text: Optional pre-generated text (will generate if not provided)
        layer_idx: Specific layer to visualize (None = all layers)
        head_idx: Specific attention head to visualize (None = all heads)
    """
    # Generate text if not provided
    if generated_text is None:
        generated_text, attention_weights = generate_text(
            model, 
            tokenizer, 
            prompt,
            max_length=20,
            return_attention_weights=True
        )
    else:
        # Encode the text to get attention weights
        input_ids = torch.tensor(tokenizer.encode(generated_text)).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            _ = model(input_ids)
            attention_weights = [model.get_attention_weights()]
    
    if not attention_weights:
        print("No attention weights available for visualization")
        return
    
    # Get last step's attention weights
    attn_weights = attention_weights[-1]
    
    # Get tokens
    tokens = tokenizer.encode(generated_text)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    
    # Determine which layers and heads to plot
    num_layers = len(attn_weights)
    if layer_idx is None:
        layers_to_plot = range(num_layers)
    else:
        layers_to_plot = [layer_idx]
    
    num_heads = attn_weights[0].size(1)
    if head_idx is None:
        heads_to_plot = range(num_heads)
    else:
        heads_to_plot = [head_idx]
    
    # Create the plot
    fig_size = (3 * len(heads_to_plot), 3 * len(layers_to_plot))
    fig, axes = plt.subplots(
        len(layers_to_plot), 
        len(heads_to_plot),
        figsize=fig_size,
        squeeze=False
    )
    
    # Plot each layer and head
    for i, layer in enumerate(layers_to_plot):
        for j, head in enumerate(heads_to_plot):
            if layer < len(attn_weights) and head < num_heads:
                ax = axes[i, j]
                
                # Get attention matrix for this layer and head
                attn = attn_weights[layer][0, head].cpu().numpy()
                
                # Limit to actual sequence length
                seq_len = min(len(tokens), attn.shape[0])
                attn = attn[:seq_len, :seq_len]
                
                # Plot attention weights
                im = ax.imshow(attn, cmap='viridis')
                
                # Add labels
                if i == len(layers_to_plot) - 1:
                    # Only add x-labels on bottom row
                    step = max(1, len(token_strs) // 10)
                    ax.set_xticks(range(0, len(token_strs[:seq_len]), step))
                    ax.set_xticklabels(token_strs[:seq_len:step], rotation=90)
                else:
                    ax.set_xticks([])
                
                if j == 0:
                    # Only add y-labels on leftmost column
                    step = max(1, len(token_strs) // 10)
                    ax.set_yticks(range(0, len(token_strs[:seq_len]), step))
                    ax.set_yticklabels(token_strs[:seq_len:step])
                else:
                    ax.set_yticks([])
                    
                ax.set_title(f"Layer {layer}, Head {head}")
    
    plt.tight_layout()
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig('attention_visualization.png')
    plt.show()
    
    print(f"Generated text:\n{generated_text}")
    return attention_weights

def complete_text(model, tokenizer, prompt, max_length=50, **kwargs):
    """Complete a given text prompt."""
    return generate_text(model, tokenizer, prompt, max_length, **kwargs)

def answer_question(model, tokenizer, question, **kwargs):
    """Generate an answer to a question."""
    prompt = f"Q: {question}\nA:"
    return generate_text(model, tokenizer, prompt, **kwargs)

def generate_story(model, tokenizer, theme, length=200, **kwargs):
    """Generate a short story based on a theme."""
    prompt = f"Write a short story about {theme}. Once upon a time,"
    return generate_text(model, tokenizer, prompt, max_length=length, **kwargs)