# Mini-GPT: Build Your Own Language Model from Scratch

This repository contains a complete implementation of a small-scale transformer-based language model, inspired by GPT architecture. It's designed for educational purposes to help you understand how modern language models work by building one from scratch.

## Project Structure

'''
mini-gpt/
│
├── dataset.py              # Dataset preparation and loading utilities
├── generator.py            # Text generation functionality 
├── main.py                 # Main script and CLI interface
├── model.py                # Model architecture implementation
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
├── trainer.py              # Training loop and optimization
└── utils.py                # Helper functions and utilities
'''

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mini-gpt.git
cd mini-gpt
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
python main.py --mode train \
  --num_layers 4 \
  --embed_dim 256 \
  --epochs 3 \
  --batch_size 16 \
  --dataset wikitext-2
```

### Generating Text

```bash
python main.py --mode generate \
  --model_path checkpoints/best_model.pt \
  --prompt "Once upon a time" \
  --max_length 100 \
  --temperature 0.8
```

### Interactive Generation

```bash
python main.py --mode interactive \
  --model_path checkpoints/best_model.pt \
  --temperature 0.8
```

### Testing the Implementation

```bash
python main.py --mode test
```

## Model Configuration

The Mini-GPT model can be configured with the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| vocab_size | Size of vocabulary | 50257 |
| max_seq_length | Maximum sequence length | 128 |
| embed_dim | Embedding dimension | 256 |
| num_heads | Number of attention heads | 8 |
| num_layers | Number of transformer layers | 4 |
| ff_dim | Feed-forward dimension | 1024 |
| dropout | Dropout probability | 0.1 |

## Examples

### Training Progress Visualization

![Training Progress](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*YBtBGpgzW_JGMTThvtZ3eg.png)

### Attention Visualization

![Attention Visualization](https://jalammar.github.io/images/gpt2/gpt2-attention-pattern.png)

### Sample Generated Text

```
Prompt: "Once upon a time,"

Generated: "Once upon a time, in the kingdom of Eldoria, there lived a young apprentice named Lukas. He spent his days studying the ancient arts of magic under the tutelage of the renowned wizard Alaric.

One fateful morning, while practicing a particularly difficult spell, Lukas accidentally opened a portal to another dimension. Through the swirling vortex, he glimpsed strange creatures and landscapes unlike anything in his world..."
```

## How It Works

Mini-GPT is built on the transformer architecture with these key components:

1. **Token Embeddings**: Convert words to vectors
2. **Positional Encodings**: Add location information
3. **Self-Attention**: Understand relationships between words
4. **Feed-Forward Networks**: Process information
5. **Layer Normalization**: Stabilize training

For more details, check out our blog series:
- [Part 1: Understanding Our Mini-GPT - The Blueprint](https://link-to-part1)
- [Part 2: Setting Up the Foundation - Environment and Data](https://link-to-part2)
- [Part 3: Building the Brain - Core Components](https://link-to-part3)
- [Part 4: Training and Generation - Bringing It to Life](https://link-to-part4)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project was inspired by [Andrej Karpathy's minGPT](https://github.com/karpathy/minGPT)
- Thanks to the [Hugging Face team](https://huggingface.co/) for their transformers library
- Architecture diagrams adapted from [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
