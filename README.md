# Read me
# ATNLP Reproduction Project: GPT-2 on BabyLM

Group project for Advanced Topics in NLP (2026/27). We implement a GPT-2 style
language model from scratch, pretrain it on the BabyLM Strict-Small dataset
(10M words), and evaluate it with training/validation loss and the BLiMP benchmark.

## Repository structure

| File | Description |
|---|---|
| `transformer.py` | GPT-2 model implementation (Task 1) |
| `atnlp26_27-transformer-question.ipynb` | Model development notebook with sanity checks (Task 1) |
| TODO | Tokenizer and data loader (Task 2) |
| TODO | Training and evaluation scripts (Task 3) |

## Requirements

- Python 3
- PyTorch

## Task 1: Model

`transformer.py` contains:

- `MultiHeadAttention`: causal multi-head self-attention with optional padding mask
- `TransformerBlock`: pre-LayerNorm transformer block (attention + feedforward network, with residual connections)
- `initialize_gpt2_weights`: GPT-2 weight initialization (provided by the course)
- `GPT2`: token + learned absolute position embeddings, a stack of transformer blocks,
  final LayerNorm, and an output layer with weight tying

### Usage

```python
from transformer import GPT2

model = GPT2(
    vocab_size=8000,
    emb_dim=512,
    num_layers=4,
    num_heads=8,
    dropout=0.1,
    max_len=256,
)
logits = model(x, mask)   # mask is optional
```

With these hyperparameters the model has about 16.8M parameters.

### Interface

| | Shape | Notes |
|---|---|---|
| Input `x` | `(batch_size, seq_len)` | Token ids (`torch.long`), `seq_len <= max_len` |
| `mask` (optional) | `(batch_size, 1, 1, seq_len)` | `1` = real token, `0` = padding. Use `None` if there is no padding |
| Output | `(batch_size, seq_len, vocab_size)` | Logits, no softmax applied |

The logits at position `t` predict the token at position `t + 1`, so training
targets must be shifted by one position.

### Sanity checks

Run all cells in `atnlp26_27-transformer-question.ipynb`:

1. **Shape check**: output shape is `(4, 80, 200)` for the small test model.
2. **Copy task**: after 15 epochs, validation token accuracy reaches 100%
   (final val loss ≈ 0.004).

## Task 2: Tokenizer and data

TODO

## Task 3: Experiments

TODO: how to train the model and reproduce Experiment 1 (training/validation loss)
and Experiment 2 (BLiMP evaluation).
