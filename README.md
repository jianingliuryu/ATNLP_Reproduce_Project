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
| `train_tokenizer.py` | Trains the byte-level BPE tokenizer (Task 2) |
| `data_loader.py` | Encodes BabyLM, builds chunks and training and validation DataLoaders (Task 2) |
| `babylm_bpe-vocab.json`, `babylm_bpe-merges.txt` | The trained tokenizer (used for training and BLiMP) |
| `tokenizer and data loader workbook + sanity checks.ipynb` | Tokenizer and data loader development notebook with sanity checks (Task 2) |
| `tokenizer and data loader workbook draft.py` | Process of converting the tokenizer and data loader workbook to .py files (Task 2) |
| TODO | Training and evaluation scripts (Task 3) |

## Requirements

- Python 3
- PyTorch
- `datasets` and `tokenizers` from Hugging Face

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

**All scripts must be run from the repo's root folder** to ensure that Python finds and uses our trained tokenizer.

### Tokenizer

`train_tokenizer.py` trains a byte-level BPE tokenizer from scratch. It is trained on the BabyLM Strict-Small training text with a vocabulary size of 8000. Also, it uses the special tokens `<pad>` (ID 0) and `<|endoftext|>` (ID 1). 

It writes two files:
- `babylm_bpe-vocab.json`
- `babylm_bpe-merges.txt`

**It is important that these are kept together in the repository**. The tokenizer would produce incorrect IDs and you e.g. use the vocab from one run of training and merges from another.

**Do not run `train_tokenizer.py`.** 
The files are already committed to the repository, so you do not have to run the file. Also, you may produce a different vocabulary if you retrain. In Experiment 2, we must use the precise tokenizer our model was trained on. Therefore, retraining the tokenizer means you also must retrain the model.

### Data loader

`data_loader.py` turns the dataset into batches for training:

1. Loads the BabyLM Strict-Small training split from Hugging Face.
2. Encodes every row with the tokenizer and appends `<|endoftext|>` after each row, giving one long stream of token ids.
3. Cuts the stream into chunks of 256 tokens (meeting the `MAX_SEQUENCE_LENGTH` requirement). Long inputs are split across chunks instead of truncated, so every token is used. The final chunk is padded and its padded targets are set to -100 so the loss ignores them.
4. Randomly holds out 5% of the chunks for validation (seed 42).
5. Wraps the chunks in DataLoaders with batch size 32.

### Usage

```python
from data_loader import get_dataloaders

train_dl, val_dl, tokenizer = get_dataloaders()

for inputs, targets, mask in train_dl:
    logits = model(inputs, mask)
    loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
```

Use `nn.CrossEntropyLoss(ignore_index=-100)` so padded targets are skipped.

| | Shape | Notes |
|---|---|---|
| `inputs` | `(32, 256)` | Token ids |
| `targets` | `(32, 256)` | `inputs` shifted by one position, -100 for padding |
| `mask` | `(32, 1, 1, 256)` | `True` = real token, `False` = padding |

Each call to `get_dataloaders()` re-encodes the corpus, which takes a few
seconds, so call it once per script.

### Quick check

```
python data_loader.py
```

This prints the batch shapes, checks that targets are the inputs shifted by
one, and checks that all ids fit the vocabulary.


## Task 3: Experiments

TODO: how to train the model and reproduce Experiment 1 (training/validation loss)
and Experiment 2 (BLiMP evaluation).
