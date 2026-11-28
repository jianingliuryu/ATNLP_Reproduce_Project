from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

# laid out in the assignment
VOCAB_SIZE = 8000
# connects to data_loader.py
TOKENIZER_PREFIX = "babylm_bpe"

ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small")
texts = ds["train"]["text"]

tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    texts,
    vocab_size=VOCAB_SIZE,
    min_frequency=2, # pairs must be seen at least twice before merging
    special_tokens=["<pad>", "<|endoftext|>"],
)

# writes to current folder
tokenizer.save_model(".", TOKENIZER_PREFIX)

print(f"Saved tokenizer with vocab size {tokenizer.get_vocab_size()}")