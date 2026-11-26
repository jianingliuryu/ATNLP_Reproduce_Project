from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

VOCAB_SIZE = 8000
TOKENIZER_PREFIX = "babylm_bpe"

ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small")
texts = ds["train"]["text"]

tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    texts,
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["<pad>", "<|endoftext|>"],
)
tokenizer.save_model(".", TOKENIZER_PREFIX)

print(f"Saved tokenizer with vocab size {tokenizer.get_vocab_size()}")