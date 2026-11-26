import os
import math
import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader

# laid out in the assignment
VOCAB_SIZE = 8000
BLOCK = 256
BATCH_SIZE = 32
SEED = 42
VAL_FRACTION = 0.05
IGNORE = -100

# we must use the same tokenizer the model was trained on in experiment 2
TOKENIZER_PREFIX = "babylm_bpe"


def load_texts():
    ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small")
    return ds["train"]["text"]


def get_tokenizer():
    return ByteLevelBPETokenizer(
        TOKENIZER_PREFIX + "-vocab.json",
        TOKENIZER_PREFIX + "-merges.txt",
    )


# encoding takes a few seconds
def get_token_stream(tokenizer):
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    texts = load_texts()
    encodings = tokenizer.encode_batch(texts)

    all_ids = []
    for enc in encodings:
        for token_id in enc.ids:
            all_ids.append(token_id)
        all_ids.append(eos_id)

    return torch.tensor(all_ids, dtype=torch.long)


def make_chunks(ids, block_size, pad_id):
    inputs = ids[:-1]
    targets = ids[1:]

    num_targets = len(targets)
    num_chunks = math.ceil(num_targets / block_size)
    n_pad = num_chunks * block_size - num_targets

    inputs = torch.cat([inputs, torch.full((n_pad,), pad_id)])
    targets = torch.cat([targets, torch.full((n_pad,), IGNORE)])

    x_all = inputs.view(num_chunks, block_size)
    y_all = targets.view(num_chunks, block_size)
    return x_all, y_all


def make_collate_fn(pad_id):
    def collate_fn(batch):
        all_x = []
        all_y = []
        for x, y in batch:
            all_x.append(x)
            all_y.append(y)

        x = torch.stack(all_x)
        y = torch.stack(all_y)
        mask = (x != pad_id).unsqueeze(1).unsqueeze(2)
        return x, y, mask
    return collate_fn


def get_dataloaders():
    torch.manual_seed(SEED)

    tokenizer = get_tokenizer()
    assert tokenizer.get_vocab_size() == VOCAB_SIZE, "tokenizer/model vocab mismatch"
    pad_id = tokenizer.token_to_id("<pad>")

    ids = get_token_stream(tokenizer)
    x_all, y_all = make_chunks(ids, BLOCK, pad_id)
    full_ds = TensorDataset(x_all, y_all)

    n_val = int(VAL_FRACTION * len(full_ds))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    collate_fn = make_collate_fn(pad_id)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)