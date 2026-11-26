from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import torch
import math
from torch.utils.data import TensorDataset, random_split, DataLoader

ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small")

VOCAB_SIZE = 8000 

texts = ds["train"]["text"]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train_from_iterator(
    texts,
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<|endoftext|>"],
)

eos_id = tokenizer.token_to_id("<|endoftext|>")
pad_id = tokenizer.token_to_id("<pad>")

encodings = tokenizer.encode_batch(texts)

all_ids = []

for enc in encodings:
    for token_id in enc.ids:
        all_ids.append(token_id)
    all_ids.append(eos_id)

ids = torch.tensor(all_ids, dtype=torch.long)

import math

IGNORE = -100

def make_chunks(ids, block_size, pad_id):
    inputs = ids[:-1]
    targets = ids[1:]

    num_targets = len(targets)
    num_chunks = math.ceil(num_targets / block_size)

    total_slots = num_chunks * block_size
    n_pad = total_slots - num_targets

    inputs = torch.cat([inputs, torch.full((n_pad,), pad_id)])
    targets = torch.cat([targets, torch.full((n_pad,), IGNORE)])

    x_all = inputs.view(num_chunks, block_size)
    y_all = targets.view(num_chunks, block_size)

    return x_all, y_all

BLOCK = 256

x_all, y_all = make_chunks(ids, BLOCK, pad_id)

full_ds = TensorDataset(x_all, y_all)

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

SEED = 42
BLOCK = 256
BATCH_SIZE = 32
VAL_FRACTION = 0.05

full_ds = TensorDataset(x_all, y_all)

n_val = int(VAL_FRACTION * len(full_ds))
n_train = len(full_ds) - n_val
train_ds, val_ds = random_split(
    full_ds, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED),
)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)