import math # for rounding up
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


def load_texts(): # returns the text column of the train split
    ds = load_dataset("BabyLM-community/BabyLM-2026-Strict-Small")
    return ds["train"]["text"]


def get_tokenizer(): # build tokenizer from saved files
    return ByteLevelBPETokenizer(
        TOKENIZER_PREFIX + "-vocab.json",
        TOKENIZER_PREFIX + "-merges.txt",
    ) # run from repo's root!


# encoding takes a few seconds
def get_token_stream(tokenizer):
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    texts = load_texts()
    encodings = tokenizer.encode_batch(texts) # one Encoding object per sentence

    all_ids = [] # one long list
    for enc in encodings:
        for token_id in enc.ids:
            all_ids.append(token_id)
        all_ids.append(eos_id)

    return torch.tensor(all_ids, dtype=torch.long) # convert to 1D tensor


def make_chunks(ids, block_size, pad_id):
    # stream without its last token and without its first token
    inputs = ids[:-1]
    targets = ids[1:]

    num_targets = len(targets)
    num_chunks = math.ceil(num_targets / block_size) # round up so leftover tokens gets its own chunk
    n_pad = num_chunks * block_size - num_targets

    inputs = torch.cat([inputs, torch.full((n_pad,), pad_id)])
    targets = torch.cat([targets, torch.full((n_pad,), IGNORE)]) # loss skips padded targets

    # from 1D tensor to a table with one row per chunk
    x_all = inputs.view(num_chunks, block_size)
    y_all = targets.view(num_chunks, block_size)
    return x_all, y_all


# build and return a function
def make_collate_fn(pad_id):
    def collate_fn(batch):
        all_x = []
        all_y = []
        # a batch is 32 (x, y) pairs
        for x, y in batch:
            all_x.append(x)
            all_y.append(y)

        # transforming 32 tensors of shape [256] to one tensor of shape [32, 256]
        x = torch.stack(all_x)
        y = torch.stack(all_y)
        # False for padding
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

    n_val = int(VAL_FRACTION * len(full_ds)) # 5% for validation
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED), # identical split for every run
    )


    collate_fn = make_collate_fn(pad_id)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn) # reshuffle every epoch for training
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return train_dl, val_dl, tokenizer


if __name__ == "__main__":
    train_dl, val_dl, tokenizer = get_dataloaders()
    x, y, mask = next(iter(train_dl))
    # sanity checks
    print(x.shape, y.shape, mask.shape)
    print(torch.equal(x[0, 1:], y[0, :-1]))
    print(x.max().item() < VOCAB_SIZE)