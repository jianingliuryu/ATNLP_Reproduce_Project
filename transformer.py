import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.linear_qkv = nn.Linear(emb_dim, 3 * emb_dim)
        self.linear_out = nn.Linear(emb_dim, emb_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, _ = x.shape


        q, k, v = self.linear_qkv(x).split(self.emb_dim, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)


        key_out = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        allowed = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        allowed = allowed.view(1, 1, T, T)
        if mask is not None:
            allowed = allowed & mask.bool()
        key_out = key_out.masked_fill(~allowed, torch.finfo(key_out.dtype).min)

        attn = F.softmax(key_out, dim=-1)
        attn = self.attn_dropout(attn)


        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.emb_dim)
        out = self.out_dropout(self.linear_out(out))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout):
        super().__init__()

        # Multi-head attention built from the given arguments.
        # Must be called `mha`: initialize_gpt2_weights accesses layer.mha.linear_out
        self.mha = MultiHeadAttention(emb_dim, num_heads, dropout)

        # Two separate LayerNorms with eps=1e-5 (each has its own learnable weight and bias)
        self.ln1 = nn.LayerNorm(emb_dim, eps=1e-5)   # applied before attention
        self.ln2 = nn.LayerNorm(emb_dim, eps=1e-5)   # applied before the FFN

        # GPT-2 uses a hidden dimension of 4 * emb_dim in the feedforward network
        forward_dim = 4 * emb_dim

        # Feedforward network. Must be called `ffn` and keep this order:
        # initialize_gpt2_weights accesses layer.ffn[2] (the second Linear layer)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),   # [0] project up to forward_dim
            nn.GELU(approximate="tanh"),       # [1] GELU variant used by GPT-2
            nn.Linear(forward_dim, emb_dim),   # [2] project back down to emb_dim
            nn.Dropout(dropout),               # [3] dropout with the given rate
        )

    def forward(self, x, mask=None):
        # normalize inside each residual branch
        # Step 1 + 2: LayerNorm -> attention (with mask), then add the original input back
        x = x + self.mha(self.ln1(x), mask)

        # Step 3 + 4: LayerNorm -> FFN, then add the result of step 2 back
        x = x + self.ffn(self.ln2(x))

        return x


def initialize_gpt2_weights(model):
    def initialize_module(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    model.apply(initialize_module)

    # two residual branches per transformer block
    residual_std = 0.02 / math.sqrt(2 * len(model.layers))
    for layer in model.layers:
        nn.init.normal_(layer.mha.linear_out.weight, mean=0.0, std=residual_std)
        nn.init.normal_(layer.ffn[2].weight, mean=0.0, std=residual_std)


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=768,
        num_layers=12,
        num_heads=12,
        dropout=0.1,
        max_len=1024,
    ):
        super().__init__()

        self.max_len = max_len

        # token and learned absolute position embeddings
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.dropout = nn.Dropout(dropout)

        # stack of transformer blocks (must be named `layers`)
        self.layers = nn.ModuleList(
            [TransformerBlock(emb_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # final layer norm and output projection (no bias)
        self.ln_f = nn.LayerNorm(emb_dim, eps=1e-5)
        self.linear_out = nn.Linear(emb_dim, vocab_size, bias=False)

        # GPT-2 weight initialization, then weight tying
        initialize_gpt2_weights(self)
        self.linear_out.weight = self.token_emb.weight

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len)
        B, T = x.shape
        assert T <= self.max_len, f"sequence length {T} exceeds max_len {self.max_len}"

        # positions 0..T-1 on the same device as the input
        positions = torch.arange(T, device=x.device)

        # token + position embeddings, then dropout: (B, T, emb_dim)
        h = self.dropout(self.token_emb(x) + self.pos_emb(positions))

        # pass through all transformer blocks with the padding mask
        for layer in self.layers:
            h = layer(h, mask)

        # final layer norm and projection to vocabulary: (B, T, vocab_size)
        h = self.ln_f(h)
        logits = self.linear_out(h)

        return logits
