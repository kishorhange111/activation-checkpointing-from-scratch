"""A GPT written as a chain of layers x_0 → x_1 → … → x_L, where x_L is the loss.

Checkpointing schedules work on chains: layer l maps x_{l-1} to x_l, and nothing else
flows between layers. Layer 1 embeds the token ids x_0, layers 2..L-1 are transformer
blocks, and layer L applies the head and returns the scalar loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab: int = 256
    ctx: int = 64
    d: int = 64
    blocks: int = 4
    heads: int = 4
    dropout: float = 0.0


class Embed(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, idx):
        pos = torch.arange(idx.shape[1], device=idx.device)
        return self.drop(self.tok(idx) + self.pos(pos))


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.h = c.heads
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv, self.proj = nn.Linear(c.d, 3 * c.d), nn.Linear(c.d, c.d)
        self.up, self.down = nn.Linear(c.d, 4 * c.d), nn.Linear(4 * c.d, c.d)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).unbind(2)
        a = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.down(F.gelu(self.up(self.ln2(x)))))


class Head(nn.Module):
    """Final norm, logits and the loss against fixed targets."""

    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln = nn.LayerNorm(c.d)
        self.out = nn.Linear(c.d, c.vocab, bias=False)
        self.targets = None

    def forward(self, x):
        logits = self.out(self.ln(x))
        return F.cross_entropy(logits.flatten(0, 1), self.targets.flatten())


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.layers = nn.ModuleList([Embed(c), *[Block(c) for _ in range(c.blocks)], Head(c)])

    @property
    def L(self):
        return len(self.layers)

    def set_targets(self, targets):
        self.layers[-1].targets = targets

    def forward(self, idx, targets):
        self.set_targets(targets)
        x = idx
        for f in self.layers:
            x = f(x)
        return x
