"""A small, independent next-item recommender over three-level semantic IDs."""

import re

import torch
from torch import nn
from torch.nn import functional as F


def _parts(sid):
    parts = re.findall(r"<[abc]_\d+>", sid) if sid.startswith("<") else sid.split("/")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError("each SID must have three nonempty levels")
    if sid.startswith("<") and "".join(parts) != sid:
        raise ValueError("invalid semantic ID")
    return parts


def build_vocabulary(rows):
    tokens = {part for row in rows for sid in [*row["history"], row["target"]] for part in _parts(sid)}
    return {token: index for index, token in enumerate(["<pad>", *sorted(tokens)])}


def encode_examples(rows, vocabulary, max_history=100):
    if not rows:
        raise ValueError("at least one example is required")
    width = min(max(len(row["history"]) for row in rows), max_history)
    if width == 0:
        raise ValueError("each example needs history")
    histories, targets = [], []
    for row in rows:
        history = row["history"][-width:]
        histories.append([[0, 0, 0]] * (width - len(history)) + [[vocabulary[p] for p in _parts(sid)] for sid in history])
        targets.append([vocabulary[p] for p in _parts(row["target"])])
    return torch.tensor(histories, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class SIDRecommender(nn.Module):
    def __init__(self, vocabulary_size, dim=64, heads=4, dropout=0.1, max_history=100):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.embedding = nn.Embedding(vocabulary_size, dim, padding_idx=0)
        self.position = nn.Embedding(max_history + 1, dim, padding_idx=0)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.heads = nn.ModuleList(nn.Linear(dim, vocabulary_size) for _ in range(3))
        self.max_history = max_history

    def forward(self, history):
        if history.ndim != 3 or history.size(-1) != 3 or history.size(1) > self.max_history:
            raise ValueError("history must be [batch, time, 3] within max_history")
        valid = history.ne(0).any(dim=-1)
        if not torch.all(valid.any(dim=1)):
            raise ValueError("each history needs at least one SID")
        positions = valid.long().cumsum(dim=1) * valid.long()
        x = self.embedding(history).sum(dim=2) / 3 + self.position(positions)
        encoded = self.encoder(x, src_key_padding_mask=~valid)
        last = valid.long().sum(dim=1) - 1
        # Left padding keeps the latest valid SID at the final position.
        last = torch.where(valid[:, -1], torch.full_like(last, history.size(1) - 1), last)
        context = encoded[torch.arange(history.size(0), device=history.device), last]
        return torch.stack([head(context) for head in self.heads], dim=1)

    def loss(self, history, target):
        logits = self(history)
        return sum(F.cross_entropy(logits[:, level, :], target[:, level]) for level in range(3)) / 3

    @torch.no_grad()
    def predict(self, history, valid_sids, vocabulary):
        """Choose only complete SIDs from the supplied catalog."""
        if not valid_sids:
            raise ValueError("valid_sids cannot be empty")
        ids = torch.tensor([[vocabulary[p] for p in _parts(sid)] for sid in valid_sids], device=history.device)
        logits = self(history)
        scores = logits[:, 0, ids[:, 0]] + logits[:, 1, ids[:, 1]] + logits[:, 2, ids[:, 2]]
        return ids[scores.argmax(dim=1)]
