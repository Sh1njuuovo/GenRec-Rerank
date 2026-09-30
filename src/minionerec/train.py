"""Train a compact semantic-ID sequence recommender on JSONL examples."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import SIDRecommender, build_vocabulary, encode_examples


def load_examples(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row["history"], list) or not row["history"]:
                    raise ValueError("history must be a nonempty list")
                rows.append({"history": row["history"], "target": row["target"]})
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"invalid example on line {number}: {exc}") from exc
    if not rows:
        raise ValueError("input contains no examples")
    return rows


def train_file(path, epochs=5, dim=64, heads=4, batch_size=128, lr=1e-3, seed=42, output=None):
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    random.seed(seed)
    rows = load_examples(path)
    vocabulary = build_vocabulary(rows)
    history, target = encode_examples(rows, vocabulary)
    model = SIDRecommender(len(vocabulary), dim=dim, heads=heads)
    loader = DataLoader(TensorDataset(history, target), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    report = {"examples": len(rows), "vocabulary_size": len(vocabulary), "epochs": []}
    for epoch in range(epochs):
        model.train()
        weighted_loss = 0.0
        for batch_history, batch_target in loader:
            optimizer.zero_grad()
            loss = model.loss(batch_history, batch_target)
            loss.backward()
            optimizer.step()
            weighted_loss += loss.item() * len(batch_target)
        report["epochs"].append({"epoch": epoch + 1, "training_loss": weighted_loss / len(rows)})
    if output is not None:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "dim": dim, "heads": heads}, destination / "model.pt")
        (destination / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2) + "\n", encoding="utf-8")
        (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(train_file(args.examples, args.epochs, args.dim, args.heads, args.batch_size, output=args.output), indent=2))


if __name__ == "__main__":
    main()
