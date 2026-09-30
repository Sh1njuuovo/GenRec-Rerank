"""Extract small fixed subsets from the Industrial_and_Scientific splits.

Reads the original CSVs as text and writes row subsets without touching the
originals. Uses only the standard library so it can run on the local Mac.

Usage:
    python3 make_subsets.py --src-root <MiniOneRec/data/Amazon> \
        --out-dir <run-dir>/subsets \
        --train-n 512 --valid-n 256 --test-n 256 --seed 42
"""

import argparse
import csv
import os
import random

CATEGORY = "Industrial_and_Scientific"
SPLITS = {
    "train": ("train", "train-n"),
    "valid": ("valid", "valid-n"),
    "test": ("test", "test-n"),
}


def find_source(src_root, split):
    split_dir = os.path.join(src_root, split)
    for name in sorted(os.listdir(split_dir)):
        if name.startswith(CATEGORY) and name.endswith(".csv"):
            return os.path.join(split_dir, name)
    raise FileNotFoundError(f"no {CATEGORY} csv under {split_dir}")


def subset_rows(src_path, out_path, n, seed):
    with open(src_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]

    rng = random.Random(seed)
    if n < len(rows):
        picked = sorted(rng.sample(range(len(rows)), n))
        rows = [rows[i] for i in picked]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(rows)

    return len(rows), len(header)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    for _, (_, flag) in SPLITS.items():
        parser.add_argument(f"--{flag}", dest=flag.replace("-", "_"), type=int)
    args = parser.parse_args()

    for split, (_, flag) in SPLITS.items():
        n = getattr(args, flag.replace("-", "_"))
        if n is None:
            continue
        src_path = find_source(args.src_root, split)
        out_path = os.path.join(args.out_dir, f"{CATEGORY}_{split}_{n}.csv")
        kept, cols = subset_rows(src_path, out_path, n, args.seed)
        print(f"{split}: {kept} rows x {cols} cols -> {out_path}")


if __name__ == "__main__":
    main()
