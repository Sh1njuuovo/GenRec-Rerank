"""Paired comparison of two evaluation result files.

The same test samples are used for both models, so a paired bootstrap on the
per-sample hit indicators is far more sensitive than comparing two aggregate
numbers. Reports the metric difference, its confidence interval, and how often
one model wins on a sample the other loses.

Usage:
    python compare_results.py --a <result_a.json> --b <result_b.json> \
        --info_file <info.txt> --topk 10 --label_a SFT --label_b RL
"""

import argparse
import json
import math
import random
import re

TOKEN_RE = re.compile(r"<[abc]_\d+>")


def load_result(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and data and isinstance(data[0], list):
        flat = []
        for shard in data:
            flat.extend(shard)
        data = flat
    return data


def normalize(text):
    return str(text).strip().strip('"').strip()


def target_of(entry):
    target = entry.get("output")
    if isinstance(target, list):
        target = target[0] if target else ""
    return normalize(target)


def rank_of(entry):
    """Zero-based rank of the target in the prediction list, or None."""
    target = target_of(entry)
    for position, pred in enumerate(entry.get("predict", [])):
        if normalize(pred) == target:
            return position
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--label_a", default="A")
    parser.add_argument("--label_b", default="B")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    data_a = load_result(args.a)
    data_b = load_result(args.b)
    if len(data_a) != len(data_b):
        raise SystemExit(f"sample count differs: {len(data_a)} vs {len(data_b)}")

    n = len(data_a)
    k = args.topk

    hit_a, hit_b, ndcg_a, ndcg_b = [], [], [], []
    win_a = win_b = both_hit = both_miss = 0

    for entry_a, entry_b in zip(data_a, data_b):
        ra, rb = rank_of(entry_a), rank_of(entry_b)
        ha = 1.0 if (ra is not None and ra < k) else 0.0
        hb = 1.0 if (rb is not None and rb < k) else 0.0
        hit_a.append(ha)
        hit_b.append(hb)
        ndcg_a.append((1.0 / math.log(ra + 2)) if (ra is not None and ra < k) else 0.0)
        ndcg_b.append((1.0 / math.log(rb + 2)) if (rb is not None and rb < k) else 0.0)

        if ha > hb:
            win_a += 1
        elif hb > ha:
            win_b += 1
        elif ha > 0:
            both_hit += 1
        else:
            both_miss += 1

    mean = lambda xs: sum(xs) / len(xs)

    def bootstrap_diff(values_a, values_b):
        rng = random.Random(args.seed)
        diffs = []
        idx = range(n)
        for _ in range(args.bootstrap):
            sample = [rng.randrange(n) for _ in idx]
            da = sum(values_a[i] for i in sample) / n
            db = sum(values_b[i] for i in sample) / n
            diffs.append(db - da)
        diffs.sort()
        lo = diffs[int(0.025 * len(diffs))]
        hi = diffs[int(0.975 * len(diffs)) - 1]
        return lo, hi

    hr_a, hr_b = mean(hit_a), mean(hit_b)
    ndcg_norm = 1.0 / math.log(2)
    nd_a, nd_b = mean(ndcg_a) / ndcg_norm, mean(ndcg_b) / ndcg_norm

    hr_lo, hr_hi = bootstrap_diff(hit_a, hit_b)
    nd_lo, nd_hi = bootstrap_diff([x / ndcg_norm for x in ndcg_a],
                                  [x / ndcg_norm for x in ndcg_b])

    report = {
        "samples": n,
        "topk": k,
        f"HR@{k}": {args.label_a: round(hr_a, 8), args.label_b: round(hr_b, 8),
                    "diff": round(hr_b - hr_a, 8),
                    "ci95": [round(hr_lo, 8), round(hr_hi, 8)],
                    "significant": not (hr_lo <= 0 <= hr_hi)},
        f"NDCG@{k}": {args.label_a: round(nd_a, 8), args.label_b: round(nd_b, 8),
                      "diff": round(nd_b - nd_a, 8),
                      "ci95": [round(nd_lo, 8), round(nd_hi, 8)],
                      "significant": not (nd_lo <= 0 <= nd_hi)},
        "per_sample": {
            f"{args.label_a}_only": win_a,
            f"{args.label_b}_only": win_b,
            "both_hit": both_hit,
            "both_miss": both_miss,
            "net_change": win_b - win_a,
        },
    }

    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
