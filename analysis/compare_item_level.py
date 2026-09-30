"""Paired comparison of two result files under the strict item-level metric.

Companion to `item_level_eval.py`. A prediction counts only when the emitted SID
decodes to exactly one item and that item is the target, so assignments with
shared SIDs gain nothing.

Usage:
    python compare_item_level.py --result-a <a.json> --result-b <b.json> \
        --index-a <a.index.json> --index-b <b.index.json> \
        --test-csv <test.csv> --label-a kmeans --label-b prebuilt --topk 10
"""

import argparse
import collections
import csv
import json
import math
import random


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


def decode_table(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    sid_to_items = collections.defaultdict(list)
    for item_id, tokens in raw.items():
        sid_to_items["".join(tokens)].append(item_id)
    return sid_to_items


def strict_hits(rows, predictions, sid_to_items, topk):
    """1.0 when the first uniquely-decoding correct prediction appears in top-k."""
    out = []
    for row, preds in zip(rows, predictions):
        target = row["item_id"].strip()
        hit = 0.0
        for position, pred in enumerate(preds):
            if position >= topk:
                break
            items = sid_to_items.get(normalize(pred))
            if items and len(items) == 1 and items[0] == target:
                hit = 1.0
                break
        out.append(hit)
    return out


def paired_bootstrap(a, b, rounds, seed=42):
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(rounds):
        sample = [rng.randrange(n) for _ in range(n)]
        da = sum(a[i] for i in sample) / n
        db = sum(b[i] for i in sample) / n
        diffs.append(db - da)
    diffs.sort()
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-a", required=True)
    parser.add_argument("--result-b", required=True)
    parser.add_argument("--index-a", required=True)
    parser.add_argument("--index-b", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.test_csv, newline="", encoding="utf-8")))
    table_a = decode_table(args.index_a)
    table_b = decode_table(args.index_b)
    preds_a = [[normalize(p) for p in e.get("predict", [])] for e in load_result(args.result_a)]
    preds_b = [[normalize(p) for p in e.get("predict", [])] for e in load_result(args.result_b)]
    assert len(rows) == len(preds_a) == len(preds_b)

    hits_a = strict_hits(rows, preds_a, table_a, args.topk)
    hits_b = strict_hits(rows, preds_b, table_b, args.topk)
    n = len(rows)
    hr_a = sum(hits_a) / n
    hr_b = sum(hits_b) / n
    lo, hi = paired_bootstrap(hits_a, hits_b, args.bootstrap)

    both = sum(1 for x, y in zip(hits_a, hits_b) if x == 1 and y == 1)
    only_a = sum(1 for x, y in zip(hits_a, hits_b) if x == 1 and y == 0)
    only_b = sum(1 for x, y in zip(hits_a, hits_b) if x == 0 and y == 1)

    report = {
        "metric": f"item_unique_HR@{args.topk}",
        "rows": n,
        args.label_a: round(hr_a, 6),
        args.label_b: round(hr_b, 6),
        "diff": round(hr_b - hr_a, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "significant": not (lo <= 0 <= hi),
        "per_sample": {
            f"{args.label_a}_only": only_a,
            f"{args.label_b}_only": only_b,
            "both": both,
            "net_change": only_b - only_a,
        },
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
