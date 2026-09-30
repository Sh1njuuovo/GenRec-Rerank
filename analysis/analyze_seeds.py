"""Seed-level analysis of the RQ-KMeans versus prebuilt SID comparison.

`compare_item_level.py` handles a single pair of runs. This script pools three
training seeds per SID assignment and reports:

  * the per-seed paired bootstrap over test samples
  * the mean difference across seeds with a two-stage bootstrap, which resamples
    seeds first and then samples inside each seed
  * the same numbers restricted to targets whose SID is unique under both
    assignments, so shared SIDs cannot contribute

Every run is scored under the strict item-level rule: a prediction counts only
when the emitted SID decodes to exactly one item and that item is the target.

Usage:
    python analyze_seeds.py \
        --prebuilt-results r1.json r2.json r3.json \
        --kmeans-results   k1.json k2.json k3.json \
        --prebuilt-index p.index.json --kmeans-index k.index.json \
        --prebuilt-test p_test.csv --kmeans-test k_test.csv \
        --labels 42 43 44
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
    table = collections.defaultdict(list)
    for item_id, tokens in raw.items():
        table["".join(tokens)].append(item_id)
    return table


def strict_hits(rows, predictions, sid_to_items, topk):
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


def unique_flags(rows, sid_to_items, item_to_sid):
    flags = []
    for row in rows:
        sid = item_to_sid.get(row["item_id"].strip())
        flags.append(sid is not None and len(sid_to_items[sid]) == 1)
    return flags


def paired_bootstrap(a, b, rounds, seed=42):
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(rounds):
        sample = [rng.randrange(n) for _ in range(n)]
        diffs.append(
            sum(b[i] for i in sample) / n - sum(a[i] for i in sample) / n
        )
    diffs.sort()
    return diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds) - 1]


def mean(values):
    return sum(values) / len(values)


def sd(values):
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def t_two_sided_p(t, df):
    """Two-sided p-value, closed form for df = 1 and 2, else a normal tail."""
    if df == 1:
        return 1.0 - (2.0 / math.pi) * math.atan(abs(t))
    if df == 2:
        return 1.0 - abs(t) / math.sqrt(t * t + 2.0)
    return math.erfc(abs(t) / math.sqrt(2.0))


def two_stage_bootstrap(hits_a, hits_b, rounds, seed=7):
    """Resample seeds, then samples inside each seed."""
    rng = random.Random(seed)
    n_seeds = len(hits_a)
    out = []
    for _ in range(rounds):
        picked = [rng.randrange(n_seeds) for _ in range(n_seeds)]
        total = 0.0
        for s in picked:
            a, b = hits_a[s], hits_b[s]
            n = len(a)
            idx = [rng.randrange(n) for _ in range(n)]
            total += sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n
        out.append(total / n_seeds)
    out.sort()
    return out[int(0.025 * rounds)], out[int(0.975 * rounds) - 1]


def subset_mean(values, keep):
    picked = [v for v, k in zip(values, keep) if k]
    return mean(picked), len(picked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prebuilt-results", nargs="+", required=True)
    parser.add_argument("--kmeans-results", nargs="+", required=True)
    parser.add_argument("--prebuilt-index", required=True)
    parser.add_argument("--kmeans-index", required=True)
    parser.add_argument("--prebuilt-test", required=True)
    parser.add_argument("--kmeans-test", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    rows_p = list(csv.DictReader(open(args.prebuilt_test, newline="", encoding="utf-8")))
    rows_k = list(csv.DictReader(open(args.kmeans_test, newline="", encoding="utf-8")))
    assert len(rows_p) == len(rows_k), "test splits differ in length"

    order_p = [r["item_id"].strip() for r in rows_p]
    order_k = [r["item_id"].strip() for r in rows_k]
    if order_p != order_k:
        raise SystemExit("test splits are not in the same sample order")
    rows = rows_p
    n = len(rows)

    table_p = decode_table(args.prebuilt_index)
    table_k = decode_table(args.kmeans_index)
    item_to_sid_p = {i: s for s, items in table_p.items() for i in items}
    item_to_sid_k = {i: s for s, items in table_k.items() for i in items}

    keep_p = unique_flags(rows, table_p, item_to_sid_p)
    keep_k = unique_flags(rows, table_k, item_to_sid_k)
    keep_both = [a and b for a, b in zip(keep_p, keep_k)]

    hits_p, hits_k = [], []
    for path in args.prebuilt_results:
        preds = [[normalize(p) for p in e.get("predict", [])] for e in load_result(path)]
        assert len(preds) == n
        hits_p.append(strict_hits(rows, preds, table_p, args.topk))
    for path in args.kmeans_results:
        preds = [[normalize(p) for p in e.get("predict", [])] for e in load_result(path)]
        assert len(preds) == n
        hits_k.append(strict_hits(rows, preds, table_k, args.topk))

    report = {
        "metric": f"item_unique_HR@{args.topk}",
        "rows": n,
        "test_split_order_verified": True,
        "shared_sid_items": {
            "prebuilt": sum(len(v) for v in table_p.values() if len(v) > 1),
            "kmeans": sum(len(v) for v in table_k.values() if len(v) > 1),
        },
        "unique_in_both_rows": sum(keep_both),
        "per_seed": [],
    }

    diffs = []
    for label, hp, hk in zip(args.labels, hits_p, hits_k):
        lo, hi = paired_bootstrap(hp, hk, args.bootstrap)
        diff = mean(hk) - mean(hp)
        diffs.append(diff)
        m_p, _ = subset_mean(hp, keep_both)
        m_k, n_clean = subset_mean(hk, keep_both)
        report["per_seed"].append(
            {
                "seed": label,
                "prebuilt": round(mean(hp), 6),
                "kmeans": round(mean(hk), 6),
                "diff": round(diff, 6),
                "ci95": [round(lo, 6), round(hi, 6)],
                "significant": not (lo <= 0 <= hi),
                "clean_subset": {
                    "rows": n_clean,
                    "prebuilt": round(m_p, 6),
                    "kmeans": round(m_k, 6),
                    "diff": round(m_k - m_p, 6),
                },
            }
        )

    mean_diff = mean(diffs)
    sd_diff = sd(diffs)
    t = mean_diff / (sd_diff / math.sqrt(len(diffs)))
    lo, hi = two_stage_bootstrap(hits_p, hits_k, args.bootstrap)
    per_seed_sign = all(d > 0 for d in diffs)

    hits_p_clean = [[v for v, k in zip(h, keep_both) if k] for h in hits_p]
    hits_k_clean = [[v for v, k in zip(h, keep_both) if k] for h in hits_k]
    clean_diffs = [mean(k) - mean(p) for p, k in zip(hits_p_clean, hits_k_clean)]
    clean_mean = mean(clean_diffs)
    clean_t = clean_mean / (sd(clean_diffs) / math.sqrt(len(clean_diffs)))
    clean_lo, clean_hi = two_stage_bootstrap(hits_p_clean, hits_k_clean, args.bootstrap)

    report["seed_level"] = {
        "seeds": len(diffs),
        "mean_diff": round(mean_diff, 6),
        "sd_diff": round(sd_diff, 6),
        "min_diff": round(min(diffs), 6),
        "max_diff": round(max(diffs), 6),
        "all_seeds_same_sign": per_seed_sign,
        "two_stage_ci95": [round(lo, 6), round(hi, 6)],
        "two_stage_significant": not (lo <= 0 <= hi),
        "t_vs_zero": round(t, 3),
        "df": len(diffs) - 1,
        "p_two_sided": round(t_two_sided_p(t, len(diffs) - 1), 4),
    }
    report["seed_level_clean_subset"] = {
        "rows": sum(keep_both),
        "mean_diff": round(clean_mean, 6),
        "two_stage_ci95": [round(clean_lo, 6), round(clean_hi, 6)],
        "two_stage_significant": not (clean_lo <= 0 <= clean_hi),
        "t_vs_zero": round(clean_t, 3),
        "p_two_sided": round(t_two_sided_p(clean_t, len(clean_diffs) - 1), 4),
    }

    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
