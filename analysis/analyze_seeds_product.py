"""Seed-level comparison under the product-level rule.

Same statistics as `analyze_seeds.py`, but a prediction counts when the SID it
emits decodes to item ids that all carry the target's product title. Duplicate
catalog entries therefore stop being penalised, while a SID that merges
genuinely different products still gets no credit.

Usage:
    python analyze_seeds_product.py --prebuilt-results ... --kmeans-results ... \
        --prebuilt-index ... --kmeans-index ... --item-meta ... \
        --prebuilt-test ... --kmeans-test ... --labels 42 43 44
"""

import argparse
import collections
import csv
import json
import math
import random
import re


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


def product_key(title):
    text = str(title).lower()
    text = re.sub(r"[\s\u0000-\u001f]+", " ", text)
    return text.strip()


def build(path, keys):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    sid_to_items = collections.defaultdict(list)
    for item_id, tokens in raw.items():
        sid_to_items["".join(tokens)].append(item_id)
    sid_to_keys = {s: sorted({keys[i] for i in v}) for s, v in sid_to_items.items()}
    item_to_sid = {i: s for s, v in sid_to_items.items() for i in v}
    return sid_to_keys, item_to_sid


def product_hits(rows, predictions, sid_to_keys, keys, topk):
    out = []
    for row, preds in zip(rows, predictions):
        target = row["item_id"].strip()
        target_key = keys.get(target)
        hit = 0.0
        for position, pred in enumerate(preds):
            if position >= topk:
                break
            ks = sid_to_keys.get(normalize(pred))
            if ks and len(ks) == 1 and ks[0] == target_key:
                hit = 1.0
                break
        out.append(hit)
    return out


def ambiguous_flags(rows, sid_to_keys, item_to_sid):
    flags = []
    for row in rows:
        sid = item_to_sid.get(row["item_id"].strip())
        flags.append(sid is None or len(sid_to_keys.get(sid, [])) > 1)
    return flags


def mean(v):
    return sum(v) / len(v)


def sd(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def t_two_sided_p(t, df):
    if df == 1:
        return 1.0 - (2.0 / math.pi) * math.atan(abs(t))
    if df == 2:
        return 1.0 - abs(t) / math.sqrt(t * t + 2.0)
    return math.erfc(abs(t) / math.sqrt(2.0))


def paired_bootstrap(a, b, rounds, seed=42):
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds) - 1]


def two_stage_bootstrap(list_a, list_b, rounds, seed=7):
    rng = random.Random(seed)
    k = len(list_a)
    out = []
    for _ in range(rounds):
        picked = [rng.randrange(k) for _ in range(k)]
        total = 0.0
        for s in picked:
            a, b = list_a[s], list_b[s]
            n = len(a)
            idx = [rng.randrange(n) for _ in range(n)]
            total += sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n
        out.append(total / k)
    out.sort()
    return out[int(0.025 * rounds)], out[int(0.975 * rounds) - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prebuilt-results", nargs="+", required=True)
    parser.add_argument("--kmeans-results", nargs="+", required=True)
    parser.add_argument("--prebuilt-index", required=True)
    parser.add_argument("--kmeans-index", required=True)
    parser.add_argument("--item-meta", required=True)
    parser.add_argument("--prebuilt-test", required=True)
    parser.add_argument("--kmeans-test", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    with open(args.item_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    keys = {i: product_key(meta[i]["title"]) for i in meta}

    rows_p = list(csv.DictReader(open(args.prebuilt_test, newline="", encoding="utf-8")))
    rows_k = list(csv.DictReader(open(args.kmeans_test, newline="", encoding="utf-8")))
    if [r["item_id"].strip() for r in rows_p] != [r["item_id"].strip() for r in rows_k]:
        raise SystemExit("test splits are not in the same sample order")
    rows = rows_p
    n = len(rows)

    keys_p, item_to_sid_p = build(args.prebuilt_index, keys)
    keys_k, item_to_sid_k = build(args.kmeans_index, keys)

    keep_p = ambiguous_flags(rows, keys_p, item_to_sid_p)
    keep_k = ambiguous_flags(rows, keys_k, item_to_sid_k)
    keep_both = [not a and not b for a, b in zip(keep_p, keep_k)]

    hits_p, hits_k = [], []
    for path in args.prebuilt_results:
        preds = [[normalize(p) for p in e.get("predict", [])] for e in load_result(path)]
        assert len(preds) == n
        hits_p.append(product_hits(rows, preds, keys_p, keys, args.topk))
    for path in args.kmeans_results:
        preds = [[normalize(p) for p in e.get("predict", [])] for e in load_result(path)]
        assert len(preds) == n
        hits_k.append(product_hits(rows, preds, keys_k, keys, args.topk))

    report = {
        "metric": f"product_unique_HR@{args.topk}",
        "rows": n,
        "rows_with_ambiguous_target_sid": {
            "prebuilt": sum(keep_p),
            "kmeans": sum(keep_k),
            "either": sum(1 for a, b in zip(keep_p, keep_k) if a or b),
        },
        "unambiguous_in_both_rows": sum(keep_both),
        "per_seed": [],
    }

    diffs = []
    for label, hp, hk in zip(args.labels, hits_p, hits_k):
        lo, hi = paired_bootstrap(hp, hk, args.bootstrap)
        diff = mean(hk) - mean(hp)
        diffs.append(diff)
        keep = keep_both
        m_p = mean([v for v, k in zip(hp, keep) if k])
        m_k = mean([v for v, k in zip(hk, keep) if k])
        report["per_seed"].append(
            {
                "seed": label,
                "prebuilt": round(mean(hp), 6),
                "kmeans": round(mean(hk), 6),
                "diff": round(diff, 6),
                "ci95": [round(lo, 6), round(hi, 6)],
                "significant": not (lo <= 0 <= hi),
                "unambiguous_subset": {
                    "rows": sum(keep),
                    "prebuilt": round(m_p, 6),
                    "kmeans": round(m_k, 6),
                    "diff": round(m_k - m_p, 6),
                },
            }
        )

    mean_diff = mean(diffs)
    t = mean_diff / (sd(diffs) / math.sqrt(len(diffs)))
    lo, hi = two_stage_bootstrap(hits_p, hits_k, args.bootstrap)
    clean_p = [[v for v, k in zip(h, keep_both) if k] for h in hits_p]
    clean_k = [[v for v, k in zip(h, keep_both) if k] for h in hits_k]
    clean_diffs = [mean(k) - mean(p) for p, k in zip(clean_p, clean_k)]
    clean_lo, clean_hi = two_stage_bootstrap(clean_p, clean_k, args.bootstrap)

    report["seed_level"] = {
        "seeds": len(diffs),
        "mean_diff": round(mean_diff, 6),
        "sd_diff": round(sd(diffs), 6),
        "min_diff": round(min(diffs), 6),
        "max_diff": round(max(diffs), 6),
        "all_seeds_same_sign": all(d > 0 for d in diffs) or all(d < 0 for d in diffs),
        "two_stage_ci95": [round(lo, 6), round(hi, 6)],
        "two_stage_significant": not (lo <= 0 <= hi),
        "t_vs_zero": round(t, 3),
        "df": len(diffs) - 1,
        "p_two_sided": round(t_two_sided_p(t, len(diffs) - 1), 4),
        "seeds_favouring_prebuilt": sum(1 for d in diffs if d < 0),
        "seeds_favouring_kmeans": sum(1 for d in diffs if d > 0),
    }
    report["seed_level_unambiguous_subset"] = {
        "rows": sum(keep_both),
        "mean_diff": round(mean(clean_diffs), 6),
        "two_stage_ci95": [round(clean_lo, 6), round(clean_hi, 6)],
        "two_stage_significant": not (clean_lo <= 0 <= clean_hi),
        "p_two_sided": round(
            t_two_sided_p(mean(clean_diffs) / (sd(clean_diffs) / math.sqrt(len(clean_diffs))),
                          len(clean_diffs) - 1),
            4,
        ),
    }

    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
