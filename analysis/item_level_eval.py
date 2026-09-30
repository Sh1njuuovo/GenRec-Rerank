"""Re-score a SID-level prediction file under item-level conventions.

The model emits a code, so the reported metric compares code strings. When one
code is shared by several items, predicting it counts as a hit for every one of
them, which makes an assignment with more collisions look better without the
model having identified anything. This script re-scores the same predictions
three ways:

  code_match     the current protocol: predicted SID equals the target's SID
  item_unique    credit only when the predicted SID decodes to exactly one item
                 and that item is the target
  item_fraction  credit 1/k when the predicted SID decodes to k items and the
                 target is one of them

It also splits every metric by whether the target's own SID is shared, so the
collision contribution is visible rather than folded into the headline.

Usage:
    python item_level_eval.py --result <result.json> --index <index.json> \
        --test-csv <test.csv> --topk 10 --label sft_prebuilt
"""

import argparse
import collections
import csv
import json
import math


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


def build_decode_table(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    sid_to_items = collections.defaultdict(list)
    item_to_sid = {}
    for item_id, tokens in raw.items():
        sid = "".join(tokens)
        sid_to_items[sid].append(item_id)
        item_to_sid[item_id] = sid
    return sid_to_items, item_to_sid


def score(rows, predictions, sid_to_items, item_to_sid, topk):
    """Return the three metric variants plus a collision split."""
    buckets = {
        "all": list(range(len(rows))),
        "unique_target": [],
        "shared_target": [],
    }
    for i, row in enumerate(rows):
        sid = item_to_sid.get(row["item_id"].strip())
        if sid is None:
            continue
        if len(sid_to_items[sid]) == 1:
            buckets["unique_target"].append(i)
        else:
            buckets["shared_target"].append(i)

    out = {}
    norm = 1.0 / math.log(2)
    for name, idx in buckets.items():
        if not idx:
            continue
        n = len(idx)
        code_hits = item_hits = 0
        code_ndcg = item_ndcg = fraction = 0.0
        for i in idx:
            target_sid = item_to_sid[rows[i]["item_id"].strip()]
            target_item = rows[i]["item_id"].strip()
            for position, pred in enumerate(predictions[i]):
                if position >= topk:
                    break
                sid = normalize(pred)
                weight = 1.0 / math.log(position + 2)
                if sid == target_sid:
                    code_hits += 1
                    code_ndcg += weight
                    break
            for position, pred in enumerate(predictions[i]):
                if position >= topk:
                    break
                sid = normalize(pred)
                items = sid_to_items.get(sid)
                if not items:
                    continue
                weight = 1.0 / math.log(position + 2)
                if len(items) == 1 and items[0] == target_item:
                    item_hits += 1
                    item_ndcg += weight
                    fraction += 1.0
                    break
                if target_item in items:
                    fraction += 1.0 / len(items)
                    break
        out[name] = {
            "rows": n,
            "code_match_HR": round(code_hits / n, 6),
            "code_match_NDCG": round((code_ndcg / n) / norm, 6),
            "item_unique_HR": round(item_hits / n, 6),
            "item_unique_NDCG": round((item_ndcg / n) / norm, 6),
            "item_fraction_HR": round(fraction / n, 6),
            "optimism_HR": round((code_hits - item_hits) / n, 6),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    sid_to_items, item_to_sid = build_decode_table(args.index)
    rows = list(csv.DictReader(open(args.test_csv, newline="", encoding="utf-8")))
    data = load_result(args.result)
    assert len(rows) == len(data), f"{len(rows)} rows vs {len(data)} predictions"
    predictions = [[normalize(p) for p in entry.get("predict", [])] for entry in data]

    report = {
        "label": args.label,
        "result": args.result,
        "index": args.index,
        "topk": args.topk,
        "shared_sid_groups": sum(1 for v in sid_to_items.values() if len(v) > 1),
        "items_in_shared_sids": sum(len(v) for v in sid_to_items.values() if len(v) > 1),
        "metrics": score(rows, predictions, sid_to_items, item_to_sid, args.topk),
    }

    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
