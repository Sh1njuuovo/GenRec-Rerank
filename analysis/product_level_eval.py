"""Score predictions at product granularity instead of item-id granularity.

The catalog contains the same product under several item ids. Items 181 and 182
for instance carry the exact same title, so a recommendation that shows either
one gives the user the same product. The strict item-level rule still counts
only one of them, which makes a shared SID look worse than it is whenever the
shared group holds duplicates.

This script keys every hit on the product title:

  * a prediction counts when the SID it emits decodes to item ids that all share
    one product key, and that key is the target's key;
  * a prediction that decodes to several distinct products is ambiguous and gets
    no credit, which is the case the strict rule is meant to punish.

Usage:
    python product_level_eval.py --result <r.json> --index <index.json> \
        --item-meta <item.json> --test-csv <test.csv> --label kmeans_43
"""

import argparse
import collections
import csv
import json
import math
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--item-meta", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    with open(args.index, "r", encoding="utf-8") as f:
        raw_index = json.load(f)
    with open(args.item_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    keys = {item: product_key(meta[item]["title"]) for item in meta}
    sid_to_items = collections.defaultdict(list)
    for item_id, tokens in raw_index.items():
        sid_to_items["".join(tokens)].append(item_id)

    sid_to_keys = {
        sid: sorted({keys[i] for i in items}) for sid, items in sid_to_items.items()
    }
    item_to_sid = {i: s for s, items in sid_to_items.items() for i in items}

    rows = list(csv.DictReader(open(args.test_csv, newline="", encoding="utf-8")))
    data = load_result(args.result)
    assert len(rows) == len(data), f"{len(rows)} rows vs {len(data)} predictions"
    predictions = [[normalize(p) for p in e.get("predict", [])] for e in data]

    topk = args.topk
    norm = 1.0 / math.log(2)
    hits = code_hits = fraction = 0
    code_ndcg = prod_ndcg = 0.0
    ambiguous_target = 0
    n = len(rows)
    for row, preds in zip(rows, predictions):
        target_item = row["item_id"].strip()
        target_sid = item_to_sid.get(target_item)
        target_key = keys.get(target_item)
        if target_sid is not None and len(sid_to_keys[target_sid]) > 1:
            ambiguous_target += 1
        for position, pred in enumerate(preds):
            if position >= topk:
                break
            weight = 1.0 / math.log(position + 2)
            if pred == target_sid:
                code_hits += 1
                code_ndcg += weight
                break
        for position, pred in enumerate(preds):
            if position >= topk:
                break
            ks = sid_to_keys.get(pred)
            if not ks:
                continue
            weight = 1.0 / math.log(position + 2)
            if len(ks) == 1 and ks[0] == target_key:
                hits += 1
                prod_ndcg += weight
                fraction += 1.0
                break
            if target_key in ks:
                fraction += 1.0 / len(ks)
                break

    report = {
        "label": args.label,
        "rows": n,
        "topk": topk,
        "rows_targeting_an_ambiguous_sid": ambiguous_target,
        "code_match_HR": round(code_hits / n, 6),
        "product_unique_HR": round(hits / n, 6),
        "product_unique_NDCG": round((prod_ndcg / n) / norm, 6),
        "product_fraction_HR": round(fraction / n, 6),
        "optimism_HR": round((code_hits - hits) / n, 6),
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
