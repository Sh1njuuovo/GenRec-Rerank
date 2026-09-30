"""Break the beam diagnostics down by how often the target appears in training.

A miss can come from a weak model or from the long tail of the catalog. This
script buckets the test rows by the number of times the target item occurs as a
training target, then reports coverage and hit rates inside each bucket.

Usage:
    python popularity_diagnostic.py --result eval_sft_full/result_final_checkpoint.json \
        --index .../Industrial_and_Scientific.index.json \
        --test-csv .../test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
        --train-csv .../train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
        --label sft_full
"""

import argparse
import collections
import csv
import json


BUCKETS = [
    ("never in train", lambda c: c == 0),
    ("1", lambda c: c == 1),
    ("2-5", lambda c: 2 <= c <= 5),
    ("6-20", lambda c: 6 <= c <= 20),
    ("21-100", lambda c: 21 <= c <= 100),
    ("over 100", lambda c: c > 100),
]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    with open(args.index, "r", encoding="utf-8") as f:
        raw_index = json.load(f)
    sid_to_items = collections.defaultdict(list)
    for item_id, tokens in raw_index.items():
        sid_to_items["".join(tokens)].append(item_id)
    item_to_sid = {i: s for s, v in sid_to_items.items() for i in v}

    train_counts = collections.Counter()
    for row in csv.DictReader(open(args.train_csv, newline="", encoding="utf-8")):
        train_counts[row["item_id"].strip()] += 1

    test_rows = list(csv.DictReader(open(args.test_csv, newline="", encoding="utf-8")))
    data = load_result(args.result)
    assert len(test_rows) == len(data)

    stats = {name: collections.Counter() for name, _ in BUCKETS}
    stats["all"] = collections.Counter()

    for row_data, entry in zip(test_rows, data):
        target_item = row_data["item_id"].strip()
        target_sid = item_to_sid.get(target_item)
        preds = [normalize(p) for p in entry.get("predict", [])]
        count = train_counts[target_item]

        hit = 0
        for position, pred in enumerate(preds):
            if position >= args.topk:
                break
            if pred == target_sid:
                hit = 1
                break
        strict = 0
        for position, pred in enumerate(preds):
            if position >= args.topk:
                break
            items = sid_to_items.get(pred)
            if items and len(items) == 1 and items[0] == target_item:
                strict = 1
                break
        cover = 1 if target_sid in preds else 0
        target_l1 = target_sid.split("<b_")[0] if target_sid else None
        l1 = 1 if target_l1 and any(p.split("<b_")[0] == target_l1 for p in preds) else 0

        for name, predicate in BUCKETS:
            if predicate(count):
                stats[name]["rows"] += 1
                stats[name]["hit"] += hit
                stats[name]["strict"] += strict
                stats[name]["cover"] += cover
                stats[name]["l1"] += l1
        stats["all"]["rows"] += 1
        stats["all"]["hit"] += hit
        stats["all"]["strict"] += strict
        stats["all"]["cover"] += cover
        stats["all"]["l1"] += l1

    def render(counter):
        n = counter["rows"] or 1
        return {
            "rows": counter["rows"],
            "share_of_rows": round(counter["rows"] / (stats["all"]["rows"] or 1), 4),
            "HR@10": round(counter["hit"] / n, 4),
            "strict_HR@10": round(counter["strict"] / n, 4),
            "coverage@50": round(counter["cover"] / n, 4),
            "level1_present": round(counter["l1"] / n, 4),
        }

    report = {
        "label": args.label,
        "buckets": {name: render(stats[name]) for name, _ in BUCKETS},
        "all": render(stats["all"]),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
