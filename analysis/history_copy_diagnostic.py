"""Measure how much of the beam is a copy of the user's own history.

The reranker leans on one feature, whether the candidate is an item the user
already interacted with. This script checks that the mechanism is real: the
model puts a history item first on nearly half of the rows, while the true next
item is a repeat for only a small share of them.

Usage:
    python history_copy_diagnostic.py \
        --result eval_sft_full/result_final_checkpoint.json \
        --index .../index/Industrial_and_Scientific.index.json \
        --csv .../test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
        --label sft_full
"""

import argparse
import collections
import csv
import json
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


def parse_ids(raw):
    return [int(v) for v in re.findall(r"\d+", raw or "")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    with open(args.index, "r", encoding="utf-8") as f:
        raw_index = json.load(f)
    item_to_sid = {int(item): "".join(tokens)
                   for item, tokens in raw_index.items()}

    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    beam = load_result(args.result)
    if len(rows) != len(beam):
        raise SystemExit(f"{len(rows)} rows vs {len(beam)} predictions")

    n = len(rows)
    rank1_copy = 0
    target_repeat = 0
    slots = 0
    copy_rows = []
    for row, entry in zip(rows, beam):
        history = parse_ids(row.get("history_item_id", ""))
        history_sids = {item_to_sid.get(item) for item in history}
        target = int(row["item_id"].strip())
        preds = [normalize(p) for p in entry.get("predict", [])]
        in_history = [1 if p in history_sids else 0 for p in preds]
        slots += sum(in_history)
        if preds and in_history and in_history[0]:
            rank1_copy += 1
            copy_rows.append(row)
        if target in history:
            target_repeat += 1

    report = {
        "label": args.label,
        "rows": n,
        "share_of_beam_slots_that_are_history_items": round(slots / (n * 50), 6),
        "rows_whose_rank1_is_a_history_item": round(rank1_copy / n, 6),
        "rows_whose_target_is_itself_a_history_item": round(target_repeat / n, 6),
        "hit_rate_of_the_copy_first_rows": None,
    }
    if copy_rows:
        hits = 0
        for row in copy_rows:
            target = int(row["item_id"].strip())
            if target in parse_ids(row.get("history_item_id", "")):
                hits += 1
        report["hit_rate_of_the_copy_first_rows"] = round(hits / len(copy_rows), 6)
    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
