"""Split a two-model comparison by whether the target item is collision-free.

Evaluation matches SID strings. When several items share one SID, generating
that SID counts as a hit for any of them, so an assignment with more collisions
gets a partly free boost. This script redoes the comparison on the subset of
test rows whose target item is collision-free in both assignments, which removes
that effect, and reports how much of the difference comes from each part.

Run from the run directory:
    python check_collision_confound.py \
        --result-a eval_sft_full/result_final_checkpoint.json --label-a prebuilt \
        --result-b eval_behavior/result_final_checkpoint.json --label-b behavior \
        --index-a <prebuilt index> --index-b <behavior index> \
        --test-csv <test csv with item_id> --topk 10
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


def load_index(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    index = {}
    collisions = set()
    seen = collections.defaultdict(list)
    for item_id, tokens in raw.items():
        sid = "".join(tokens)
        index[item_id] = sid
        seen[sid].append(item_id)
    for sid, items in seen.items():
        if len(items) > 1:
            collisions.update(items)
    return index, collisions, {sid: items for sid, items in seen.items() if len(items) > 1}


def rank_of(entry):
    target = normalize(entry["output"])
    for position, pred in enumerate(entry.get("predict", [])):
        if normalize(pred) == target:
            return position
    return None


def rates(entries, topk):
    hits = 0
    ndcg = 0.0
    for entry in entries:
        rank = rank_of(entry)
        if rank is not None and rank < topk:
            hits += 1
            ndcg += 1.0 / math.log(rank + 2)
    n = len(entries)
    norm = 1.0 / math.log(2)
    return hits, hits / n if n else 0.0, (ndcg / n) / norm if n else 0.0


def paired_bootstrap(hit_a, hit_b, rounds, seed=42):
    rng = random.Random(seed)
    n = len(hit_a)
    diffs = []
    for _ in range(rounds):
        sample = [rng.randrange(n) for _ in range(n)]
        da = sum(hit_a[i] for i in sample) / n
        db = sum(hit_b[i] for i in sample) / n
        diffs.append(db - da)
    diffs.sort()
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-a", required=True)
    parser.add_argument("--result-b", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--index-a", required=True)
    parser.add_argument("--index-b", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    index_a, coll_a, groups_a = load_index(args.index_a)
    index_b, coll_b, groups_b = load_index(args.index_b)

    rows = list(csv.DictReader(open(args.test_csv, newline="", encoding="utf-8")))
    data_a = load_result(args.result_a)
    data_b = load_result(args.result_b)
    assert len(rows) == len(data_a) == len(data_b), "row counts differ"

    clean_idx, coll_idx = [], []
    for i, row in enumerate(rows):
        item = row["item_id"].strip()
        if item in coll_a or item in coll_b:
            coll_idx.append(i)
        else:
            clean_idx.append(i)

    report = {
        "rows": len(rows),
        "collision_groups": {"A": len(groups_a), "B": len(groups_b)},
        "items_in_collisions": {"A": len(coll_a), "B": len(coll_b)},
        "unique_targets_clean": len({rows[i]["item_id"] for i in clean_idx}),
        "subsets": {},
    }

    for name, idx in (("all", list(range(len(rows)))), ("clean", clean_idx), ("collision", coll_idx)):
        entries_a = [data_a[i] for i in idx]
        entries_b = [data_b[i] for i in idx]
        hits_a, hr_a, nd_a = rates(entries_a, args.topk)
        hits_b, hr_b, nd_b = rates(entries_b, args.topk)

        hit_a = [1.0 if (r := rank_of(e)) is not None and r < args.topk else 0.0 for e in entries_a]
        hit_b = [1.0 if (r := rank_of(e)) is not None and r < args.topk else 0.0 for e in entries_b]
        lo, hi = paired_bootstrap(hit_a, hit_b, args.bootstrap)

        report["subsets"][name] = {
            "rows": len(idx),
            f"{args.label_a}_hits": hits_a,
            f"{args.label_b}_hits": hits_b,
            f"{args.label_a}_HR": round(hr_a, 6),
            f"{args.label_b}_HR": round(hr_b, 6),
            "HR_diff": round(hr_b - hr_a, 6),
            "HR_diff_ci95": [round(lo, 6), round(hi, 6)],
            f"{args.label_a}_NDCG": round(nd_a, 6),
            f"{args.label_b}_NDCG": round(nd_b, 6),
            "NDCG_diff": round(nd_b - nd_a, 6),
        }

    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
