"""Explain why the seed-level gain sits entirely on rows with a shared SID.

Under the strict item-level rule a target whose own SID decodes to several
items can never be counted, so every row in a shared prebuilt group is a
guaranteed miss for the prebuilt model. The seed-level result moved the other
way when those rows were dropped, so this script inspects them directly.

It reports, for the rows whose prebuilt SID is shared:
  * how many distinct target items they cover and how often each appears in the
    test split
  * the size of the shared group each target belongs to
  * the strict hit rate of the kmeans models on exactly those rows
  * the same statistics for the rows with a unique prebuilt SID, as a control
"""

import argparse
import collections
import csv
import json


def normalize(text):
    return str(text).strip().strip('"').strip()


def load_result(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and data and isinstance(data[0], list):
        flat = []
        for shard in data:
            flat.extend(shard)
        data = flat
    return data


def decode_table(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    table = collections.defaultdict(list)
    for item_id, tokens in raw.items():
        table["".join(tokens)].append(item_id)
    return table


def strict_hit(preds, target, table, topk):
    for position, pred in enumerate(preds):
        if position >= topk:
            break
        items = table.get(normalize(pred))
        if items and len(items) == 1 and items[0] == target:
            return 1.0
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--prebuilt-index", required=True)
    parser.add_argument("--kmeans-index", required=True)
    parser.add_argument("--prebuilt-results", nargs="+", required=True)
    parser.add_argument("--kmeans-results", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--top-items", type=int, default=12)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.test_csv, newline="", encoding="utf-8")))
    targets = [r["item_id"].strip() for r in rows]
    table_p = decode_table(args.prebuilt_index)
    table_k = decode_table(args.kmeans_index)
    item_to_sid_p = {i: s for s, items in table_p.items() for i in items}

    frequency = collections.Counter(targets)
    print(f"test rows {len(rows)}, distinct targets {len(frequency)}")
    top = frequency.most_common(args.top_items)
    print("most frequent targets in the test split")
    for item, count in top:
        sid_p = item_to_sid_p.get(item)
        group_p = len(table_p[sid_p]) if sid_p else 0
        sid_k = next((s for s, items in table_k.items() if item in items), None)
        group_k = len(table_k[sid_k]) if sid_k else 0
        print(
            f"  {item:>10}  rows={count:>4}  prebuilt_group={group_p}  kmeans_group={group_k}"
        )

    shared_partner = collections.Counter()
    shared_idx = []
    for i, target in enumerate(targets):
        sid = item_to_sid_p.get(target)
        if sid is None:
            continue
        if len(table_p[sid]) > 1:
            shared_idx.append(i)
            for other in table_p[sid]:
                if other != target:
                    shared_partner[other] += 1
    unique_idx = [i for i in range(len(rows)) if i not in set(shared_idx)]

    print()
    print(f"rows with a shared prebuilt SID: {len(shared_idx)}")
    print(f"rows with a unique prebuilt SID: {len(unique_idx)}")
    shared_targets = collections.Counter(targets[i] for i in shared_idx)
    print(f"distinct targets behind those rows: {len(shared_targets)}")
    print("group sizes of the shared targets")
    sizes = collections.Counter(
        len(table_p[item_to_sid_p[t]]) for t in shared_targets
    )
    for size, count in sorted(sizes.items()):
        print(f"  group size {size}: {count} distinct targets")
    print("targets appearing most often among the shared rows")
    for item, count in shared_targets.most_common(args.top_items):
        sid = item_to_sid_p[item]
        print(
            f"  {item:>10}  shared_rows={count:>4}  test_rows={frequency[item]:>4}"
            f"  group={sorted(table_p[sid])}"
        )

    print()
    print("strict hits by group, per seed")
    print(
        f"{'seed':>6} {'shared rows':>12} {'shared hit':>11} {'shared HR':>10}"
        f" {'unique rows':>12} {'unique hit':>11} {'unique HR':>10}"
    )
    for label, ppath, kpath in zip(
        args.labels, args.prebuilt_results, args.kmeans_results
    ):
        preds_p = [normalize_list(e) for e in load_result(ppath)]
        preds_k = [normalize_list(e) for e in load_result(kpath)]
        hits_p_s = sum(strict_hit(preds_p[i], targets[i], table_p, args.topk) for i in shared_idx)
        hits_k_s = sum(strict_hit(preds_k[i], targets[i], table_k, args.topk) for i in shared_idx)
        hits_p_u = sum(strict_hit(preds_p[i], targets[i], table_p, args.topk) for i in unique_idx)
        hits_k_u = sum(strict_hit(preds_k[i], targets[i], table_k, args.topk) for i in unique_idx)
        print(
            f"{label:>6} {len(shared_idx):>12} {hits_k_s:>11.0f} {hits_k_s / len(shared_idx):>10.4f}"
            f" {len(unique_idx):>12} {hits_k_u:>11.0f} {hits_k_u / len(unique_idx):>10.4f}"
        )
        print(
            f"{'':>6} prebuilt on shared rows: strict {hits_p_s:.0f}"
            f"   prebuilt on unique rows: strict {hits_p_u:.0f} ({hits_p_u / len(unique_idx):.4f})"
        )


def normalize_list(entry):
    return [normalize(p) for p in entry.get("predict", [])]


if __name__ == "__main__":
    main()
