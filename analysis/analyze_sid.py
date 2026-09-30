"""Analyze an existing SID assignment without a GPU.

Answers the questions that decide how much the SID space limits the reported
metrics: how the three codebooks are used, how frequent the collisions are, and
whether colliding items are semantically close or unrelated.

Usage:
    python analyze_sid.py \
        --index data/Amazon/index/Industrial_and_Scientific.index.json \
        --item  data/Amazon/index/Industrial_and_Scientific.item.json \
        --info  data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
        --test  data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
        --json_out results/sid_analysis.json
"""

import argparse
import collections
import csv
import json
import math
import re

LEVEL_RE = re.compile(r"<([abc])_(\d+)>")


def load_index(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_items(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_info(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                rows.append({"sid": parts[0].strip(), "title": parts[1].strip()})
    return rows


def entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)


def describe_level(level_name, codes, sample_size):
    counts = collections.Counter(codes)
    used = len(counts)
    top = counts.most_common(8)
    return {
        "level": level_name,
        "unique_codes_used": used,
        f"fraction_of_{sample_size}_codes_used": round(used / sample_size, 4),
        "entropy_nats": round(entropy(counts), 4),
        "max_entropy_nats": round(math.log(sample_size), 4),
        "top_codes": [{"code": f"<{level_name}_{code}>", "items": n} for code, n in top],
        "unused_code_examples": sorted(set(range(sample_size)) - set(counts))[:10],
    }


def parse_sid(sid):
    found = LEVEL_RE.findall(sid)
    by_level = {}
    for level, code in found:
        by_level[level] = int(code)
    return by_level


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--item", default="")
    parser.add_argument("--info", default="")
    parser.add_argument("--test", default="")
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    index = load_index(args.index)
    items = load_items(args.item) if args.item else {}

    report = {"num_items": len(index)}

    sid_of = {}
    levels = {"a": [], "b": [], "c": []}
    for item_id, sids in index.items():
        sid = "".join(sids)
        sid_of[item_id] = sid
        parsed = parse_sid(sid)
        for level, code in parsed.items():
            levels[level].append(code)

    report["codebook_usage"] = [
        describe_level(level, levels[level], args.codebook_size) for level in "abc"
    ]

    sid_to_items = collections.defaultdict(list)
    for item_id, sid in sid_of.items():
        sid_to_items[sid].append(item_id)

    collisions = {sid: ids for sid, ids in sid_to_items.items() if len(ids) > 1}
    collision_items = sum(len(ids) for ids in collisions.values())
    report["sid_space"] = {
        "unique_sids": len(sid_to_items),
        "collision_groups": len(collisions),
        "items_in_collisions": collision_items,
        "collision_item_fraction": round(collision_items / len(index), 6),
        "sid_reuse_fraction": round(1 - len(sid_to_items) / len(index), 6),
    }

    detail = []
    for sid, ids in sorted(collisions.items(), key=lambda kv: -len(kv[1]))[:20]:
        titles = [items.get(i, {}).get("title", "?") for i in ids] if items else []
        detail.append({"sid": sid, "item_ids": ids, "titles": titles})
    report["collision_detail"] = detail

    if args.info:
        info_rows = load_info(args.info)
        info_sids = collections.Counter(row["sid"] for row in info_rows)
        report["info_file"] = {
            "rows": len(info_rows),
            "unique_sids": len(info_sids),
            "duplicate_sid_rows": sum(n - 1 for n in info_sids.values() if n > 1),
        }

    if args.test:
        with open(args.test, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            targets = collections.Counter()
            history_lengths = []
            for row in reader:
                targets[row["item_sid"].strip()] += 1
                raw = row.get("history_item_sid", "")
                history_lengths.append(raw.count("<a_"))
        total = sum(targets.values())
        missing = [sid for sid in targets if sid not in sid_to_items]
        report["test_set"] = {
            "rows": total,
            "unique_target_items": len(targets),
            "unique_target_sids": len(sid_to_items and targets),
            "targets_missing_from_index": len(missing),
            "target_sid_repeat_rows": sum(n - 1 for n in targets.values() if n > 1),
            "history_length_min": min(history_lengths) if history_lengths else None,
            "history_length_max": max(history_lengths) if history_lengths else None,
            "history_length_mean": round(sum(history_lengths) / len(history_lengths), 2)
            if history_lengths else None,
        }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
