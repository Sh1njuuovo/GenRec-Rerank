"""Replace the SID columns of the existing splits with a new SID assignment.

The shipped splits already carry `item_id`, `history_item_id` and the SID
strings. Rewriting the SID columns through the item ids keeps the samples, their
order and the train/valid/test split exactly as they are, and avoids needing the
raw Amazon18 files that `convert_dataset.py` reads.

Everything except `item_sid` and `history_item_sid` must come out byte-identical,
so the script asserts that on every row and fails loudly instead of writing a
quietly different dataset.

Usage:
    python remap_sids.py \
        --src-root /path/MiniOneRec/data/Amazon \
        --new-index /path/sid_exp/index/Industrial_and_Scientific_exp.index.json \
        --out-dir /path/new_data \
        --category Industrial_and_Scientific \
        --item-json /path/MiniOneRec/data/Amazon/index/Industrial_and_Scientific.item.json \
        --info-out /path/new_data/Industrial_and_Scientific_exp.info.txt
"""

import argparse
import csv
import json
import os
import re

LEVEL_RE = re.compile(r"<[abc]_\d+>")
SPLITS = ("train", "valid", "test")


def find_split_file(src_root, split, category):
    split_dir = os.path.join(src_root, split)
    for name in sorted(os.listdir(split_dir)):
        if name.startswith(category) and name.endswith(".csv"):
            return os.path.join(split_dir, name)
    raise FileNotFoundError(f"no {category} csv under {split_dir}")


def parse_id_list(raw):
    """history_item_id is stored as a bracketed list of integers."""
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [item.strip().strip("'").strip('"') for item in text.split(",") if item.strip()]


def parse_sid_list(raw):
    """history_item_sid is stored as a bracketed list of quoted SID strings."""
    quoted = re.findall(r"'([^']*)'", raw)
    if quoted:
        return quoted
    # Fallback for an unquoted list: three tokens make one SID.
    tokens = LEVEL_RE.findall(raw)
    return ["".join(tokens[i:i + 3]) for i in range(0, len(tokens), 3)]


def sid_string(tokens):
    return "".join(tokens)


def remap_split(src_path, out_path, new_index, category, report):
    with open(src_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    stats = {
        "rows": len(rows),
        "history_items": 0,
        "max_history_len": 0,
        "missing_item_ids": 0,
        "targets_changed": 0,
        "history_len_mismatch": 0,
    }
    missing = set()

    out_rows = []
    for row in rows:
        item_id = row["item_id"].strip()
        if item_id not in new_index:
            missing.add(item_id)
            stats["missing_item_ids"] += 1
            continue

        history_ids = parse_id_list(row["history_item_id"])
        old_history_sids = parse_sid_list(row["history_item_sid"])
        if len(history_ids) != len(old_history_sids):
            stats["history_len_mismatch"] += 1

        new_history = []
        ok = True
        for hid in history_ids:
            if hid not in new_index:
                missing.add(hid)
                ok = False
                break
            new_history.append(sid_string(new_index[hid]))
        if not ok:
            continue

        stats["history_items"] += len(history_ids)
        stats["max_history_len"] = max(stats["max_history_len"], len(history_ids))

        new_row = dict(row)
        old_target = row["item_sid"].strip()
        new_target = sid_string(new_index[item_id])
        if new_target != old_target:
            stats["targets_changed"] += 1
        new_row["item_sid"] = new_target
        new_row["history_item_sid"] = "[" + ", ".join(f"'{s}'" for s in new_history) + "]"

        # Every other column must be untouched.
        for key in fieldnames:
            if key in ("item_sid", "history_item_sid"):
                continue
            assert new_row[key] == row[key], f"{src_path}: column {key} changed"

        out_rows.append(new_row)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        # The shipped splits use plain LF; keep that so the only difference
        # against the originals is the SID columns themselves.
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)

    stats["missing_examples"] = sorted(missing)[:5]
    report[category + "_" + os.path.basename(src_path)] = stats
    return stats


def write_info_file(new_index, item_json_path, out_path, dataset_name):
    """Rebuild the `sid\tTITLE\titem_id` table used by evaluation and RL."""
    with open(item_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    lines = []
    for item_id in sorted(new_index, key=lambda x: int(x)):
        sid = sid_string(new_index[item_id])
        title = items.get(item_id, {}).get("title", "")
        lines.append(f"{sid}\t{title}\t{item_id}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--new-index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--category", default="Industrial_and_Scientific")
    parser.add_argument("--out-tag", default="_exp",
                        help="inserted before the first underscore of the split name")
    parser.add_argument("--item-json", default="")
    parser.add_argument("--info-out", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    with open(args.new_index, "r", encoding="utf-8") as f:
        new_index = json.load(f)

    report = {
        "new_index": args.new_index,
        "new_index_items": len(new_index),
        "splits": {},
    }

    for split in SPLITS:
        src_path = find_split_file(args.src_root, split, args.category)
        out_name = os.path.basename(src_path).replace(
            args.category, args.category + args.out_tag, 1
        )
        out_path = os.path.join(args.out_dir, split, out_name)
        stats = remap_split(src_path, out_path, new_index, args.category, report["splits"])
        print(f"{split}: {stats}")

    if args.info_out:
        count = write_info_file(new_index, args.item_json, args.info_out, args.category)
        report["info_rows"] = count
        print(f"info file: {count} rows -> {args.info_out}")

    total_missing = sum(s["missing_item_ids"] for s in report["splits"].values())
    total_mismatch = sum(s["history_len_mismatch"] for s in report["splits"].values())
    print(f"missing ids: {total_missing}, history length mismatches: {total_mismatch}")
    report["missing_ids_total"] = total_missing
    report["history_len_mismatch_total"] = total_mismatch

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.json_out}")

    return 0 if total_missing == 0 and total_mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
