"""Independent audit of an evaluation result JSON.

calc.py only counts predictions that are absent from the item table, and it
stops scanning a sample as soon as it finds the target. This script scans every
returned candidate on every sample, so legality and duplication are checked over
the full prediction list. HR/NDCG are recomputed with the same convention as
calc.py so the two numbers can be compared.

Usage:
    python check_predictions.py --result <result.json> --info_file <info.txt> \
        --json_out <audit.json>
"""

import argparse
import json
import math
import re

TOKEN_RE = re.compile(r"<[abc]_\d+>")


def load_result(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and data and isinstance(data[0], list):
        flat = []
        for shard in data:
            flat.extend(shard)
        data = flat
    return data


def load_items(info_file):
    sid_strings = []
    with open(info_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t")
            if parts and parts[0].strip():
                sid_strings.append(parts[0].strip())
    return sid_strings


def normalize(pred):
    return pred.strip().strip('"').strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--json_out", default="")
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50])
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    data = load_result(args.result)
    sid_strings = load_items(args.info_file)
    valid_sids = set(sid_strings)

    valid_tokens = set()
    for sid in valid_sids:
        valid_tokens.update(TOKEN_RE.findall(sid))

    n = len(data)
    beam_sizes = set()
    illegal_sid = 0
    illegal_token = 0
    wrong_length = 0
    duplicate_total = 0
    duplicate_samples = 0
    total_preds = 0
    target_found = 0
    empty_samples = 0
    offenders = []

    hits = {k: 0 for k in args.topk}
    ndcg = {k: 0.0 for k in args.topk}

    for entry in data:
        preds = [normalize(p) for p in entry.get("predict", [])]
        target = entry.get("output")
        if isinstance(target, list):
            target = target[0] if target else ""
        target = normalize(str(target))

        beam_sizes.add(len(preds))
        total_preds += len(preds)
        if not preds:
            empty_samples += 1

        seen = set()
        for pred in preds:
            if pred in seen:
                duplicate_total += 1
                continue
            seen.add(pred)
        if len(preds) != len(seen):
            duplicate_samples += 1

        first_hit = None
        for position, pred in enumerate(preds):
            if pred not in valid_sids:
                illegal_sid += 1
                if len(offenders) < args.examples:
                    offenders.append({"pred": pred, "target": target})
            tokens = TOKEN_RE.findall(pred)
            if len(tokens) != 3:
                wrong_length += 1
            if any(tok not in valid_tokens for tok in tokens):
                illegal_token += 1
            if first_hit is None and pred == target:
                first_hit = position

        if first_hit is not None:
            target_found += 1
        for k in args.topk:
            if first_hit is not None and first_hit < k:
                hits[k] += 1
                ndcg[k] += 1.0 / math.log(first_hit + 2)

    report = {
        "result_file": args.result,
        "samples": n,
        "beam_sizes": sorted(beam_sizes),
        "total_predictions": total_preds,
        "avg_predictions_per_sample": round(total_preds / n, 3) if n else 0,
        "empty_prediction_samples": empty_samples,
        "illegal_sid_count": illegal_sid,
        "illegal_sid_rate": round(illegal_sid / total_preds, 6) if total_preds else None,
        "illegal_token_count": illegal_token,
        "wrong_token_count": wrong_length,
        "duplicate_predictions": duplicate_total,
        "duplicate_rate": round(duplicate_total / total_preds, 6) if total_preds else None,
        "samples_with_duplicates": duplicate_samples,
        "target_found": target_found,
        "metrics": {},
        "offender_examples": offenders,
    }

    denom = 1.0 / math.log(2)
    for k in args.topk:
        report["metrics"][f"HR@{k}"] = round(hits[k] / n, 8) if n else None
        report["metrics"][f"NDCG@{k}"] = round((ndcg[k] / n) / denom, 8) if n else None

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
