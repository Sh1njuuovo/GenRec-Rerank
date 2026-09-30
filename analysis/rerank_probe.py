"""Test whether the beam candidates can be reordered with item vectors.

The beam diagnostic shows that when the target SID is generated at all it is
ranked inside the top ten only about half the time. This script asks whether a
cheap relevance signal recovers part of that gap: score every candidate by the
cosine similarity between its item vector and the mean vector of the user's
history, then look at the metric under the new order.

No training and no GPU. Reads the official precomputed item vectors directly
from the `.npy` file with the standard library so it runs anywhere.

Usage:
    python rerank_probe.py --result eval_sft_full/result_final_checkpoint.json \
        --index MiniOneRec/data/Amazon/index/Industrial_and_Scientific.index.json \
        --test-csv MiniOneRec/data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
        --embeddings MiniOneRec/data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
        --label sft_full
"""

import argparse
import ast
import array
import collections
import csv
import json
import math
import operator
import re
import struct


def read_npy(path):
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise SystemExit(f"{path} is not a npy file")
        major, _ = f.read(2)
        if major == 1:
            header_len = struct.unpack("<H", f.read(2))[0]
        else:
            header_len = struct.unpack("<I", f.read(4))[0]
        header = f.read(header_len).decode("latin1")
        descr = re.search(r"'descr':\s*'([^']+)'", header).group(1)
        fortran = re.search(r"'fortran_order':\s*(\w+)", header).group(1)
        shape = ast.literal_eval(re.search(r"'shape':\s*(\([^)]*\))", header).group(1))
        if descr not in ("<f2", "<f4") or fortran != "False":
            raise SystemExit(f"unsupported layout descr={descr} fortran={fortran}")
        count = 1
        for dim in shape:
            count *= dim
        width = 2 if descr == "<f2" else 4
        payload = f.read(count * width)
        code = "e" if descr == "<f2" else "f"
        # array.array has no half-float code on every build, struct always does
        values = struct.unpack(f"<{count}{code}", payload)
        data = array.array("f", values)
    return shape, data


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


def parse_history(raw):
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, nargs="+", default=[10, 60])
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    shape, flat = read_npy(args.embeddings)
    rows_n, dim = shape
    print(f"embeddings {shape}")

    with open(args.index, "r", encoding="utf-8") as f:
        raw_index = json.load(f)
    sid_to_items = collections.defaultdict(list)
    for item_id, tokens in raw_index.items():
        sid_to_items["".join(tokens)].append(item_id)
    item_to_sid = {i: s for s, v in sid_to_items.items() for i in v}

    test_rows = list(csv.DictReader(open(args.test_csv, newline="", encoding="utf-8")))
    data = load_result(args.result)
    assert len(test_rows) == len(data)

    norms = array.array("f", [0.0]) * rows_n
    for i in range(rows_n):
        base = i * dim
        chunk = flat[base:base + dim]
        total = sum(map(operator.mul, chunk, chunk))
        norms[i] = math.sqrt(total) if total > 0 else 0.0

    def row(i):
        base = int(i) * dim
        return flat[base:base + dim]

    def cosine(a, b, norm_a, norm_b):
        if norm_a == 0 or norm_b == 0:
            return 0.0
        dot = sum(map(operator.mul, a, b))
        return dot / (norm_a * norm_b)

    def mean_history_vector(history):
        vec = array.array("f", [0.0]) * dim
        used = 0
        for item in history:
            idx = int(item)
            if idx < 0 or idx >= rows_n:
                continue
            base = idx * dim
            chunk = flat[base:base + dim]
            vec = array.array("f", [v + c for v, c in zip(vec, chunk)])
            used += 1
        if used == 0:
            return None, 0.0
        vec = array.array("f", [v / used for v in vec])
        total = sum(map(operator.mul, vec, vec))
        return vec, math.sqrt(total)

    def candidate_score(pred, hist_vec, hist_norm, use_mean):
        items = sid_to_items.get(pred)
        if not items:
            return None
        if len(items) == 1:
            idx = int(items[0])
        else:
            # shared code: use the average vector of the group
            vec = array.array("f", [0.0]) * dim
            for item in items:
                base = int(item) * dim
                chunk = flat[base:base + dim]
                vec = array.array("f", [v + c for v, c in zip(vec, chunk)])
            vec = array.array("f", [v / len(items) for v in vec])
            total = sum(map(operator.mul, vec, vec))
            norm = math.sqrt(total) if total > 0 else 0.0
            return cosine(vec, hist_vec, norm, hist_norm)
        return cosine(row(idx), hist_vec, norms[idx], hist_norm)

    baseline_code = baseline_strict = 0
    variants = collections.defaultdict(lambda: [0, 0])  # -> [code, strict]

    for row_data, entry in zip(test_rows, data):
        target_item = row_data["item_id"].strip()
        target_sid = item_to_sid.get(target_item)
        preds = [normalize(p) for p in entry.get("predict", [])]

        def code_hit(order):
            for position, idx in enumerate(order[: args.topk]):
                if preds[idx] == target_sid:
                    return 1
            return 0

        def strict_hit(order):
            for idx in order[: args.topk]:
                items = sid_to_items.get(preds[idx])
                if items and len(items) == 1 and items[0] == target_item:
                    return 1
            return 0

        natural = list(range(len(preds)))
        baseline_code += code_hit(natural)
        baseline_strict += strict_hit(natural)

        if target_sid not in preds:
            continue

        history = parse_history(row_data.get("history_item_id", ""))
        hist_vec, hist_norm = mean_history_vector(history)
        if hist_vec is None:
            continue

        sims = [candidate_score(p, hist_vec, hist_norm, True) for p in preds]
        sim_order = sorted(
            natural,
            key=lambda i: (-(sims[i] if sims[i] is not None else -2.0), i),
        )
        code_order = sorted(
            natural,
            key=lambda i: (0 if sims[i] is None else 1, sims[i] if sims[i] is not None else 0.0),
            reverse=True,
        )
        variants["similarity_only"][0] += code_hit(sim_order)
        variants["similarity_only"][1] += strict_hit(sim_order)

        sim_rank = [0] * len(preds)
        for position, idx in enumerate(sim_order):
            sim_rank[idx] = position + 1
        for k in args.rrf_k:
            fused = sorted(
                natural,
                key=lambda i: -(1.0 / (k + i + 1) + 1.0 / (k + sim_rank[i])),
            )
            variants[f"rrf_k{k}"][0] += code_hit(fused)
            variants[f"rrf_k{k}"][1] += strict_hit(fused)

    n = len(test_rows)
    report = {
        "label": args.label,
        "rows": n,
        "baseline": {
            "code_HR": round(baseline_code / n, 6),
            "strict_HR": round(baseline_strict / n, 6),
        },
        "variants": {
            name: {
                "code_HR": round(values[0] / n, 6),
                "strict_HR": round(values[1] / n, 6),
                "code_delta": round((values[0] - baseline_code) / n, 6),
                "strict_delta": round((values[1] - baseline_strict) / n, 6),
            }
            for name, values in sorted(variants.items())
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
