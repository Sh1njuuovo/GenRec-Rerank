"""Rerank beam candidates with the co-occurrence structure of the training set.

The text-vector reranker in `rerank_probe.py` barely moves the metric, so this
script tries the other cheap relevance signal: which items appear together in a
user history. Pairs are counted over unique items inside one training history,
turned into positive pointwise mutual information, and a candidate is scored by
the sum of its PPMI against the user's history items.

Only training data is used, and the next item is never part of the history, so
nothing from the test label leaks into the score.

Usage:
    python rerank_cooccurrence.py --result eval_sft_full/result_final_checkpoint.json \
        --index .../Industrial_and_Scientific.index.json \
        --test-csv .../test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
        --train-csv .../train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
        --label sft_full
"""

import argparse
import collections
import csv
import json
import math
import random
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


def build_ppmi(train_csv):
    pairs = collections.Counter()
    item_total = collections.Counter()
    history_count = 0
    target_count = collections.Counter()
    with open(train_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target_count[row["item_id"].strip()] += 1
            hist = sorted(set(parse_ids(row.get("history_item_id", ""))))
            if len(hist) < 2:
                continue
            history_count += 1
            for i in range(len(hist)):
                item_total[hist[i]] += 1
                for j in range(i + 1, len(hist)):
                    pairs[(hist[i], hist[j])] += 1
    total_pairs = sum(pairs.values())
    ppmi = {}
    for (a, b), c in pairs.items():
        denom = item_total[a] * item_total[b]
        if denom <= 0:
            continue
        value = math.log((c * total_pairs) / denom)
        if value > 0:
            ppmi[(a, b)] = value
            ppmi[(b, a)] = value
    return ppmi, target_count, total_pairs, len(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, nargs="+", default=[10, 60])
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    ppmi, target_count, total_pairs, pair_count = build_ppmi(args.train_csv)
    print(f"co-occurring pairs {pair_count}, retained PPMI entries {len(ppmi) // 2}")

    with open(args.index, "r", encoding="utf-8") as f:
        raw_index = json.load(f)
    sid_to_items = collections.defaultdict(list)
    for item_id, tokens in raw_index.items():
        sid_to_items["".join(tokens)].append(item_id)
    item_to_sid = {i: s for s, v in sid_to_items.items() for i in v}

    test_rows = list(csv.DictReader(open(args.test_csv, newline="", encoding="utf-8")))
    data = load_result(args.result)
    assert len(test_rows) == len(data)
    n = len(test_rows)

    def candidate_items(pred):
        items = sid_to_items.get(pred)
        if not items:
            return []
        return [int(i) for i in items]

    variants = collections.defaultdict(lambda: [0, 0])
    hits = collections.defaultdict(list)
    strict_ceiling = 0
    unique_target_rows = 0
    user_of_row = []
    baseline_code = baseline_strict = 0

    for row_data, entry in zip(test_rows, data):
        target_item = row_data["item_id"].strip()
        target_sid = item_to_sid.get(target_item)
        preds = [normalize(p) for p in entry.get("predict", [])]
        user_of_row.append(row_data.get("user_id", ""))

        def code_hit(order):
            for idx in order[: args.topk]:
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
        hits["baseline"].append(strict_hit(natural))

        decoded = sid_to_items.get(target_sid) or []
        scoreable = len(decoded) == 1 and decoded[0] == target_item
        if scoreable:
            unique_target_rows += 1
        if scoreable and target_sid in preds:
            strict_ceiling += 1
        if target_sid not in preds:
            continue

        history = parse_ids(row_data.get("history_item_id", ""))
        history = [h for h in history if h != int(target_item)]
        if not history:
            continue

        scores = []
        for pred in preds:
            items = candidate_items(pred)
            if not items:
                scores.append(None)
                continue
            total = 0.0
            for item in items:
                running = 0.0
                for h in history:
                    value = ppmi.get((item, h))
                    if value:
                        running += value
                total += running / len(items)
            scores.append(total)

        cooc_order = sorted(
            natural,
            key=lambda i: (-(scores[i] if scores[i] is not None else -1.0), i),
        )
        pop_order = sorted(
            natural,
            key=lambda i: (-max(
                (target_count.get(str(it), 0) for it in candidate_items(preds[i])),
                default=0,
            ), i),
        )
        variants["cooc_only"][0] += code_hit(cooc_order)
        variants["cooc_only"][1] += strict_hit(cooc_order)
        hits["cooc_only"].append(strict_hit(cooc_order))
        variants["popularity_only"][0] += code_hit(pop_order)
        variants["popularity_only"][1] += strict_hit(pop_order)
        hits["popularity_only"].append(strict_hit(pop_order))

        cooc_rank = [0] * len(preds)
        for position, idx in enumerate(cooc_order):
            cooc_rank[idx] = position + 1
        for k in args.rrf_k:
            fused = sorted(
                natural,
                key=lambda i: -(1.0 / (k + i + 1) + 1.0 / (k + cooc_rank[i])),
            )
            variants[f"rrf_k{k}"][0] += code_hit(fused)
            variants[f"rrf_k{k}"][1] += strict_hit(fused)
            hits[f"rrf_k{k}"].append(strict_hit(fused))

    # rows that were skipped keep their baseline value, which equals zero when the
    # target never appears in the beam
    for name, values in hits.items():
        if len(values) < n:
            values.extend([0.0] * (n - len(values)))

    clusters = collections.defaultdict(list)
    for position, user in enumerate(user_of_row):
        clusters[user].append(position)
    cluster_lists = list(clusters.values())

    def paired_ci(a, b, rounds=2000, seed=13):
        rng = random.Random(seed)
        diffs = []
        for _ in range(rounds):
            idx = [rng.randrange(n) for _ in range(n)]
            diffs.append(
                sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n
            )
        diffs.sort()
        return diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds) - 1]

    def cluster_ci(a, b, rounds=2000, seed=17):
        rng = random.Random(seed)
        diffs = []
        for _ in range(rounds):
            picked = [cluster_lists[rng.randrange(len(cluster_lists))]
                      for _ in range(len(cluster_lists))]
            idx = [i for group in picked for i in group]
            m = len(idx)
            diffs.append(
                sum(b[i] for i in idx) / m - sum(a[i] for i in idx) / m
            )
        diffs.sort()
        return diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds) - 1]

    report = {
        "label": args.label,
        "rows": n,
        "users": len(cluster_lists),
        "baseline": {
            "code_HR": round(baseline_code / n, 6),
            "strict_HR": round(baseline_strict / n, 6),
        },
        "rows_with_strictly_scoreable_target": round(unique_target_rows / n, 6),
        "strict_reachable_ceiling": round(strict_ceiling / n, 6),
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
    for name in sorted(hits):
        if name == "baseline":
            continue
        lo, hi = paired_ci(hits["baseline"], hits[name])
        clo, chi = cluster_ci(hits["baseline"], hits[name])
        report["variants"][name]["strict_ci95"] = [round(lo, 6), round(hi, 6)]
        report["variants"][name]["strict_ci95_by_user"] = [round(clo, 6), round(chi, 6)]
        report["variants"][name]["significant"] = not (clo <= 0 <= chi)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
