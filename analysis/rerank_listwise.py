"""Listwise reranker trained on the model's own beam candidates.

The first reranker (stage 16) was trained with pooled negatives and fused with
the beam rank through a weight picked on the validation split. Two things are
different here.

1. Training data is the model's own beam on a training split, so the negatives
   are exactly the candidates the reranker will see at inference time. Rows
   where the target is not in the beam are dropped, since no ordering can
   recover them.
2. The objective is listwise: one softmax over the candidates of a row, so the
   beam rank becomes a feature instead of a separate fusion weight, and the
   model learns the trade off between the beam rank and the collaborative
   features.

Three new features come out of the stage 17 measurement that the model's errors
at the first level are complementary to a plain co occurrence prior (76% each,
86.5% together). The prior ranks at the item, first level and second level are
given to the model directly.

Usage:
    python scripts/rerank_listwise.py train \
        --beam eval_trainbeam50/result_final_checkpoint.json \
        --csv subsets/Industrial_and_Scientific_train_10000.csv \
        --train-csv .../train/...csv --index .../index.json --model results/listwise_model.json

    python scripts/rerank_listwise.py select \
        --beam eval_valid_sft/result_final_checkpoint.json --csv .../valid/...csv \
        --model results/listwise_model.json --config results/listwise_config.json

    python scripts/rerank_listwise.py apply \
        --beam eval_sft_full/result_final_checkpoint.json --csv .../test/...csv \
        --config results/listwise_config.json --label test
"""

import argparse
import collections
import csv
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rerank_learned as base


FEATURES = [
    # collaborative evidence
    "ppmi_sum",
    "ppmi_max",
    "cooc_norm",
    "pop_log",
    "in_history",
    "last_same",
    "l1_frac",
    "l2_frac",
    "prev_l1",
    "prev_l2",
    "hist_len_log",
    "sid_shared",
    "pop_l1_rate",
    "pop_l2_rate",
    # beam position
    "rank_model",
    "is_top1",
    # co occurrence prior ranks, the complementary signal
    "prior_item_rank_log",
    "prior_l1_rank_log",
    "prior_l2_rank_log",
    # branch budget inside this row
    "branch_l1_slots",
    "branch_l2_slots",
]

SUBSETS = {
    "core_plus_prior": ["ppmi_sum", "pop_log", "in_history", "rank_model",
                        "prior_item_rank_log", "prior_l1_rank_log",
                        "prior_l2_rank_log"],
    "core_plus_prior_branch": ["ppmi_sum", "pop_log", "in_history", "rank_model",
                               "prior_item_rank_log", "prior_l1_rank_log",
                               "prior_l2_rank_log", "branch_l1_slots",
                               "branch_l2_slots", "is_top1"],
    "prior_only": ["rank_model", "prior_item_rank_log", "prior_l1_rank_log",
                   "prior_l2_rank_log"],
    "core_only": ["ppmi_sum", "pop_log", "in_history", "rank_model"],
    "all": list(FEATURES),
}


def prior_ranks(stats, catalog, history, popular_item_rank, popular_code_rank):
    """Rank items, first level codes and second level prefixes from co-occurrence."""
    scores = collections.defaultdict(float)
    for item in set(history):
        for neighbour, value in stats.adjacency.get(item, ()):
            scores[neighbour] += value
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    base_rank = len(ordered)
    item_rank = {item: position + 1 for position, (item, _) in enumerate(ordered)}
    level1_scores = collections.defaultdict(float)
    level2_scores = collections.defaultdict(float)
    for item, value in scores.items():
        sid = catalog.item_to_sid.get(item)
        if sid is None:
            continue
        level1_scores[catalog.sid_l1[sid]] += value
        level2_scores[catalog.sid_l2[sid]] += value
    level1_order = sorted(level1_scores.items(), key=lambda pair: (-pair[1], pair[0]))
    level2_order = sorted(level2_scores.items(), key=lambda pair: (-pair[1], pair[0]))
    level1_rank = {key: position + 1 for position, (key, _) in enumerate(level1_order)}
    level2_rank = {key: position + 1 for position, (key, _) in enumerate(level2_order)}
    level1_base = len(level1_order)
    level2_base = len(level2_order)

    def item_value(item):
        if item in item_rank:
            return item_rank[item]
        return base_rank + popular_item_rank.get(item, 9999)

    def level1_value(code):
        if code in level1_rank:
            return level1_rank[code]
        return level1_base + popular_code_rank.get(code, 999)

    def level2_value(key):
        if key in level2_rank:
            return level2_rank[key]
        return level2_base + popular_code_rank.get(key[0], 999)

    return item_value, level1_value, level2_value


def build_rows(args):
    catalog = base.Catalog(args.index, args.item_meta)
    stats = base.TrainStats(args.train_csv, catalog)
    popular_item_rank = {item: position + 1 for position, item in
                         enumerate(stats.all_items)}
    # rank of a first level code by how often it is a target
    level1_order = [code for code, _ in stats.l1_target.most_common()]
    popular_code_rank = {code: position + 1
                         for position, code in enumerate(level1_order)}

    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    beam = base.load_result(args.beam)
    if len(rows) != len(beam):
        raise SystemExit(f"{len(rows)} csv rows vs {len(beam)} predictions")

    examples = []
    for row, entry in zip(rows, beam):
        target_item = int(row["item_id"].strip())
        history = base.parse_ids(row.get("history_item_id", ""))
        context = base.RowContext(stats, catalog, history)
        item_value, level1_value, level2_value = prior_ranks(
            stats, catalog, history, popular_item_rank, popular_code_rank)
        candidates = []
        seen = set()
        for sid in entry.get("predict", []):
            sid = base.normalize(sid)
            if sid in seen:
                continue
            seen.add(sid)
            candidates.append(sid)
        if not candidates:
            continue
        branch_l1 = collections.Counter()
        branch_l2 = collections.Counter()
        for sid in candidates:
            branch_l1[sid[:7]] += 1
            branch_l2[sid[:14]] += 1

        target_sid = catalog.item_to_sid.get(target_item)
        features = []
        labels = []
        sids = []
        for rank, sid in enumerate(candidates):
            values = context.sid_vector(sid)
            if values is None:
                continue
            values = list(values)
            values.append(math.log1p(rank + 1))
            values.append(1.0 if rank == 0 else 0.0)
            items = [int(v) for v in catalog.sid_to_items[sid]]
            values.append(math.log1p(min(item_value(item) for item in items)))
            values.append(math.log1p(level1_value(catalog.sid_l1[sid])))
            values.append(math.log1p(level2_value(catalog.sid_l2[sid])))
            values.append(float(branch_l1[sid[:7]]))
            values.append(float(branch_l2[sid[:14]]))
            features.append(values)
            labels.append(1.0 if sid == target_sid else 0.0)
            sids.append(sid)
        examples.append({"features": features, "labels": labels,
                         "sids": sids, "target_sid": target_sid})
    return examples


def standardize(examples, indexes):
    count = 0
    total = [0.0] * len(indexes)
    total_sq = [0.0] * len(indexes)
    for example in examples:
        for values in example["features"]:
            count += 1
            for position, index in enumerate(indexes):
                value = values[index]
                total[position] += value
                total_sq[position] += value * value
    mean = [value / count for value in total]
    scale = []
    for position in range(len(indexes)):
        variance = total_sq[position] / count - mean[position] ** 2
        scale.append(math.sqrt(variance) if variance > 1e-9 else 1.0)
    return mean, scale


def fit(examples, names, epochs, lr, l2):
    indexes = [FEATURES.index(name) for name in names]
    mean, scale = standardize(examples, indexes)
    usable = [example for example in examples if any(example["labels"])]
    weights = [0.0] * len(indexes)
    for epoch in range(epochs):
        grad = [0.0] * len(indexes)
        loss = 0.0
        for example in usable:
            scores = []
            scaled = []
            for values in example["features"]:
                row = [(values[index] - mean[position]) / scale[position]
                       for position, index in enumerate(indexes)]
                scaled.append(row)
                scores.append(sum(w * v for w, v in zip(weights, row)))
            top = max(scores)
            exponentials = [math.exp(min(30.0, s - top)) for s in scores]
            total = sum(exponentials)
            probabilities = [value / total for value in exponentials]
            loss += -math.log(max(probabilities[example["labels"].index(1.0)], 1e-12))
            for row, probability, label in zip(scaled, probabilities,
                                               example["labels"]):
                error = probability - label
                for position in range(len(indexes)):
                    grad[position] += error * row[position]
        for position in range(len(indexes)):
            weights[position] -= lr * (grad[position] / len(usable)
                                       + l2 * weights[position])
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch} listwise loss {loss / len(usable):.4f}")
    return {"features": names, "index": indexes, "mean": mean, "scale": scale,
            "weights": weights, "rows_used": len(usable), "epochs": epochs}


def score(examples, model):
    orders = []
    for example in examples:
        scores = []
        for values in example["features"]:
            total = 0.0
            for position, index in enumerate(model["index"]):
                total += (model["weights"][position]
                          * (values[index] - model["mean"][position])
                          / model["scale"][position])
            scores.append(total)
        orders.append(scores)
    return orders


def subset_model(full, names):
    """Rebuild a model restricted to a named feature subset."""
    weight_of = dict(zip(full["features"], full["weights"]))
    return {
        "features": list(names),
        "index": [FEATURES.index(name) for name in names],
        "weights": [weight_of[name] for name in names],
        "mean": full["mean"],
        "scale": full["scale"],
    }


def orders_from_scores(examples, scores):
    """Rank each row's candidates by the model score."""
    orders = []
    for example, row_scores in zip(examples, scores):
        pairs = sorted(zip(row_scores, example["sids"]), key=lambda pair: -pair[0])
        orders.append([sid for _, sid in pairs])
    return orders


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    def common(target):
        target.add_argument("--beam", required=True)
        target.add_argument("--csv", required=True)
        target.add_argument("--train-csv", required=True)
        target.add_argument("--index", required=True)
        target.add_argument("--item-meta", required=True)

    train_parser = sub.add_parser("train")
    common(train_parser)
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--epochs", type=int, default=60)
    train_parser.add_argument("--lr", type=float, default=0.5)
    train_parser.add_argument("--l2", type=float, default=1e-4)

    select_parser = sub.add_parser("select")
    common(select_parser)
    select_parser.add_argument("--model", required=True)
    select_parser.add_argument("--config", required=True)

    apply_parser = sub.add_parser("apply")
    common(apply_parser)
    apply_parser.add_argument("--config", required=True)
    apply_parser.add_argument("--label", default="")
    apply_parser.add_argument("--json_out", default="")

    args = parser.parse_args()

    if args.mode == "train":
        examples = build_rows(args)
        print(f"rows {len(examples)}, with target in beam "
              f"{sum(1 for e in examples if any(e['labels']))}")
        model = fit(examples, FEATURES, args.epochs, args.lr, args.l2)
        with open(args.model, "w", encoding="utf-8") as f:
            json.dump(model, f, indent=1)
        print("weights")
        for name, value in sorted(zip(model["features"], model["weights"]),
                                  key=lambda pair: -abs(pair[1])):
            print(f"  {name:22s} {value:+.4f}")
        print(f"wrote {args.model}")
        return

    # evaluation modes share the row construction
    catalog = base.Catalog(args.index, args.item_meta)
    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    examples = build_rows(args)
    beam = base.load_result(args.beam)
    base_orders = [[base.normalize(sid) for sid in entry.get("predict", [])]
                   for entry in beam]
    baseline = base.score_orders(catalog, rows, base_orders, 10)

    if args.mode == "select":
        with open(args.model, "r", encoding="utf-8") as f:
            full = json.load(f)
        results = []
        for name, names in SUBSETS.items():
            model = subset_model(full, names)
            scores = score(examples, model)
            orders = orders_from_scores(examples, scores)
            report = base.score_orders(catalog, rows, orders, 10)
            results.append({
                "subset": name,
                "item_unique_HR": report["item_unique_HR"],
                "code_match_HR": report["code_match_HR"],
                "product_unique_HR": report.get("product_unique_HR"),
            })
            print(f"{name:26s} strict {report['item_unique_HR']:.5f} "
                  f"code {report['code_match_HR']:.5f} "
                  f"product {report.get('product_unique_HR')}")
        best = max(results, key=lambda item: (item["item_unique_HR"],
                                              item["product_unique_HR"] or 0.0))
        best = dict(best)
        best["features"] = list(SUBSETS[best["subset"]])
        best["model"] = args.model
        best["selected_on"] = args.csv
        best["baseline_item_unique_HR"] = baseline["item_unique_HR"]
        with open(args.config, "w", encoding="utf-8") as f:
            json.dump({"selected": best, "grid": results}, f, indent=1)
        print(json.dumps(best, indent=2))
        print(f"wrote {args.config}")
        return

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)["selected"]
    with open(config["model"], "r", encoding="utf-8") as f:
        full = json.load(f)
    names = config.get("features")
    if not names:
        names = SUBSETS[config["subset"]]
    model = subset_model(full, names)
    scores = score(examples, model)
    orders = orders_from_scores(examples, scores)
    report = base.compare(catalog, rows, baseline, orders, 10, args.label,
                          "listwise")
    report["config"] = {"subset": config.get("subset"), "model": config.get("model")}
    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
