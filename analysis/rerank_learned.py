#!/usr/bin/env python3
"""Learn a collaborative reranker on top of the frozen beam candidates.

The beam diagnostic shows two things at once. Most of the loss is on the
generation side, and about 8.7 points of strict item-level headroom sits in the
ordering of the candidates that were generated. A hand-tuned co-occurrence
probe already recovered 1.3 of those points, so this script replaces the hand
tuning with a small learned scorer and a single validation step.

Modes:

  fit      Train a logistic scorer on the training split alone. Positives are
           the true next item. Negatives mirror what a beam contains, so they
           are drawn from popular items, from co-occurrence neighbours of the
           history, and from the history itself.
  ceiling  Measure how much new recall a co-occurrence candidate pool adds on
           top of the beam. Pure diagnostic, no training, no GPU.
  select   Evaluate the fusion between the learned score and the model's own
           beam rank on the validation split, then freeze one configuration.
  apply    Re-score the candidates with the frozen configuration and report all
           three evaluation protocols with paired confidence intervals.

Everything runs on the standard library so it works on the laptop without a
GPU. Only the beam export itself needs the server.

Usage:
    python scripts/rerank_learned.py fit \
        --train-csv .../train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
        --index .../index/Industrial_and_Scientific.index.json \
        --model results/rerank_model.json

    python scripts/rerank_learned.py select \
        --result eval_valid_sft/result_final_checkpoint.json \
        --csv .../valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
        --train-csv .../train/...csv --index .../index.json \
        --item-meta .../index/Industrial_and_Scientific.item.json \
        --model results/rerank_model.json --config results/rerank_config.json

    python scripts/rerank_learned.py apply \
        --result eval_sft_full/result_final_checkpoint.json \
        --csv .../test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
        --train-csv .../train/...csv --index .../index.json \
        --item-meta .../index/Industrial_and_Scientific.item.json \
        --model results/rerank_model.json --config results/rerank_config.json \
        --label sft_full
"""

import argparse
import bisect
import collections
import csv
import json
import math
import random
import re


FEATURES = [
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
]


# --------------------------------------------------------------------------
# readers


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


def product_key(title):
    text = str(title).lower()
    text = re.sub(r"[\s\u0000-\u001f]+", " ", text)
    return text.strip()


# --------------------------------------------------------------------------
# catalog and training statistics


class Catalog:
    """Item ids grouped by SID plus the SID prefix structure."""

    def __init__(self, index_path, item_meta_path=None):
        with open(index_path, "r", encoding="utf-8") as f:
            raw_index = json.load(f)
        self.sid_to_items = collections.defaultdict(list)
        self.item_to_sid = {}
        for item_id, tokens in raw_index.items():
            sid = "".join(tokens)
            self.sid_to_items[sid].append(item_id)
            self.item_to_sid[int(item_id)] = sid
        self.sid_tokens = {sid: self._tokens_of(sid) for sid in self.sid_to_items}
        self.sid_l1 = {}
        self.sid_l2 = {}
        for sid in self.sid_to_items:
            parts = re.findall(r"<([abc])_(\d+)>", sid)
            key1 = ""
            key2 = ""
            for level, value in parts:
                if level == "a":
                    key1 = value
                    key2 = value
                elif level == "b":
                    key2 += "_" + value
            self.sid_l1[sid] = key1
            self.sid_l2[sid] = key2
        self.key_to_sid = None
        if item_meta_path:
            with open(item_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            keys = {item: product_key(meta[item]["title"]) for item in meta}
            self.sid_to_keys = {
                sid: sorted({keys[i] for i in items})
                for sid, items in self.sid_to_items.items()
            }
            self.item_to_key = {
                int(item): self.sid_to_keys[sid][0]
                for sid, items in self.sid_to_items.items()
                if len(self.sid_to_keys[sid]) == 1
                for item in items
            }
        else:
            self.sid_to_keys = None
            self.item_to_key = {}

    def _tokens_of(self, sid):
        return re.findall(r"<[abc]_\d+>", sid)


class TrainStats:
    """Everything the scorer needs, computed from the training split only."""

    def __init__(self, train_csv, catalog):
        self.catalog = catalog
        pairs = collections.Counter()
        self.item_target = collections.Counter()
        self.item_hist = collections.Counter()
        total_targets = 0
        histories = 0
        for row in csv.DictReader(open(train_csv, newline="", encoding="utf-8")):
            target = int(row["item_id"].strip())
            self.item_target[target] += 1
            total_targets += 1
            hist = sorted(set(parse_ids(row.get("history_item_id", ""))))
            if not hist:
                continue
            histories += 1
            for i, item in enumerate(hist):
                self.item_hist[item] += 1
                for j in range(i + 1, len(hist)):
                    pairs[(hist[i], hist[j])] += 1
        self.total_targets = total_targets
        self.histories = histories
        total_pairs = sum(pairs.values())
        self.pairs = pairs
        ppmi = {}
        for (a, b), count in pairs.items():
            denom = self.item_hist[a] * self.item_hist[b]
            if denom <= 0:
                continue
            value = math.log((count * total_pairs) / denom)
            if value > 0:
                ppmi[(a, b)] = value
                ppmi[(b, a)] = value
        self.ppmi = ppmi
        adjacency = collections.defaultdict(list)
        for (a, b), value in ppmi.items():
            adjacency[a].append((b, value))
        for item in adjacency:
            adjacency[item].sort(key=lambda pair: (-pair[1], pair[0]))
        self.adjacency = adjacency
        # popularity sampler over items, used for negative mining
        items = sorted(self.item_target)
        weights = [self.item_target[i] for i in items]
        cumulative = []
        running = 0.0
        for weight in weights:
            running += weight
            cumulative.append(running)
        self.pop_items = items
        self.pop_cumulative = cumulative
        self.pop_total = running
        self.top_popular = [item for item, _ in self.item_target.most_common(60)]
        self.all_items = sorted(catalog.item_to_sid)
        # coarse code priors
        self.l1_target = collections.Counter()
        self.l2_target = collections.Counter()
        for item, count in self.item_target.items():
            sid = catalog.item_to_sid.get(item)
            if sid is None:
                continue
            self.l1_target[catalog.sid_l1[sid]] += count
            self.l2_target[catalog.sid_l2[sid]] += count

    def sample_popular(self, rng):
        position = bisect.bisect_left(
            self.pop_cumulative, rng.random() * self.pop_total
        )
        if position >= len(self.pop_items):
            position = len(self.pop_items) - 1
        return self.pop_items[position]

    def sample_neighbour(self, rng, history):
        """A co-occurrence neighbour of one history item, if there is any."""
        neighbours = None
        for item in rng.sample(history, len(history)):
            pool = self.adjacency.get(item)
            if pool:
                neighbours = pool
                break
        if not neighbours:
            return None
        head = neighbours[:200]
        weights = [value for _, value in head]
        total = sum(weights)
        pick = rng.random() * total
        running = 0.0
        for (item, value), weight in zip(head, weights):
            running += weight
            if running >= pick:
                return item
        return head[-1][0]


# --------------------------------------------------------------------------
# features


class RowContext:
    """History side values that every candidate of one row shares."""

    def __init__(self, stats, catalog, history):
        self.stats = stats
        self.catalog = catalog
        self.history = history
        self.unique = sorted(set(history))
        self.length = len(history)
        self.sqrt_len = math.sqrt(self.length) if self.length else 1.0
        self.history_set = set(history)
        self.last = history[-1] if history else None
        self.l1_count = collections.Counter()
        self.l2_count = collections.Counter()
        for item in self.unique:
            sid = catalog.item_to_sid.get(item)
            if sid is None:
                continue
            self.l1_count[catalog.sid_l1[sid]] += 1
            self.l2_count[catalog.sid_l2[sid]] += 1
        self.denom = max(len(self.unique), 1)

    def vector(self, item):
        """Features for one candidate item."""
        stats = self.stats
        catalog = self.catalog
        ppmi_sum = 0.0
        ppmi_max = 0.0
        cooc_norm = 0.0
        for other in self.unique:
            value = stats.ppmi.get((item, other))
            if value:
                ppmi_sum += value
                if value > ppmi_max:
                    ppmi_max = value
            count = stats.pairs.get((item, other)) or stats.pairs.get((other, item))
            if count:
                cooc_norm += count / max(stats.item_hist[other], 1)
        sid = catalog.item_to_sid.get(item)
        l1 = catalog.sid_l1.get(sid, "")
        l2 = catalog.sid_l2.get(sid, "")
        last_sid = catalog.item_to_sid.get(self.last) if self.last is not None else None
        return [
            ppmi_sum / self.sqrt_len,
            ppmi_max,
            cooc_norm / self.sqrt_len,
            math.log1p(stats.item_target.get(item, 0)),
            1.0 if item in self.history_set else 0.0,
            1.0 if self.last is not None and item == self.last else 0.0,
            self.l1_count.get(l1, 0) / self.denom,
            self.l2_count.get(l2, 0) / self.denom,
            1.0 if last_sid is not None and catalog.sid_l1[last_sid] == l1 else 0.0,
            1.0 if last_sid is not None and catalog.sid_l2[last_sid] == l2 else 0.0,
            math.log1p(self.length),
            1.0 if len(catalog.sid_to_items.get(sid, ())) > 1 else 0.0,
            math.log1p(stats.l1_target.get(l1, 0)) - math.log1p(stats.total_targets),
            math.log1p(stats.l2_target.get(l2, 0)) - math.log1p(stats.total_targets),
        ]

    def sid_vector(self, sid):
        """Average the item features over the items that share one SID."""
        items = self.catalog.sid_to_items.get(sid)
        if not items:
            return None
        if len(items) == 1:
            return self.vector(int(items[0]))
        total = [0.0] * len(FEATURES)
        for item in items:
            values = self.vector(int(item))
            for position, value in enumerate(values):
                total[position] += value
        return [value / len(items) for value in total]


# --------------------------------------------------------------------------
# training


def build_pool(stats, context, pool_size=60, popular_size=40):
    """A candidate pool that looks like a beam.

    Beam candidates are mostly popular items together with a few items that
    share structure with the history, so the negatives used for training are
    drawn from the same mixture.
    """
    neighbours = []
    for item in context.unique:
        neighbours.extend(stats.adjacency.get(item, ())[:200])
    neighbours.sort(key=lambda pair: (-pair[1], pair[0]))
    pool = []
    seen = set()
    for item, _ in neighbours:
        if item in seen:
            continue
        seen.add(item)
        pool.append(item)
        if len(pool) >= pool_size:
            break
    for item in stats.top_popular[:popular_size]:
        if item in seen:
            continue
        seen.add(item)
        pool.append(item)
    return pool


def build_examples(stats, catalog, train_csv, neg_per_pos, rng, limit=0,
                   pool_mode="mixed"):
    """Yield (features, label) pairs for one pass over the training split."""
    wanted = {"popular": 0.4, "neighbour": 0.3, "history": 0.3}
    rows = list(csv.DictReader(open(train_csv, newline="", encoding="utf-8")))
    if limit:
        rows = rows[:limit]
    for row in rows:
        target = int(row["item_id"].strip())
        history = parse_ids(row.get("history_item_id", ""))
        if not history:
            continue
        context = RowContext(stats, catalog, history)
        yield context.vector(target), 1.0
        seen = {target}
        pool = None
        if pool_mode == "beamlike":
            pool = [item for item in build_pool(stats, context) if item not in seen]
        for _ in range(neg_per_pos):
            if pool:
                item = pool[rng.randrange(len(pool))]
                if item in seen:
                    continue
                seen.add(item)
                yield context.vector(item), 0.0
                continue
            pick = rng.random()
            if pick < wanted["popular"]:
                item = stats.sample_popular(rng)
            elif pick < wanted["popular"] + wanted["neighbour"]:
                item = stats.sample_neighbour(rng, list(context.unique))
                if item is None:
                    item = stats.sample_popular(rng)
            else:
                item = rng.choice(history)
            if item in seen:
                continue
            seen.add(item)
            yield context.vector(item), 0.0


def fit(args):
    catalog = Catalog(args.index, args.item_meta)
    stats = TrainStats(args.train_csv, catalog)
    selected = args.features or list(FEATURES)
    for name in selected:
        if name not in FEATURES:
            raise SystemExit(f"unknown feature {name}")
    indexes = [FEATURES.index(name) for name in selected]
    print(f"train histories {stats.histories}, targets {stats.total_targets}")
    print(f"co-occurrence pairs {len(stats.pairs)}, "
          f"positive PPMI entries {len(stats.ppmi) // 2}")
    print(f"features {selected}")
    rng = random.Random(args.seed)
    # pass 1, mean and variance of every feature on the training sample
    count = 0
    total = [0.0] * len(FEATURES)
    total_sq = [0.0] * len(FEATURES)
    for features, _ in build_examples(stats, catalog, args.train_csv,
                                      args.neg_per_pos, rng, args.limit_rows,
                                      args.pool_mode):
        count += 1
        for i, value in enumerate(features):
            total[i] += value
            total_sq[i] += value * value
    mean = [value / count for value in total]
    scale = []
    for i in range(len(FEATURES)):
        variance = total_sq[i] / count - mean[i] * mean[i]
        scale.append(math.sqrt(variance) if variance > 1e-9 else 1.0)
    print(f"fit examples {count}")

    weights = [0.0] * len(selected)
    bias = 0.0
    rng = random.Random(args.seed + 1)
    for epoch in range(args.epochs):
        rng2 = random.Random(args.seed + 100 + epoch)
        seen = 0
        loss = 0.0
        for features, label in build_examples(stats, catalog, args.train_csv,
                                              args.neg_per_pos, rng2,
                                              args.limit_rows, args.pool_mode):
            seen += 1
            z = bias
            scaled = []
            for position, index in enumerate(indexes):
                value = (features[index] - mean[index]) / scale[index]
                scaled.append(value)
                z += weights[position] * value
            z = max(-30.0, min(30.0, z))
            prob = 1.0 / (1.0 + math.exp(-z))
            error = prob - label
            loss += -math.log(prob if label > 0.5 else 1.0 - prob)
            step = args.lr * (1.0 / math.sqrt(1.0 + seen / 20000.0))
            for i, value in enumerate(scaled):
                weights[i] -= step * (error * value + args.l2 * weights[i])
            bias -= step * error
        print(f"epoch {epoch} logloss {loss / max(seen, 1):.4f} examples {seen}")

    model = {
        "features": selected,
        "index": indexes,
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "bias": bias,
        "train_csv": args.train_csv,
        "neg_per_pos": args.neg_per_pos,
        "epochs": args.epochs,
        "seed": args.seed,
        "pool_mode": args.pool_mode,
    }
    write_json(args.model, model)
    print("weights, largest first")
    for name, value in sorted(zip(selected, weights), key=lambda pair: -abs(pair[1])):
        print(f"  {name:14s} {value:+.4f}")


def load_model(path):
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    if "index" not in model:
        model["index"] = [FEATURES.index(name) for name in model["features"]]
    if len(model["index"]) != len(model["weights"]):
        raise SystemExit("model feature list does not match this script")
    return model


def logit(model, values):
    z = model["bias"]
    for position, index in enumerate(model["index"]):
        z += (model["weights"][position]
              * (values[index] - model["mean"][index]) / model["scale"][index])
    return z


# --------------------------------------------------------------------------
# scoring one split


def score_candidates(stats, catalog, beam, rows, model, augment=0):
    """Collect the feature vector of every candidate of every row.

    The model is applied afterwards so that swapping or ablating weights does
    not require recomputing the features.
    """
    scored_rows = []
    info = {"rows_with_target_in_beam": 0, "augmented_rows": 0}
    for row, entry in zip(rows, beam):
        target_item = int(row["item_id"].strip())
        target_sid = catalog.item_to_sid.get(target_item)
        history = parse_ids(row.get("history_item_id", ""))
        candidates = []
        seen_sid = set()
        for sid in entry.get("predict", []):
            sid = normalize(sid)
            if sid in seen_sid:
                continue
            seen_sid.add(sid)
            candidates.append(sid)
        if target_sid in seen_sid:
            info["rows_with_target_in_beam"] += 1
        context = RowContext(stats, catalog, history)
        scored = []
        for rank, sid in enumerate(candidates):
            values = context.sid_vector(sid)
            if values is None:
                continue
            scored.append([sid, 0.0, rank + 1, values])
        if augment:
            pool = set()
            for item in context.unique:
                for neighbour, _ in stats.adjacency.get(item, ())[:3000]:
                    pool.add(neighbour)
            fresh = []
            for item in pool:
                sid = catalog.item_to_sid.get(item)
                if sid is None or sid in seen_sid:
                    continue
                values = context.vector(item)
                fresh.append([sid, 0.0, len(candidates) + 1, values])
            fresh.sort(key=lambda candidate: -logit(model, candidate[3]))
            keep = []
            taken = set()
            for candidate in fresh:
                if candidate[0] in taken:
                    continue
                taken.add(candidate[0])
                keep.append(candidate)
                if len(keep) >= augment:
                    break
            if keep:
                info["augmented_rows"] += 1
            scored.extend(keep)
        scored_rows.append(scored)
    apply_model(scored_rows, model)
    return scored_rows, info


def apply_model(scored_rows, model):
    for scored in scored_rows:
        for candidate in scored:
            candidate[1] = logit(model, candidate[3])


def orders_from_scores(scored_rows, lam, rrf_k):
    """Baseline order, learned-only order and their fusion, per row.

    The fusion is the same reciprocal rank rule the hand-tuned probe used, so
    lambda equal to zero reproduces the untouched beam order exactly.
    """
    base_orders = []
    learned_orders = []
    fused_orders = []
    for scored in scored_rows:
        learned_rank = {}
        for position, candidate in enumerate(
            sorted(scored, key=lambda item: (-item[1], item[2]))
        ):
            learned_rank[candidate[0]] = position + 1
        by_model = sorted(scored, key=lambda item: item[2])
        by_learned = sorted(scored,
                            key=lambda item: (learned_rank[item[0]], item[2]))
        base_orders.append([candidate[0] for candidate in by_model])
        learned_orders.append([candidate[0] for candidate in by_learned])
        fused = sorted(
            scored,
            key=lambda item: -(1.0 / (rrf_k + item[2])
                               + lam / (rrf_k + learned_rank[item[0]])),
        )
        fused_orders.append([candidate[0] for candidate in fused])
    return {
        "base": base_orders,
        "learned": learned_orders,
        "fused": fused_orders,
    }


def score_orders(catalog, rows, orders, topk):
    """Three protocols plus the strict ceiling of this candidate pool."""
    n = len(rows)
    code_hits = 0
    strict_hits = 0
    product_hits = 0
    fraction_hits = 0.0
    ceiling = 0
    per_row_strict = [0.0] * n
    per_row_code = [0.0] * n
    per_row_product = [0.0] * n
    users = []
    for position, (row, order) in enumerate(zip(rows, orders)):
        target_item = row["item_id"].strip()
        target_key = None
        if catalog.sid_to_keys is not None:
            target_key = catalog.item_to_key.get(int(target_item))
        target_sid = catalog.item_to_sid.get(int(target_item))
        users.append(row.get("user_id", ""))
        if target_sid is not None:
            items = catalog.sid_to_items[target_sid]
            if len(items) == 1 and items[0] == target_item and target_sid in order:
                ceiling += 1
        for rank, sid in enumerate(order):
            if rank >= topk:
                break
            if sid == target_sid:
                code_hits += 1
                per_row_code[position] = 1.0
                break
        for rank, sid in enumerate(order):
            if rank >= topk:
                break
            items = catalog.sid_to_items.get(sid)
            if not items:
                continue
            if len(items) == 1 and items[0] == target_item:
                strict_hits += 1
                per_row_strict[position] = 1.0
                break
        if catalog.sid_to_keys is not None:
            for rank, sid in enumerate(order):
                if rank >= topk:
                    break
                keys = catalog.sid_to_keys.get(sid)
                if not keys:
                    continue
                if len(keys) == 1 and keys[0] == target_key:
                    product_hits += 1
                    per_row_product[position] = 1.0
                    fraction_hits += 1.0
                    break
                if target_key in keys:
                    fraction_hits += 1.0 / len(keys)
                    break
    report = {
        "rows": n,
        "code_match_HR": round(code_hits / n, 6),
        "item_unique_HR": round(strict_hits / n, 6),
        "strict_reachable_ceiling": round(ceiling / n, 6),
        "per_row": {
            "code": per_row_code,
            "strict": per_row_strict,
            "product": per_row_product,
        },
        "users": users,
    }
    if catalog.sid_to_keys is not None:
        report["product_unique_HR"] = round(product_hits / n, 6)
        report["product_fraction_HR"] = round(fraction_hits / n, 6)
    return report


def paired_ci(a, b, rounds=2000, seed=13):
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(rounds):
        total = 0.0
        for _ in range(n):
            index = rng.randrange(n)
            total += b[index] - a[index]
        diffs.append(total / n)
    diffs.sort()
    return diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds) - 1]


def cluster_ci(a, b, users, rounds=2000, seed=17):
    groups = collections.defaultdict(list)
    for position, user in enumerate(users):
        groups[user].append(position)
    lists = list(groups.values())
    rng = random.Random(seed)
    diffs = []
    for _ in range(rounds):
        picked = [lists[rng.randrange(len(lists))] for _ in range(len(lists))]
        indexes = [i for group in picked for i in group]
        count = len(indexes)
        total = 0.0
        for i in indexes:
            total += b[i] - a[i]
        diffs.append(total / count)
    diffs.sort()
    return diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds) - 1]


def compare(catalog, rows, base, new_orders, topk, label, variant, base_orders=None):
    """Score one variant against the shared baseline."""
    if base is None:
        base = score_orders(catalog, rows, base_orders, topk)
    new = score_orders(catalog, rows, new_orders, topk)
    out = {
        "label": label,
        "variant": variant,
        "rows": base["rows"],
        "topk": topk,
        "strict_reachable_ceiling": base["strict_reachable_ceiling"],
        "baseline": {
            "code_match_HR": base["code_match_HR"],
            "item_unique_HR": base["item_unique_HR"],
        },
        "variant_metrics": {
            "code_match_HR": new["code_match_HR"],
            "item_unique_HR": new["item_unique_HR"],
        },
        "delta": {
            "code_match_HR": round(new["code_match_HR"] - base["code_match_HR"], 6),
            "item_unique_HR": round(new["item_unique_HR"] - base["item_unique_HR"], 6),
        },
    }
    if "product_unique_HR" in base:
        out["baseline"]["product_unique_HR"] = base["product_unique_HR"]
        out["baseline"]["product_fraction_HR"] = base["product_fraction_HR"]
        out["variant_metrics"]["product_unique_HR"] = new["product_unique_HR"]
        out["variant_metrics"]["product_fraction_HR"] = new["product_fraction_HR"]
        out["delta"]["product_unique_HR"] = round(
            new["product_unique_HR"] - base["product_unique_HR"], 6
        )
        out["delta"]["product_fraction_HR"] = round(
            new["product_fraction_HR"] - base["product_fraction_HR"], 6
        )
    for name in ("item_unique", "code_match", "product_unique"):
        if name == "product_unique" and "product_unique_HR" not in base:
            continue
        key = {"item_unique": "strict", "code_match": "code",
               "product_unique": "product"}[name]
        low, high = paired_ci(base["per_row"][key], new["per_row"][key])
        low_u, high_u = cluster_ci(base["per_row"][key], new["per_row"][key],
                                   base["users"])
        out[f"{name}_delta_ci95"] = [round(low, 6), round(high, 6)]
        out[f"{name}_delta_ci95_by_user"] = [round(low_u, 6), round(high_u, 6)]
    return out


# --------------------------------------------------------------------------
# modes


def read_split(args):
    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    beam = load_result(args.result)
    if len(rows) != len(beam):
        raise SystemExit(f"{len(rows)} csv rows vs {len(beam)} predictions")
    return rows, beam


def run_apply(args):
    catalog = Catalog(args.index, args.item_meta)
    stats = TrainStats(args.train_csv, catalog)
    model = load_model(args.model)
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    if args.lambda_override is not None:
        config["lambda"] = args.lambda_override
    if args.rrf_k_override is not None:
        config["rrf_k"] = args.rrf_k_override
    rows, beam = read_split(args)
    scored_rows, info = score_candidates(stats, catalog, beam, rows, model,
                                         augment=args.augment)
    orders = orders_from_scores(scored_rows, config.get("lambda", 0.0),
                                config.get("rrf_k", 10))
    base = score_orders(catalog, rows, orders["base"], args.topk)
    report = {
        "label": args.label,
        "config": config,
        "augment": args.augment,
        "candidate_pool": {
            "rows_with_target_in_beam": info["rows_with_target_in_beam"],
            "strict_reachable_ceiling": base["strict_reachable_ceiling"],
            "code_reachable_ceiling": round(
                info["rows_with_target_in_beam"] / base["rows"], 6),
        },
        "variants": {},
    }
    for name in ("base", "learned", "fused"):
        report["variants"][name] = compare(
            catalog, rows, base, orders[name], args.topk, args.label, name
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        write_json(args.json_out, report)
    return report


def run_select(args):
    """Freeze one configuration on the validation split.

    The candidate configurations are the pre-declared feature subsets crossed
    with the fusion weight and the rank constant of the reciprocal rank term.
    Everything is scored on the validation split and one configuration is
    written out. The test split is not touched here.
    """
    catalog = Catalog(args.index, args.item_meta)
    stats = TrainStats(args.train_csv, catalog)
    rows, beam = read_split(args)
    models = [(path, load_model(path)) for path in args.models]
    baseline = None
    scored_by_model = {}
    learned = {}
    for path, model in models:
        scored_rows, info = score_candidates(stats, catalog, beam, rows, model,
                                             augment=args.augment)
        scored_by_model[path] = scored_rows
        orders = orders_from_scores(scored_rows, 0.0, 10)
        if baseline is None:
            baseline = score_orders(catalog, rows, orders["base"], args.topk)
        learned[path] = score_orders(catalog, rows, orders["learned"], args.topk)
    print(f"baseline valid strict {baseline['item_unique_HR']:.6f} "
          f"code {baseline['code_match_HR']:.6f} "
          f"product {baseline.get('product_unique_HR')} "
          f"ceiling {baseline['strict_reachable_ceiling']:.6f}")
    for path, model in models:
        report = learned[path]
        print(f"learned only {path} strict {report['item_unique_HR']:.6f} "
              f"code {report['code_match_HR']:.6f} "
              f"product {report.get('product_unique_HR')}")
    grid = []
    for path, model in models:
        scored_rows = scored_by_model[path]
        for lam in args.lambdas:
            for rrf_k in args.rrf_ks:
                orders = orders_from_scores(scored_rows, lam, rrf_k)
                report = score_orders(catalog, rows, orders["fused"], args.topk)
                grid.append({
                    "model": path,
                    "features": model["features"],
                    "lambda": lam,
                    "rrf_k": rrf_k,
                    "item_unique_HR": report["item_unique_HR"],
                    "code_match_HR": report["code_match_HR"],
                    "product_unique_HR": report.get("product_unique_HR"),
                })
                print(f"{path.split('/')[-1]:16s} lambda {lam:<5} k {rrf_k:<4} "
                      f"strict {report['item_unique_HR']:.6f} "
                      f"code {report['code_match_HR']:.6f} "
                      f"product {report.get('product_unique_HR')}")
    best = max(grid, key=lambda item: (item["item_unique_HR"],
                                       item["product_unique_HR"] or 0.0,
                                       -item["lambda"], -item["rrf_k"]))
    best = dict(best)
    best["augment"] = args.augment
    best["selected_on"] = args.csv
    best["learned_only_item_unique_HR"] = learned[best["model"]]["item_unique_HR"]
    best["baseline_item_unique_HR"] = baseline["item_unique_HR"]
    best["baseline_code_match_HR"] = baseline["code_match_HR"]
    best["grid_size"] = len(grid)
    print(json.dumps(best, indent=2))
    write_json(args.config, best)
    write_json(args.grid_out, {"baseline": {
        "item_unique_HR": baseline["item_unique_HR"],
        "code_match_HR": baseline["code_match_HR"],
        "product_unique_HR": baseline.get("product_unique_HR"),
        "strict_reachable_ceiling": baseline["strict_reachable_ceiling"],
    }, "learned_only": {
        path: {
            "item_unique_HR": learned[path]["item_unique_HR"],
            "code_match_HR": learned[path]["code_match_HR"],
            "product_unique_HR": learned[path].get("product_unique_HR"),
        } for path, _ in models
    }, "grid": sorted(grid, key=lambda item: -item["item_unique_HR"]),
       "selected": best})


def run_ceiling(args):
    catalog = Catalog(args.index, args.item_meta)
    stats = TrainStats(args.train_csv, catalog)
    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    beam = load_result(args.result) if args.result else None
    sizes = [int(value) for value in args.sizes]
    hit = {size: 0 for size in sizes}
    hit_union = {size: 0 for size in sizes}
    rows_with_pool = 0
    for position, row in enumerate(rows):
        target_item = int(row["item_id"].strip())
        target_sid = catalog.item_to_sid.get(target_item)
        history = parse_ids(row.get("history_item_id", ""))
        context = RowContext(stats, catalog, history)
        scores = collections.defaultdict(float)
        for item in context.unique:
            for neighbour, value in stats.adjacency.get(item, ()):
                scores[neighbour] += value
        if not scores:
            pool = []
        else:
            pool = [item for item, _ in sorted(scores.items(),
                                               key=lambda pair: (-pair[1], pair[0]))]
            rows_with_pool += 1
        if beam is not None:
            beam_sids = {normalize(sid) for sid in beam[position].get("predict", [])}
        else:
            beam_sids = set()
        for size in sizes:
            top = pool[:size]
            sids = {catalog.item_to_sid.get(item) for item in top}
            if target_sid in sids:
                hit[size] += 1
            if target_sid in (sids | beam_sids):
                hit_union[size] += 1
    report = {
        "rows": len(rows),
        "rows_with_nonempty_cooccurrence_pool": rows_with_pool,
        "cooccurrence_pool": {size: round(hit[size] / len(rows), 6) for size in sizes},
        "cooccurrence_pool_plus_beam50": {
            size: round(hit_union[size] / len(rows), 6) for size in sizes
        },
    }
    if beam is not None:
        report["beam_only"] = round(
            sum(1 for position, row in enumerate(rows)
                if catalog.item_to_sid.get(int(row["item_id"].strip())) in
                {normalize(sid) for sid in beam[position].get("predict", [])})
            / len(rows), 6
        )
    print(json.dumps(report, indent=2))
    if args.json_out:
        write_json(args.json_out, report)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(target, need_result=True):
        if need_result:
            target.add_argument("--result", required=True)
        target.add_argument("--train-csv", required=True)
        target.add_argument("--index", required=True)
        target.add_argument("--item-meta", default="")
        target.add_argument("--topk", type=int, default=10)
        target.add_argument("--augment", type=int, default=0)
        target.add_argument("--json_out", default="")

    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--train-csv", required=True)
    fit_parser.add_argument("--index", required=True)
    fit_parser.add_argument("--item-meta", default="")
    fit_parser.add_argument("--model", required=True)
    fit_parser.add_argument("--neg-per-pos", type=int, default=12)
    fit_parser.add_argument("--epochs", type=int, default=4)
    fit_parser.add_argument("--lr", type=float, default=0.15)
    fit_parser.add_argument("--l2", type=float, default=1e-5)
    fit_parser.add_argument("--seed", type=int, default=42)
    fit_parser.add_argument("--limit-rows", type=int, default=0)
    fit_parser.add_argument("--pool-mode", choices=["mixed", "beamlike"],
                            default="mixed")
    fit_parser.add_argument("--features", nargs="+", default=None,
                            help="subset of the feature zoo to fit")
    fit_parser.set_defaults(func=fit)

    ceiling_parser = sub.add_parser("ceiling")
    add_common(ceiling_parser)
    ceiling_parser.add_argument("--csv", required=True)
    ceiling_parser.add_argument("--sizes", type=int, nargs="+",
                                default=[10, 20, 50, 100, 200, 500])
    ceiling_parser.set_defaults(func=run_ceiling)

    select_parser = sub.add_parser("select")
    add_common(select_parser)
    select_parser.add_argument("--csv", required=True)
    select_parser.add_argument("--models", nargs="+", required=True)
    select_parser.add_argument("--config", required=True)
    select_parser.add_argument("--grid-out", required=True)
    select_parser.add_argument("--lambdas", type=float, nargs="+",
                               default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.5, 4.0])
    select_parser.add_argument("--rrf-ks", type=int, nargs="+", default=[10, 60])
    select_parser.set_defaults(func=run_select)

    apply_parser = sub.add_parser("apply")
    add_common(apply_parser)
    apply_parser.add_argument("--csv", required=True)
    apply_parser.add_argument("--model", default="")
    apply_parser.add_argument("--config", required=True)
    apply_parser.add_argument("--label", default="")
    apply_parser.add_argument("--lambda-override", type=float, default=None)
    apply_parser.add_argument("--rrf-k-override", type=int, default=None)
    apply_parser.set_defaults(func=run_apply)

    args = parser.parse_args()
    if args.mode == "apply" and not args.model:
        with open(args.config, "r", encoding="utf-8") as f:
            args.model = json.load(f).get("model", "")
        if not args.model:
            raise SystemExit("pass --model or record it in the config file")
    args.func(args)


if __name__ == "__main__":
    main()
