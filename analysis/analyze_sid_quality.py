"""Measure how well a SID assignment aligns with user behaviour.

Collision rate and codebook usage describe the assignment itself. They say
nothing about whether items that share a prefix are the items a user actually
interacts with next. This script measures that alignment, using only the
training interactions and the index, so it runs on CPU.

Four behaviour-alignment statistics, each reported for the level-1 code and for
the full three-token SID:

  co_prefix      share of item pairs inside one user history that share a prefix
  majority_acc   accuracy of predicting the next target from the last history
                 item with a majority lookup table
  nmi            normalised mutual information between the same two variables
  entropy        entropy of the target distribution, for context

Usage:
    python analyze_sid_quality.py --index <index.json> --train-csv <train.csv> \
        --label "prebuilt" --json_out results/sid_quality_prebuilt.json
"""

import argparse
import collections
import csv
import json
import math
import random
import re

TOKENS = re.compile(r"<[abc]_\d+>")


def sid_of(index, item_id):
    return index[item_id]


def prefix(tokens, level):
    return "".join(tokens[:level])


def entropy(counts):
    total = sum(counts.values())
    return -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)


def mutual_information(pairs):
    joint = collections.Counter(pairs)
    xs = collections.Counter()
    ys = collections.Counter()
    for (x, y), n in joint.items():
        xs[x] += n
        ys[y] += n
    total = sum(joint.values())
    mi = 0.0
    for (x, y), n in joint.items():
        pxy = n / total
        mi += pxy * math.log(pxy / ((xs[x] / total) * (ys[y] / total)))
    return mi


def majority_accuracy(pairs):
    by_state = collections.defaultdict(collections.Counter)
    for x, y in pairs:
        by_state[x][y] += 1
    correct = sum(c.most_common(1)[0][1] for c in by_state.values())
    return correct / len(pairs) if pairs else 0.0


def heldout_majority_accuracy(pairs, ratio=0.8):
    """Fit the lookup table on a prefix of the data, score it on the rest.

    In-sample majority accuracy rewards assignments with more classes, because
    a larger table memorises more. Holding out removes that advantage, so two
    assignments with different codebook usage can be compared directly.
    """
    cut = int(len(pairs) * ratio)
    train, test = pairs[:cut], pairs[cut:]
    if not train or not test:
        return None
    by_state = collections.defaultdict(collections.Counter)
    for x, y in train:
        by_state[x][y] += 1
    table = {x: c.most_common(1)[0][0] for x, c in by_state.items()}
    fallback = collections.Counter(y for _, y in train).most_common(1)[0][0]
    correct = sum(1 for x, y in test if table.get(x, fallback) == y)
    return correct / len(test)


def title_tokens(title):
    return {t for t in re.split(r"[^a-z0-9]+", title.lower()) if len(t) >= 3}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def title_coherence(index, titles, level, n_pairs, seed):
    """Do items under the same prefix look alike in text?

    Compares the mean title Jaccard of same-prefix pairs against random
    cross-prefix pairs. A positive margin means the prefix groups items with
    overlapping wording, which is the property a language model can exploit.
    """
    clusters = collections.defaultdict(list)
    for item_id, tokens in index.items():
        if item_id in titles:
            clusters[prefix(tokens, level)].append(item_id)
    keys = [k for k, v in clusters.items() if len(v) > 1]
    if not keys:
        return None

    rng = random.Random(seed)
    within, cross = [], []
    for _ in range(n_pairs):
        key = rng.choice(keys)
        a, b = rng.sample(clusters[key], 2)
        within.append(jaccard(title_tokens(titles[a]), title_tokens(titles[b])))
    all_ids = [i for i in index if i in titles]
    while len(cross) < n_pairs:
        a, b = rng.sample(all_ids, 2)
        if prefix(index[a], level) != prefix(index[b], level):
            cross.append(jaccard(title_tokens(titles[a]), title_tokens(titles[b])))
    mean_within = sum(within) / len(within)
    mean_cross = sum(cross) / len(cross)
    return {
        "title_jaccard_within_prefix": round(mean_within, 6),
        "title_jaccard_across_prefix": round(mean_cross, 6),
        "title_coherence_gain": round(mean_within - mean_cross, 6),
        "pairs_sampled": n_pairs,
    }


def load_histories(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hist = [h.strip() for h in re.findall(r"\d+", row["history_item_id"])]
            rows.append((hist, row["item_id"].strip()))
    return rows


def analyse(index, rows, level, titles=None, n_pairs=100000, seed=42):
    pair_total = pair_same = 0
    last_level_pairs = []
    last_full_pairs = []
    target_level = collections.Counter()
    target_full = collections.Counter()

    for hist, target in rows:
        if target not in index:
            continue
        hist = [h for h in hist if h in index]
        if not hist:
            continue
        tgt_tokens = index[target]
        target_level[prefix(tgt_tokens, level)] += 1
        target_full["".join(tgt_tokens)] += 1

        # co-occurrence alignment: do items in the same history share a prefix
        for i in range(len(hist)):
            for j in range(i + 1, len(hist)):
                pair_total += 1
                if prefix(index[hist[i]], level) == prefix(index[hist[j]], level):
                    pair_same += 1

        last = index[hist[-1]]
        last_level_pairs.append((prefix(last, level), prefix(tgt_tokens, level)))
        last_full_pairs.append(("".join(last), prefix(tgt_tokens, level)))

    y_counts = collections.Counter(y for _, y in last_level_pairs)
    h_y = entropy(y_counts)
    mi = mutual_information(last_level_pairs)

    return {
        "rows": len(rows),
        "co_prefix_pair_rate": round(pair_same / pair_total, 6) if pair_total else None,
        "co_prefix_pairs": pair_total,
        "majority_acc_last_level": round(majority_accuracy(last_level_pairs), 6),
        "majority_acc_last_full": round(majority_accuracy(last_full_pairs), 6),
        "heldout_acc_last_level": round(heldout_majority_accuracy(last_level_pairs), 6),
        "nmi_last_level": round(mi / h_y, 6) if h_y > 0 else None,
        "target_level_entropy": round(h_y, 4),
        "target_level_classes": len(y_counts),
        "target_full_classes": len(target_full),
        "context_states": len(set(x for x, _ in last_full_pairs)),
        "title_coherence": title_coherence(index, titles, level, n_pairs, seed)
        if titles else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--item-json", default="")
    parser.add_argument("--label", default="sid")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    with open(args.index, "r", encoding="utf-8") as f:
        index = json.load(f)
    rows = load_histories(args.train_csv)
    titles = None
    if args.item_json:
        with open(args.item_json, "r", encoding="utf-8") as f:
            items = json.load(f)
        titles = {k: v.get("title", "") for k, v in items.items()}

    report = {"label": args.label, "index": args.index, "level": args.level}
    report.update(analyse(index, rows, args.level, titles))

    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
