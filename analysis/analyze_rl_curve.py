"""Read the training-time metric log and report whether RL beat its own start.

`rec_metric_callback.py` writes one JSON object per evaluation to
`rec_metrics.jsonl`. The row at step 0 is the untouched SFT policy, and every
later row is measured on the same fixed subset with the same decoding settings,
so comparisons inside one file are apples to apples.

Usage:
    python analyze_rl_curve.py --log <run>/rec_metrics.jsonl
    python analyze_rl_curve.py --log <a>/rec_metrics.jsonl --log <b>/rec_metrics.jsonl
    python analyze_rl_curve.py --log ... --metric hr@10 --json_out curve.json
"""

import argparse
import json
import os


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return [r for r in rows if "error" not in r and "hr@10" in r]


def summarize(path, metric):
    rows = load_rows(path)
    if not rows:
        return None

    rows.sort(key=lambda r: r.get("step", 0))
    base = rows[0]
    best = max(rows, key=lambda r: r.get(metric, float("-inf")))
    last = rows[-1]

    return {
        "log": path,
        "evaluations": len(rows),
        "metric": metric,
        "start": base.get(metric),
        "best": best.get(metric),
        "best_step": best.get("step"),
        "final": last.get(metric),
        "best_gain": best.get(metric) - base.get(metric),
        "final_gain": last.get(metric) - base.get(metric),
        "target_in_group_start": base.get("target_in_group"),
        "target_in_group_best": best.get("target_in_group"),
        "target_in_group_final": last.get("target_in_group"),
        "rows": rows,
    }


def print_table(summary):
    rows = summary["rows"]
    print(f"\n{os.path.basename(os.path.dirname(summary['log']))} "
          f"({summary['evaluations']} evaluations)")
    print(f"{'step':>7} {'hr@1':>8} {'hr@10':>8} {'ndcg@10':>9} "
          f"{'hr@50':>8} {'in_group':>9} {'sec':>6}")
    for r in rows:
        in_group = r.get("target_in_group")
        in_group = float("nan") if in_group is None else in_group
        print(f"{r.get('step', -1):>7} {r.get('hr@1', float('nan')):>8.4f} "
              f"{r.get('hr@10', float('nan')):>8.4f} "
              f"{r.get('ndcg@10', float('nan')):>9.4f} "
              f"{r.get('hr@50', float('nan')):>8.4f} "
              f"{in_group:>9.4f} {r.get('seconds', 0):>6.1f}")

    print(f"\nstart {summary['metric']} = {summary['start']:.4f} at step {rows[0].get('step')}")
    print(f"best  {summary['metric']} = {summary['best']:.4f} at step {summary['best_step']} "
          f"(gain {summary['best_gain']:+.4f})")
    print(f"final {summary['metric']} = {summary['final']:.4f} "
          f"(gain {summary['final_gain']:+.4f})")
    if summary["best_gain"] <= 0:
        print("note: no checkpoint beat the SFT start on this metric")
    elif summary["final_gain"] < 0:
        print("note: the final checkpoint is worse than the SFT start while the "
              "best checkpoint is better, so checkpoint selection matters")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", required=True)
    parser.add_argument("--metric", default="hr@10")
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    summaries = []
    for path in args.log:
        summary = summarize(path, args.metric)
        if summary is None:
            print(f"{path}: no usable rows")
            continue
        summaries.append(summary)
        print_table(summary)

    if len(summaries) > 1:
        print(f"\n{'run':<28}{'start':>9}{'best':>9}{'final':>9}{'best_step':>11}")
        for s in summaries:
            name = os.path.basename(os.path.dirname(s["log"]))
            print(f"{name:<28}{s['start']:>9.4f}{s['best']:>9.4f}"
                  f"{s['final']:>9.4f}{s['best_step']:>11}")
        n = len(summaries)
        print(f"{'mean':<28}{sum(s['start'] for s in summaries) / n:>9.4f}"
              f"{sum(s['best'] for s in summaries) / n:>9.4f}"
              f"{sum(s['final'] for s in summaries) / n:>9.4f}")
        wins = sum(1 for s in summaries if s["best_gain"] > 0)
        print(f"\nruns where the best checkpoint beat the SFT start: {wins}/{n}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
