"""Convert recommendation CSV rows with semantic IDs into JSONL examples."""

import argparse
import ast
import csv
import json
from pathlib import Path

from .model import _parts


def prepare_sid_csv(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for number, row in enumerate(csv.DictReader(stream), 2):
            try:
                history = ast.literal_eval(row["history_item_sid"])
                target = row["item_sid"].strip()
                if not isinstance(history, list) or not history:
                    raise ValueError("history_item_sid must be a nonempty list")
                for sid in [*history, target]:
                    _parts(sid)
                rows.append({"history": history, "target": target})
            except (KeyError, SyntaxError, ValueError, TypeError) as exc:
                raise ValueError(f"invalid CSV row {number}: {exc}") from exc
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = prepare_sid_csv(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} examples to {args.output}")


if __name__ == "__main__":
    main()
