# GenRec-Rerank

GenRec-Rerank explores next-item recommendation with three-level semantic IDs and offline reranking analysis. The runnable core reads a user's item history as semantic IDs, predicts the next ID, and restricts selection to complete IDs from a supplied catalog. The repository also contains independent scripts for analyzing candidate lists, ID collisions, and reranking results.

## Method

1. **Semantic ID input.** Each item has an existing three-level ID such as `<a_1><b_2><c_3>`. The data converter reads histories and targets from CSV. This project consumes those IDs; it does not train a semantic ID tokenizer.
2. **History encoding.** The three tokens of each item are embedded and averaged. Position embeddings and a two-layer Transformer encoder produce a representation of the interaction history.
3. **Next-item prediction.** Three output heads predict one token per ID level. Training averages the three cross-entropy losses.
4. **Valid-item selection.** At prediction time, the model scores supplied complete IDs by summing their three token logits. It returns the highest-scoring valid ID, so it cannot assemble an ID absent from that catalog.
5. **Offline analysis.** Scripts in `analysis/` compare results across runs and inspect item-level metrics, ID collisions, candidate coverage, and reranking behavior. They are standalone analysis tools rather than stages called by the training command.

The model and candidate selection logic are in [`src/minionerec/model.py`](src/minionerec/model.py).

## Data flow

CSV with `history_item_sid` and `item_sid` → JSONL examples → vocabulary and padded histories → model training → checkpoint

The converter expects a nonempty list in `history_item_sid` and a single target in `item_sid`. It validates that every ID has three levels. A JSONL record looks like this:

```json
{"history": ["<a_1><b_1><c_2>", "<a_2><b_3><c_1>"], "target": "<a_3><b_2><c_4>"}
```

Histories are left padded with ID 0 and limited to 100 items during encoding. The vocabulary is built from the provided training file and saved with the checkpoint.

## Quick start

Use Python 3.9 or newer. Run these commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

PYTHONPATH=src python -m minionerec.train examples/demo.jsonl \
  --epochs 1 --dim 16 --heads 2 --output runs/demo
```

The demo data is synthetic. The command writes `model.pt`, `vocabulary.json`, and `metrics.json` to `runs/demo/`.

To use a CSV file containing semantic IDs:

```bash
PYTHONPATH=src python -m minionerec.prepare data/train.csv data/train.jsonl
PYTHONPATH=src python -m minionerec.train \
  data/train.jsonl --epochs 5 --output runs/train
```

Run the tests with `PYTHONPATH=src python -m unittest discover -s tests -v`. Individual offline tools can be inspected with commands such as `python analysis/compare_results.py --help`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/minionerec/prepare.py` | Validate and convert semantic ID CSV data |
| `src/minionerec/model.py` | Encode histories, predict ID levels, and select valid candidates |
| `src/minionerec/train.py` | Train and save the model, vocabulary, and loss report |
| `analysis/` | Independent metric, collision, and reranking analyses |
| `results/` | Selected statistics from earlier experiments |
| `tests/` | Data, model, prediction, and training checks |

The archived statistics in `results/` were not produced by the current compact model. This repository does not include semantic ID construction, language-model fine-tuning, reinforcement learning, or an integrated reranking training stage. The training command reports loss only; a held-out evaluation pipeline is still needed for performance claims.

## Paper

[MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation](https://arxiv.org/abs/2510.24431)
