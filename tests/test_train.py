import json
import tempfile
import unittest
from pathlib import Path

from minionerec.train import train_file


class TrainingTests(unittest.TestCase):
    def test_one_epoch_on_small_dataset(self):
        rows = [
            {"history": ["a/b/c", "a/b/d"], "target": "a/b/e"},
            {"history": ["a/b/d", "a/b/e"], "target": "a/b/c"},
            {"history": ["a/b/e", "a/b/c"], "target": "a/b/d"},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "examples.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            report = train_file(path, epochs=1, dim=8, heads=2, batch_size=2)
        self.assertEqual(report["examples"], 3)
        self.assertEqual(len(report["epochs"]), 1)
