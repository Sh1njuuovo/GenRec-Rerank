import unittest

import torch

from minionerec.model import SIDRecommender, build_vocabulary, encode_examples


class SIDModelTests(unittest.TestCase):
    def test_vocabulary_and_encoding(self):
        rows = [{"history": ["a/b/c", "a/b/d"], "target": "a/b/e"}]
        vocab = build_vocabulary(rows)
        history, target = encode_examples(rows, vocab)
        self.assertEqual(tuple(history.shape), (1, 2, 3))
        self.assertEqual(tuple(target.shape), (1, 3))
        self.assertEqual(vocab["<pad>"], 0)

    def test_angle_bracket_semantic_ids(self):
        rows = [{"history": ["<a_1><b_2><c_3>"], "target": "<a_1><b_2><c_4>"}]
        vocab = build_vocabulary(rows)
        history, target = encode_examples(rows, vocab)
        self.assertEqual(tuple(history.shape), (1, 1, 3))
        self.assertNotEqual(target[0, 2].item(), history[0, 0, 2].item())

    def test_training_step_and_valid_prediction(self):
        rows = [{"history": ["a/b/c", "a/b/d"], "target": "a/b/e"}]
        vocab = build_vocabulary(rows)
        history, target = encode_examples(rows, vocab)
        model = SIDRecommender(len(vocab), dim=16, heads=2, dropout=0)
        loss = model.loss(history, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(tuple(model.predict(history, ["a/b/e"], vocab).shape), (1, 3))
        self.assertEqual(model.predict(history, ["a/b/e"], vocab)[0].tolist(), target[0].tolist())


if __name__ == "__main__":
    unittest.main()
