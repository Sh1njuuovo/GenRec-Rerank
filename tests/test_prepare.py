import tempfile
import unittest
from pathlib import Path

from minionerec.prepare import prepare_sid_csv


class PrepareTests(unittest.TestCase):
    def test_reads_semantic_id_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "items.csv"
            source.write_text('history_item_sid,item_sid\n"[\'<a_1><b_2><c_3>\']",<a_1><b_2><c_4>\n')
            rows = prepare_sid_csv(source)
        self.assertEqual(rows, [{"history": ["<a_1><b_2><c_3>"], "target": "<a_1><b_2><c_4>"}])
