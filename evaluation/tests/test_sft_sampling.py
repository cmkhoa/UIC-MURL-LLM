import json
import unittest
from collections import Counter

from scripts.make_sft_dataset import balance_taxonomy, build_sft_records, ROOT


class SamplingTests(unittest.TestCase):
    def test_balance_retains_sources_and_spreads_repetition(self):
        records = [
            {"example_id": str(i), "messages": [{"content": json.dumps({"error_taxonomy": label})}]}
            for i, label in enumerate(["a", "a", "a", "a", "a", "b", "b", "c"])
        ]
        balanced, metadata = balance_taxonomy(records, 42)
        self.assertEqual(metadata["sampled_counts"], {"a": 5, "b": 5, "c": 5})
        self.assertEqual(len(balanced), 15)
        self.assertEqual({r["example_id"] for r in balanced}, {r["example_id"] for r in records})
        counts = Counter(r["example_id"] for r in balanced)
        self.assertLessEqual(abs(counts["5"] - counts["6"]), 1)
        self.assertEqual((balanced, metadata), balance_taxonomy(records, 42))
        self.assertNotEqual(balanced, balance_taxonomy(records, 43)[0])
        self.assertEqual(records[0]["example_id"], "0")

    def test_rejects_other_splits_before_reading(self):
        for name in ["validation", "test"]:
            with self.assertRaises(ValueError):
                build_sft_records(ROOT / f"data/processed/{name}.jsonl")

    def test_empty(self):
        with self.assertRaises(ValueError):
            balance_taxonomy([], 42)

    def test_target_retains_all_sources_without_downsampling(self):
        records = [
            {"example_id": str(i), "messages": [{"content": json.dumps({"error_taxonomy": label})}]}
            for i, label in enumerate(["a", "a", "a", "b"])
        ]
        balanced, metadata = balance_taxonomy(records, 42, 4)
        self.assertEqual(metadata["sampled_counts"], {"a": 4, "b": 4})
        self.assertEqual(set(row["example_id"] for row in balanced), {"0", "1", "2", "3"})
        balanced, metadata = balance_taxonomy(records, 42, 2)
        self.assertEqual(metadata["sampled_counts"], {"a": 3, "b": 2})
        self.assertEqual(set(row["example_id"] for row in balanced), {"0", "1", "2", "3"})
