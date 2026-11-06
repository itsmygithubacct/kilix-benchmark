"""College multiple-choice scoring."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from kilix_benchmark.mcq import load_suite, score_completion, score_report

SUITE = Path(__file__).resolve().parents[1] / "src" / "kilix_benchmark" / "suites" / "college_mcq.json"


class CollegeMcqTests(unittest.TestCase):
    def test_suite_has_twenty_four_distinct_items(self) -> None:
        suite = load_suite(SUITE)
        ids = [item["id"] for item in suite["items"]]
        self.assertEqual(len(ids), 24)
        self.assertEqual(len(set(ids)), 24)
        for item in suite["items"]:
            self.assertIn(item["answer"], {"A", "B", "C", "D"})
            self.assertIn("choice", item["prompt"])

    def test_json_choice_accepts_a_fence_and_a_think_block(self) -> None:
        item = {"answer": "C"}
        self.assertTrue(score_completion(item, '{"choice":"C"}')[0])
        self.assertTrue(score_completion(item, '```json\n{"choice": "C"}\n```')[0])
        wrapped = '<think>twelve</think>\n```json\n{"choice":"C"}\n```'
        self.assertTrue(score_completion(item, wrapped)[0])
        self.assertFalse(score_completion(item, '{"choice":"A"}')[0])
        self.assertFalse(score_completion(item, '{"choice":"C","note":"ok"}')[0])
        self.assertFalse(score_completion(item, "C")[0])

    def test_report_counts_correct_rows(self) -> None:
        suite = {"items": [{"id": "one", "answer": "A", "subject": "logic"}]}
        report = score_report(suite, "demo", [{"id": "one", "completion": '{"choice":"A"}'}])
        self.assertEqual(report["score"], {"correct": 1, "count": 1})
        missing = score_report(suite, "demo", [])
        self.assertEqual(missing["score"]["correct"], 0)
        self.assertTrue(json.dumps(report))
