"""codex exec replies, without calling the program."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kilix_benchmark import codex_exec


class CodexExecTests(unittest.TestCase):
    def test_argv_pins_model_effort_and_closes_the_prompt(self) -> None:
        command = codex_exec.argv("Reply with JSON.", "gpt-6-luna", "high", Path("last.txt"))
        self.assertEqual(command[0:2], ["codex", "exec"])
        self.assertIn("--sandbox", command)
        self.assertIn("read-only", command)
        self.assertEqual(command[command.index("-m") + 1], "gpt-6-luna")
        self.assertIn('model_reasoning_effort="high"', command)
        self.assertEqual(command[-2:], ["--", "Reply with JSON."])
        with self.assertRaises(ValueError):
            codex_exec.argv("x", "gpt-6-luna", "turbo", Path("last.txt"))

    def test_record_keeps_a_finished_item(self) -> None:
        suite = {
            "schema": "kilix.benchmark.suite/v1",
            "id": "demo",
            "items": [
                {"id": "one", "answer": "A", "subject": "logic", "prompt": "one"},
                {"id": "two", "answer": "B", "subject": "logic", "prompt": "two"},
            ],
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            suite_path = root / "suite.json"
            suite_path.write_text(json.dumps(suite), encoding="utf-8")
            output = root / "ref.json"
            output.write_text(
                json.dumps(
                    {
                        "schema": codex_exec.REFERENCE_SCHEMA,
                        "model": "gpt-6-luna",
                        "effort": "low",
                        "items": [{"id": "one", "completion": '{"choice":"A"}', "seconds": 1, "error": ""}],
                    }
                ),
                encoding="utf-8",
            )
            seen = []

            def fake(prompt: str, *, model: str, effort: str, timeout: float) -> str:
                del model, effort, timeout
                seen.append(prompt)
                return '{"choice":"B"}'

            original = codex_exec.complete
            codex_exec.complete = fake
            try:
                document = codex_exec.record(suite_path, "gpt-6-luna", "low", output, workers=1)
            finally:
                codex_exec.complete = original
        self.assertEqual(seen, ["two"])
        self.assertEqual(document["score"], {"correct": 2, "count": 2})
        self.assertEqual([row["id"] for row in document["items"]], ["one", "two"])
