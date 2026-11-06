"""Score a multiple-choice suite whose answer is one JSON object.

The object has a single key, choice. A leading think block and one markdown
fence are removed before parsing. Any other shape fails.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SUITE_SCHEMA = "kilix.benchmark.suite/v1"
_THINK = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)
_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```$")


def load_suite(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SUITE_SCHEMA:
        raise ValueError("suite file has the wrong schema")
    if not document.get("items"):
        raise ValueError("suite file has no items")
    return document


def visible_answer(text: str) -> str:
    value = _THINK.sub("", text.replace("\r\n", "\n").strip(), count=1).strip()
    value = _FENCE_OPEN.sub("", value, count=1)
    value = _FENCE_CLOSE.sub("", value, count=1)
    return value.strip()


def score_completion(item: dict, completion: str) -> tuple[bool, str]:
    try:
        value = json.loads(visible_answer(completion))
    except json.JSONDecodeError:
        return False, "not json"
    if not isinstance(value, dict) or set(value) != {"choice"}:
        return False, "json keys differ"
    got = value["choice"]
    if not isinstance(got, str):
        return False, "choice is not a string"
    ok = got.strip().casefold() == str(item["answer"]).strip().casefold()
    return ok, "choice" if ok else "wrong choice"


def score_report(suite: dict, model: str, rows: list[dict]) -> dict:
    by_id = {row["id"]: row for row in rows}
    items = []
    correct = 0
    for item in suite["items"]:
        row = by_id.get(item["id"], {})
        ok, reason = score_completion(item, row.get("completion") or "")
        correct += int(ok)
        items.append(
            {
                "id": item["id"],
                "subject": item.get("subject", ""),
                "correct": ok,
                "reason": reason,
                "completion": row.get("completion") or "",
                "seconds": row.get("seconds"),
                "error": row.get("error") or "",
            }
        )
    return {
        "model": model,
        "score": {"correct": correct, "count": len(suite["items"])},
        "items": items,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a college multiple-choice suite")
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args(argv)
    suite = load_suite(args.suite)
    rows = json.loads(args.results.read_text(encoding="utf-8"))
    report = score_report(suite, args.model, rows)
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
