"""Get one reply from `codex exec`.

The prompt is the whole request. stdin is closed so exec does not wait for
more input. The last message is the reply. A read-only sandbox and an empty
working directory keep the run from editing the suite tree.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kilix_benchmark.mcq import load_suite, score_report

EFFORTS = ("low", "medium", "high", "xhigh")
REFERENCE_SCHEMA = "kilix.benchmark.reference/v1"


def argv(prompt: str, model: str, effort: str, output: Path) -> list[str]:
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {', '.join(EFFORTS)}")
    if not model or model.startswith("-"):
        raise ValueError("model must be a plain name")
    return [
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "-m",
        model,
        "-c",
        f"model_reasoning_effort={json.dumps(effort)}",
        "-o",
        str(output),
        "--",
        prompt,
    ]


def complete(prompt: str, *, model: str, effort: str, timeout: float = 300) -> str:
    with tempfile.TemporaryDirectory(prefix="kilix-benchmark-") as raw:
        work = Path(raw)
        last = work / "last.txt"
        command = argv(prompt, model, effort, last)
        completed = subprocess.run(
            command,
            cwd=work,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(detail[-500:] or f"codex exec exited {completed.returncode}")
        if not last.is_file():
            raise RuntimeError("codex exec wrote no last message")
        return last.read_text(encoding="utf-8")


def _reference(suite_name: str, model: str, effort: str, rows: list[dict]) -> dict:
    return {
        "schema": REFERENCE_SCHEMA,
        "suite": suite_name,
        "model": model,
        "effort": effort,
        "items": rows,
    }


def record(suite_path: Path, model: str, effort: str, output: Path, *, workers: int = 2, timeout: float = 300) -> dict:
    suite = load_suite(suite_path)
    done: dict[str, dict] = {}
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("model") == model and previous.get("effort") == effort:
            for row in previous.get("items") or []:
                if row.get("id") and not row.get("error") and row.get("completion"):
                    done[row["id"]] = row
    pending = [item for item in suite["items"] if item["id"] not in done]

    def one(item: dict) -> dict:
        started = time.monotonic()
        try:
            text = complete(item["prompt"], model=model, effort=effort, timeout=timeout)
            error = ""
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
            text, error = "", str(exc)[-500:]
        return {
            "id": item["id"],
            "completion": text,
            "seconds": round(time.monotonic() - started, 3),
            "error": error,
        }

    rows = list(done.values())
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        futures = {pool.submit(one, item): item["id"] for item in pending}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            document = _reference(suite.get("id") or suite_path.stem, model, effort, rows)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            print(f"{model} {effort} {row['id']} {'error' if row['error'] else 'ok'}", flush=True)
    order = {item["id"]: index for index, item in enumerate(suite["items"])}
    rows.sort(key=lambda row: order.get(row["id"], len(order)))
    document = _reference(suite.get("id") or suite_path.stem, model, effort, rows)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    report = score_report(suite, f"{model}:{effort}", rows)
    document["score"] = report["score"]
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record codex exec replies for a suite")
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True, choices=EFFORTS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)
    document = record(
        args.suite,
        args.model,
        args.effort,
        args.output,
        workers=args.workers,
        timeout=args.timeout,
    )
    score = document["score"]
    print(f"{args.model} {args.effort} {score['correct']}/{score['count']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
