"""Numbers from tab-bench runs, and the paired comparison of two arms.

Each run directory holds ``result.json`` (driver summary), ``driver.jsonl``
(writes, frame-file events, process samples) and ``events.jsonl`` (what the
desktop did). All times are CLOCK_MONOTONIC nanoseconds from the same host,
so driver and desktop records can be compared directly.

Terms used in the output:

- *render*: one ``Desk.render`` pass that drew, from its start to the end of
  the send (``t1 - t0``); *compose* is the part before the send.
- *written*: the desktop finished writing the frame to the terminal.
- *taken*: the terminal deleted the frame file, which it does right after it
  has mapped the file. It is not the moment pixels reach the screen.
- *latency*: from the desktop parsing an input batch that holds a key press
  or pointer report to the first drawn frame that starts after it.
- *transport*: from the driver's write to the desktop parsing that input. It
  includes the remote-control hop and the terminal, and also any time the
  input waited because the desktop was still busy with the previous frame,
  so under pointer load it is not a pure transport time.

Comparison: run values are paired in run order (each pair is one run of each
arm, the leading arm alternates). A difference is called only when the
two-sided sign test over pairs gives p <= 0.05, the bootstrap 95% interval of
the median paired difference excludes zero, and the median difference is at
least 5% of the reference arm's median. Otherwise the verdict is "no
difference shown".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

NS_PER_MS = 1_000_000
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
LATENCY_WINDOW_NS = 2_000 * NS_PER_MS
SIGN_ALPHA = 0.05
PRACTICAL = 0.05
BOOTSTRAP_SEED = 20261112
BOOTSTRAP_ROUNDS = 10_000

# Metrics the protocol names in advance. Everything else is descriptive.
PRIMARY = (
    ("type-paced.latency_written_ms.median", "Paced keystroke to frame written (ms, run median)"),
    ("startup.launch_to_desktop_ms", "Tab launch to first desktop frame written (ms)"),
    ("hover.render_ms.median", "Pointer-motion frame render time (ms, run median)"),
    ("run.cpu_s", "Desktop process CPU time, whole run (s)"),
    ("run.vmhwm_mb", "Desktop process peak resident memory (MiB)"),
)


# ── small statistics ────────────────────────────────────────────────────────

def percentile(values, q):
    """Linear-interpolated percentile, q in [0, 100]. None for no values."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    if len(data) == 1:
        return float(data[0])
    pos = (len(data) - 1) * q / 100
    low = math.floor(pos)
    high = math.ceil(pos)
    return float(data[low] + (data[high] - data[low]) * (pos - low))


def median(values):
    return percentile(values, 50)


def spread(values) -> dict:
    data = [v for v in values if v is not None]
    if not data:
        return {"n": 0}
    return {"n": len(data), "median": median(data), "p25": percentile(data, 25),
            "p75": percentile(data, 75), "p95": percentile(data, 95),
            "min": float(min(data)), "max": float(max(data))}


def sign_test(differences) -> dict:
    """Two-sided exact binomial sign test; zero differences are dropped."""
    nonzero = [d for d in differences if d != 0]
    n = len(nonzero)
    positive = sum(1 for d in nonzero if d > 0)
    if n == 0:
        return {"n": 0, "positive": 0, "p": 1.0}
    tail = min(positive, n - positive)
    p = 2 * sum(math.comb(n, k) for k in range(tail + 1)) / 2 ** n
    return {"n": n, "positive": positive, "p": min(1.0, p)}


def bootstrap_median(differences, rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
    data = list(differences)
    if not data:
        return None, None
    rng = random.Random(seed)
    stats = sorted(median([rng.choice(data) for _ in data]) for _ in range(rounds))
    return percentile(stats, 2.5), percentile(stats, 97.5)


def compare(reference, candidate) -> dict:
    """Paired comparison: candidate minus reference, pair by pair."""
    pairs = [(a, b) for a, b in zip(reference, candidate) if a is not None and b is not None]
    if not pairs:
        return {"pairs": 0, "verdict": "no data"}
    diffs = [b - a for a, b in pairs]
    ref_median = median([a for a, _ in pairs])
    cand_median = median([b for _, b in pairs])
    diff_median = median(diffs)
    low, high = bootstrap_median(diffs)
    sign = sign_test(diffs)
    relative = diff_median / ref_median if ref_median else None
    excludes_zero = low is not None and (low > 0 or high < 0)
    called = (sign["p"] <= SIGN_ALPHA and excludes_zero and relative is not None
              and abs(relative) >= PRACTICAL)
    if called:
        verdict = "candidate higher" if diff_median > 0 else "candidate lower"
    else:
        verdict = "no difference shown"
    return {"pairs": len(pairs), "reference_median": ref_median, "candidate_median": cand_median,
            "median_difference": diff_median, "ci95": [low, high], "relative": relative,
            "ratio": cand_median / ref_median if ref_median else None,
            "sign_test": sign, "verdict": verdict}


# ── one run ─────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _of(records, kind):
    return [r for r in records if r.get("e") == kind]


def _ms(ns):
    return None if ns is None else ns / NS_PER_MS


def _meaningful(batch) -> bool:
    """A batch holds a key press/repeat or any pointer report."""
    for item in batch["ev"]:
        if item[0] == "k" and item[3] != 3:
            return True
        if item[0] == "m":
            return True
    return False


def phase_windows(driver: list[dict]) -> list[tuple[str, int, int]]:
    """[(phase, begin, until)]: a phase lasts until the next one begins."""
    begins = [r for r in _of(driver, "phase") if r["what"] == "begin"]
    ends = [r for r in _of(driver, "phase") if r["what"] == "end"]
    out = []
    for i, begin in enumerate(begins):
        if i + 1 < len(begins):
            until = begins[i + 1]["t"]
        else:
            until = ends[-1]["t"] if ends else begin["t"]
        out.append((begin["phase"], begin["t"], until))
    return out


def _cpu_between(samples, begin, until):
    before = [s for s in samples if s["t"] <= begin]
    upto = [s for s in samples if s["t"] <= until]
    if not before or not upto:
        return None
    a, b = before[-1], upto[-1]
    return ((b["utime"] + b["stime"]) - (a["utime"] + a["stime"])) / CLK_TCK


def phase_metrics(name, begin, until, frames, inputs, sends, deleted, samples) -> dict:
    in_frames = [f for f in frames if begin <= f["t0"] < until]
    in_inputs = [i for i in inputs if begin <= i["t"] < until]
    in_sends = [s for s in sends if begin <= s["t"] < until]
    written, taken, no_frame = [], [], 0
    for batch in in_inputs:
        if not _meaningful(batch):
            continue
        frame = next((f for f in in_frames if f["t0"] >= batch["t"]), None)
        if frame is None or frame["t0"] - batch["t"] > LATENCY_WINDOW_NS:
            no_frame += 1
            continue
        written.append(_ms(frame["t1"] - batch["t"]))
        gone = deleted.get(frame.get("file"))
        if gone is not None:
            taken.append(_ms(gone - batch["t"]))
    transport = []
    for send in in_sends:
        batch = next((i for i in in_inputs if i["t"] >= send["t"]), None)
        if batch is not None:
            transport.append(_ms(batch["t"] - send["t"]))
    received_keys = sum(1 for b in in_inputs for item in b["ev"] if item[0] == "k" and item[3] == 1)
    received_pointer = sum(1 for b in in_inputs for item in b["ev"] if item[0] == "m")
    duration_s = (until - begin) / 1e9
    out = {
        "frames": len(in_frames),
        "frames_per_s": len(in_frames) / duration_s if duration_s > 0 else None,
        "render_ms": spread([_ms(f["t1"] - f["t0"]) for f in in_frames]),
        "compose_ms": spread([_ms(f["tb"] - f["t0"]) for f in in_frames]),
        "send_ms": spread([_ms(f["t1"] - f["tb"]) for f in in_frames]),
        "text_ms": spread([_ms(f["text_ns"]) for f in in_frames]),
        "measure_ms": spread([_ms(f["measure_ns"]) for f in in_frames]),
        "text_calls": spread([f["text_n"] for f in in_frames]),
        "neuronte_masks": sum(f.get("mask_n", 0) for f in in_frames),
        "latency_written_ms": spread(written),
        "latency_taken_ms": spread(taken),
        "inputs_without_frame": no_frame,
        "transport_ms": spread(transport),
        "keys_sent": sum(s["keys"] for s in in_sends),
        "key_presses_received": received_keys,
        "pointer_reports_received": received_pointer,
        "writes": len(in_sends),
        "cpu_s": _cpu_between(samples, begin, until),
    }
    if name == "type-burst" and in_inputs:
        last = max(i["t"] for i in in_inputs if _meaningful(i)) if any(
            _meaningful(i) for i in in_inputs) else None
        first = min(i["t"] for i in in_inputs)
        after = [f for f in in_frames if last is not None and f["t0"] >= last]
        if after:
            span = after[0]["t1"] - first
            out["burst_ms"] = _ms(span)
            out["burst_keys_per_s"] = received_keys / (span / 1e9) if span > 0 else None
    return out


def run_metrics(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    events = read_jsonl(run_dir / "events.jsonl")
    driver = read_jsonl(run_dir / "driver.jsonl")
    out = {"run": result.get("run"), "arm": result.get("arm"), "index": result.get("index"),
           "warmup": bool(result.get("warmup")), "status": result.get("status"),
           "notes": result.get("notes", []), "checks": {}}
    frames = _of(events, "frame")
    inputs = _of(events, "input")
    sends = _of(driver, "send")
    files = _of(driver, "file")
    deleted = {f["file"]: f["t"] for f in files if f["what"] == "deleted"}
    written_files = {f["file"] for f in files if f["what"] == "written"}
    samples = [s for s in _of(driver, "proc") if s.get("who") == "desktop"]
    launch = _of(driver, "launch")
    start = _of(events, "start")
    imported = _of(events, "imported")
    screens = [s for s in _of(events, "screen") if s["name"] == "startup"]
    blits = _of(events, "blit")
    exits = _of(events, "exit")
    fonts = (_of(events, "fonts") or [{}])[0]

    startup = {}
    if launch and start:
        t_launch = launch[0]["t"]
        startup["launch_to_probe_ms"] = _ms(start[0]["t"] - t_launch)
        if imported:
            startup["import_ms"] = _ms(imported[0]["t1"] - imported[0]["t0"])
        if blits:
            startup["launch_to_first_frame_ms"] = _ms(blits[0]["t1"] - t_launch)
        if screens:
            first = next((f for f in frames if f["t0"] >= screens[0]["t1"]), None)
            if first is not None:
                startup["launch_to_desktop_ms"] = _ms(first["t1"] - t_launch)
                startup["first_desktop_render_ms"] = _ms(first["t1"] - first["t0"])
                startup["splash_ms"] = _ms(screens[0]["t1"] - screens[0]["t0"])
                gone = deleted.get(first.get("file"))
                if gone is not None:
                    startup["launch_to_desktop_taken_ms"] = _ms(gone - t_launch)
    out["startup"] = startup

    phases = {}
    for name, begin, until in phase_windows(driver):
        phases[name] = phase_metrics(name, begin, until, frames, inputs, sends, deleted, samples)
    out["phases"] = phases

    run = {}
    if exits:
        ex = exits[-1]
        run["cpu_s"] = ex.get("utime", 0) + ex.get("stime", 0)
        run["vmhwm_mb"] = (ex.get("vmhwm_kb") or 0) / 1024
        run["exit_status"] = ex.get("status")
        run["neuronte_sizes"] = [
            [s["em"], s["italic"]] for s in (ex.get("neuronte_sizes") or []) if s["loaded"]]
    run["frames"] = len(frames)
    run["frames_written"] = len(written_files)
    run["frames_taken"] = len(written_files & set(deleted))
    run["text_path"] = ("neuronte" if fonts.get("FONT") == "NeuronteFont"
                        and sum(f.get("mask_n", 0) for f in frames) > 0 else
                        "freetype" if fonts.get("FONT") == "FreeTypeFont" else "unknown")
    run["fonts"] = {k: fonts.get(k) for k in ("FONT", "BOLD", "SMALL", "START_W", "CLOCK_W")}
    run["desk"] = result.get("desk")
    out["whole"] = run

    pastes = sum(1 for b in inputs for item in b["ev"] if item[0] == "p")
    focus = sum(1 for b in inputs for item in b["ev"] if item[0] == "f")
    sent = sum(s["keys"] for s in sends)
    received = sum(1 for b in inputs for item in b["ev"] if item[0] == "k" and item[3] == 1)
    checks = out["checks"]
    checks["complete"] = result.get("status") == "complete"
    checks["all_keys_received"] = sent == received
    checks["no_paste_or_focus_input"] = pastes == 0 and focus == 0
    checks["all_frames_taken"] = run["frames_written"] == run["frames_taken"]
    checks["notepad_opened"] = any(
        w[0] == "Notepad" for f in frames for w in f.get("wins", ()))
    out["valid"] = all(checks.values())
    out["summary"] = {"keys_sent": sent, "key_presses_received": received, "pastes": pastes,
                      "focus_events": focus}
    host = {}
    for side in ("host_before", "host_after"):
        state = result.get(side) or {}
        host[side] = {"loadavg": state.get("loadavg"), "gpu": state.get("gpu"),
                      "drm": state.get("drm"), "utc": state.get("utc"),
                      "power": state.get("power")}
    out["host"] = host
    loads = [r["loadavg"][0] for r in _of(driver, "load")]
    out["host"]["load1_during"] = spread(loads)
    return out


def lookup(metrics: dict, dotted: str):
    """``type-paced.latency_written_ms.median`` -> value, across sections."""
    head, _, rest = dotted.partition(".")
    if head == "startup":
        node = metrics.get("startup", {})
    elif head == "run":
        node = metrics.get("whole", {})
    else:
        node = metrics.get("phases", {}).get(head)
    for part in rest.split(".") if rest else ():
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node if isinstance(node, (int, float)) else None


# ── campaign ────────────────────────────────────────────────────────────────

def pair_runs(runs: list[dict], reference: str, candidate: str):
    """Measured runs in index order, two at a time; (pairs, problems)."""
    measured = sorted((r for r in runs if not r["warmup"]), key=lambda r: r["index"])
    pairs, problems = [], []
    for i in range(0, len(measured) - 1, 2):
        a, b = measured[i], measured[i + 1]
        arms = {a["arm"]: a, b["arm"]: b}
        if set(arms) != {reference, candidate}:
            problems.append(f"runs {a['run']} and {b['run']} are not one of each arm")
            continue
        if not (a["valid"] and b["valid"]):
            bad = [r["run"] for r in (a, b) if not r["valid"]]
            problems.append(f"pair {a['run']}/{b['run']} dropped: invalid {', '.join(bad)}")
            continue
        pairs.append((arms[reference], arms[candidate]))
    if len(measured) % 2:
        problems.append(f"run {measured[-1]['run']} has no partner")
    sizes = {tuple(r["whole"].get("desk") or ()) for pair in pairs for r in pair}
    if len(sizes) > 1:
        problems.append(f"desk sizes differ between runs: {sorted(sizes)}")
    return pairs, problems


DESCRIPTIVE = (
    "startup.launch_to_probe_ms", "startup.import_ms", "startup.launch_to_first_frame_ms",
    "startup.first_desktop_render_ms", "startup.launch_to_desktop_taken_ms",
    "menu.latency_written_ms.median", "menu.render_ms.median", "menu.text_ms.median",
    "open-notepad.cpu_s",
    "type-paced.latency_written_ms.p95", "type-paced.latency_taken_ms.median",
    "type-paced.render_ms.median", "type-paced.compose_ms.median", "type-paced.text_ms.median",
    "type-paced.measure_ms.median", "type-paced.transport_ms.median", "type-paced.cpu_s",
    "type-burst.burst_ms", "type-burst.burst_keys_per_s", "type-burst.frames",
    "caret.latency_written_ms.median", "caret.render_ms.median",
    "hover.latency_written_ms.median", "hover.latency_written_ms.p95", "hover.frames",
    "hover.pointer_reports_received", "hover.text_ms.median", "hover.compose_ms.median",
    "hover.send_ms.median", "hover.transport_ms.median", "hover.cpu_s",
    "drag.latency_written_ms.median", "drag.render_ms.median", "drag.frames",
    "run.frames",
)


def analyze(out_dir: Path, reference: str, candidate: str) -> dict:
    out_dir = Path(out_dir)
    runs = [run_metrics(path) for path in sorted((out_dir / "runs").iterdir())
            if (path / "result.json").exists()]
    pairs, problems = pair_runs(runs, reference, candidate)
    comparison = {}
    for dotted, label in PRIMARY:
        comparison[dotted] = {"label": label, "primary": True, **compare(
            [lookup(a, dotted) for a, _ in pairs], [lookup(b, dotted) for _, b in pairs])}
    for dotted in DESCRIPTIVE:
        comparison[dotted] = {"label": dotted, "primary": False, **compare(
            [lookup(a, dotted) for a, _ in pairs], [lookup(b, dotted) for _, b in pairs])}
    per_arm = {}
    for arm in (reference, candidate):
        chosen = [r for r in runs if r["arm"] == arm and not r["warmup"] and r["valid"]]
        per_arm[arm] = {dotted: spread([lookup(r, dotted) for r in chosen])
                        for dotted, _ in PRIMARY}
    return {"schema": "kilix.benchmark.tab-analysis/v1", "reference": reference,
            "candidate": candidate, "pairs": len(pairs), "problems": problems,
            "runs": runs, "per_arm": per_arm, "comparison": comparison}


def _fmt(value, digits=2):
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summary_markdown(analysis: dict) -> str:
    ref, cand = analysis["reference"], analysis["candidate"]
    lines = [f"# Tab bench summary: {cand} against {ref}", "",
             f"Complete valid pairs: {analysis['pairs']}.", ""]
    if analysis["problems"]:
        lines += ["Problems:", ""] + [f"- {p}" for p in analysis["problems"]] + [""]
    lines += ["## Runs", "", "| run | arm | warm-up | status | valid | failed checks | load1 before |",
              "|---|---|---|---|---|---|---|"]
    for r in analysis["runs"]:
        failed = ", ".join(k for k, v in r["checks"].items() if not v) or "–"
        load = (r["host"]["host_before"].get("loadavg") or [None])[0]
        lines.append(f"| {r['run']} | {r['arm']} | {'yes' if r['warmup'] else 'no'} | "
                     f"{r['status']} | {'yes' if r['valid'] else 'no'} | {failed} | {_fmt(load)} |")
    for title, primary in (("Primary metrics", True), ("Descriptive metrics", False)):
        lines += ["", f"## {title}", "",
                  f"| metric | {ref} median | {cand} median | median diff | 95% CI | ratio | "
                  "sign test (+/n, p) | verdict |", "|---|---|---|---|---|---|---|---|"]
        for dotted, row in analysis["comparison"].items():
            if row["primary"] != primary or row.get("pairs", 0) == 0:
                continue
            ci = row["ci95"]
            sign = row["sign_test"]
            lines.append(
                f"| {row['label']} | {_fmt(row['reference_median'])} | "
                f"{_fmt(row['candidate_median'])} | {_fmt(row['median_difference'])} | "
                f"{_fmt(ci[0])} to {_fmt(ci[1])} | {_fmt(row['ratio'], 3)} | "
                f"{sign['positive']}/{sign['n']}, {_fmt(sign['p'], 4)} | {row['verdict']} |")
    lines += ["", "## Spread per arm (valid measured runs)", "",
              "| metric | arm | n | median | p25 | p75 | min | max |",
              "|---|---|---|---|---|---|---|---|"]
    for dotted, label in PRIMARY:
        for arm in (ref, cand):
            s = analysis["per_arm"][arm][dotted]
            lines.append(f"| {label} | {arm} | {s.get('n', 0)} | {_fmt(s.get('median'))} | "
                         f"{_fmt(s.get('p25'))} | {_fmt(s.get('p75'))} | {_fmt(s.get('min'))} | "
                         f"{_fmt(s.get('max'))} |")
    return "\n".join(lines) + "\n"


def public_reference(analysis: dict, plan: dict, host: dict) -> dict:
    """A record with no local paths: commits, host kind, verdicts, spreads."""
    def commit(entry):
        return {"commit": (entry or {}).get("commit"),
                "dirty_paths": (entry or {}).get("dirty_paths")}
    record = {
        "schema": "kilix.benchmark.tab-reference/v1",
        "suite": plan.get("suite"),
        "started_utc": plan.get("started_utc"),
        "arms": {name: commit(entry) for name, entry in (plan.get("arms") or {}).items()},
        "neuronte": commit(plan.get("neuronte_home")),
        "neuronte_library_sha256": plan.get("neuronte_library_sha256"),
        "harness": commit(plan.get("harness")),
        "host": host,
        "reference_arm": analysis["reference"],
        "candidate_arm": analysis["candidate"],
        "pairs": analysis["pairs"],
        "problems": analysis["problems"],
        "runs": [{"run": r["run"], "arm": r["arm"], "index": r["index"], "warmup": r["warmup"],
                  "valid": r["valid"],
                  "failed_checks": [k for k, v in r["checks"].items() if not v],
                  "load1_before": (r["host"]["host_before"].get("loadavg") or [None])[0],
                  "values": {key: lookup(r, key) for key in analysis["comparison"]}}
                 for r in analysis["runs"]],
        "comparison": analysis["comparison"],
        "per_arm": analysis["per_arm"],
    }
    text = json.dumps(record)
    for marker in ("/home/", "/tmp/", "/root/", os.path.expanduser("~")):
        if marker and marker != "~" and marker in text:
            raise ValueError(f"reference would contain a local path ({marker})")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze tab-bench runs")
    parser.add_argument("--out", required=True, type=Path, help="campaign directory (has runs/)")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--campaign", type=Path,
                        help="campaign-NN.json to name commits in a public reference")
    parser.add_argument("--host", type=Path, help="JSON describing the host kind, no names")
    parser.add_argument("--public", type=Path, help="write a path-free reference here")
    args = parser.parse_args(argv)
    analysis = analyze(args.out, args.reference, args.candidate)
    if args.public:
        if not args.campaign:
            parser.error("--public needs --campaign")
        plan = json.loads(args.campaign.read_text(encoding="utf-8"))
        host = json.loads(args.host.read_text(encoding="utf-8")) if args.host else {}
        record = public_reference(analysis, plan, host)
        args.public.parent.mkdir(parents=True, exist_ok=True)
        args.public.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (args.out / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    (args.out / "SUMMARY.md").write_text(summary_markdown(analysis), encoding="utf-8")
    print(summary_markdown(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
