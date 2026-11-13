"""Turn the runs of ``video_bench`` into numbers and one report.

Every figure is taken over a player's own measuring window: from the moment
the terminal took its first picture to the moment its quit key was written.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from kilix_benchmark.video_bench import ARMS, SHM_PREFIX

TICK = os.sysconf("SC_CLK_TCK")


def load(run_dir: Path):
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    with (run_dir / "events.jsonl").open(encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle]
    return result, events


def percentile(values, q: float):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def pictures(events, prefix: str) -> list[dict]:
    """One record per shared-memory picture, in the order they were made.

    A name is reused once the terminal has unlinked it, so each ``created``
    opens a new record for that name."""
    current: dict[str, dict] = {}
    out = []
    for event in events:
        if event["e"] != "shm" or not event["file"].startswith(prefix):
            continue
        name = event["file"]
        if event["what"] == "created":
            current[name] = {"created": event["t"]}
            out.append(current[name])
        elif name in current:
            current[name].setdefault(event["what"], event["t"])
            if "size" in event:
                current[name]["size"] = event["size"]
    return out


def window(result: dict, events, arm: str):
    start = (result.get("first_picture_t") or {}).get(arm)
    stop = next((e["t"] for e in events if e["e"] == "key" and e["who"] == arm and e["key"] == "quit"), None)
    return start, stop


def cadence(times_ns, fps: float, span_s: float) -> dict:
    """How evenly pictures reached the terminal."""
    gaps = [(b - a) / 1e6 for a, b in zip(times_ns, times_ns[1:])]
    period = 1000.0 / fps
    return {"pictures": len(times_ns),
            "per_second": round(len(times_ns) / span_s, 2) if span_s > 0 else None,
            "gap_ms_median": round(statistics.median(gaps), 2) if gaps else None,
            "gap_ms_p95": round(percentile(gaps, 0.95), 2) if gaps else None,
            "gap_ms_max": round(max(gaps), 2) if gaps else None,
            "gaps_over_1_5_periods": sum(g > 1.5 * period for g in gaps),
            "gaps_over_100_ms": sum(g > 100 for g in gaps)}


def cpu(events, who: str, start: int, stop: int) -> dict:
    """CPU seconds per second of the window, by program and by thread name."""
    samples = [e for e in events if e["e"] == "tree" and e["who"] == who and start <= e["t"] <= stop]
    if len(samples) < 2:
        return {"total": None, "by_program": {}, "by_thread": {}, "rss_mb_peak": None}
    span = (samples[-1]["t"] - samples[0]["t"]) / 1e9
    first_seen: dict[str, list] = {}
    last_seen: dict[str, list] = {}
    for sample in samples:
        for pid, entry in sample["procs"].items():
            first_seen.setdefault(pid, entry)
            last_seen[pid] = entry
    programs: dict[str, float] = {}
    for pid, entry in last_seen.items():
        used = (entry[1] - first_seen[pid][1]) / TICK / span
        programs[entry[0]] = round(programs.get(entry[0], 0.0) + used, 4)
    by_thread = {}
    if "threads" in samples[0] and "threads" in samples[-1]:
        for name, ticks in samples[-1]["threads"].items():
            used = (ticks - samples[0]["threads"].get(name, 0)) / TICK / span
            if used >= 0.005:
                by_thread[name] = round(used, 4)
    peak = max(sum(entry[2] for entry in s["procs"].values()) for s in samples) / 1024
    out = {"total": round(sum(programs.values()), 4), "by_program": programs,
           "by_thread": dict(sorted(by_thread.items(), key=lambda item: -item[1])),
           "rss_mb_peak": round(peak, 1)}
    if samples[0].get("io") and samples[-1].get("io"):
        out["write_calls_mb_per_s"] = round((samples[-1]["io"][0] - samples[0]["io"][0]) / span / 1e6, 2)
        out["storage_mb_per_s"] = round((samples[-1]["io"][1] - samples[0]["io"][1]) / span / 1e6, 2)
    return out


def neuronte_log(run_dir: Path) -> list[dict]:
    path = run_dir / "neuronte-frames.jsonl"
    if not path.is_file():
        return []
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue                                    # a last line cut short
    return out


def seeks(events, arm: str, shown: list[dict], log: list[dict]) -> list[dict]:
    """From each seek key to the first picture after it that is from the new place."""
    out = []
    deleted = sorted(p["deleted"] for p in shown if "deleted" in p)
    for key in (e for e in events if e["e"] == "key" and e["who"] == arm and e["key"] != "quit"):
        entry = {"key": key["key"], "ms": None}
        if arm == "mpv":
            restart = next((e["t"] for e in events if e["e"] == "mpv-event"
                            and e["event"] == "playback-restart" and e["t"] > key["t"]), None)
            if restart is not None:
                picture = next((t for t in deleted if t >= restart), None)
                entry["decoder_ms"] = round((restart - key["t"]) / 1e6, 1)
                if picture is not None:
                    entry["ms"] = round((picture - key["t"]) / 1e6, 1)
        else:
            before = [r for r in log if r["t"] * 1e9 <= key["t"]]
            after = [r for r in log if r["t"] * 1e9 > key["t"] and "shown" in r]
            if before and after:
                jump = next((r for r in after if abs(r["pts"] - before[-1]["pts"]) > 1.0), None)
                if jump is not None:
                    entry["ms"] = round(jump["shown"] * 1e3 - key["t"] / 1e6, 1)
                    following = [r for r in log if r["n"] > jump["n"]][:60]
                    entry["dropped_in_next_60"] = sum(bool(r.get("dropped")) for r in following)
                    entry["first_late_ms"] = jump.get("late_ms")
        out.append(entry)
    return out


def arm_metrics(run_dir: Path, result: dict, events, arm: str) -> dict:
    start, stop = window(result, events, arm)
    out = {"arm": arm, "run": result["run"], "phase": result["phase"], "warmup": result["warmup"],
           "side": next((side for side, who in result["sides"].items() if who == arm), None)}
    launched = (result.get("launch_t") or {}).get(arm)
    if start is None or stop is None:
        out["error"] = "no picture reached the terminal" if start is None else "no quit key"
        return out
    span = (stop - start) / 1e9
    every = pictures(events, SHM_PREFIX[arm])
    shown = [p for p in every if start <= p.get("deleted", -1) <= stop]
    sizes = [p["size"] for p in shown if "size" in p]
    out.update(seconds=round(span, 3),
               startup_ms=round((start - launched) / 1e6, 1) if launched is not None else None,
               cadence=cadence([p["deleted"] for p in shown], result["clip"]["fps"], span),
               picture_bytes=int(statistics.median(sizes)) if sizes else None,
               megabytes_per_second=round(sum(sizes) / span / 1e6, 2) if sizes else None,
               cpu=cpu(events, arm, start, stop))
    log = neuronte_log(run_dir) if arm == "neuronte" else []
    if arm == "mpv":
        taken = [(p["deleted"] - p["closed"]) / 1e6 for p in shown if "closed" in p]
        counters = [e for e in events if e["e"] == "mpv" and start <= e["t"] <= stop]
        if counters:
            last = counters[-1]
            sync = [abs(e["avsync"]) * 1e3 for e in counters if isinstance(e.get("avsync"), (int, float))]
            out["player"] = {"dropped_by_output": last.get("frame-drop-count"),
                             "dropped_by_decoder": last.get("decoder-frame-drop-count"),
                             "mistimed": last.get("mistimed-frame-count"),
                             "delayed": last.get("vo-delayed-frame-count"),
                             "sync_ms_p95": round(percentile(sync, 0.95), 2) if sync else None,
                             "sync_ms_max": round(max(sync), 2) if sync else None}
    else:
        inside = [r for r in log if start <= r["t"] * 1e9 <= stop]
        seen = [r for r in inside if "shown" in r]
        ends = sorted(p["deleted"] for p in shown)
        taken = []
        position = 0
        for record in seen:                                 # the first unlink after each write
            wrote = record["shown"] * 1e9
            while position < len(ends) and ends[position] < wrote:
                position += 1
            if position == len(ends):
                break
            taken.append((ends[position] - wrote) / 1e6)
            position += 1
        late = [r["late_ms"] for r in seen]
        stages = {}
        for name in ("read_ms", "fit_ms", "draw_ms", "pack_ms", "write_ms"):
            values = [r[name] for r in seen if name in r]
            if values:
                stages[name] = {"mean": round(statistics.fmean(values), 3),
                                "p95": round(percentile(values, 0.95), 3)}
        out["player"] = {"source_frames": len(inside), "shown": len(seen),
                         "dropped_late": sum(bool(r.get("dropped")) for r in inside),
                         "skipped_terminal_busy": sum(bool(r.get("skipped")) for r in inside),
                         "late_ms_median": round(statistics.median(late), 2) if late else None,
                         "late_ms_p95": round(percentile(late, 0.95), 2) if late else None,
                         "late_ms_max": round(max(late), 2) if late else None,
                         "stages_ms": stages}
    out["terminal_take_ms"] = {"median": round(statistics.median(taken), 2),
                               "p95": round(percentile(taken, 0.95), 2),
                               "max": round(max(taken), 2)} if taken else None
    out["seeks"] = seeks(events, arm, shown, log)
    return out


def captures(events, windows: dict) -> dict | None:
    """Captures of each side taken wholly inside that side's measuring window.

    A pane is rightly blank before its player's first picture and after it quits."""
    shots = [e for e in events if e["e"] == "shot"]
    if not shots:
        return None
    out = {"captures": len(shots),
           "capture_ms_median": round(statistics.median((s["t"] - s["t0"]) / 1e6 for s in shots), 1)}
    for side in ("left", "right"):
        start, stop = windows.get(side) or (None, None)
        inside = [s for s in shots if start is not None and stop is not None and start <= s["t0"] and s["t"] <= stop]
        out[side] = {"captures": len(inside), "flat": sum(s[side]["flat"] for s in inside),
                     "flat_files": [s["file"] for s in inside if s[side]["flat"] and "file" in s],
                     "least_spread": min((s[side]["spread"] for s in inside), default=None)}
    return out


def run_metrics(run_dir: Path) -> dict:
    result, events = load(run_dir)
    arms = [arm_metrics(run_dir, result, events, arm) for arm in result["arms"]]
    out = {"run": result["run"], "phase": result["phase"], "warmup": result["warmup"],
           "status": result["status"], "notes": result["notes"], "sides": result["sides"],
           "geometry": result.get("geometry"), "arms": arms}
    windows = [window(result, events, arm) for arm in result["arms"]]
    if all(a is not None and b is not None for a, b in windows):
        out["terminal"] = cpu(events, "terminal", min(a for a, _ in windows), max(b for _, b in windows))
    if result.get("idle_until_t"):
        idle = cpu(events, "terminal", 0, result["idle_until_t"])
        if idle["total"] is not None:
            out["terminal_idle"] = idle
    loads = [e["loadavg"][0] for e in events if e["e"] == "load"]
    out["load_1m"] = round(statistics.fmean(loads), 2) if loads else None
    shots = captures(events, {side: window(result, events, who) for side, who in result["sides"].items()
                              if who in ARMS})
    if shots is not None:
        for side, who in result["sides"].items():
            shots[side]["who"] = who
        out["captures"] = shots
    return out


# ── the report ──────────────────────────────────────────────────────────────

def dig(record: dict, path: str):
    for key in path.split("."):
        if not isinstance(record, dict) or record.get(key) is None:
            return None
        record = record[key]
    return record


ROWS = (
    ("first picture after launch, ms", "startup_ms", 0),
    ("pictures the terminal took, per second", "cadence.per_second", 2),
    ("gap between pictures, median ms", "cadence.gap_ms_median", 1),
    ("gap between pictures, 95th percentile ms", "cadence.gap_ms_p95", 1),
    ("longest gap, ms", "cadence.gap_ms_max", 0),
    ("gaps over 1.5 frame periods", "cadence.gaps_over_1_5_periods", 1),
    ("bytes per picture", "picture_bytes", 0),
    ("megabytes handed to the terminal per second", "megabytes_per_second", 1),
    ("terminal took a picture after, median ms", "terminal_take_ms.median", 2),
    ("terminal took a picture after, 95th percentile ms", "terminal_take_ms.p95", 2),
    ("player CPU, cores", "cpu.total", 3),
    ("player memory, peak MB", "cpu.rss_mb_peak", 0),
)


def median_of(records, path: str):
    values = [v for v in (dig(r, path) for r in records) if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def spread_of(records, path: str):
    values = [v for v in (dig(r, path) for r in records) if isinstance(v, (int, float))]
    return (min(values), max(values)) if values else None


def cell(records, path: str, digits: int) -> str:
    middle = median_of(records, path)
    if middle is None:
        return "–"
    low, high = spread_of(records, path)
    text = f"{middle:,.{digits}f}"
    return text if low == high else f"{text} ({low:,.{digits}f}–{high:,.{digits}f})"


def merged(records, path: str) -> dict:
    """Median of each named share across runs (programs, threads, stages)."""
    names: dict[str, list] = {}
    for record in records:
        for name, value in (dig(record, path) or {}).items():
            names.setdefault(name, []).append(value["mean"] if isinstance(value, dict) else value)
    return {name: round(statistics.median(values), 4) for name, values in
            sorted(names.items(), key=lambda item: -statistics.median(item[1]))}


def report(metrics: list[dict], campaign: dict | None) -> str:
    lines = ["# kilix-mpv and the Neuronte player, side by side", ""]
    if campaign:
        video = campaign["video"]
        lines += [f"Clip: {video['width']}x{video['height']} {video['codec']}, {video['fps']:g} frames a "
                  f"second, sha256 `{video['sha256'][:16]}`; {campaign['seconds']:g} s from "
                  f"{campaign['start']:g} s. kilix-mpv `{(campaign['mpv']['sha256'] or '')[:16]}`; "
                  f"Neuronte `{(campaign['neuronte_home']['commit'] or '')[:10]}`"
                  + (f" with {campaign['neuronte_home']['dirty_paths']} changed files"
                     if campaign["neuronte_home"]["dirty_paths"] else "") + ".", ""]
    failed = [m for m in metrics if m["status"] != "ok"]
    if failed:
        lines += ["Runs that did not finish: " + ", ".join(m["run"] for m in failed) + ".", ""]
    measured = [m for m in metrics if not m["warmup"] and m["status"] == "ok"]
    for phase, title in (("solo", "One player at a time"), ("together", "Both at once"),
                         ("watch", "Both at once, in front and captured (timing is disturbed)")):
        runs = [m for m in measured if m["phase"] == phase]
        if not runs:
            continue
        per_arm = {arm: [a for m in runs for a in m["arms"] if a["arm"] == arm and "error" not in a]
                   for arm in ARMS}
        count = max(len(v) for v in per_arm.values())
        lines += [f"## {title}", "",
                  f"Median of {count} runs; the range follows in brackets when runs differ.", "",
                  "| | kilix-mpv | Neuronte |", "|---|---|---|"]
        for label, path, digits in ROWS:
            lines.append(f"| {label} | {cell(per_arm['mpv'], path, digits)} | "
                         f"{cell(per_arm['neuronte'], path, digits)} |")
        if phase == "solo":
            lines.append("| terminal CPU while it played, cores | "
                         + " | ".join(cell([m for m in runs if m["arms"][0]["arm"] == arm],
                                           "terminal.total", 3) for arm in ARMS) + " |")
        else:
            lines.append(f"| terminal CPU with both playing, cores | {cell(runs, 'terminal.total', 3)} | |")
        if phase == "solo":
            for label, path, digits in (
                    ("terminal CPU just before, cores", "terminal_idle.total", 3),
                    ("terminal write calls while it played, MB a second", "terminal.write_calls_mb_per_s", 1),
                    ("terminal write calls just before, MB a second", "terminal_idle.write_calls_mb_per_s", 1)):
                lines.append(f"| {label} | " + " | ".join(
                    cell([m for m in runs if m["arms"][0]["arm"] == arm], path, digits) for arm in ARMS) + " |")
        lines.append("")
        for arm in ARMS:
            if not per_arm[arm]:
                continue
            programs = merged(per_arm[arm], "cpu.by_program")
            lines.append(f"{'kilix-mpv' if arm == 'mpv' else 'Neuronte'} CPU by program, cores: "
                         + ", ".join(f"{name} {value:.3f}" for name, value in programs.items()) + ".")
            if arm == "mpv":
                parts = merged(per_arm[arm], "cpu.by_thread")
                if parts:
                    lines.append("kilix-mpv CPU by thread, cores: "
                                 + ", ".join(f"{name} {value:.3f}" for name, value in parts.items()) + ".")
                for label, path in (("pictures dropped by the output", "player.dropped_by_output"),
                                    ("dropped by the decoder", "player.dropped_by_decoder"),
                                    ("delayed", "player.delayed"),
                                    ("A/V difference, worst ms", "player.sync_ms_max")):
                    lines.append(f"kilix-mpv {label}: {cell(per_arm[arm], path, 1)}.")
            else:
                stages = merged(per_arm[arm], "player.stages_ms")
                lines.append("Neuronte time per shown picture, ms: "
                             + ", ".join(f"{name[:-3]} {value:.2f}" for name, value in stages.items()) + ".")
                for label, path in (("source frames dropped for lateness", "player.dropped_late"),
                                    ("skipped because the terminal was behind", "player.skipped_terminal_busy"),
                                    ("picture behind the sound, median ms", "player.late_ms_median"),
                                    ("picture behind the sound, 95th percentile ms", "player.late_ms_p95")):
                    lines.append(f"Neuronte {label}: {cell(per_arm[arm], path, 1)}.")
            found = [s for a in per_arm[arm] for s in a.get("seeks", ()) if s.get("ms") is not None]
            if found:
                values = [s["ms"] for s in found]
                text = (f"{'kilix-mpv' if arm == 'mpv' else 'Neuronte'} key to first picture from the "
                        f"new place, ms: median {statistics.median(values):.0f} "
                        f"({min(values):.0f}–{max(values):.0f}), {len(values)} seeks")
                drops = [s["dropped_in_next_60"] for s in found if "dropped_in_next_60" in s]
                if drops:
                    text += f"; frames dropped in the next 60: median {statistics.median(drops):g}"
                lines.append(text + ".")
            lines.append("")
        thread_share = merged(runs, "terminal.by_thread")
        if thread_share:
            lines += ["Terminal CPU by thread, cores: "
                      + ", ".join(f"{name} {value:.3f}" for name, value in list(thread_share.items())[:6]) + ".", ""]
        if phase == "watch":
            for arm in ARMS:
                flat = total = 0
                for m in runs:
                    for side in ("left", "right"):
                        if m.get("captures", {}).get(side, {}).get("who") == arm:
                            flat += m["captures"][side]["flat"]
                            total += m["captures"][side]["captures"]
                lines.append(f"{'kilix-mpv' if arm == 'mpv' else 'Neuronte'}: {flat} of {total} "
                             "captures taken while it played showed one flat colour where the "
                             "picture belongs.")
            lines.append("")
    loads = [m["load_1m"] for m in measured if m["load_1m"] is not None]
    if loads:
        lines += [f"Host load during measured runs: {min(loads):.1f} to {max(loads):.1f}.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise video_bench runs")
    parser.add_argument("out", type=Path, help="the directory given to video_bench --out")
    args = parser.parse_args(argv)
    runs = sorted(p for p in (args.out / "runs").iterdir() if (p / "result.json").is_file())
    metrics = [run_metrics(run) for run in runs]
    campaigns = sorted(args.out.glob("campaign-*.json"))
    campaign = json.loads(campaigns[-1].read_text(encoding="utf-8")) if campaigns else None
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    text = report(metrics, campaign)
    (args.out / "REPORT.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
