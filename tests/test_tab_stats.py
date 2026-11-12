"""Run metrics from synthetic logs, and the paired comparison rule."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kilix_benchmark import tab_stats as S

MS = 1_000_000


def write_run(root: Path, name: str, arm: str, index: int, *, warmup=False, key_ms=10.0,
              lose_key=False, keep_file=False) -> Path:
    """A minimal run: launch, splash, one paced phase with three keys, quit."""
    run = root / "runs" / name
    run.mkdir(parents=True)
    t = 1_000_000 * MS
    driver = [{"e": "launch", "t": t}]
    events = [
        {"e": "start", "t": t + 50 * MS, "pid": 7},
        {"e": "imported", "t0": t + 60 * MS, "t1": t + 160 * MS},
        {"e": "fonts", "FONT": "NeuronteFont" if arm == "n" else "FreeTypeFont"},
        {"e": "blit", "t0": t + 200 * MS, "t1": t + 210 * MS, "kind": "full", "file": "full-1.rgb"},
        {"e": "screen", "name": "startup", "t0": t + 200 * MS, "t1": t + 1400 * MS},
        {"e": "frame", "t0": t + 1500 * MS, "tb": t + 1590 * MS, "t1": t + 1600 * MS,
         "kind": "full", "file": "full-2.rgb", "text_n": 9, "text_ns": 40 * MS,
         "measure_n": 0, "measure_ns": 0, "mask_n": 9 if arm == "n" else 0, "fonts": {},
         "dispatch_n": 0, "dispatch_ns": 0,
         "wins": [["Notepad", "Untitled", 1, 1, 10, 10, False]]},
    ]
    files = [("full-1.rgb", t + 210 * MS), ("full-2.rgb", t + 1600 * MS)]
    begin = t + 10_000 * MS
    driver.append({"e": "phase", "t": begin, "phase": "type-paced", "what": "begin"})
    for k in range(3):
        sent = begin + (k + 1) * 120 * MS
        driver.append({"e": "send", "t": sent, "t1": sent + 1000, "phase": "type-paced",
                       "label": "a", "keys": 1, "bytes": 14})
        if lose_key and k == 2:
            continue
        got = sent + 1 * MS
        events.append({"e": "input", "t": got, "n": 2,
                       "ev": [["k", "a", 1, 1], ["k", "a", 1, 3]]})
        f0 = got + 1 * MS
        f1 = got + int(key_ms * MS)
        name_k = f"band-{k}.rgb"
        events.append({"e": "frame", "t0": f0, "tb": f1 - 1 * MS, "t1": f1, "kind": "band",
                       "file": name_k, "text_n": 3, "text_ns": 2 * MS, "measure_n": 1,
                       "measure_ns": MS // 2, "mask_n": 3 if arm == "n" else 0, "fonts": {},
                       "dispatch_n": 1, "dispatch_ns": 100_000})
        files.append((name_k, f1 + 2 * MS))
    end = begin + 1000 * MS
    driver.append({"e": "phase", "t": end, "phase": "type-paced", "what": "end"})
    driver.append({"e": "phase", "t": end, "phase": "quit", "what": "begin"})
    driver.append({"e": "phase", "t": end + 5 * MS, "phase": "quit", "what": "end"})
    for label, when in files:
        driver.append({"e": "file", "t": when - 100, "what": "written", "file": label})
        if not (keep_file and label == "band-0.rgb"):
            driver.append({"e": "file", "t": when, "what": "deleted", "file": label})
    driver.append({"e": "proc", "t": begin - MS, "who": "desktop", "utime": 100, "stime": 10,
                   "rss_kb": 1})
    driver.append({"e": "proc", "t": end - MS, "who": "desktop", "utime": 150, "stime": 10,
                   "rss_kb": 1})
    events.append({"e": "exit", "t": end + 9 * MS, "status": 0, "utime": 3.0, "stime": 0.5,
                   "vmhwm_kb": 102400, "neuronte_sizes": None})
    result = {"run": name, "arm": arm, "index": index, "warmup": warmup,
              "status": "complete", "notes": [], "desk": [100, 50],
              "host_before": {"loadavg": [1.0, 1.0, 1.0]}, "host_after": {}}
    (run / "result.json").write_text(json.dumps(result))
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (run / "driver.jsonl").write_text("\n".join(json.dumps(e) for e in driver) + "\n")
    return run


class StatisticsTests(unittest.TestCase):
    def test_percentile_and_spread(self) -> None:
        self.assertEqual(S.median([3, 1, 2]), 2.0)
        self.assertEqual(S.median([1, 2, 3, 4]), 2.5)
        self.assertEqual(S.percentile([0, 10], 95), 9.5)
        self.assertIsNone(S.median([]))
        self.assertEqual(S.spread([None])["n"], 0)

    def test_sign_test_exact_values(self) -> None:
        self.assertAlmostEqual(S.sign_test([1] * 9 + [-1])["p"], 0.021484375)
        self.assertAlmostEqual(S.sign_test([1] * 8 + [-1] * 2)["p"], 0.109375)
        self.assertEqual(S.sign_test([1] * 10)["positive"], 10)
        self.assertEqual(S.sign_test([0, 0])["n"], 0)

    def test_compare_calls_only_clear_and_practical_differences(self) -> None:
        reference = [10.0, 10.2, 9.9, 10.1, 10.0, 9.8, 10.3, 10.0, 9.9, 10.1]
        slower = [r * 1.3 for r in reference]
        self.assertEqual(S.compare(reference, slower)["verdict"], "candidate higher")
        faster = [r * 0.7 for r in reference]
        self.assertEqual(S.compare(reference, faster)["verdict"], "candidate lower")
        tiny = [r + 0.1 for r in reference]           # 1%: consistent but not practical
        self.assertEqual(S.compare(reference, tiny)["verdict"], "no difference shown")
        noisy = [r + (3 if i % 2 else -3) for i, r in enumerate(reference)]
        self.assertEqual(S.compare(reference, noisy)["verdict"], "no difference shown")
        self.assertEqual(S.compare([None], [1.0])["verdict"], "no data")


class RunMetricsTests(unittest.TestCase):
    def test_latency_startup_cpu_and_checks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run = write_run(Path(raw), "03-r", "r", 3, key_ms=10)
            m = S.run_metrics(run)
        self.assertTrue(m["valid"], m["checks"])
        paced = m["phases"]["type-paced"]
        self.assertEqual(paced["frames"], 3)
        self.assertAlmostEqual(paced["latency_written_ms"]["median"], 10.0)
        self.assertAlmostEqual(paced["latency_taken_ms"]["median"], 12.0)
        self.assertAlmostEqual(paced["transport_ms"]["median"], 1.0)
        self.assertAlmostEqual(paced["render_ms"]["median"], 9.0)
        self.assertAlmostEqual(paced["cpu_s"], 0.5)
        self.assertEqual(paced["key_presses_received"], 3)
        self.assertAlmostEqual(m["startup"]["launch_to_desktop_ms"], 1600.0)
        self.assertAlmostEqual(m["startup"]["launch_to_first_frame_ms"], 210.0)
        self.assertAlmostEqual(m["startup"]["import_ms"], 100.0)
        self.assertAlmostEqual(m["whole"]["cpu_s"], 3.5)
        self.assertAlmostEqual(m["whole"]["vmhwm_mb"], 100.0)
        self.assertEqual(m["whole"]["text_path"], "freetype")
        self.assertEqual(S.lookup(m, "type-paced.latency_written_ms.median"), 10.0)
        self.assertEqual(S.lookup(m, "run.vmhwm_mb"), 100.0)
        self.assertIsNone(S.lookup(m, "hover.render_ms.median"))

    def test_lost_key_and_unread_frame_fail_checks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            lost = S.run_metrics(write_run(Path(raw), "03-r", "r", 3, lose_key=True))
            kept = S.run_metrics(write_run(Path(raw), "04-r", "r", 4, keep_file=True))
        self.assertFalse(lost["checks"]["all_keys_received"])
        self.assertFalse(lost["valid"])
        self.assertFalse(kept["checks"]["all_frames_taken"])

    def test_neuronte_text_path_needs_masks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            m = S.run_metrics(write_run(Path(raw), "03-n", "n", 3))
        self.assertEqual(m["whole"]["text_path"], "neuronte")


class CampaignTests(unittest.TestCase):
    def test_pairs_in_order_and_invalid_pairs_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_run(root, "01-r", "r", 1, warmup=True)
            write_run(root, "02-n", "n", 2, warmup=True)
            for i, (arm, ms) in enumerate([("r", 10), ("n", 13), ("n", 13), ("r", 10),
                                           ("r", 10), ("n", 13)], start=3):
                write_run(root, f"{i:02d}-{arm}", arm, i, key_ms=ms)
            write_run(root, "09-r", "r", 9, key_ms=10)
            write_run(root, "10-n", "n", 10, key_ms=13, lose_key=True)
            analysis = S.analyze(root, "r", "n")
        self.assertEqual(analysis["pairs"], 3)
        self.assertEqual(len(analysis["problems"]), 1)
        self.assertIn("10-n", analysis["problems"][0])
        row = analysis["comparison"]["type-paced.latency_written_ms.median"]
        self.assertAlmostEqual(row["median_difference"], 3.0)
        self.assertEqual(row["sign_test"]["positive"], 3)
        self.assertEqual(row["verdict"], "no difference shown")   # 3 pairs cannot reach p<=0.05
        text = S.summary_markdown(analysis)
        self.assertIn("Paced keystroke to frame written", text)
        self.assertIn("| 01-r | r | yes |", text)


if __name__ == "__main__":
    unittest.main()
