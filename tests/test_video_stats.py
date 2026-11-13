"""The video benchmark's arithmetic, on runs written by hand."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kilix_benchmark import video_stats

MS = 1_000_000
S = 1_000_000_000


def shm(t_ms: float, what: str, name: str, size=None) -> dict:
    event = {"e": "shm", "t": int(t_ms * MS), "what": what, "file": name}
    if size is not None:
        event["size"] = size
    return event


def steady(prefix: str, start_ms: float, count: int, period_ms: float, take_ms: float, size: int):
    """Pictures made one period apart on three reused names."""
    out = []
    for index in range(count):
        made = start_ms + index * period_ms
        name = f"{prefix}9-x-{index % 3}"
        out += [shm(made, "created", name), shm(made + 1, "closed", name, size),
                shm(made + 1 + take_ms, "deleted", name)]
    return out


def tree(t_ms: float, who: str, procs: dict, threads=None, io=None) -> dict:
    event = {"e": "tree", "t": int(t_ms * MS), "who": who, "procs": procs}
    if threads is not None:
        event["threads"] = threads
    if io is not None:
        event["io"] = io
    return event


class PictureTests(unittest.TestCase):
    def test_a_reused_name_is_a_new_picture_each_time(self) -> None:
        events = steady("kilix-mpv-", 0, 7, 10, 3, 500) + [shm(999, "created", "other-1")]
        events.sort(key=lambda e: e["t"])
        found = video_stats.pictures(events, "kilix-mpv-")
        self.assertEqual(len(found), 7)
        self.assertEqual([p["created"] // MS for p in found], [0, 10, 20, 30, 40, 50, 60])
        self.assertTrue(all(p["deleted"] - p["closed"] == 3 * MS and p["size"] == 500 for p in found))

    def test_a_picture_never_taken_has_no_delete(self) -> None:
        found = video_stats.pictures([shm(0, "created", "kilix-mpv-a"), shm(1, "closed", "kilix-mpv-a", 9)],
                                     "kilix-mpv-")
        self.assertEqual(found, [{"created": 0, "closed": MS, "size": 9}])


class CadenceTests(unittest.TestCase):
    def test_even_pictures_have_no_long_gaps(self) -> None:
        got = video_stats.cadence([i * 40 * MS for i in range(26)], 25.0, 1.0)
        self.assertEqual((got["pictures"], got["per_second"]), (26, 26.0))
        self.assertEqual((got["gap_ms_median"], got["gap_ms_max"]), (40.0, 40.0))
        self.assertEqual((got["gaps_over_1_5_periods"], got["gaps_over_100_ms"]), (0, 0))

    def test_a_missed_picture_and_a_stall_are_counted(self) -> None:
        times = [0, 40, 80, 160, 200, 400]                  # one doubled gap, one 200 ms stall
        got = video_stats.cadence([t * MS for t in times], 25.0, 0.4)
        self.assertEqual(got["gaps_over_1_5_periods"], 2)
        self.assertEqual(got["gaps_over_100_ms"], 1)
        self.assertEqual(got["gap_ms_max"], 200.0)

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(video_stats.percentile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertAlmostEqual(video_stats.percentile([10, 20], 0.95), 19.5)
        self.assertIsNone(video_stats.percentile([], 0.5))


class CpuTests(unittest.TestCase):
    def test_cores_by_program_thread_and_memory_peak(self) -> None:
        events = [
            tree(0, "neuronte", {"5": ["python3", 100, 2048], "6": ["ffmpeg", 10, 1024]}),
            tree(1000, "neuronte", {"5": ["python3", 100 + video_stats.TICK // 2, 4096],
                                    "6": ["ffmpeg", 10 + video_stats.TICK // 4, 1024]}),
            tree(2000, "neuronte", {"5": ["python3", 100 + video_stats.TICK, 2048],
                                    "6": ["ffmpeg", 10 + video_stats.TICK // 2, 1024]}),
            tree(5000, "neuronte", {"5": ["python3", 999999, 1]}),          # after the window
            tree(0, "mpv", {"7": ["kilix-mpv", 0, 1]}),                      # somebody else
        ]
        got = video_stats.cpu(events, "neuronte", 0, 2000 * MS)
        self.assertEqual(got["by_program"], {"python3": 0.5, "ffmpeg": 0.25})
        self.assertEqual(got["total"], 0.75)
        self.assertEqual(got["rss_mb_peak"], 5.0)

    def test_threads_and_write_volume_come_from_the_first_and_last_sample(self) -> None:
        events = [tree(0, "terminal", {"1": ["kitty", 0, 1]}, {"kitty": 0, "DiskCacheWrite": 50, "idle": 3},
                       [1_000_000, 0]),
                  tree(1000, "terminal", {"1": ["kitty", video_stats.TICK, 1]},
                       {"kitty": video_stats.TICK // 2, "DiskCacheWrite": 50 + video_stats.TICK // 10, "idle": 3},
                       [41_000_000, 2_000_000])]
        got = video_stats.cpu(events, "terminal", 0, 1000 * MS)
        self.assertEqual(got["by_thread"], {"kitty": 0.5, "DiskCacheWrite": 0.1})
        self.assertEqual((got["write_calls_mb_per_s"], got["storage_mb_per_s"]), (40.0, 2.0))

    def test_one_sample_is_not_a_measurement(self) -> None:
        self.assertIsNone(video_stats.cpu([tree(0, "mpv", {"1": ["kilix-mpv", 0, 1]})], "mpv", 0, S)["total"])


def write_run(root: Path, events, log=None, **changes) -> Path:
    run = root / "07-together-mpv+neuronte-left-mpv"
    run.mkdir()
    result = {"run": run.name, "phase": "together", "warmup": False, "status": "ok", "notes": [],
              "arms": ["mpv", "neuronte"], "sides": {"left": "mpv", "right": "neuronte"},
              "clip": {"fps": 25.0}, "launch_t": {"mpv": 0, "neuronte": 0},
              "first_picture_t": {"mpv": 1004 * MS, "neuronte": 503 * MS}, "geometry": {}}
    result.update(changes)
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with (run / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in sorted(events, key=lambda e: e.get("t", 0)):
            handle.write(json.dumps(event) + "\n")
    if log is not None:
        with (run / "neuronte-frames.jsonl").open("w", encoding="utf-8") as handle:
            for record in log:
                handle.write(json.dumps(record) + "\n")
            handle.write('{"n": 999, "t": 1.0, "pts"')       # the player was stopped mid-line
    return run


def scenario():
    """mpv: 25 pictures a second for 2 s, taken after 3 ms, one seek.
    Neuronte: the same, taken after 2 ms, two frames dropped, one seek."""
    events = steady("kilix-mpv-", 1000, 51, 40, 3, 1500) + steady("neuronte-play-", 500, 51, 40, 2, 1200)
    events += [{"e": "key", "t": 3004 * MS, "who": "mpv", "key": "quit"},
               {"e": "key", "t": 2503 * MS, "who": "neuronte", "key": "quit"},
               {"e": "key", "t": 2000 * MS, "who": "mpv", "key": "right"},
               {"e": "mpv-event", "t": 1900 * MS, "event": "playback-restart"},      # before the key
               {"e": "mpv-event", "t": 2150 * MS, "event": "playback-restart"},
               {"e": "key", "t": 1500 * MS, "who": "neuronte", "key": "right"}]
    for index in range(9):
        t = 1000 + index * 250
        events.append({"e": "mpv", "t": t * MS, "time-pos": 20 + index / 4, "frame-drop-count": index,
                       "decoder-frame-drop-count": 0, "mistimed-frame-count": None,
                       "vo-delayed-frame-count": 1, "avsync": 0.001 * index, "estimated-vf-fps": 25})
        events.append(tree(t, "mpv", {"7": ["kilix-mpv", index * video_stats.TICK // 4, 1024 * 200]},
                           {"vo": index * video_stats.TICK // 5, "kilix-mpv": index * video_stats.TICK // 20}))
        events.append(tree(t - 500, "neuronte", {"8": ["python3", index * video_stats.TICK // 50, 1024 * 100]}))
        events.append(tree(t - 500, "terminal", {"1": ["kitty", index * video_stats.TICK // 20, 1024]},
                           {"kitty": index * video_stats.TICK // 20}, [index * 10_000_000, 0]))
        events.append({"e": "load", "t": t * MS, "loadavg": [2.0, 2.0, 2.0]})
    log = []
    for index in range(51):
        shown = (500 + index * 40 + 1) / 1000
        record = {"n": index + 1, "t": shown, "pts": 20 + index * 0.04 + (5 if index >= 30 else 0),
                  "clock": 0, "read_ms": 0.1}
        if index in (31, 32):
            record["dropped"] = True
        else:
            record.update(fit_ms=0.3, draw_ms=2.0, pack_ms=0.5, write_ms=0.4, late_ms=1.5, shown=shown)
        log.append(record)
    return events, log


class RunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        events, log = scenario()
        self.run_dir = write_run(Path(self.temp.name), events, log)
        self.metrics = video_stats.run_metrics(self.run_dir)
        self.mpv, self.neuronte = self.metrics["arms"]

    def test_each_arm_is_measured_over_its_own_window(self) -> None:
        self.assertEqual((self.mpv["seconds"], self.neuronte["seconds"]), (2.0, 2.0))
        self.assertEqual((self.mpv["startup_ms"], self.neuronte["startup_ms"]), (1004.0, 503.0))
        self.assertEqual(self.mpv["cadence"]["pictures"], 51)
        self.assertEqual(self.neuronte["cadence"]["pictures"], 51)
        self.assertEqual(self.mpv["picture_bytes"], 1500)
        self.assertEqual(self.neuronte["picture_bytes"], 1200)
        self.assertEqual(self.mpv["side"], "left")

    def test_the_terminal_before_the_players_start_is_kept_apart(self) -> None:
        events, log = scenario()
        with tempfile.TemporaryDirectory() as raw:
            run = write_run(Path(raw), events, log, idle_until_t=1000 * MS)
            metrics = video_stats.run_metrics(run)
        self.assertEqual(metrics["terminal_idle"]["total"], 0.2)
        self.assertNotIn("terminal_idle", self.metrics)

    def test_how_long_the_terminal_took(self) -> None:
        self.assertEqual(self.mpv["terminal_take_ms"]["median"], 3.0)
        self.assertEqual(self.neuronte["terminal_take_ms"]["median"], 2.0)

    def test_player_counters(self) -> None:
        self.assertEqual(self.mpv["player"]["dropped_by_output"], 8)
        self.assertEqual(self.mpv["player"]["sync_ms_max"], 8.0)
        self.assertEqual(self.neuronte["player"]["dropped_late"], 2)
        # the window opens when the terminal takes the first picture, just after the player wrote it
        self.assertEqual(self.neuronte["player"]["shown"], 48)
        self.assertEqual(self.neuronte["player"]["stages_ms"]["draw_ms"]["mean"], 2.0)

    def test_cpu_shares(self) -> None:
        self.assertEqual(self.mpv["cpu"]["total"], 1.0)
        self.assertEqual(self.mpv["cpu"]["by_thread"], {"vo": 0.8, "kilix-mpv": 0.2})
        self.assertEqual(self.neuronte["cpu"]["total"], 0.08)
        self.assertEqual(self.metrics["terminal"]["total"], 0.2)
        self.assertEqual(self.metrics["terminal"]["write_calls_mb_per_s"], 40.0)

    def test_seek_is_timed_to_the_first_picture_from_the_new_place(self) -> None:
        mpv = self.mpv["seeks"][0]
        self.assertEqual(mpv["decoder_ms"], 150.0)          # the restart after the key, not the one before
        self.assertEqual(mpv["ms"], 164.0)                  # the first unlink at or after that restart
        neuronte = self.neuronte["seeks"][0]
        self.assertEqual(neuronte["ms"], 201.0)             # frame 31 is the first shown from +5 s
        self.assertEqual(neuronte["dropped_in_next_60"], 2)

    def test_report_names_both_players_and_the_measured_values(self) -> None:
        text = video_stats.report([self.metrics], None)
        self.assertIn("## Both at once", text)
        self.assertIn("| bytes per picture | 1,500 | 1,200 |", text)
        self.assertIn("| player CPU, cores | 1.000 | 0.080 |", text)
        self.assertIn("kilix-mpv CPU by thread, cores: vo 0.800, kilix-mpv 0.200.", text)
        self.assertIn("Neuronte source frames dropped for lateness: 2.0.", text)
        self.assertIn("kilix-mpv key to first picture from the new place, ms: median 164", text)
        self.assertNotIn("One player at a time", text)

    def test_warmups_and_failed_runs_stay_out_of_the_tables(self) -> None:
        warm = dict(self.metrics, warmup=True)
        failed = dict(self.metrics, status="failed", run="09-x")
        text = video_stats.report([warm, failed], None)
        self.assertNotIn("| bytes per picture", text)
        self.assertIn("Runs that did not finish: 09-x.", text)


class MissingTests(unittest.TestCase):
    def test_an_arm_that_showed_nothing_is_an_error_not_a_zero(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events, log = scenario()
            run = write_run(Path(raw), events, log, first_picture_t={"neuronte": 503 * MS})
            metrics = video_stats.run_metrics(run)
            self.assertEqual(metrics["arms"][0]["error"], "no picture reached the terminal")
            self.assertNotIn("terminal", metrics)
            self.assertIn("| bytes per picture | – | 1,200 |", video_stats.report([metrics], None))

    def test_captures_are_counted_for_the_player_on_that_side(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events, log = scenario()
            # mpv plays 1004-3004 ms on the left, Neuronte 503-2503 ms on the right. Both panes are
            # blank in the first capture (before either started) and the right one in the last
            # (after Neuronte quit); only the capture at 1400 ms is a blank during play.
            for index, at in enumerate((100, 1000, 1400, 1800, 2200, 2600)):
                blank_right = at in (100, 1400, 2600)
                events.append({"e": "shot", "t0": at * MS, "t": (at + 100) * MS, "n": index + 1,
                               "size": [800, 400], "file": f"shot-{index + 1:04d}.png",
                               "left": {"flat": at == 100, "spread": 0.0 if at == 100 else 40.0},
                               "right": {"flat": blank_right, "spread": 0.0 if blank_right else 35.0}})
            run = write_run(Path(raw), events, log, phase="watch")
            metrics = video_stats.run_metrics(run)
            self.assertEqual(metrics["captures"]["right"], {"captures": 4, "flat": 1, "flat_files": ["shot-0003.png"],
                                                            "least_spread": 0.0, "who": "neuronte"})
            self.assertEqual(metrics["captures"]["left"]["captures"], 4)
            self.assertEqual(metrics["captures"]["left"]["flat"], 0)
            self.assertEqual(metrics["captures"]["capture_ms_median"], 100.0)
            text = video_stats.report([metrics], None)
            self.assertIn("Neuronte: 1 of 4 captures taken while it played showed one flat colour", text)
            self.assertIn("kilix-mpv: 0 of 4 captures taken while it played showed one flat colour", text)


if __name__ == "__main__":
    unittest.main()
