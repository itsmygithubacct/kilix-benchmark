"""Driver pieces that do not need a terminal: order, sandbox, inotify, tails."""

from __future__ import annotations

import json
import os
import struct
import tempfile
import time
import unittest
from pathlib import Path

from kilix_benchmark import tab_bench


class OrderTests(unittest.TestCase):
    def test_warmups_then_alternating_pairs(self) -> None:
        self.assertEqual(tab_bench.order(["a", "b"], 3), ["a", "b", "a", "b", "b", "a", "a", "b"])
        self.assertEqual(tab_bench.order(["a", "b"], 0), ["a", "b"])
        with self.assertRaises(ValueError):
            tab_bench.order(["a"], 1)


class SandboxTests(unittest.TestCase):
    def test_every_writable_path_is_inside_the_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "run"
            suite = {"seed_files": ["one.txt", "two.txt"]}
            frames = tab_bench.prepare_sandbox(root, suite)
            self.assertTrue(frames.is_dir())
            self.assertEqual(oct(frames.stat().st_mode & 0o777), "0o700")
            self.assertEqual(sorted(p.name for p in (root / "desktop").iterdir()),
                             ["one.txt", "two.txt"])
            cache = Path(raw) / "cache"
            env = tab_bench.sandbox_env(root, cache, "/src/kilix", "/lib/state.so", {"X": "1"})
            shared = {"KILIX_HOME", "KILIX_STATE_LIBRARY", "GPU_TERMINAL_SOURCE_HOME",
                      "KILIX95_CACHE_HOME", "X"}
            for name, value in env.items():
                if name in shared:
                    continue
                self.assertTrue(value.startswith(str(root)), f"{name}={value}")
            self.assertEqual(env["KILIX95_CACHE_HOME"], str(cache))
            self.assertEqual(env["X"], "1")
            self.assertEqual(Path(env["KILIX95_SESSION_HOME"]) / "frames", frames)
            again = tab_bench.prepare_sandbox(root, {"seed_files": []})
            self.assertEqual(list((root / "desktop").iterdir()), [])
            self.assertEqual(again, frames)


class InotifyTests(unittest.TestCase):
    def test_parse_packed_events(self) -> None:
        def event(wd, mask, name):
            raw = name.encode() + b"\0"
            raw += b"\0" * (-len(raw) % 16)
            return struct.pack("iIII", wd, mask, 0, len(raw)) + raw
        data = event(1, tab_bench.IN_CREATE | tab_bench.IN_ISDIR, "tty-graphics-protocol-x") + \
            event(2, tab_bench.IN_DELETE, "band-7.rgb")
        self.assertEqual(tab_bench.parse_inotify(data), [
            (1, tab_bench.IN_CREATE | tab_bench.IN_ISDIR, "tty-graphics-protocol-x"),
            (2, tab_bench.IN_DELETE, "band-7.rgb")])

    def test_watch_sees_a_file_written_and_deleted_in_a_new_directory(self) -> None:
        seen = []
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            watch = tab_bench.FrameWatch(root, seen.append)
            watch.start()
            try:
                sub = root / "tty-graphics-protocol-test"
                sub.mkdir()
                time.sleep(0.3)
                path = sub / "full-1.rgb"
                path.write_bytes(b"\0" * 12)
                path.unlink()
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and not any(
                        r.get("what") == "deleted" for r in seen):
                    time.sleep(0.05)
            finally:
                watch.stop()
        whats = [r["what"] for r in seen if r.get("file") == "full-1.rgb"]
        self.assertIn("written", whats)
        self.assertIn("deleted", whats)
        self.assertLess(whats.index("written"), whats.index("deleted"))


class TreeTests(unittest.TestCase):
    TREE = [{"id": 1, "is_focused": True, "tabs": [
        {"id": 1, "is_focused": False, "windows": [{"id": 3, "is_focused": False}]},
        {"id": 2, "is_focused": True, "windows": [{"id": 4, "is_focused": False},
                                                  {"id": 5, "is_focused": True}]}]}]

    def test_focused_and_found(self) -> None:
        self.assertEqual(tab_bench.focused_window(self.TREE), 5)
        self.assertEqual(tab_bench.find_window(self.TREE, 3)[2]["id"], 3)
        self.assertIsNone(tab_bench.find_window(self.TREE, 9))


class EventsTailTests(unittest.TestCase):
    def test_partial_lines_wait_for_their_end(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            tail = tab_bench.Events(path)
            self.assertEqual(tail.poll(), [])
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"e": "start", "pid": 1}) + "\n" + '{"e": "fra')
            self.assertEqual([r["e"] for r in tail.poll()], ["start"])
            with open(path, "a", encoding="utf-8") as handle:
                handle.write('me", "t0": 1}\n')
            self.assertEqual(tail.poll(), [{"e": "frame", "t0": 1}])
            self.assertEqual(len(tail.find("frame")), 1)


class ProcTests(unittest.TestCase):
    def test_reads_own_process(self) -> None:
        sample = tab_bench.read_proc(os.getpid())
        self.assertIsNotNone(sample)
        self.assertGreater(sample[2], 0)
        self.assertIsNone(tab_bench.read_proc(2 ** 22 + 12345))


if __name__ == "__main__":
    unittest.main()
