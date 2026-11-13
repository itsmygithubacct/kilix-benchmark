"""Driver pieces of the video benchmark that need no terminal."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from kilix_benchmark import video_bench


def wait_for(predicate, seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class PlanTests(unittest.TestCase):
    def test_warmups_orders_and_sides_alternate(self) -> None:
        runs = video_bench.plan(2, 1)
        solo = [r for r in runs if r["phase"] == "solo"]
        self.assertEqual([r["arms"][0] for r in solo],
                         ["mpv", "neuronte", "neuronte", "mpv", "mpv", "neuronte"])
        self.assertEqual([r["warmup"] for r in solo], [True, True, False, False, False, False])
        self.assertEqual([r["left"] for r in solo],
                         ["mpv", "neuronte", "idle", "idle", "mpv", "neuronte"])
        together = [r for r in runs if r["phase"] == "together"]
        self.assertEqual([r["left"] for r in together], ["mpv", "neuronte", "mpv"])
        self.assertEqual([r["warmup"] for r in together], [True, False, False])
        self.assertEqual([r["phase"] for r in runs][-1], "watch")
        self.assertFalse(runs[-1]["warmup"])

    def test_no_pairs_leaves_only_warmups(self) -> None:
        self.assertTrue(all(r["warmup"] for r in video_bench.plan(0, 0)))
        self.assertEqual(len(video_bench.plan(0, 0)), 3)


class CommandTests(unittest.TestCase):
    def test_mpv_uses_the_launcher_line_shared_memory_and_no_sound_device(self) -> None:
        command = video_bench.mpv_command("/usr/bin/kilix-mpv", "/v.mp4", "/s/mpv.sock", 20, ["--sws-fast=yes"])
        self.assertEqual(command[0], "/usr/bin/kilix-mpv")
        self.assertEqual(command[-1], "/v.mp4")
        for part in ("--vo=kitty", "--vo-kitty-use-shm=yes", "--no-config", "--ao=null",
                     "--input-ipc-server=/s/mpv.sock", "--start=20", "--sws-fast=yes"):
            self.assertIn(part, command)
        self.assertFalse(any(part.startswith("--length") for part in command))

    def test_neuronte_logs_every_frame_and_takes_extra_options(self) -> None:
        command = video_bench.neuronte_command("/usr/bin/python3", "/n", "/v.mp4", "/o/f.jsonl",
                                               "/o/s.json", 20, ["--transport", "inline"])
        self.assertEqual(command[:2], ["/usr/bin/python3", "/n/tools/neuronte-play.py"])
        self.assertEqual(command[-1], "/v.mp4")
        self.assertEqual(command[command.index("--log") + 1], "/o/f.jsonl")
        self.assertEqual(command[command.index("--audio-output") + 1], "null")
        self.assertEqual(command[-3:-1], ["--transport", "inline"])

    def test_seek_spec(self) -> None:
        self.assertEqual(video_bench.seek_spec("8:right"), (8.0, "right"))
        with self.assertRaises(Exception):
            video_bench.seek_spec("8:sideways")


class RunAttributionTests(unittest.TestCase):
    def test_only_owned_players_count_including_frames_before_pid_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run = video_bench.Run(spec={"phase":"solo", "arms":["neuronte"],
                                       "left":"neuronte", "warmup":False},
                                  index=1, out_dir=Path(raw), kitten=None, args=None, clip={})
            def picture(pid, moment, prefix="neuronte-play-"):
                return {"e":"shm", "what":"deleted", "file":f"{prefix}{pid}-a-0", "t":moment}
            try:
                run.note(picture(420, 1))  # Same player type, even a PID prefix match.
                run.note(picture(9, 2, "kilix-mpv-"))  # Another test's player.
                run.note(picture(42, 3))
                self.assertEqual(run.first_picture, {})
                run.set_players({"neuronte":42})
                self.assertEqual(run.first_picture, {"neuronte":3})
                run.note(picture(420, 4))
                run.note(picture(9, 5, "kilix-mpv-"))
                run.note(picture(42, 6))
                run.events.flush()
                records = [json.loads(line) for line in (run.dir / "events.jsonl").read_text().splitlines()]
                self.assertEqual([r["t"] for r in records], [3, 6])
                self.assertEqual(run.first_picture, {"neuronte":3})
            finally:
                run.events.close()
                run.short.rmdir()


class ShmWatchTests(unittest.TestCase):
    def test_times_only_the_players_pictures_and_reads_their_size(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            seen = []
            watch = video_bench.ShmWatch(seen.append, ["kilix-mpv-", "neuronte-play-"], root=raw)
            watch.start()
            try:
                for name, size in (("kilix-mpv-1-a-0", 300), ("somebody-else", 10), ("neuronte-play-2-b-1", 120)):
                    with open(os.path.join(raw, name), "wb") as handle:
                        handle.write(b"x" * size)
                    time.sleep(0.05)
                os.unlink(os.path.join(raw, "kilix-mpv-1-a-0"))
                self.assertTrue(wait_for(lambda: any(e["what"] == "deleted" for e in seen)))
            finally:
                watch.stop()
        self.assertEqual({e["file"] for e in seen}, {"kilix-mpv-1-a-0", "neuronte-play-2-b-1"})
        mpv = [e for e in seen if e["file"] == "kilix-mpv-1-a-0"]
        self.assertEqual([e["what"] for e in mpv], ["created", "closed", "deleted"])
        self.assertEqual(mpv[1]["size"], 300)
        self.assertEqual([e["t"] for e in mpv], sorted(e["t"] for e in mpv))
        self.assertEqual(next(e for e in seen if e["file"].startswith("neuronte") and e["what"] == "closed")["size"], 120)


class ProcessTests(unittest.TestCase):
    def test_stat_fields_survive_a_name_with_spaces_and_brackets(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = os.path.join(raw, "stat")
            fields = ["S", "7"] + ["0"] * 9 + ["11", "22"] + ["0"] * 8 + ["333"] + ["0"] * 20
            Path(path).write_text("42 (odd) name (x)) " + " ".join(fields), encoding="ascii")
            self.assertEqual(video_bench.stat_fields(path), ("odd) name (x)", 7, 11, 22, 333))
            self.assertIsNone(video_bench.stat_fields(os.path.join(raw, "absent")))

    def test_tree_and_find_player_see_through_a_wrapper(self) -> None:
        command = [sys.executable, "-c", "import time; time.sleep(30)"]
        wrapper = subprocess.Popen(["sh", "-c", '"$@" & wait', "sh", *command])
        try:
            player = video_bench.find_player(wrapper.pid, command, wait=5)
            self.assertIsNotNone(player)
            self.assertNotEqual(player, wrapper.pid)
            self.assertIn(player, video_bench.tree(wrapper.pid))
            self.assertIsNone(video_bench.find_player(wrapper.pid, command + ["other"], wait=0.1))
            self.assertIn("python", " ".join(video_bench.threads(player)))
            self.assertEqual(len(video_bench.written(player)), 2)
        finally:
            subprocess.run(["kill", str(player)] if player else ["true"])
            wrapper.kill()
            wrapper.wait()

    def test_sampler_reports_a_tree_by_program(self) -> None:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        seen = []
        try:
            sampler = video_bench.TreeSampler(seen.append)
            sampler.roots["arm"] = child.pid
            sampler.single["terminal"] = os.getpid()
            sampler.threaded.add("arm")
            sampler.sample()
        finally:
            child.kill()
            child.wait()
        arm = next(e for e in seen if e.get("who") == "arm")
        self.assertEqual(list(arm["procs"]), [str(child.pid)])
        self.assertEqual(len(arm["procs"][str(child.pid)]), 3)
        self.assertIn("threads", arm)
        self.assertIn("io", next(e for e in seen if e.get("who") == "terminal"))
        self.assertTrue(any(e["e"] == "load" for e in seen))


class FakeMpv(threading.Thread):
    """Answers get_property like mpv and sends one event."""

    def __init__(self, path: str):
        super().__init__(daemon=True)
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(path)
        self.server.listen(1)
        self.asked = []

    def run(self):
        connection, _ = self.server.accept()
        connection.sendall(b'{"event":"playback-restart"}\n')
        buffer = b""
        while True:
            try:
                data = connection.recv(4096)
            except OSError:
                break
            if not data:
                break
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                request = json.loads(line)
                self.asked.append(request["command"])
                value = {"time-pos": 21.5, "frame-drop-count": 3}.get(request["command"][-1])
                reply = {"request_id": request["request_id"], "error": "success", "data": value}
                try:
                    connection.sendall(json.dumps(reply).encode() + b"\n")
                except OSError:
                    return


class MpvIpcTests(unittest.TestCase):
    def test_samples_properties_and_passes_events_on(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = os.path.join(raw, "s")
            fake = FakeMpv(path)
            fake.start()
            seen = []
            ipc = video_bench.MpvIpc(path, seen.append, period=0.02)
            self.assertTrue(ipc.connect(2))
            ipc.start()
            try:
                self.assertTrue(wait_for(lambda: sum(e["e"] == "mpv" for e in seen) >= 2))
            finally:
                ipc.stop()
        sample = next(e for e in seen if e["e"] == "mpv")
        self.assertEqual(sample["time-pos"], 21.5)
        self.assertEqual(sample["frame-drop-count"], 3)
        self.assertEqual(set(video_bench.MPV_PROPERTIES) - set(sample), set())
        self.assertTrue(any(e["e"] == "mpv-event" and e["event"] == "playback-restart" for e in seen))
        self.assertIn(["observe_property", 0, "seeking"], fake.asked)

    def test_connect_gives_up_when_nothing_listens(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            self.assertFalse(video_bench.MpvIpc(os.path.join(raw, "none"), print).connect(0.1))


class FlatCentreTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            self.skipTest("Pillow is not installed")
        self.Image, self.ImageDraw = Image, ImageDraw

    def test_a_blank_half_is_flat_and_a_picture_is_not(self) -> None:
        image = self.Image.new("RGB", (800, 400), (12, 12, 12))
        draw = self.ImageDraw.Draw(image)
        for x in range(400, 800, 8):                        # stripes on the right half only
            draw.rectangle((x, 0, x + 3, 400), fill=(240, 200, 40))
        self.assertTrue(video_bench.flat_centre(image, "left")[0])
        flat, spread = video_bench.flat_centre(image, "right")
        self.assertFalse(flat)
        self.assertGreater(spread, 20)

    def test_a_picture_that_stops_short_of_the_middle_still_counts_as_blank(self) -> None:
        image = self.Image.new("RGB", (800, 400), (0, 0, 0))
        self.ImageDraw.Draw(image).rectangle((0, 0, 60, 60), fill=(255, 255, 255))
        self.assertTrue(video_bench.flat_centre(image, "left")[0])


if __name__ == "__main__":
    unittest.main()
