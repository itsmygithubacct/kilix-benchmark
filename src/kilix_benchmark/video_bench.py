"""Play one video with kilix-mpv and with the Neuronte player, side by side.

Each run opens its own Kilix tab with two panes of equal size. A solo run
plays the file in one pane while the other idles; a together run plays it in
both at once. Sides and order alternate. The driver records, on one
monotonic clock:

- every picture each player hands the terminal through shared memory: when
  the object was created, when the player closed it, and when the terminal
  unlinked it (the terminal unlinks a picture once it has read it), with its
  size in bytes, through inotify on /dev/shm;
- CPU ticks and resident memory of each player's whole process tree, by
  program name, of kilix-mpv's threads by thread name, and of the terminal;
- kilix-mpv's own counters over its IPC socket (position, dropped, mistimed
  and delayed frames, A/V difference, picture size);
- the Neuronte player's own per-frame log and summary;
- when scripted seek keys were written to each pane, and when each player
  showed a picture again;
- host load, battery and GPU state before and after.

``--watch`` adds runs in which the tab is brought to the front and the
window is captured several times a second; a capture whose picture area is
one flat colour is counted for the side it happened on. Those runs disturb
the timing and are reported apart from the others.

Sound goes to a null output in both players, so both keep time by an audio
clock and nothing is heard.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from kilix_benchmark.tab_bench import (
    IN_CLOEXEC, IN_CLOSE_WRITE, IN_CREATE, IN_DELETE, IN_NONBLOCK, Kitten, file_sha256,
    find_window, focused_window, host_state, identity, mono, parse_inotify, wait_for_quiet,
)

RUN_SCHEMA = "kilix.benchmark.video-run/v1"
ARMS = ("mpv", "neuronte")
SHM_PREFIX = {"mpv": "kilix-mpv-", "neuronte": "neuronte-play-"}
MPV_PROPERTIES = ("time-pos", "frame-drop-count", "decoder-frame-drop-count",
                  "mistimed-frame-count", "vo-delayed-frame-count", "avsync",
                  "estimated-vf-fps")
SEEK_KEYS = {"right": "\x1b[C", "left": "\x1b[D", "up": "\x1b[A", "down": "\x1b[B"}


# ── what the players are asked to do ────────────────────────────────────────

def mpv_command(mpv: str, video: str, ipc: str, start: float = 0.0, extra=()) -> list[str]:
    """The launcher's own kilix-mpv line, plus a socket and a silent output."""
    return [mpv, "--no-config", "--load-scripts=no", "--vo=kitty", "--vo-kitty-use-shm=yes",
            "--terminal=yes", "--input-terminal=yes", "--force-window=no", "--audio-display=no",
            "--term-osd=no", "--term-status-msg=", "--msg-level=all=warn", "--msg-color=no",
            "--ao=null", f"--input-ipc-server={ipc}", f"--start={start:g}", *extra, video]


def neuronte_command(python: str, home: str, video: str, log: str, stats: str,
                     start: float = 0.0, extra=()) -> list[str]:
    return [python, str(Path(home) / "tools" / "neuronte-play.py"), "--no-status",
            "--audio-output", "null", "--start", f"{start:g}", "--log", log, "--stats", stats,
            *extra, video]


def plan(pairs: int, watch: int) -> list[dict]:
    """Solo runs in alternating order, then together runs with sides swapped.

    The first run of each kind is a warm-up. Watch runs come last."""
    runs = []
    solo = ["mpv", "neuronte"]
    for index in range(pairs + 1):
        first, second = solo if index % 2 == 0 else solo[::-1]
        for arm in (first, second):
            runs.append({"phase": "solo", "arms": [arm], "left": arm if index % 2 == 0 else None,
                         "warmup": index == 0})
    for index in range(pairs + 1):
        runs.append({"phase": "together", "arms": list(ARMS), "left": ARMS[index % 2],
                     "warmup": index == 0})
    for index in range(watch):
        runs.append({"phase": "watch", "arms": list(ARMS), "left": ARMS[index % 2],
                     "warmup": False})
    for run in runs:
        if run["left"] is None:
            run["left"] = "idle"
    return runs


def probe(video: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,codec_name:format=duration", "-of", "json", video],
        capture_output=True, text=True, timeout=60, check=True)
    doc = json.loads(out.stdout)
    stream = doc["streams"][0]
    num, _, den = stream["r_frame_rate"].partition("/")
    return {"width": stream["width"], "height": stream["height"], "codec": stream["codec_name"],
            "fps": float(num) / float(den or 1), "duration": float(doc["format"]["duration"])}


# ── observers ───────────────────────────────────────────────────────────────

class ShmWatch(threading.Thread):
    """Times the shared-memory pictures of the players under test."""

    def __init__(self, sink, prefixes, root: str = "/dev/shm"):
        super().__init__(daemon=True)
        self.sink, self.prefixes, self.root = sink, tuple(prefixes), root
        self.stopping = False
        libc = ctypes.CDLL(None, use_errno=True)
        libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self.fd = libc.inotify_init1(IN_NONBLOCK | IN_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1 failed")
        if libc.inotify_add_watch(self.fd, os.fsencode(root),
                                  IN_CREATE | IN_CLOSE_WRITE | IN_DELETE) < 0:
            raise OSError(ctypes.get_errno(), f"inotify_add_watch failed for {root}")

    def run(self):
        names = {IN_CREATE: "created", IN_CLOSE_WRITE: "closed", IN_DELETE: "deleted"}
        while not self.stopping:
            ready, _, _ = select.select([self.fd], [], [], 0.1)
            if not ready:
                continue
            try:
                data = os.read(self.fd, 1 << 16)
            except BlockingIOError:
                continue
            now = mono()
            for _wd, mask, name in parse_inotify(data):
                if not name.startswith(self.prefixes):
                    continue
                for bit, label in names.items():
                    if mask & bit:
                        record = {"e": "shm", "t": now, "what": label, "file": name}
                        if bit == IN_CLOSE_WRITE:
                            try:
                                record["size"] = os.stat(os.path.join(self.root, name)).st_size
                            except OSError:
                                pass
                        self.sink(record)

    def stop(self):
        self.stopping = True
        self.join(timeout=2)
        os.close(self.fd)


def stat_fields(path: str):
    """(comm, ppid, utime, stime, rss_pages) from one /proc stat file, or None."""
    try:
        with open(path, encoding="ascii", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None
    try:
        head, tail = text.rsplit(")", 1)
        fields = tail.split()
        return head.split("(", 1)[1], int(fields[1]), int(fields[11]), int(fields[12]), int(fields[21])
    except (IndexError, ValueError):
        return None


def tree(pid: int) -> list[int]:
    """The process and everything it started that is still alive."""
    found, queue = [], [pid]
    while queue:
        current = queue.pop()
        if current in found:
            continue
        found.append(current)
        try:
            tasks = os.listdir(f"/proc/{current}/task")
        except OSError:
            continue
        for task in tasks:
            try:
                with open(f"/proc/{current}/task/{task}/children", encoding="ascii") as handle:
                    queue.extend(int(child) for child in handle.read().split())
            except (OSError, ValueError):
                pass
    return found


def find_player(root: int, command: list[str], wait: float = 10.0):
    """The pid under a pane's first process that is running exactly this command.

    A held pane's own process is a wrapper that outlives the player."""
    wanted = b"\0".join(os.fsencode(part) for part in command) + b"\0"
    deadline = time.monotonic() + wait
    while True:
        for pid in tree(root):
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as handle:
                    if handle.read() == wanted:
                        return pid
            except OSError:
                pass
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.01)


def threads(pid: int) -> dict:
    """{thread name: ticks} for a live process; names shared by several threads add up."""
    out: dict[str, int] = {}
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for task in tasks:
        fields = stat_fields(f"/proc/{pid}/task/{task}/stat")
        if fields:
            out[fields[0]] = out.get(fields[0], 0) + fields[2] + fields[3]
    return out


def written(pid: int):
    """Bytes a process has passed to write calls and bytes sent to storage, or None."""
    try:
        with open(f"/proc/{pid}/io", encoding="ascii") as handle:
            fields = dict(line.split(": ") for line in handle.read().splitlines())
        return [int(fields["wchar"]), int(fields["write_bytes"])]
    except (OSError, KeyError, ValueError):
        return None


class TreeSampler(threading.Thread):
    """CPU and memory of whole process trees, and of some processes' threads."""

    def __init__(self, sink, period: float = 0.25):
        super().__init__(daemon=True)
        self.sink, self.period = sink, period
        self.roots: dict[str, int] = {}
        self.single: dict[str, int] = {}
        self.threaded: set[str] = set()
        self.stopping = False
        self.page_kb = os.sysconf("SC_PAGE_SIZE") // 1024

    def sample(self):
        now = mono()
        for who, root in list(self.roots.items()):
            procs = {}
            for pid in tree(root):
                fields = stat_fields(f"/proc/{pid}/stat")
                if fields:
                    procs[str(pid)] = [fields[0], fields[2] + fields[3], fields[4] * self.page_kb]
            if procs:
                record = {"e": "tree", "t": now, "who": who, "procs": procs}
                if who in self.threaded:
                    record["threads"] = threads(root)
                self.sink(record)
        for who, pid in list(self.single.items()):
            fields = stat_fields(f"/proc/{pid}/stat")
            if fields:
                self.sink({"e": "tree", "t": now, "who": who,
                           "procs": {str(pid): [fields[0], fields[2] + fields[3],
                                                fields[4] * self.page_kb]},
                           "threads": threads(pid), "io": written(pid)})
        self.sink({"e": "load", "t": now, "loadavg": os.getloadavg()})

    def run(self):
        while not self.stopping:
            self.sample()
            time.sleep(self.period)

    def stop(self):
        self.stopping = True
        self.join(timeout=2)


class MpvIpc(threading.Thread):
    """Polls a few of mpv's properties and passes its events on."""

    def __init__(self, path: str, sink, period: float = 0.25):
        super().__init__(daemon=True)
        self.path, self.sink, self.period = path, sink, period
        self.sock = None
        self.buffer = b""
        self.stopping = False
        self.next_id = 1

    def connect(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.path)
            except OSError:
                sock.close()
                time.sleep(0.02)
                continue
            sock.settimeout(1.0)
            self.sock = sock
            return True
        return False

    def _send(self, *command) -> int:
        request = self.next_id
        self.next_id += 1
        self.sock.sendall(json.dumps({"command": list(command), "request_id": request}).encode() + b"\n")
        return request

    def _lines(self, wait: float):
        ready, _, _ = select.select([self.sock], [], [], wait)
        if ready:
            data = self.sock.recv(1 << 16)
            if not data:
                raise ConnectionError("mpv closed its socket")
            self.buffer += data
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            try:
                yield json.loads(line)
            except ValueError:
                continue

    def run(self):
        pending: dict[int, str] = {}
        try:
            for name in ("seeking", "paused-for-cache"):
                self._send("observe_property", 0, name)
            pending[self._send("get_property", "osd-dimensions")] = "osd-dimensions"
            pending[self._send("get_property", "hwdec-current")] = "hwdec-current"
            pending[self._send("get_property", "video-params")] = "video-params"
            last = 0.0
            sample: dict = {}
            while not self.stopping:
                if time.monotonic() - last >= self.period:
                    last = time.monotonic()
                    for name in MPV_PROPERTIES:
                        pending[self._send("get_property", name)] = name
                for message in self._lines(0.05):
                    now = mono()
                    if "event" in message:
                        record = {"e": "mpv-event", "t": now, "event": message["event"]}
                        for key in ("name", "data", "reason"):
                            if key in message:
                                record[key] = message[key]
                        self.sink(record)
                        continue
                    name = pending.pop(message.get("request_id"), None)
                    if name is None:
                        continue
                    if name not in MPV_PROPERTIES:
                        self.sink({"e": "mpv-once", "t": now, "name": name,
                                   "data": message.get("data"), "error": message.get("error")})
                        continue
                    sample[name] = message.get("data")
                    if name == MPV_PROPERTIES[-1]:
                        self.sink({"e": "mpv", "t": now, **sample})
                        sample = {}
        except (OSError, ConnectionError):
            pass

    def stop(self):
        self.stopping = True
        self.join(timeout=2)
        if self.sock is not None:
            self.sock.close()


def flat_centre(image, side: str, tolerance: float = 2.0):
    """(is one flat colour, spread) for the middle of one half of a capture.

    The picture is centred in its pane, so the middle of each half of the
    window is inside it for any ordinary aspect ratio."""
    width, height = image.size
    half = width // 2
    left = (half // 4) if side == "left" else half + half // 4
    box = (left, height * 2 // 5, left + half // 2, height * 3 // 5)
    grey = image.crop(box).convert("L")
    low, high = grey.getextrema()
    histogram = grey.histogram()
    total = sum(histogram)
    mean = sum(i * n for i, n in enumerate(histogram)) / total
    spread = (sum(n * (i - mean) ** 2 for i, n in enumerate(histogram)) / total) ** 0.5
    return spread < tolerance and high - low < 8 * tolerance, round(spread, 3)


class Watcher(threading.Thread):
    """Captures the focused window and notes flat pictures on either side."""

    def __init__(self, sink, directory: Path, display_env: dict, period: float = 0.2):
        super().__init__(daemon=True)
        self.sink, self.directory, self.period = sink, directory, period
        self.env = {**os.environ, **display_env}
        self.stopping = False
        self.kept = 0

    def run(self):
        from PIL import Image
        number = 0
        while not self.stopping:
            started = time.monotonic()
            number += 1
            path = self.directory / f"shot-{number:04d}.png"
            before = mono()
            done = subprocess.run(["scrot", "-u", "-o", str(path)], env=self.env,
                                  capture_output=True, timeout=20)
            after = mono()
            if done.returncode != 0 or not path.is_file():
                self.sink({"e": "shot-error", "t": after,
                           "text": done.stderr.decode(errors="replace")[-200:]})
                time.sleep(self.period)
                continue
            try:
                with Image.open(path) as image:
                    image.load()
                    record = {"e": "shot", "t0": before, "t": after, "n": number,
                              "size": list(image.size)}
                    for side in ("left", "right"):
                        flat, spread = flat_centre(image, side)
                        record[side] = {"flat": flat, "spread": spread}
            except OSError as exc:
                self.sink({"e": "shot-error", "t": after, "text": str(exc)})
                continue
            keep = record["left"]["flat"] or record["right"]["flat"] or number in (5, 25)
            if keep and self.kept < 12:
                self.kept += 1
                record["file"] = path.name
            else:
                path.unlink(missing_ok=True)
            self.sink(record)
            time.sleep(max(0.0, self.period - (time.monotonic() - started)))

    def stop(self):
        self.stopping = True
        self.join(timeout=25)


# ── one run ─────────────────────────────────────────────────────────────────

class Run:
    def __init__(self, *, spec, index, out_dir, kitten, args, clip):
        self.spec, self.index, self.kitten, self.args, self.clip = spec, index, kitten, args, clip
        self.run_id = f"{index:02d}-{spec['phase']}-" + "+".join(spec["arms"]) + f"-left-{spec['left']}"
        self.dir = out_dir / self.run_id
        self.dir.mkdir(parents=True)
        self.short = Path(tempfile.mkdtemp(prefix="kvb"))      # a short path for the socket
        self.lock = threading.Lock()
        self.events = open(self.dir / "events.jsonl", "w", encoding="utf-8")
        self.first_picture: dict[str, int] = {}
        self.interrupted = False

    def note(self, record: dict):
        if record["e"] == "shm" and record["what"] == "deleted":
            for arm, prefix in SHM_PREFIX.items():
                if arm not in self.first_picture and record["file"].startswith(prefix):
                    self.first_picture[arm] = record["t"]
        with self.lock:
            self.events.write(json.dumps(record, separators=(",", ":")) + "\n")

    def command(self, arm: str) -> list[str]:
        a = self.args
        if arm == "mpv":
            return mpv_command(a.mpv, a.video, str(self.short / "mpv.sock"), a.start, a.mpv_option)
        if arm == "neuronte":
            return neuronte_command(a.python, a.neuronte_home, a.video,
                                    str(self.dir / "neuronte-frames.jsonl"),
                                    str(self.dir / "neuronte-stats.json"), a.start,
                                    a.neuronte_option)
        return ["sleep", "100000"]

    def execute(self) -> dict:
        spec, a = self.spec, self.args
        result = {"schema": RUN_SCHEMA, "run": self.run_id, **spec, "status": "failed", "notes": [],
                  "seconds": a.seconds, "start": a.start, "clip": self.clip,
                  "host_before": host_state()}
        stale = {name for name in os.listdir("/dev/shm") if name.startswith(tuple(SHM_PREFIX.values()))}
        watch = ShmWatch(self.note, SHM_PREFIX.values())
        sampler = TreeSampler(self.note)
        sampler.single["terminal"] = a.terminal_pid
        sampler.threaded.add("mpv")
        ipc = watcher = first = None
        previous_focus = None
        panes = {}
        watch.start()
        sampler.start()
        try:
            sides = [spec["left"], next((x for x in (*spec["arms"], "idle") if x != spec["left"]), "idle")]
            if spec["left"] == "idle":
                sides = ["idle", spec["arms"][0]]
            result["sides"] = {"left": sides[0], "right": sides[1]}
            commands = {name: self.command(name) for name in sides}
            result["commands"] = commands
            if spec["phase"] == "watch":
                previous_focus = focused_window(self.kitten.ls())
            time.sleep(a.idle_s)                            # the terminal with neither player, for comparison
            result["idle_until_t"] = mono()
            result["launch_t"] = {}
            result["launch_t"][sides[0]] = mono()
            # --hold keeps a pane (and so the other pane's size) after its player leaves
            first = int(self.kitten.call(
                "launch", "--type=tab", "--keep-focus", "--hold",
                f"--tab-title=video-bench {self.run_id}", f"--cwd={self.dir}",
                *commands[sides[0]]).strip().splitlines()[-1])
            self.kitten.call("goto-layout", "--match", f"window_id:{first}", "splits")
            result["launch_t"][sides[1]] = mono()
            second = int(self.kitten.call(
                "launch", "--type=window", "--keep-focus", "--hold", "--match", f"window_id:{first}",
                "--location=vsplit", f"--cwd={self.dir}", *commands[sides[1]]).strip().splitlines()[-1])
            panes = {sides[0]: first, sides[1]: second}
            result["panes"] = panes
            tree_now = self.kitten.ls()
            geometry = {}
            for name, window in panes.items():
                found = find_window(tree_now, window)
                if found is None:
                    result["notes"].append(f"{name} pane closed at once")
                    continue
                geometry[name] = {"columns": found[2]["columns"], "lines": found[2]["lines"],
                                  "pane_pid": found[2]["pid"]}
                if name in ARMS:
                    player = find_player(found[2]["pid"], commands[name])
                    if player is None:
                        result["notes"].append(f"{name} never started")
                        continue
                    geometry[name]["pid"] = player
                    sampler.roots[name] = player
            result["geometry"] = geometry
            if spec["phase"] == "watch":
                self.kitten.call("focus-tab", "--match", f"window_id:{first}")
                watcher = Watcher(self.note, self.dir, {"DISPLAY": a.display})
                watcher.start()
            if "mpv" in spec["arms"]:
                ipc = MpvIpc(str(self.short / "mpv.sock"), self.note)
                if ipc.connect():
                    ipc.start()
                else:
                    result["notes"].append("no mpv socket")
                    ipc = None
            # Each player gets --seconds from its own first picture, then its quit key.
            deadline = time.monotonic() + a.seconds + 90
            alive = dict(sampler.roots)
            todo = {name: sorted(a.seek) + [(a.seconds, "quit")] for name in alive}
            while alive and time.monotonic() < deadline:
                for name in list(alive):
                    started = self.first_picture.get(name)
                    while started is not None and todo[name] and mono() - started >= todo[name][0][0] * 1e9:
                        _, key = todo[name].pop(0)
                        self.note({"e": "key", "t": mono(), "who": name, "key": key})
                        self.kitten.call("send-text", "--match", f"id:{panes[name]}",
                                         "q" if key == "quit" else SEEK_KEYS[key])
                for name, pid in list(alive.items()):
                    if not os.path.exists(f"/proc/{pid}"):
                        self.note({"e": "exit", "t": mono(), "who": name})
                        del alive[name]
                time.sleep(0.02)
            result["first_picture_t"] = dict(self.first_picture)
            if alive:
                result["notes"].append("timed out: " + ", ".join(alive))
            elif len(self.first_picture) == len(spec["arms"]):
                result["status"] = "ok"
            else:
                result["notes"].append("no picture from: " + ", ".join(
                    name for name in spec["arms"] if name not in self.first_picture))
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            result["notes"].append(f"{type(exc).__name__}: {exc}")
        except (KeyboardInterrupt, SystemExit):
            result["notes"].append("interrupted")
            self.interrupted = True
        finally:
            if watcher is not None:
                watcher.stop()
            if ipc is not None:
                ipc.stop()
            time.sleep(0.3)
            sampler.stop()
            watch.stop()
            if first is not None:
                try:
                    self.kitten.call("close-tab", "--match", f"window_id:{first}")
                except RuntimeError:
                    pass                                    # both players left: the tab is gone
                for window in panes.values():
                    try:
                        self.kitten.call("close-window", "--match", f"id:{window}")
                    except RuntimeError:
                        pass
            if previous_focus is not None:
                try:
                    self.kitten.call("focus-window", "--match", f"id:{previous_focus}")
                except RuntimeError as exc:
                    result["notes"].append(f"focus not restored: {exc}")
            result["host_after"] = host_state()
            self.events.close()
            shutil.rmtree(self.short, ignore_errors=True)
            leftovers = [name for name in os.listdir("/dev/shm")
                         if name.startswith(tuple(SHM_PREFIX.values())) and name not in stale]
            if leftovers:
                result["notes"].append(f"{len(leftovers)} pictures left in /dev/shm")
        return result


# ── campaign ────────────────────────────────────────────────────────────────

def seek_spec(text: str):
    at, _, key = text.partition(":")
    if key not in SEEK_KEYS:
        raise argparse.ArgumentTypeError(f"seek key must be one of {', '.join(SEEK_KEYS)}")
    return float(at), key


def campaign(args) -> int:
    out = Path(args.out).expanduser().resolve()
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    clip = probe(args.video)
    forward = sum({"right": 5, "up": 60}.get(key, 0) for _, key in args.seek)
    if args.start + args.seconds + forward + 5 > clip["duration"]:
        raise SystemExit("--start, --seconds and the seeks run past the end of the file")
    for at, _ in args.seek:
        if at >= args.seconds:
            raise SystemExit("a --seek time is after the end of the run")
    kitten = Kitten(args.kitten, args.password_file, dict(os.environ))
    sequence = [run for run in plan(args.pairs, args.watch) if not args.only or run["phase"] in args.only]
    if not sequence:
        raise SystemExit("--only leaves nothing to run")
    existing = sorted(p.name for p in runs_dir.iterdir())
    first_index = 1 + max((int(name.split("-", 1)[0]) for name in existing), default=0)
    record = {"video": {"name": Path(args.video).name, "sha256": file_sha256(Path(args.video)), **clip},
              "seconds": args.seconds, "start": args.start, "seek": args.seek,
              "mpv": {"path": args.mpv, "sha256": file_sha256(Path(args.mpv)),
                      "version": subprocess.run([args.mpv, "--version"], capture_output=True,
                                                text=True).stdout.splitlines()[:1]},
              "mpv_source": identity(args.mpv_source) if args.mpv_source else None,
              "neuronte_home": identity(args.neuronte_home),
              "neuronte_library_sha256": file_sha256(Path(args.neuronte_home) / "build" / "libneuronte.so"),
              "neuronte_options": args.neuronte_option, "mpv_options": args.mpv_option,
              "harness": identity(str(Path(__file__).resolve().parents[2])),
              "sequence": sequence, "first_index": first_index, "pairs": args.pairs,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / f"campaign-{first_index:02d}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    def log(text):
        print(time.strftime("%H:%M:%S"), text, flush=True)

    failed = 0
    for offset, spec in enumerate(sequence):
        quiet = wait_for_quiet(args.load_limit, args.load_resume, args.load_wait_s, log)
        run = Run(spec=spec, index=first_index + offset, out_dir=runs_dir, kitten=kitten,
                  args=args, clip=clip)
        log(f"run {run.run_id} ({'warm-up' if spec['warmup'] else 'measured'}) starting")
        result = run.execute()
        result["quiet_check"] = quiet
        (run.dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        failed += result["status"] != "ok"
        log(f"run {run.run_id}: {result['status']} {'; '.join(result['notes'])}")
        if run.interrupted:
            return 130
        time.sleep(args.gap_s)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="kilix-mpv and the Neuronte player, side by side")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=20.0, help="how much of the file each run plays")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--pairs", type=int, default=3, help="measured runs of each kind, after one warm-up")
    parser.add_argument("--watch", type=int, default=0,
                        help="extra together runs brought to the front and captured; they take "
                             "the screen for their length")
    parser.add_argument("--only", action="append", choices=["solo", "together", "watch"],
                        help="run only this kind; repeat for more than one")
    parser.add_argument("--seek", type=seek_spec, action="append", default=[],
                        help="SECONDS:KEY, such as 8:right; written to each player's pane")
    parser.add_argument("--mpv", default="/usr/bin/kilix-mpv")
    parser.add_argument("--mpv-source", help="the kilix-mpv checkout, recorded for the report")
    parser.add_argument("--mpv-option", action="append", default=[],
                        help="one extra argument for kilix-mpv, as --mpv-option=--sws-fast=yes")
    parser.add_argument("--neuronte-home", required=True)
    parser.add_argument("--neuronte-option", action="append", default=[],
                        help="one extra argument for the Neuronte player; repeat as needed")
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--kitten", required=True, help="the running terminal's own kitten")
    parser.add_argument("--password-file", default=os.environ.get("KILIX_RC_PASSWORD_FILE", ""))
    parser.add_argument("--terminal-pid", type=int, default=int(os.environ.get("KITTY_PID", "0")))
    parser.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    parser.add_argument("--load-limit", type=float, default=10.0)
    parser.add_argument("--load-resume", type=float, default=8.0)
    parser.add_argument("--load-wait-s", type=float, default=300.0)
    parser.add_argument("--gap-s", type=float, default=2.0)
    parser.add_argument("--idle-s", type=float, default=2.0,
                        help="how long the terminal is sampled before each run's players start")
    args = parser.parse_args(argv)
    if args.pairs < 0 or args.watch < 0:
        parser.error("--pairs and --watch must not be negative")
    if not args.terminal_pid:
        parser.error("--terminal-pid is needed outside a Kilix pane")
    if not (Path(args.neuronte_home) / "tools" / "neuronte-play.py").is_file():
        parser.error("--neuronte-home has no tools/neuronte-play.py")
    for tool in ("ffprobe", args.mpv):
        if not shutil.which(tool):
            parser.error(f"{tool} was not found")
    args.video = str(Path(args.video).expanduser().resolve())
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # leave through the run's cleanup
    return campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
