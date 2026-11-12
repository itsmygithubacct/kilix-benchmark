"""Run Kilix 95 checkouts in Kilix tabs and drive one scripted session.

Each run opens its own tab with remote control (the launch keeps focus where
it was), starts ``tab_probe.py`` there on one checkout, waits for the first
desktop frame, writes the suite's keys and pointer reports to the tab through
one long-lived ``kitten @ send-text --stdin``, and closes the desktop with
its own quit key. Meanwhile the driver records:

- when each scripted write was made;
- when the terminal created, finished, and deleted every frame file (the
  terminal deletes a frame file once it has mapped it), through inotify;
- the desktop process's CPU ticks and resident memory every 200 ms, and the
  load average every second;
- host load, battery and GPU state before and after the run.

Runs of different arms are interleaved and the leading arm alternates
(AB, BA, AB, ...). The first run of each arm is a warm-up that also takes the
screenshot; ``tab_stats`` leaves warm-ups out of the comparison.

Nothing here touches the live Kilix 95 data: every run gets a fresh private
storage tree. The only persistent state is one bytecode cache per arm.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import select
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

from kilix_benchmark import tab_keys

SUITE_SCHEMA = "kilix.benchmark.tab/v1"
RUN_SCHEMA = "kilix.benchmark.tab-run/v1"
PROBE = Path(__file__).resolve().with_name("tab_probe.py")
mono = time.monotonic_ns


# ── suite ───────────────────────────────────────────────────────────────────

def load_suite(path: Path) -> dict:
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    if suite.get("schema") != SUITE_SCHEMA:
        raise ValueError("suite file has the wrong schema")
    phases = suite.get("phases") or []
    if not phases:
        raise ValueError("suite has no phases")
    seen = set()
    for phase in phases:
        if phase["id"] in seen:
            raise ValueError(f"phase {phase['id']} appears twice")
        seen.add(phase["id"])
        for step in phase["steps"]:
            kinds = {"keys", "text", "burst", "hover", "drag", "wait_ms"} & set(step)
            if len(kinds) != 1:
                raise ValueError(f"phase {phase['id']}: a step needs exactly one action")
            for name in step.get("keys", ()):
                key, mods = tab_keys.parse_key(name)
                tab_keys.named_key(key, mods)
            for ch in step.get("text", "") + step.get("burst", ""):
                tab_keys.type_char(ch)
    if phases[-1]["id"] != "quit":
        raise ValueError("the last phase must be quit")
    return suite


def compile_phase(phase: dict, desk=None, windows=None) -> list[dict]:
    """Scheduled writes for one phase: [{after_ms, data, label, keys}].

    ``after_ms`` is the wait before the write, measured from the previous
    write. ``desk`` is (width, height); ``windows`` the probe's window list,
    needed only by hover and drag steps.
    """
    out = []
    for step in phase["steps"]:
        if "wait_ms" in step:
            out.append({"after_ms": step["wait_ms"], "data": b"", "label": "wait", "keys": 0})
        elif "keys" in step:
            interval = step.get("interval_ms", 0)
            for _ in range(step.get("repeat", 1)):
                for name in step["keys"]:
                    key, mods = tab_keys.parse_key(name)
                    out.append({"after_ms": interval, "label": name, "keys": 1,
                                "data": tab_keys.press_named(key, mods).encode()})
        elif "text" in step:
            interval = step.get("interval_ms", 0)
            for ch in step["text"]:
                out.append({"after_ms": interval, "label": ch, "keys": 1,
                            "data": tab_keys.type_char(ch).encode()})
        elif "burst" in step:
            text, chunk = step["burst"], max(1, step.get("chunk", 1))
            for start in range(0, len(text), chunk):
                piece = text[start:start + chunk]
                out.append({"after_ms": 0, "label": "burst", "keys": len(piece),
                            "data": "".join(tab_keys.type_char(c) for c in piece).encode()})
        elif "hover" in step:
            spec = step["hover"]
            if desk is None:
                raise ValueError("hover needs the desk size")
            width, height = desk
            (fx0, fy0), (fx1, fy1) = spec["from"], spec["to"]
            count = spec["count"]
            for i in range(count):
                f = i / max(1, count - 1)
                x = round((fx0 + (fx1 - fx0) * f) * (width - 1))
                y = round((fy0 + (fy1 - fy0) * f) * (height - 1))
                out.append({"after_ms": spec["interval_ms"], "label": "hover", "keys": 0,
                            "data": tab_keys.mouse(x, y, motion=True).encode()})
        elif "drag" in step:
            spec = step["drag"]
            win = _find_window(windows or [], spec["window"])
            if win is None:
                raise LookupError(f"no {spec['window']} window to drag")
            x = win[2] + spec["grip"][0]
            y = win[3] + spec["grip"][1]
            interval = spec["interval_ms"]
            out.append({"after_ms": interval, "label": "drag-press", "keys": 0,
                        "data": tab_keys.mouse(x, y, press=True).encode()})
            for _ in range(spec["count"]):
                x += spec["step"][0]
                y += spec["step"][1]
                out.append({"after_ms": interval, "label": "drag", "keys": 0,
                            "data": tab_keys.mouse(x, y, motion=True, held=True).encode()})
            out.append({"after_ms": interval, "label": "drag-release", "keys": 0,
                        "data": tab_keys.mouse(x, y, press=False).encode()})
    return out


def _find_window(windows, class_name):
    for win in reversed(windows):
        if win[0] == class_name and not win[6]:
            return win
    return None


# ── terminal remote control ────────────────────────────────────────────────

class Kitten:
    def __init__(self, kitten: str, password_file: str, env: dict | None = None):
        self.kitten = kitten
        self.password_file = password_file
        self.env = env

    def argv(self, *args: str) -> list[str]:
        return [self.kitten, "@", "--password-file", self.password_file, *args]

    def call(self, *args: str, timeout: float = 30) -> str:
        done = subprocess.run(self.argv(*args), capture_output=True, text=True,
                              timeout=timeout, env=self.env, stdin=subprocess.DEVNULL)
        if done.returncode != 0:
            raise RuntimeError(f"kitten {args[0]} failed: {(done.stderr or done.stdout).strip()[-300:]}")
        return done.stdout

    def ls(self) -> list:
        return json.loads(self.call("ls"))

    def launch_tab(self, title: str, cwd: str, command: list[str], run_id: str) -> int:
        out = self.call("launch", "--type=tab", "--keep-focus", f"--tab-title={title}",
                        f"--cwd={cwd}", f"--env=K95BENCH_RUN={run_id}", *command)
        return int(out.strip().splitlines()[-1])

    def streamer(self, window_id: int) -> subprocess.Popen:
        return subprocess.Popen(
            self.argv("send-text", "--match", f"id:{window_id}", "--stdin"),
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            bufsize=0, env=self.env)


def find_window(tree: list, window_id: int):
    for os_window in tree:
        for tab in os_window.get("tabs", ()):
            for win in tab.get("windows", ()):
                if win.get("id") == window_id:
                    return os_window, tab, win
    return None


def focused_window(tree: list):
    for os_window in tree:
        if not os_window.get("is_focused"):
            continue
        for tab in os_window.get("tabs", ()):
            if not tab.get("is_focused"):
                continue
            for win in tab.get("windows", ()):
                if win.get("is_focused"):
                    return win.get("id")
    return None


# ── observers ───────────────────────────────────────────────────────────────

IN_OPEN, IN_CLOSE_WRITE, IN_CLOSE_NOWRITE = 0x20, 0x08, 0x10
IN_CREATE, IN_DELETE, IN_ISDIR = 0x100, 0x200, 0x40000000
IN_NONBLOCK, IN_CLOEXEC = 0o4000, 0o2000000
_EVENT = struct.Struct("iIII")


def parse_inotify(data: bytes):
    """[(wd, mask, name)] from one read of an inotify descriptor."""
    out, offset = [], 0
    while offset + _EVENT.size <= len(data):
        wd, mask, _cookie, length = _EVENT.unpack_from(data, offset)
        offset += _EVENT.size
        name = data[offset:offset + length].split(b"\0", 1)[0].decode("utf-8", "replace")
        offset += length
        out.append((wd, mask, name))
    return out


class FrameWatch(threading.Thread):
    """Times frame files: app open/close-write, terminal open, delete."""

    def __init__(self, root: Path, sink):
        super().__init__(daemon=True)
        self.root = root
        self.sink = sink
        self.stopping = False
        libc = ctypes.CDLL(None, use_errno=True)
        self._add = libc.inotify_add_watch
        self._add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self.fd = libc.inotify_init1(IN_NONBLOCK | IN_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1 failed")
        self.root_wd = self._watch(root, IN_CREATE)
        self.dirs = {}

    def _watch(self, path: Path, mask: int) -> int:
        wd = self._add(self.fd, os.fsencode(path), mask)
        if wd < 0:
            raise OSError(ctypes.get_errno(), f"inotify_add_watch failed for {path}")
        return wd

    def run(self):
        mask = IN_OPEN | IN_CLOSE_WRITE | IN_CLOSE_NOWRITE | IN_DELETE
        names = {IN_OPEN: "open", IN_CLOSE_WRITE: "written", IN_CLOSE_NOWRITE: "closed",
                 IN_DELETE: "deleted"}
        while not self.stopping:
            ready, _, _ = select.select([self.fd], [], [], 0.1)
            if not ready:
                continue
            try:
                data = os.read(self.fd, 1 << 16)
            except BlockingIOError:
                continue
            now = mono()
            for wd, event_mask, name in parse_inotify(data):
                if wd == self.root_wd and event_mask & IN_CREATE and event_mask & IN_ISDIR:
                    try:
                        self.dirs[self._watch(self.root / name, mask)] = name
                    except OSError as exc:
                        self.sink({"e": "watch-error", "t": now, "text": str(exc)})
                    continue
                for bit, label in names.items():
                    if event_mask & bit and not event_mask & IN_ISDIR:
                        self.sink({"e": "file", "t": now, "what": label, "file": name})

    def stop(self):
        self.stopping = True
        self.join(timeout=2)
        os.close(self.fd)


def read_proc(pid: int):
    """(utime_ticks, stime_ticks, rss_kb) of a live process, or None."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        with open(f"/proc/{pid}/statm", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return int(fields[11]), int(fields[12]), pages * os.sysconf("SC_PAGE_SIZE") // 1024


class ProcSampler(threading.Thread):
    def __init__(self, sink, period: float = 0.2):
        super().__init__(daemon=True)
        self.sink = sink
        self.period = period
        self.pids: dict[str, int] = {}
        self.stopping = False

    def run(self):
        last_load = 0.0
        while not self.stopping:
            now = mono()
            for label, pid in list(self.pids.items()):
                sample = read_proc(pid)
                if sample is not None:
                    self.sink({"e": "proc", "t": now, "who": label, "utime": sample[0],
                               "stime": sample[1], "rss_kb": sample[2]})
            if time.monotonic() - last_load >= 1.0:
                last_load = time.monotonic()
                self.sink({"e": "load", "t": now, "loadavg": os.getloadavg()})
            time.sleep(self.period)

    def stop(self):
        self.stopping = True
        self.join(timeout=2)


def host_state() -> dict:
    state = {"t": mono(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "loadavg": os.getloadavg()}
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,enforced.power.limit,power.draw,pstate,"
             "clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        state["gpu"] = gpu.stdout.strip() if gpu.returncode == 0 else f"rc={gpu.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        state["gpu"] = f"unavailable: {exc}"
    drm = {}
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]")):
        entry = {}
        for name in ("gt_act_freq_mhz", "gt_cur_freq_mhz", "gt_max_freq_mhz"):
            try:
                entry[name] = (card / name).read_text().strip()
            except OSError:
                pass
        try:
            entry["vendor"] = (card / "device" / "vendor").read_text().strip()
        except OSError:
            pass
        if entry:
            drm[card.name] = entry
    state["drm"] = drm
    power = {}
    base = Path("/sys/class/power_supply")
    for supply in sorted(base.iterdir()) if base.is_dir() else ():
        entry = {}
        for name in ("online", "status", "capacity", "energy_now", "power_now"):
            try:
                entry[name] = (supply / name).read_text().strip()
            except OSError:
                pass
        if entry:
            power[supply.name] = entry
    state["power"] = power
    try:
        top = subprocess.run(["ps", "-eo", "pcpu,comm", "--sort=-pcpu", "--no-headers"],
                             capture_output=True, text=True, timeout=10)
        state["top"] = [line.split(None, 1) for line in top.stdout.splitlines()[:8]]
    except (OSError, subprocess.TimeoutExpired):
        state["top"] = []
    return state


class Events:
    """Tail the probe's JSON-lines file."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.partial = b""
        self.records: list[dict] = []

    def poll(self) -> list[dict]:
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self.offset)
                data = handle.read()
        except FileNotFoundError:
            return []
        self.offset += len(data)
        data = self.partial + data
        lines = data.split(b"\n")
        self.partial = lines.pop()
        fresh = [json.loads(line) for line in lines if line.strip()]
        self.records.extend(fresh)
        return fresh

    def find(self, kind: str):
        return [record for record in self.records if record.get("e") == kind]


# ── one run ─────────────────────────────────────────────────────────────────

def sandbox_env(root: Path, cache: Path, kilix_home: str, state_library: str,
                extra: dict) -> dict:
    gpu = root / "gpu"
    kilix = gpu / "kilix"
    k95 = gpu / "kilix-95"
    env = {
        "HOME": str(root / "home"),
        "GPU_TERMINAL_HOME": str(gpu),
        "GPU_TERMINAL_SETTINGS_FILE": str(gpu / "settings.conf"),
        "GPU_TERMINAL_SOURCE_HOME": str(Path(kilix_home).parent),
        "KILIX_HOME": kilix_home,
        "KILIX_STATE_LIBRARY": state_library,
        "KILIX_STORAGE_HOME": str(kilix),
        "KILIX_CONFIG_HOME": str(kilix / "config"),
        "KILIX_STATE_DIRECTORY": str(kilix / "state"),
        "KILIX_CACHE_HOME": str(kilix / "cache"),
        "KILIX_DATA_HOME": str(kilix / "data"),
        "KILIX_SESSION_HOME": str(kilix / "session"),
        "KILIX_BUILD_DIRECTORY": str(kilix / "build"),
        "KILIX95_STORAGE_HOME": str(k95),
        "KILIX95_CONFIG_HOME": str(k95 / "config"),
        "KILIX95_STATE_HOME": str(k95 / "state"),
        "KILIX95_CACHE_HOME": str(cache),
        "KILIX95_DATA_HOME": str(k95 / "data"),
        "KILIX95_SESSION_HOME": str(k95 / "session"),
        "KILIX_DESKTOP_DIR": str(root / "desktop"),
        "KILIX_RECYCLE_DIR": str(root / "recycle"),
        "XDG_CONFIG_HOME": str(root / "xdg" / "config"),
        "XDG_STATE_HOME": str(root / "xdg" / "state"),
        "XDG_CACHE_HOME": str(root / "xdg" / "cache"),
        "XDG_DATA_HOME": str(root / "xdg" / "data"),
        "XDG_DATA_DIRS": str(root / "xdg" / "system"),
    }
    env.update(extra)
    return env


def prepare_sandbox(root: Path, suite: dict) -> Path:
    if root.exists():
        shutil.rmtree(root)
    for part in ("home", "desktop", "recycle", "gpu/kilix-95/session/frames"):
        (root / part).mkdir(parents=True, mode=0o700, exist_ok=True)
    for path in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        os.chmod(path, 0o700)
    for name in suite.get("seed_files", ()):
        (root / "desktop" / name).write_text(name + "\n", encoding="utf-8")
    return root / "gpu" / "kilix-95" / "session" / "frames"


class Run:
    def __init__(self, *, suite, arm, desktop, index, out_dir, scratch, kitten, kilix_home,
                 state_library, neuronte_home, python, shot_after=None, display_env=None):
        self.suite = suite
        self.arm = arm
        self.desktop = desktop
        self.index = index
        self.run_id = f"{index:02d}-{arm}"
        self.dir = out_dir / self.run_id
        self.scratch = scratch
        self.kitten = kitten
        self.kilix_home = kilix_home
        self.state_library = state_library
        self.neuronte_home = neuronte_home
        self.python = python
        self.shot_after = shot_after
        self.display_env = display_env or {}
        self.lock = threading.Lock()
        self.driver_log = None
        self.window_id = None
        self.streamer = None

    def note(self, record: dict):
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        with self.lock:
            self.driver_log.write(line)
            self.driver_log.flush()

    def execute(self) -> dict:
        self.dir.mkdir(parents=True, exist_ok=False)
        root = self.scratch / f"run-{self.run_id}"
        frames = prepare_sandbox(root, self.suite)
        cache = self.scratch / f"cache-{self.arm}"
        cache.mkdir(mode=0o700, exist_ok=True)
        events_path = root / "events.jsonl"
        extra = dict(self.suite.get("env", {}))
        extra["NEURONTE_HOME"] = self.neuronte_home
        config = {
            "arm": self.arm, "run": self.run_id, "desktop": self.desktop,
            "events": str(events_path),
            "env": sandbox_env(root, cache, self.kilix_home, self.state_library, extra),
            "env_unset": list(self.suite.get("env_unset", ())),
            "args": [],
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        shutil.copy(config_path, self.dir / "config.json")
        self.driver_log = open(self.dir / "driver.jsonl", "w", encoding="utf-8")
        result = {"schema": RUN_SCHEMA, "run": self.run_id, "arm": self.arm, "index": self.index,
                  "status": "failed", "notes": []}
        result["host_before"] = host_state()
        watch = FrameWatch(frames, self.note)
        sampler = ProcSampler(self.note)
        watch.start()
        sampler.start()
        events = Events(events_path)
        try:
            result.update(self._drive(events, sampler, config_path))
        except Exception as exc:  # the run is recorded as failed, with why
            result["notes"].append(f"{type(exc).__name__}: {exc}")
            self.note({"e": "error", "t": mono(), "text": str(exc)})
        finally:
            if self.streamer is not None and self.streamer.poll() is None:
                self.streamer.kill()
            self._close_window(result)
            time.sleep(0.5)
            sampler.stop()
            watch.stop()
            events.poll()
            result["host_after"] = host_state()
            self.driver_log.close()
            if events_path.exists():
                shutil.copy(events_path, self.dir / "events.jsonl")
            (self.dir / "result.json").write_text(json.dumps(result, indent=2, default=str),
                                                  encoding="utf-8")
            shutil.rmtree(root, ignore_errors=True)
        return result

    def _wait(self, events: Events, predicate, timeout_ms: int, what: str):
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            events.poll()
            found = predicate(events)
            if found:
                return found
            time.sleep(0.02)
        raise TimeoutError(f"timed out waiting for {what}")

    def _drive(self, events: Events, sampler: ProcSampler, config_path: Path) -> dict:
        command = [self.python, "-I", str(PROBE), str(config_path)]
        title = f"k95bench {self.run_id}"
        launched = mono()
        self.note({"e": "launch", "t": launched})
        self.window_id = self.kitten.launch_tab(title, str(config_path.parent), command, self.run_id)
        self.note({"e": "launched", "t": mono(), "window": self.window_id})
        start = self._wait(events, lambda ev: ev.find("start"), 30000, "probe start")[0]
        sampler.pids["desktop"] = start["pid"]

        def first_desktop_frame(ev):
            screens = [s for s in ev.find("screen") if s["name"] == "startup"]
            if not screens:
                return None
            return [f for f in ev.find("frame") if f["t0"] >= screens[0]["t1"]]

        first = self._wait(events, first_desktop_frame, self.suite.get("ready_timeout_ms", 60000),
                           "first desktop frame")[0]
        self.note({"e": "ready", "t": mono(), "first_frame_t1": first["t1"]})
        desk = events.find("desk")[0]
        settle_until = first["t1"] + self.suite.get("settle_ms", 8000) * 1_000_000
        while mono() < settle_until:
            time.sleep(0.05)
        events.poll()

        streamer = self.streamer = self.kitten.streamer(self.window_id)
        out = {"launch_t": launched, "window": self.window_id, "desk": [desk["w"], desk["h"]],
               "phases": []}
        for phase in self.suite["phases"]:
            events.poll()
            wins = [f["wins"] for f in events.find("frame") if "wins" in f]
            schedule = compile_phase(phase, (desk["w"], desk["h"]), wins[-1] if wins else [])
            began = mono()
            self.note({"e": "phase", "t": began, "phase": phase["id"], "what": "begin"})
            self._play(schedule, streamer, phase["id"])
            ended = mono()
            self.note({"e": "phase", "t": ended, "phase": phase["id"], "what": "end"})
            out["phases"].append({"id": phase["id"], "t0": began, "t1": ended,
                                  "writes": sum(1 for s in schedule if s["data"]),
                                  "keys": sum(s["keys"] for s in schedule)})
            if phase["id"] == self.shot_after:
                try:
                    out["screenshot"] = self._screenshot()
                except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
                    out["screenshot"] = {"error": str(exc)[-300:]}
                    self.note({"e": "screenshot-error", "t": mono(), "text": str(exc)})
        streamer.stdin.close()
        exit_record = self._wait(events, lambda ev: ev.find("exit"),
                                 self.suite.get("exit_timeout_ms", 20000), "desktop exit")[0]
        try:
            streamer.wait(timeout=5)
        except subprocess.TimeoutExpired:
            streamer.kill()
        stderr = streamer.stderr.read().decode("utf-8", "replace").strip()
        if stderr:
            out.setdefault("streamer_stderr", stderr[-500:])
        out["exit_status"] = exit_record.get("status")
        out["status"] = "complete" if exit_record.get("status") == 0 else "failed"
        return out

    def _play(self, schedule: list[dict], streamer: subprocess.Popen, phase: str):
        due = mono()
        fd = streamer.stdin.fileno()
        for item in schedule:
            due += item["after_ms"] * 1_000_000
            while True:
                remaining = due - mono()
                if remaining <= 0:
                    break
                time.sleep(min(remaining / 1e9, 0.005) if remaining > 2_000_000 else 0)
            if not item["data"]:
                continue
            started = mono()
            view = memoryview(item["data"])
            while view:
                written = os.write(fd, view)
                view = view[written:]
            self.note({"e": "send", "t": started, "t1": mono(), "phase": phase,
                       "label": item["label"], "keys": item["keys"], "bytes": len(item["data"])})

    def _screenshot(self) -> dict:
        tree = self.kitten.ls()
        previous = focused_window(tree)
        shot = self.dir / "screenshot.png"
        record = {"previous_focus": previous, "path": shot.name}
        self.kitten.call("focus-window", "--match", f"id:{self.window_id}")
        record["focused_t"] = mono()
        time.sleep(1.0)
        env = dict(os.environ)
        env.update(self.display_env)
        done = subprocess.run(["scrot", "-o", str(shot)], env=env, capture_output=True,
                              text=True, timeout=30)
        record["scrot_rc"] = done.returncode
        record["shot_t"] = mono()
        if previous is not None:
            try:
                self.kitten.call("focus-window", "--match", f"id:{previous}")
                record["restored"] = True
            except RuntimeError as exc:
                record["restored"] = f"failed: {exc}"
        self.note({"e": "screenshot", **record})
        return record

    def _close_window(self, result: dict):
        if self.window_id is None:
            return
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                tree = self.kitten.ls()
            except (RuntimeError, ValueError, subprocess.TimeoutExpired):
                break
            if find_window(tree, self.window_id) is None:
                result["window_closed"] = "by exit"
                return
            time.sleep(0.25)
        try:
            self.kitten.call("close-window", "--match", f"id:{self.window_id}")
            result["window_closed"] = "closed by driver"
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            result["window_closed"] = f"close failed: {exc}"


# ── campaign ────────────────────────────────────────────────────────────────

def identity(path: str) -> dict:
    """Commit and cleanliness of a checkout, for the campaign record."""
    def git(*args):
        done = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else None
    return {"path": str(path), "commit": git("rev-parse", "HEAD"),
            "dirty_paths": len((git("status", "--porcelain") or "").splitlines())}


def file_sha256(path: Path):
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def order(arms: list[str], pairs: int) -> list[str]:
    """Warm-up per arm, then pairs whose leading arm alternates."""
    if len(arms) != 2:
        raise ValueError("a campaign compares exactly two arms")
    a, b = arms
    sequence = [a, b]
    for pair in range(pairs):
        sequence += [a, b] if pair % 2 == 0 else [b, a]
    return sequence


def wait_for_quiet(limit: float, resume: float, longest_s: float, log) -> dict:
    load = os.getloadavg()[0]
    record = {"load_at_check": load, "waited_s": 0.0}
    if load <= limit:
        return record
    started = time.monotonic()
    while time.monotonic() - started < longest_s:
        time.sleep(10)
        load = os.getloadavg()[0]
        if load <= resume:
            break
    record["waited_s"] = round(time.monotonic() - started, 1)
    record["load_after_wait"] = load
    log(f"load {record['load_at_check']:.2f} > {limit}: waited {record['waited_s']} s, now {load:.2f}")
    return record


def campaign(args) -> int:
    suite = load_suite(args.suite)
    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not name or not path or not (Path(path) / "main.py").is_file():
            raise SystemExit(f"--arm needs NAME=KILIX95_CHECKOUT with main.py: {spec}")
        arms[name] = str(Path(path).resolve())
    out = Path(args.out).expanduser().resolve()
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch).expanduser().resolve()
    scratch.mkdir(parents=True, mode=0o700, exist_ok=True)
    kitten_env = dict(os.environ)
    kitten = Kitten(args.kitten, args.password_file, kitten_env)
    sequence = order(list(arms), args.pairs)
    existing = sorted(p.name for p in runs_dir.iterdir())
    first_index = 1 + max((int(name.split("-", 1)[0]) for name in existing), default=0)
    plan = {"suite": suite["id"], "arms": {name: identity(path) for name, path in arms.items()},
            "neuronte_home": identity(args.neuronte_home),
            "neuronte_library_sha256": file_sha256(Path(args.neuronte_home) / "build" / "libneuronte.so"),
            "kilix_home": identity(args.kilix_home),
            "harness": identity(str(Path(__file__).resolve().parents[2])),
            "sequence": sequence, "first_index": first_index, "pairs": args.pairs,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / f"campaign-{first_index:02d}.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    def log(text):
        print(time.strftime("%H:%M:%S"), text, flush=True)

    warmed = set()
    for offset, arm in enumerate(sequence):
        index = first_index + offset
        quiet = wait_for_quiet(args.load_limit, args.load_resume, args.load_wait_s, log)
        warmup = arm not in warmed
        warmed.add(arm)
        run = Run(suite=suite, arm=arm, desktop=arms[arm], index=index, out_dir=runs_dir,
                  scratch=scratch, kitten=kitten, kilix_home=args.kilix_home,
                  state_library=args.state_library, neuronte_home=args.neuronte_home,
                  python=args.python, shot_after=args.shot_after if warmup else None,
                  display_env={"DISPLAY": args.display})
        log(f"run {run.run_id} ({'warm-up' if warmup else 'measured'}) starting")
        result = run.execute()
        result["warmup"] = warmup
        result["quiet_check"] = quiet
        (run.dir / "result.json").write_text(json.dumps(result, indent=2, default=str),
                                             encoding="utf-8")
        log(f"run {run.run_id}: {result['status']} {'; '.join(result['notes'])}")
        time.sleep(args.gap_s)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kilix 95 in a Kilix tab: interleaved runs")
    parser.add_argument("--suite", type=Path,
                        default=Path(__file__).resolve().parent / "suites" / "kilix95_tab.json")
    parser.add_argument("--arm", action="append", required=True,
                        help="NAME=PATH to a Kilix 95 checkout; give two")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scratch", required=True,
                        help="private directory for run sandboxes; must be a place the "
                             "terminal deletes frame files from (/tmp or the session home)")
    parser.add_argument("--kilix-home", required=True)
    parser.add_argument("--state-library", required=True)
    parser.add_argument("--neuronte-home", required=True)
    parser.add_argument("--kitten", required=True)
    parser.add_argument("--password-file", default=os.environ.get("KILIX_RC_PASSWORD_FILE", ""))
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    parser.add_argument("--shot-after", default="type-paced",
                        help="phase after which each arm's warm-up run takes a screenshot")
    parser.add_argument("--load-limit", type=float, default=10.0)
    parser.add_argument("--load-resume", type=float, default=8.0)
    parser.add_argument("--load-wait-s", type=float, default=300.0)
    parser.add_argument("--gap-s", type=float, default=3.0)
    args = parser.parse_args(argv)
    if len(args.arm) != 2:
        parser.error("give exactly two --arm values")
    if args.pairs < 0:
        parser.error("--pairs must not be negative (0 runs the warm-ups only)")
    return campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
