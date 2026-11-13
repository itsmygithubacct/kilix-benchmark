"""Equal-resolution CPU/video measurements through private, acknowledging PTYs.

The consumer checks actual RGB geometry and samples reconstructed frames.
Throughput includes that consumer; these are not physical display FPS.
FFmpeg has no presenter. ffplay uses SDL's offscreen software renderer and
reports filtered frames, so its results are labelled separately.
"""
from __future__ import annotations

import argparse
import base64
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pty
import re
import resource
import select
import signal
import statistics
import struct
import subprocess
import sys
import termios
import time
import zlib

from kilix_benchmark.tab_bench import file_sha256, host_state, identity, mono
from kilix_benchmark.tab_stats import compare
from kilix_benchmark.video_bench import TreeSampler, probe
from kilix_benchmark.video_stats import cadence, cpu

FLAGS = "lanczos+accurate_rnd+full_chroma_int+full_chroma_inp"


class Frames:
    def __init__(self, size, directory, samples=()):
        self.size = tuple(size)
        self.directory = Path(directory)
        self.samples = set(samples)
        self.times, self.hashes = [], []

    def frame(self, raw, size):
        if tuple(size) != self.size or len(raw) != math.prod(self.size) * 3:
            raise ValueError(f"unexpected RGB frame: {size}, {len(raw)} bytes; wanted {self.size}")
        index = len(self.times)
        self.times.append(mono())
        self.hashes.append(hashlib.sha256(raw).hexdigest())
        if index in self.samples:
            (self.directory / f"frame-{index:06d}.rgb").write_bytes(raw)


class Graphics(Frames):
    """A terminal-protocol consumer, including the older mpv packet framing."""
    def __init__(self, size, directory, samples=(), shm_root="/dev/shm", prefixes=()):
        super().__init__(size, directory, samples)
        self.shm_root, self.prefixes = Path(shm_root), tuple(prefixes)
        self.pending = bytearray()
        self.payload = bytearray()
        self.header = None
        self.trailing_nuls = 0
        self.global_deletes = 0

    def feed(self, chunk):
        self.pending.extend(chunk)
        while True:
            start = self.pending.find(b"\x1b_G")
            if start < 0:
                del self.pending[:-2]
                return
            end = self.pending.find(b"\x1b\\", start + 3)
            if end < 0:
                del self.pending[:start]
                return
            control, sep, data = bytes(self.pending[start + 3:end]).partition(b";")
            del self.pending[:end + 2]
            if not sep:
                raise ValueError("graphics packet has no payload separator")
            fields = dict(x.split(b"=", 1) for x in control.split(b",") if b"=" in x)
            if fields.get(b"a") == b"d":
                self.global_deletes += b"i" not in fields
                continue
            if fields.get(b"a") in (b"T", b"t"):
                if self.header is not None:
                    raise ValueError("new frame before prior frame completed")
                self.header = fields
            if self.header is None:
                continue
            self.payload.extend(data)
            # More-data applies to direct payloads only. Upstream mpv sets
            # m=1 even on a complete shared-memory command, as Kitty accepts.
            if self.header.get(b"t") != b"s" and fields.get(b"m", b"0") != b"0":
                continue
            header = self.header
            if header.get(b"f") != b"24":
                raise ValueError("benchmark requires RGB24")
            if any(int(header.get(k, b"0")) for k in (b"c", b"r")):
                raise ValueError("terminal cell stretching would change the comparison")
            encoded = bytes(self.payload)
            if encoded.endswith(b"\0"):  # known upstream wire defect, recorded
                self.trailing_nuls += 1
                encoded = encoded[:-1]
            raw = base64.b64decode(encoded, validate=True)
            if header.get(b"t") == b"s":
                name = raw.decode("ascii")
                if "/" in name or not self.prefixes or not name.startswith(self.prefixes):
                    raise ValueError("unowned shared-memory name")
                path = self.shm_root / name
                try:
                    raw = path.read_bytes()
                    self.frame(raw, (int(header[b"s"]), int(header[b"v"])))
                finally:
                    path.unlink(missing_ok=True)  # acknowledge only this published name
            else:
                if header.get(b"o") == b"z":
                    raw = zlib.decompress(raw)
                self.frame(raw, (int(header[b"s"]), int(header[b"v"])))
            self.header = None
            self.payload.clear()

    def finish(self):
        if self.header is not None:
            raise ValueError("truncated graphics frame")


class Raw(Frames):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = bytearray()

    def feed(self, chunk):
        self.pending.extend(chunk)
        length = math.prod(self.size) * 3
        while len(self.pending) >= length:
            self.frame(bytes(self.pending[:length]), self.size)
            del self.pending[:length]

    def finish(self):
        if self.pending:
            raise ValueError("truncated raw RGB frame")


def command(arm, args, directory, fps):
    kind, path = arm["kind"], arm["path"]
    extra = arm.get("options", [])
    w, h = args.size
    scale = f"scale={w}:{h}:flags={FLAGS}"
    if getattr(args, "chroma_location", None):
        scale += ":in_chroma_loc=" + args.chroma_location
    if kind == "ffmpeg":
        return [path, "-v", "error", "-nostdin", *extra,
                *(["-readrate", "1"] if args.mode == "realtime" else []),
                "-ss", str(args.start), "-i", args.video, "-an", "-sn",
                "-vf", scale, "-frames:v", str(args.frames), "-pix_fmt", "rgb24",
                "-f", "rawvideo", "-"]
    if kind == "mpv":
        return [path, "--no-config", "--load-scripts=no", "--audio=no", "--hwdec=no",
                "--vo=kitty", "--vo-kitty-use-shm=yes", f"--vo-kitty-width={w}",
                f"--vo-kitty-height={h}", "--keepaspect=no", "--sws-scaler=lanczos",
                "--sws-allow-zimg=no", "--sws-fast=no", "--osd-level=0", "--osc=no",
                "--really-quiet", "--framedrop=no", f"--start={args.start}",
                f"--frames={args.frames}", *(["--untimed"] if args.mode == "throughput" else []),
                *extra, args.video]
    if kind == "neuronte":
        return [sys.executable, "-B", str(Path(path) / "tools/neuronte-play.py"),
                "--pixel-size", f"{w}x{h}", "--no-status", "--no-audio", "--transport", "shm",
                "--start", str(args.start), "--frames", str(args.frames),
                "--stats", str(directory / "player.json"), "--log", str(directory / "frames.jsonl"),
                *(["--untimed"] if args.mode == "throughput" else []), *extra, args.video]
    if kind == "ffplay":
        return [path, "-hide_banner", "-loglevel", "info", "-autoexit", "-an",
                "-noframedrop", "-ss", str(args.start), "-t", str(args.frames / fps),
                "-x", str(w), "-y", str(h), "-vf", scale + ",format=rgb24,showinfo", *extra, args.video]
    raise ValueError(kind)


def run(arm, args, directory, clip, warmup):
    directory.mkdir()
    events = []
    sampler = TreeSampler(events.append, period=.05)
    samples = sorted({0, args.frames // 3, 2 * args.frames // 3, args.frames - 1}) if warmup else []
    cmd = command(arm, args, directory, clip["fps"])
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", TERM="xterm-kitty")
    env.update(SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", SDL_RENDER_DRIVER="software")
    terminal = arm["kind"] in ("mpv", "neuronte")
    master = slave = None
    process = None
    error = None
    capture = None
    showinfo = bytearray()
    filtered = []
    filtered_total = 0
    result = dict(arm=arm["name"], kind=arm["kind"], mode=args.mode, warmup=warmup,
                  command=cmd, size=args.size, host_before=host_state())
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    driver_before = time.process_time()
    first = mono()
    stop = None
    try:
        with (directory / "stderr.log").open("wb") as log:
            if terminal:
                master, slave = pty.openpty()
                fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 56, 105, *args.size))
                process = subprocess.Popen(cmd, stdin=slave, stdout=slave, stderr=log, env=env,
                                           start_new_session=True,
                                           preexec_fn=lambda: fcntl.ioctl(0, termios.TIOCSCTTY, 0))
                os.close(slave); slave = None
                capture = Graphics(args.size, directory, samples,
                                   prefixes=(f"kilix-mpv-{process.pid}-", f"neuronte-play-{process.pid}-",
                                             "mpv-kitty-0x"))
                fd = master
            elif arm["kind"] == "ffmpeg":
                process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                           stderr=log, env=env, start_new_session=True)
                capture = Raw(args.size, directory, samples)
                fd = process.stdout.fileno()
            else:
                process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, env=env, start_new_session=True)
                fd = process.stderr.fileno()
            sampler.roots = {arm["name"]: process.pid}
            sampler.start()
            deadline = time.monotonic() + max(60, args.frames / clip["fps"] * 5)
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("player did not finish")
                if not select.select([fd], [], [], .05)[0]:
                    if process.poll() is not None:
                        break
                    continue
                try:
                    data = os.read(fd, 1 << 18)
                except OSError as exc:
                    if terminal and exc.errno == errno.EIO:
                        break
                    raise
                if not data:
                    break
                if capture:
                    capture.feed(data)
                else:
                    log.write(data)
                    showinfo.extend(data)
                    while b"\n" in showinfo:
                        line, _, rest = showinfo.partition(b"\n"); showinfo[:] = rest
                        match = re.search(rb"\bn:\s*(\d+).*?pts_time:([-\d.]+).*?\bs:(\d+)x(\d+)", line)
                        if match:
                            filtered_total += 1
                            n, pts, w, h = match.groups()
                            if args.start <= float(pts) < args.start + args.frames / clip["fps"] - 1e-6:
                                if (int(w), int(h)) != tuple(args.size):
                                    raise ValueError("ffplay filtered a different frame size")
                                filtered.append(dict(t=mono(), pts=float(pts), n=int(n)))
            code = process.wait(timeout=5)
            stop = mono()
            if code:
                raise ValueError(f"player exited {code}; see stderr.log")
            if capture:
                capture.finish()
                extra_redraw = (len(capture.times) == args.frames + 1 and
                                capture.hashes[-1] == capture.hashes[-2])
                result["extra_final_redraw"] = extra_redraw
                if len(capture.times) != args.frames and not extra_redraw:
                    raise ValueError(f"received {len(capture.times)} frames, expected {args.frames}")
            elif len(filtered) != args.frames:
                raise ValueError(f"ffplay filtered {len(filtered)} frames, expected {args.frames}")
    except Exception as exc:
        error = str(exc)
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        if sampler.is_alive():
            sampler.stop()
        if master is not None:
            os.close(master)
        if slave is not None:
            os.close(slave)
        if process:
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
    stop = stop or mono()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    elapsed = (stop - first) / 1e9
    used = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    times = capture.times if capture else [e["t"] for e in filtered]
    source_times = times[:args.frames]
    result.update(status="error" if error else "ok", error=error, frames=len(times),
                  wall_seconds=elapsed, child_cpu_seconds=used,
                  cpu_ms_per_source_frame=used * 1000 / args.frames,
                  driver_cpu_seconds=time.process_time() - driver_before,
                  first_frame_ms=(times[0] - first) / 1e6 if times else None,
                  throughput_fps=len(source_times) / elapsed,
                  measured_boundary="RGB consumer" if capture else "ffplay filter output",
                  steady_cpu=cpu(events, arm["name"], times[0], times[-1]) if len(times) > 1 else None,
                  cadence=cadence(source_times, clip["fps"], (source_times[-1]-source_times[0])/1e9)
                  if len(source_times) > 1 else None)
    if capture:
        result["hashes"] = capture.hashes
        result["sample_indices"] = samples
        if isinstance(capture, Graphics):
            result.update(trailing_nuls=capture.trailing_nuls, global_deletes=capture.global_deletes)
    else:
        result["filtered_frames"] = filtered
        result["filtered_total"] = filtered_total
    (directory / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in events))
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def quality(reference, candidate):
    if len(reference) != len(candidate) or not reference:
        raise ValueError("quality inputs must have the same nonzero byte length")
    squared = total = largest = 0
    for a, b in zip(reference, candidate):
        diff = abs(a - b)
        total += diff; squared += diff * diff; largest = max(largest, diff)
    mse = squared / len(reference)
    return dict(exact=squared == 0, mae=total / len(reference), max_error=largest,
                psnr_db=10 * math.log10(255**2 / mse) if mse else None)


def report(out, results, arms):
    measured = [r for r in results if not r["warmup"] and r["status"] == "ok"]
    summary = {}
    for arm in arms:
        rows = [r for r in measured if r["arm"] == arm["name"]]
        summary[arm["name"]] = {key: statistics.median(r[key] for r in rows) if rows else None
                                for key in ("child_cpu_seconds", "cpu_ms_per_source_frame",
                                            "throughput_fps", "first_frame_ms")}
        summary[arm["name"]]["successful_runs"] = len(rows)
    # Compare by repetition; never drop failed runs and silently shift pairing.
    comparisons = {}
    for reference in arms:
        for candidate in arms:
            if reference["name"] >= candidate["name"]:
                continue
            left = {r["repetition"]: r for r in results if not r["warmup"] and r["arm"] == reference["name"]}
            right = {r["repetition"]: r for r in results if not r["warmup"] and r["arm"] == candidate["name"]}
            pairs = [(left[i], right[i]) for i in sorted(left.keys() & right.keys())
                     if left[i]["status"] == right[i]["status"] == "ok"]
            comparisons[reference["name"] + ":" + candidate["name"]] = compare(
                [a["cpu_ms_per_source_frame"] for a, b in pairs],
                [b["cpu_ms_per_source_frame"] for a, b in pairs])
    checks = {}
    baseline = next((r for r in results if r["kind"] == "ffmpeg" and r["warmup"] and r["status"] == "ok"), None)
    if baseline:
        for row in results:
            if not row["warmup"] or row["status"] != "ok" or row["kind"] == "ffplay":
                continue
            checks[row["arm"]] = {
                "all_frame_hashes_equal": row["hashes"][:len(baseline["hashes"])] == baseline["hashes"],
                "samples": {str(i): quality(
                    (out / baseline["directory"] / f"frame-{i:06d}.rgb").read_bytes(),
                    (out / row["directory"] / f"frame-{i:06d}.rgb").read_bytes())
                    for i in row["sample_indices"]}}
    analysis = dict(summary=summary, comparisons=comparisons, quality_against_ffmpeg=checks)
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    lines = ["# Equal-resolution video comparison", "",
             "Private RGB consumers; not physical display FPS. FFmpeg omits presentation. "
             "ffplay uses SDL dummy/software rendering and counts filtered frames, not displayed frames.",
             "", "| Arm | Valid measured runs | CPU ms/source frame | Consumer FPS | First frame ms |",
             "|---|---:|---:|---:|---:|"]
    for name, row in summary.items():
        fmt = lambda key: f"{row[key]:.2f}" if row[key] is not None else "—"
        lines.append(f"| {name} | {row['successful_runs']} | {fmt('cpu_ms_per_source_frame')} | "
                     f"{fmt('throughput_fps')} | {fmt('first_frame_ms')} |")
    lines += ["", "Exact commands, versions, hashes, host conditions, per-run logs, paired CPU "
              "statistics and sampled RGB quality are in the adjacent JSON files."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return analysis


def arm_spec(value):
    parts = value.split(":", 2)
    if len(parts) != 3 or not re.fullmatch(r"[a-zA-Z0-9_-]+", parts[0]) or parts[1] not in ("ffmpeg", "mpv", "neuronte", "ffplay"):
        raise argparse.ArgumentTypeError("use NAME:ffmpeg|mpv|neuronte|ffplay:PATH")
    return dict(zip(("name", "kind", "path"), parts))


def size_spec(value):
    try:
        w, h = map(int, value.lower().split("x"))
        if 0 < w <= 65535 and 0 < h <= 65535:
            return [w, h]
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("use positive WIDTHxHEIGHT (at most 65535 each)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", type=arm_spec, action="append", required=True)
    p.add_argument("--option", action="append", default=[], metavar="NAME:ARG",
                   help="extra argument for one arm; repeat for an option and its value")
    p.add_argument("--size", type=size_spec, default=[945, 531])
    p.add_argument("--start", type=float, default=20)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--runs", type=int, default=6)
    p.add_argument("--mode", choices=("throughput", "realtime"), default="throughput")
    a = p.parse_args(argv)
    if a.frames < 1 or a.runs < 1 or a.start < 0:
        p.error("frames/runs must be positive and start nonnegative")
    if len({arm["name"] for arm in a.arm}) != len(a.arm):
        p.error("arm names must be unique")
    by_name = {arm["name"]: arm for arm in a.arm}
    for option in a.option:
        name, sep, value = option.partition(":")
        if not sep or name not in by_name:
            p.error("--option needs an existing arm NAME:ARG")
        by_name[name].setdefault("options", []).append(value)
    if a.mode == "throughput" and any(arm["kind"] == "ffplay" for arm in a.arm):
        p.error("ffplay is a realtime software-display reference; use --mode realtime")
    if a.start != 0 and any(arm["kind"] == "ffplay" for arm in a.arm):
        p.error("ffplay retains seek pre-roll; use --start 0 on a common keyframe-cut clip")
    a.video = str(Path(a.video).resolve())
    a.out.mkdir(parents=True, exist_ok=False)
    clip = probe(a.video)
    if a.start + a.frames / clip["fps"] > clip["duration"] + 1 / clip["fps"]:
        p.error("requested frames extend beyond the source")
    color = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=color_space,color_range,color_primaries,color_transfer,chroma_location,avg_frame_rate",
                            "-of", "json", a.video], capture_output=True, text=True, check=True)
    stream_metadata = json.loads(color.stdout)["streams"][0]
    a.chroma_location = stream_metadata.get("chroma_location")
    if a.chroma_location not in ("left", "center", "topleft", "top", "bottomleft", "bottom"):
        a.chroma_location = None
    metadata = dict(arguments={**vars(a), "out": str(a.out)}, clip=clip,
                    source_sha256=file_sha256(Path(a.video)), tools=[],
                    stream_metadata=stream_metadata)
    for arm in a.arm:
        arm["path"] = str(Path(arm["path"]).resolve())
        if arm["kind"] == "neuronte":
            version = identity(arm["path"])
        else:
            flag = "--version" if arm["kind"] == "mpv" else "-version"
            done = subprocess.run([arm["path"], flag], capture_output=True, text=True, timeout=15)
            version = dict(sha256=file_sha256(Path(arm["path"])), output=(done.stdout + done.stderr).splitlines()[:12])
        metadata["tools"].append(dict(arm=arm, identity=version))
    (a.out / "campaign.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = []
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    for repetition in range(a.runs + 1):
        # Rotate the first arm so steady host drift does not always favor one.
        offset = repetition % len(a.arm)
        for arm in a.arm[offset:] + a.arm[:offset]:
            name = f"{repetition:02d}-{arm['name']}"
            print(f"{name}: starting {'warm-up' if repetition == 0 else 'measurement'}", flush=True)
            result = run(arm, a, a.out / name, clip, repetition == 0)
            result.update(directory=name, repetition=repetition)
            results.append(result)
            (a.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(f"{name}: {result['status']}, {result['frames']} frames, "
                  f"{result['cpu_ms_per_source_frame']:.2f} CPU ms/frame; {result['error'] or ''}", flush=True)
    report(a.out, results, a.arm)
    return int(any(r["status"] != "ok" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
