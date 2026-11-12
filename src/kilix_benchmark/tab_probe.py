"""Run one Kilix 95 checkout in the current Kilix tab and log what it does.

``python3 -I tab_probe.py CONFIG.json``. The tab bench launches this as the
tab's command. It applies the run's environment, imports the checkout's
``main``, wraps a few methods with monotonic timers and then runs the
desktop exactly as ``main.py`` would. Every record is one JSON line appended
to the events file named in the config.

What is wrapped, identically in every arm:

- ``Desk.render`` (only passes that draw), ``Desk.blit`` and the frame file
  name, so each drawn frame has compose and send times and the file the
  terminal deletes once it has read it.
- ``DeskTerm.read_input``: when each batch of input was parsed.
- ``Desk.dispatch_key`` / ``Desk.dispatch_mouse``: time spent handling input.
- ``ImageDraw.text`` and the font measuring calls: time spent drawing and
  measuring text in each frame, by font class.
- ``neuronte_face._mask`` when the checkout has it: a call counter, which is
  the evidence that text went through Neuronte.

One startup side effect is switched off in every arm: the default voice
install, which would run the Kilix launcher from inside the benchmark.

This file must stay standard library only and must not import the rest of
the package: it runs with ``-I`` and its own directory removed from the path.
"""

import json
import os
import resource
import sys
import time

mono = time.monotonic_ns


class Log:
    def __init__(self, path):
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        self.fd = os.open(path, flags, 0o600)

    def emit(self, _event, **fields):
        fields["e"] = _event
        line = json.dumps(fields, separators=(",", ":"), default=str) + "\n"
        os.write(self.fd, line.encode())


class FrameState:
    """Counters that one drawn frame accumulates."""

    def __init__(self):
        self.reset()
        self.depth_text = 0
        self.depth_measure = 0
        self.wins = None

    def reset(self):
        self.text_n = 0
        self.text_ns = 0
        self.measure_n = 0
        self.measure_ns = 0
        self.mask_n = 0
        self.fonts = {}
        self.dispatch_n = 0
        self.dispatch_ns = 0


def _status_kb(field):
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _usage():
    own = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "utime": own.ru_utime, "stime": own.ru_stime, "maxrss_kb": own.ru_maxrss,
        "child_utime": kids.ru_utime, "child_stime": kids.ru_stime,
        "vmrss_kb": _status_kb("VmRSS"), "vmhwm_kb": _status_kb("VmHWM"),
        "threads": _status_kb("Threads"),
    }


def _summarize_input(events):
    out = []
    for ev in events:
        kind = ev.get("kind")
        if kind == "key":
            out.append(["k", ev.get("key"), ev.get("mods", 1), ev.get("evt", 1)])
        elif kind == "mouse":
            out.append(["m", ev.get("b"), ev.get("x"), ev.get("y"), bool(ev.get("press"))])
        elif kind == "paste":
            out.append(["p", len(ev.get("text") or "")])
        elif kind == "focus":
            out.append(["f", bool(ev.get("focused"))])
        else:
            out.append(["?", str(kind)])
    return out


def _font_name(font):
    return type(font).__name__ if font is not None else "default"


def install_text_hooks(frame, ImageDraw, ImageFont, extra_font_classes=()):
    """Time text drawing and measuring. Nested calls count once."""
    draw_text = ImageDraw.ImageDraw.text

    def text(self, xy, *args, **kwargs):
        if frame.depth_text:
            return draw_text(self, xy, *args, **kwargs)
        font = kwargs.get("font")
        if font is None and len(args) > 2:
            font = args[2]
        frame.depth_text += 1
        started = mono()
        try:
            return draw_text(self, xy, *args, **kwargs)
        finally:
            frame.text_ns += mono() - started
            frame.text_n += 1
            name = _font_name(font)
            frame.fonts[name] = frame.fonts.get(name, 0) + 1
            frame.depth_text -= 1

    ImageDraw.ImageDraw.text = text

    def timed_measure(original):
        def measure(self, *args, **kwargs):
            if frame.depth_measure or frame.depth_text:
                return original(self, *args, **kwargs)
            frame.depth_measure += 1
            started = mono()
            try:
                return original(self, *args, **kwargs)
            finally:
                frame.measure_ns += mono() - started
                frame.measure_n += 1
                frame.depth_measure -= 1
        return measure

    for cls in (ImageFont.FreeTypeFont,) + tuple(extra_font_classes):
        for name in ("getlength", "getbbox"):
            if hasattr(cls, name):
                setattr(cls, name, timed_measure(getattr(cls, name)))


def install_neuronte_counter(frame, module):
    original = module._mask

    def mask(*args, **kwargs):
        frame.mask_n += 1
        return original(*args, **kwargs)

    module._mask = mask


def _windows(desk):
    out = []
    for win in list(getattr(desk.wm, "windows", ())):
        out.append([type(win).__name__, getattr(win, "title", ""), win.x, win.y, win.w, win.h,
                    bool(getattr(win, "minimized", False))])
    return out


def install_desk_hooks(k95, log, frame):
    Desk = k95.Desk
    graphics = k95.kilix_graphics
    last = {"path": None, "band_rows": None}

    frame_path = Desk._frame_path

    def _frame_path(self, kind, sequence):
        path = frame_path(self, kind, sequence)
        last["path"] = path
        return path

    Desk._frame_path = _frame_path

    if hasattr(graphics, "build_frame_edit_file"):
        build_edit = graphics.build_frame_edit_file

        def build_frame_edit_file(path, w, h, *args, **kwargs):
            last["band_rows"] = h
            return build_edit(path, w, h, *args, **kwargs)

        graphics.build_frame_edit_file = build_frame_edit_file

    blit = Desk.blit
    current = {"in_render": False, "blit": None}

    def _blit(self, img=None, force_full=False):
        seq = self.seq
        band = getattr(self, "_band_n", 0)
        last["path"] = None
        last["band_rows"] = None
        started = mono()
        result = blit(self, img, force_full)
        ended = mono()
        if self.seq != seq:
            kind, rows = "full", self.h
        elif getattr(self, "_band_n", 0) != band:
            kind, rows = "band", last["band_rows"]
        else:
            kind, rows = "none", 0
        name = os.path.basename(last["path"]) if last["path"] and kind != "none" else None
        record = {"t0": started, "t1": ended, "kind": kind, "rows": rows, "file": name,
                  "img": img is not None, "force": bool(force_full)}
        if current["in_render"]:
            current["blit"] = record
        else:
            log.emit("blit", **record)
        return result

    Desk.blit = _blit

    render = Desk.render

    def _render(self):
        if not self.dirty:
            return render(self)
        # Counters cover everything since the previous drawn frame: the input
        # handling that asked for this frame as well as the drawing itself.
        current["in_render"] = True
        current["blit"] = None
        started = mono()
        try:
            return render(self)
        finally:
            ended = mono()
            current["in_render"] = False
            sent = current["blit"] or {}
            record = {
                "t0": started, "t1": ended, "tb": sent.get("t0", ended),
                "kind": sent.get("kind", "none"), "rows": sent.get("rows", 0),
                "file": sent.get("file"),
                "text_n": frame.text_n, "text_ns": frame.text_ns,
                "measure_n": frame.measure_n, "measure_ns": frame.measure_ns,
                "mask_n": frame.mask_n, "fonts": frame.fonts,
                "dispatch_n": frame.dispatch_n, "dispatch_ns": frame.dispatch_ns,
            }
            frame.reset()
            wins = _windows(self)
            if wins != frame.wins:
                frame.wins = wins
                record["wins"] = wins
            log.emit("frame", **record)

    Desk.render = _render

    def timed_dispatch(original):
        def dispatch(self, ev):
            started = mono()
            try:
                return original(self, ev)
            finally:
                frame.dispatch_ns += mono() - started
                frame.dispatch_n += 1
        return dispatch

    Desk.dispatch_key = timed_dispatch(Desk.dispatch_key)
    Desk.dispatch_mouse = timed_dispatch(Desk.dispatch_mouse)

    screen = Desk._show_system_screen

    def _show_system_screen(self, name, seconds):
        started = mono()
        try:
            return screen(self, name, seconds)
        finally:
            log.emit("screen", name=name, t0=started, t1=mono(), seconds=seconds)

    Desk._show_system_screen = _show_system_screen

    init = Desk.__init__

    def _init(self, *args, **kwargs):
        started = mono()
        init(self, *args, **kwargs)
        log.emit("desk", t0=started, t1=mono(), w=self.w, h=self.h,
                 cols=getattr(self.term, "cols", None), rows=getattr(self.term, "rows", None))

    Desk.__init__ = _init

    read_input = k95.DeskTerm.read_input

    def _read_input(self):
        events = read_input(self)
        if events:
            log.emit("input", t=mono(), n=len(events), ev=_summarize_input(events))
        return events

    k95.DeskTerm.read_input = _read_input


def _fonts_record(theme, face_module):
    record = {}
    for name in ("FONT", "BOLD", "SMALL"):
        font = getattr(theme, name, None)
        record[name] = _font_name(font)
        record[name + "_size"] = getattr(font, "size", None)
    record["START_W"] = getattr(theme, "START_W", None)
    record["CLOCK_W"] = getattr(theme, "CLOCK_W", None)
    record["neuronte_face"] = face_module is not None
    if face_module is not None:
        lib = getattr(face_module, "_lib", None)
        record["neuronte_root"] = getattr(face_module, "_ROOT", None)
        record["neuronte_lib_loaded"] = lib is not None
        record["neuronte_lib"] = getattr(lib, "_name", None)
    return record


def _neuronte_sizes(face_module):
    if face_module is None:
        return None
    out = []
    for (em, italic), size in sorted(getattr(face_module, "_sizes", {}).items()):
        out.append({"em": em, "italic": italic, "loaded": size is not None,
                    "weights": face_module.weight_path(em, italic) if size is not None else None})
    return out


def main(argv=None):
    started = mono()
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: tab_probe.py CONFIG.json", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as handle:
        config = json.load(handle)
    log = Log(config["events"])
    log.emit("start", t=started, pid=os.getpid(), ppid=os.getppid(), arm=config["arm"],
             run=config["run"], python=sys.version.split()[0])
    for key in config.get("env_unset", ()):
        os.environ.pop(key, None)
    os.environ.update(config.get("env", {}))

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    desktop = os.path.abspath(config["desktop"])
    sys.path.insert(0, desktop)
    sys.pycache_prefix = os.path.join(os.environ["KILIX95_CACHE_HOME"], "pycache")
    sys.argv = [os.path.join(desktop, "main.py")] + list(config.get("args", ()))

    frame = FrameState()
    from PIL import ImageDraw, ImageFont
    imported = mono()
    import main as k95
    log.emit("imported", t0=imported, t1=mono())

    face_module = sys.modules.get("neuronte_face")
    extra = (face_module.NeuronteFont,) if face_module is not None else ()
    install_text_hooks(frame, ImageDraw, ImageFont, extra)
    if face_module is not None:
        install_neuronte_counter(frame, face_module)
    install_desk_hooks(k95, log, frame)

    import system_voice
    system_voice.Controller.install_default_voice = lambda self: False
    log.emit("patched", items=["system_voice.Controller.install_default_voice"])
    log.emit("fonts", **_fonts_record(k95.T, face_module))

    status = 0
    try:
        k95.main()
    except SystemExit as exc:
        status = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException as exc:  # recorded, then re-raised for the terminal
        log.emit("error", type=type(exc).__name__, text=str(exc)[-500:])
        status = 1
        raise
    finally:
        log.emit("exit", t=mono(), status=status, neuronte_sizes=_neuronte_sizes(face_module),
                 **_usage())
    return status


if __name__ == "__main__":
    raise SystemExit(main())
