"""The probe's wrappers, on stand-in classes rather than a real desktop."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

PROBE_PATH = Path(__file__).resolve().parents[1] / "src" / "kilix_benchmark" / "tab_probe.py"
spec = importlib.util.spec_from_file_location("tab_probe_under_test", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class FakeTerm:
    cols, rows = 10, 5

    def read_input(self):
        return [{"kind": "key", "key": "a", "mods": 1, "evt": 1},
                {"kind": "key", "key": "a", "mods": 1, "evt": 3},
                {"kind": "mouse", "b": 35, "x": 1, "y": 2, "press": True}]


class FakeGraphics:
    @staticmethod
    def build_frame_edit_file(path, w, h, x, y, image):
        return f"edit {path} {h}"


def make_desk_module(draw_text):
    class Window:
        def __init__(self):
            self.title, self.x, self.y, self.w, self.h = "Untitled", 1, 2, 30, 40
            self.minimized = False

    class Notepad(Window):
        pass

    class Desk:
        def __init__(self, term=None):
            self.term = term
            self.w, self.h = 100, 50
            self.dirty = True
            self.seq = 0
            self._band_n = 0
            self.wm = types.SimpleNamespace(windows=[])

        def _frame_path(self, kind, sequence):
            return f"/frames/{kind}-{sequence}.rgb"

        def blit(self, img=None, force_full=False):
            if force_full or img is not None:
                self.seq += 1
                self._frame_path("full", self.seq)
            else:
                self._band_n += 1
                path = self._frame_path("band", self._band_n)
                graphics.build_frame_edit_file(path, self.w, 7, 0, 3, 1)

        def render(self):
            if not self.dirty:
                return
            draw_text("label")
            self.dirty = False
            self.blit()

        def dispatch_key(self, ev):
            draw_text("measure-free")

        def dispatch_mouse(self, ev):
            pass

        def _show_system_screen(self, name, seconds):
            self.blit(img=object())

    graphics = FakeGraphics()
    return types.SimpleNamespace(Desk=Desk, DeskTerm=FakeTerm, kilix_graphics=graphics,
                                 Notepad=Notepad)


class FakeFont:
    def getlength(self, text):
        return len(text)

    def getbbox(self, text):
        return (0, 0, len(text), 1)


class FakeImageDraw:
    class ImageDraw:
        def text(self, xy, text, fill=None, font=None):
            if "\n" in text:
                for line in text.split("\n"):
                    self.text(xy, line, fill, font)
            return font.getlength(text) if font else None


class ProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "events.jsonl"
        self.log = probe.Log(str(self.path))
        self.frame = probe.FrameState()
        font_module = types.SimpleNamespace(FreeTypeFont=type("FreeTypeFont", (FakeFont,), {}))
        self.font = font_module.FreeTypeFont()
        self.draw_module = types.SimpleNamespace(ImageDraw=type(
            "ImageDraw", (FakeImageDraw.ImageDraw,), {}))
        probe.install_text_hooks(self.frame, self.draw_module, font_module)
        drawer = self.draw_module.ImageDraw()
        self.k95 = make_desk_module(lambda s: drawer.text((0, 0), s + "\nx", font=self.font))
        probe.install_desk_hooks(self.k95, self.log, self.frame)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def records(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_render_records_text_band_and_windows(self) -> None:
        desk = self.k95.Desk(term=FakeTerm())
        desk.wm.windows.append(self.k95.Notepad())
        desk.dispatch_key(object())
        desk.render()
        desk.render()                 # not dirty: no record
        frames = [r for r in self.records() if r["e"] == "frame"]
        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertEqual(frame["kind"], "band")
        self.assertEqual(frame["rows"], 7)
        self.assertEqual(frame["file"], "band-1.rgb")
        self.assertEqual(frame["text_n"], 2)          # nested line calls count once each
        self.assertEqual(frame["fonts"], {"FreeTypeFont": 2})
        self.assertEqual(frame["measure_n"], 0)       # measuring inside text() is text time
        self.assertEqual(frame["dispatch_n"], 1)
        self.assertLessEqual(frame["t0"], frame["tb"])
        self.assertLessEqual(frame["tb"], frame["t1"])
        self.assertEqual(frame["wins"], [["Notepad", "Untitled", 1, 2, 30, 40, False]])
        desk.dirty = True
        desk.render()
        second = [r for r in self.records() if r["e"] == "frame"][-1]
        self.assertNotIn("wins", second)              # unchanged windows are not repeated
        self.assertEqual(second["dispatch_n"], 0)

    def test_blits_outside_render_and_startup_records(self) -> None:
        desk = self.k95.Desk(term=FakeTerm())
        desk._show_system_screen("startup", 1.2)
        desk.blit(force_full=True)
        records = self.records()
        kinds = [r["e"] for r in records]
        self.assertEqual(kinds, ["desk", "blit", "screen", "blit"])
        self.assertEqual(records[1]["kind"], "full")
        self.assertEqual(records[1]["file"], "full-1.rgb")
        self.assertTrue(records[1]["img"])
        self.assertTrue(records[3]["force"])
        self.assertEqual(records[2]["name"], "startup")
        self.assertEqual(records[0]["w"], 100)

    def test_input_summary(self) -> None:
        self.k95.DeskTerm().read_input()
        record = self.records()[-1]
        self.assertEqual(record["e"], "input")
        self.assertEqual(record["ev"], [["k", "a", 1, 1], ["k", "a", 1, 3],
                                        ["m", 35, 1, 2, True]])

    def test_measure_outside_text_is_counted(self) -> None:
        self.font.getlength("abc")
        self.font.getbbox("abc")
        self.assertEqual(self.frame.measure_n, 2)

    def test_neuronte_counter(self) -> None:
        module = types.SimpleNamespace(_mask=lambda *a: "mask")
        probe.install_neuronte_counter(self.frame, module)
        self.assertEqual(module._mask(1, 2), "mask")
        self.assertEqual(self.frame.mask_n, 1)


if __name__ == "__main__":
    unittest.main()
