"""The Kilix 95 tab suite file and the writes it compiles to."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from kilix_benchmark import tab_bench

SUITE = Path(__file__).resolve().parents[1] / "src" / "kilix_benchmark" / "suites" / "kilix95_tab.json"


def phase(suite, name):
    return next(p for p in suite["phases"] if p["id"] == name)


class SuiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.suite = tab_bench.load_suite(SUITE)

    def test_phases_in_order_and_quit_last(self) -> None:
        ids = [p["id"] for p in self.suite["phases"]]
        self.assertEqual(ids, ["menu", "open-notepad", "type-paced", "type-burst", "caret",
                               "hover", "drag", "quit"])

    def test_key_counts(self) -> None:
        menu = tab_bench.compile_phase(phase(self.suite, "menu"))
        self.assertEqual(sum(s["keys"] for s in menu), 42)
        paced = tab_bench.compile_phase(phase(self.suite, "type-paced"))
        typed = [s for s in paced if s["data"]]
        self.assertEqual(len(typed), 90)
        self.assertTrue(all(s["after_ms"] == 120 for s in typed))
        burst = tab_bench.compile_phase(phase(self.suite, "type-burst"))
        self.assertEqual(sum(s["keys"] for s in burst), 410)
        self.assertEqual(len([s for s in burst if s["data"]]), 41)
        self.assertTrue(all(s["after_ms"] == 0 for s in burst if s["data"]))

    def test_hover_stays_inside_the_desk(self) -> None:
        writes = tab_bench.compile_phase(phase(self.suite, "hover"), desk=(1908, 1008))
        reports = [s["data"].decode() for s in writes if s["data"]]
        self.assertEqual(len(reports), 120)
        for report in reports:
            code, x, y = report[3:-1].split(";")
            self.assertEqual(code, "35")
            self.assertTrue(0 <= int(x) < 1908 and 0 <= int(y) < 1008)
        with self.assertRaises(ValueError):
            tab_bench.compile_phase(phase(self.suite, "hover"))

    def test_drag_grips_the_named_window(self) -> None:
        windows = [["Notepad", "Untitled - Notepad", 100, 50, 520, 380, False]]
        writes = [s for s in tab_bench.compile_phase(phase(self.suite, "drag"), windows=windows)
                  if s["data"]]
        self.assertEqual(writes[0]["data"], b"\x1b[<0;160;60M")
        self.assertEqual(writes[1]["data"], b"\x1b[<32;164;62M")
        self.assertEqual(writes[-1]["data"], b"\x1b[<0;400;180m")
        self.assertEqual(len(writes), 62)
        with self.assertRaises(LookupError):
            tab_bench.compile_phase(phase(self.suite, "drag"), windows=[])
        minimized = [["Notepad", "x", 0, 0, 10, 10, True]]
        with self.assertRaises(LookupError):
            tab_bench.compile_phase(phase(self.suite, "drag"), windows=minimized)

    def test_bad_suites_are_refused(self) -> None:
        cases = []
        two = copy.deepcopy(self.suite)
        two["phases"][0]["steps"][0]["wait_ms"] = 5
        cases.append(two)
        unknown = copy.deepcopy(self.suite)
        unknown["phases"][0]["steps"][0]["keys"] = ["f13"]
        cases.append(unknown)
        no_quit = copy.deepcopy(self.suite)
        no_quit["phases"].pop()
        cases.append(no_quit)
        twice = copy.deepcopy(self.suite)
        twice["phases"].insert(0, copy.deepcopy(twice["phases"][0]))
        cases.append(twice)
        schema = copy.deepcopy(self.suite)
        schema["schema"] = "other"
        cases.append(schema)
        with tempfile.TemporaryDirectory() as raw:
            for number, case in enumerate(cases):
                path = Path(raw) / f"{number}.json"
                path.write_text(json.dumps(case), encoding="utf-8")
                with self.assertRaises((ValueError, KeyError), msg=f"case {number}"):
                    tab_bench.load_suite(path)

    def test_environment_switches_off_outside_effects(self) -> None:
        env = self.suite["env"]
        self.assertEqual(env["KILIX_NO_SOUND"], "1")
        self.assertEqual(env["KILIX_HOST_CLIP"], "0")
        self.assertEqual(env["KILIX_MODEL_WIZARD_OFFER"], "0")
        for name in ("DBUS_SESSION_BUS_ADDRESS", "DISPLAY", "KITTY_LISTEN_ON",
                     "KILIX_RC_PASSWORD_FILE"):
            self.assertIn(name, self.suite["env_unset"])


if __name__ == "__main__":
    unittest.main()
