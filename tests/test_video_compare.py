import base64
import json
from pathlib import Path
import tempfile
import unittest
import zlib

from kilix_benchmark.video_compare import Graphics, Raw, quality, report


def packet(control, data=b""):
    return b"\x1b_G" + control + b";" + data + b"\x1b\\"


class CaptureTests(unittest.TestCase):
    def test_fragmented_compressed_frame_and_old_trailing_nul(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = Graphics((3, 1), tmp)
            pixels = bytes(range(9))
            encoded = base64.b64encode(zlib.compress(pixels))
            wire = (packet(b"a=T,f=24,s=3,v=1,o=z,m=1", encoded[:4]) +
                    packet(b"m=0", encoded[4:] + b"\0") + packet(b"a=d"))
            for byte in wire:
                sink.feed(bytes([byte]))
            sink.finish()
            self.assertEqual(len(sink.times), 1)
            self.assertEqual(sink.trailing_nuls, 1)
            self.assertEqual(sink.global_deletes, 1)

    def test_shm_is_read_before_acknowledgment_and_never_reads_other_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "owned-frame"
            path.write_bytes(bytes(range(9)))
            sink = Graphics((3, 1), tmp, [0], shm_root=tmp, prefixes=("owned-",))
            sink.feed(packet(b"a=T,f=24,s=3,v=1,t=s,m=1", base64.b64encode(path.name.encode())))
            # Kitty ignores m for shared-memory transfers. Older mpv sends
            # this form without a following final packet.
            self.assertFalse(path.exists())
            sink.feed(packet(b"m=0"))
            self.assertFalse(path.exists())
            self.assertEqual((Path(tmp) / "frame-000000.rgb").read_bytes(), bytes(range(9)))
            for name in ("../owned-secret", "unowned-frame"):
                sink = Graphics((3, 1), tmp, shm_root=tmp, prefixes=("owned-",))
                with self.assertRaisesRegex(ValueError, "unowned"):
                    sink.feed(packet(b"a=T,f=24,s=3,v=1,t=s", base64.b64encode(name.encode())))

    def test_wrong_geometry_cell_stretch_and_partial_frames_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            for control in (b"a=T,f=24,s=1,v=3", b"a=T,f=24,s=3,v=1,c=3", b"a=T,f=32,s=3,v=1"):
                sink = Graphics((3, 1), tmp)
                with self.assertRaises(ValueError):
                    sink.feed(packet(control, base64.b64encode(bytes(9))))
            sink = Graphics((3, 1), tmp)
            sink.feed(packet(b"a=T,f=24,s=3,v=1,m=1", b"AA=="))
            with self.assertRaisesRegex(ValueError, "truncated"):
                sink.finish()
            raw = Raw((3, 1), tmp)
            raw.feed(bytes(11))
            self.assertEqual(len(raw.times), 1)
            with self.assertRaisesRegex(ValueError, "truncated"):
                raw.finish()

    def test_missing_shm_is_fatal_until_the_requested_interval_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = Graphics((1, 1), tmp, shm_root=tmp, prefixes=("owned-",), expected_frames=1)
            missing = packet(b"a=T,f=24,s=1,v=1,t=s", base64.b64encode(b"owned-missing"))
            with self.assertRaises(FileNotFoundError):
                sink.feed(missing)
            sink = Graphics((1, 1), tmp, shm_root=tmp, prefixes=("owned-",), expected_frames=1)
            sink.feed(packet(b"a=T,f=24,s=1,v=1", base64.b64encode(bytes(3))))
            sink.feed(missing)
            sink.finish()
            self.assertEqual(len(sink.times), 1)
            self.assertEqual(sink.unread_after_requested_frames, 1)

    def test_quality_uses_every_channel(self):
        self.assertTrue(quality(b"\x00\x7f\xff", b"\x00\x7f\xff")["exact"])
        q = quality(bytes(6), b"\x00\x00\x00\x00\x00\x06")
        self.assertFalse(q["exact"])
        self.assertEqual((q["mae"], q["max_error"]), (1, 6))
        with self.assertRaises(ValueError):
            quality(b"ab", b"a")

    def test_failed_runs_do_not_shift_paired_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for repetition, (a, b) in enumerate(((1, 9), (8, 10), (3, 4)), 1):
                for arm, value in (("a", a), ("b", b)):
                    rows.append(dict(arm=arm, kind="mpv", warmup=False, repetition=repetition,
                                     status="error" if (arm, repetition) == ("b", 1) else "ok",
                                     child_cpu_seconds=value, cpu_ms_per_source_frame=value,
                                     throughput_fps=30, first_frame_ms=1))
            result = report(Path(tmp), rows, [dict(name="a"), dict(name="b")])
            comparison = result["comparisons"]["a:b"]
            self.assertEqual(comparison["pairs"], 2)
            self.assertEqual(comparison["median_difference"], 1.5)


if __name__ == "__main__":
    unittest.main()
