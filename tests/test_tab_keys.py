"""Key and pointer bytes for a Kilix tab, against the protocol's spelling."""

from __future__ import annotations

import unittest

from kilix_benchmark import tab_keys as K


class KeyEncodingTests(unittest.TestCase):
    def test_plain_letter_press_and_release(self) -> None:
        self.assertEqual(K.char_key("a"), "\x1b[97u")
        self.assertEqual(K.char_key("a", release=True), "\x1b[97;1:3u")
        self.assertEqual(K.type_char("a"), "\x1b[97u\x1b[97;1:3u")

    def test_shifted_keys_report_base_and_shifted_codes(self) -> None:
        self.assertEqual(K.char_key("A"), "\x1b[97:65;2u")
        self.assertEqual(K.char_key("!"), "\x1b[49:33;2u")
        self.assertEqual(K.char_key("("), "\x1b[57:40;2u")
        self.assertEqual(K.char_key("&", release=True), "\x1b[55:38;2:3u")
        self.assertEqual(K.char_key("-"), "\x1b[45u")
        self.assertEqual(K.char_key(" "), "\x1b[32u")

    def test_named_keys(self) -> None:
        self.assertEqual(K.named_key("enter"), "\x1b[13u")
        self.assertEqual(K.type_char("\n"), "\x1b[13u\x1b[13;1:3u")
        self.assertEqual(K.named_key("escape", K.CTRL), "\x1b[27;5u")
        self.assertEqual(K.named_key("down"), "\x1b[B")
        self.assertEqual(K.named_key("down", release=True), "\x1b[1;1:3B")
        self.assertEqual(K.named_key("left", K.SHIFT), "\x1b[1;2D")
        self.assertEqual(K.press_named("q", K.CTRL | K.ALT), "\x1b[113;7u\x1b[113;7:3u")

    def test_parse_key_names(self) -> None:
        self.assertEqual(K.parse_key("ctrl+escape"), ("escape", K.CTRL))
        self.assertEqual(K.parse_key("ctrl+alt+q"), ("q", K.CTRL | K.ALT))
        self.assertEqual(K.parse_key("down"), ("down", 0))
        with self.assertRaises(ValueError):
            K.parse_key("hyper+x")
        with self.assertRaises(ValueError):
            K.named_key("f13")
        with self.assertRaises(ValueError):
            K.char_key("é")

    def test_pointer_reports(self) -> None:
        self.assertEqual(K.mouse(10, 20, motion=True), "\x1b[<35;10;20M")
        self.assertEqual(K.mouse(10, 20, motion=True, held=True), "\x1b[<32;10;20M")
        self.assertEqual(K.mouse(5, 6), "\x1b[<0;5;6M")
        self.assertEqual(K.mouse(5, 6, press=False), "\x1b[<0;5;6m")
        with self.assertRaises(ValueError):
            K.mouse(-1, 0)


if __name__ == "__main__":
    unittest.main()
