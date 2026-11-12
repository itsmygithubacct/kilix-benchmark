"""Bytes a Kilix tab sends for a key or a pointer report.

Kilix 95 asks the terminal for keyboard flags 15 (disambiguate, event types,
alternate keys, every key as an escape code) and for SGR pixel mouse reports.
These encoders produce what the terminal itself would write for one US-layout
key or one pointer report under those modes, so text written to the tab with
remote-control ``send-text`` arrives as key and mouse events, not as a paste.
"""

from __future__ import annotations

SHIFT, ALT, CTRL = 1, 2, 4

# US layout: the unshifted key that produces each shifted symbol.
_SHIFTED = {
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7",
    "*": "8", "(": "9", ")": "0", "_": "-", "+": "=", "{": "[", "}": "]",
    "|": "\\", ":": ";", '"': "'", "<": ",", ">": ".", "?": "/", "~": "`",
}
# Keys that keep their legacy CSI form under the keyboard protocol.
_LETTER_KEYS = {"up": "A", "down": "B", "right": "C", "left": "D", "home": "H", "end": "F"}
# Keys reported as CSI <code> u.
_CODE_KEYS = {"enter": 13, "tab": 9, "backspace": 127, "escape": 27, "space": 32}


def _mods_field(mods: int, release: bool) -> str:
    value = mods + 1
    if release:
        return f"{value}:3"
    return "" if value == 1 else str(value)


def _csi_u(code: int, mods: int = 0, release: bool = False, shifted: int = 0) -> str:
    key = f"{code}:{shifted}" if shifted else str(code)
    field = _mods_field(mods, release)
    return f"\x1b[{key};{field}u" if field else f"\x1b[{key}u"


def _csi_letter(final: str, mods: int = 0, release: bool = False) -> str:
    field = _mods_field(mods, release)
    return f"\x1b[1;{field}{final}" if field else f"\x1b[{final}"


def named_key(name: str, mods: int = 0, release: bool = False) -> str:
    """One press (or release) of a named key such as ``down`` or ``escape``."""
    name = name.lower()
    if name in _LETTER_KEYS:
        return _csi_letter(_LETTER_KEYS[name], mods, release)
    if name in _CODE_KEYS:
        return _csi_u(_CODE_KEYS[name], mods, release)
    if len(name) == 1 and name.isascii() and name.isprintable():
        return char_key(name, mods, release)
    raise ValueError(f"no encoding for key {name!r}")


def char_key(ch: str, mods: int = 0, release: bool = False) -> str:
    """One printable ASCII character as the terminal reports its key."""
    if len(ch) != 1 or not ch.isascii() or not ch.isprintable():
        raise ValueError(f"not a printable ASCII character: {ch!r}")
    if ch.isalpha() and ch.isupper():
        return _csi_u(ord(ch.lower()), mods | SHIFT, release, shifted=ord(ch))
    if ch in _SHIFTED:
        return _csi_u(ord(_SHIFTED[ch]), mods | SHIFT, release, shifted=ord(ch))
    return _csi_u(ord(ch), mods, release)


def type_char(ch: str) -> str:
    """Press then release, as one keystroke."""
    if ch == "\n":
        return named_key("enter") + named_key("enter", release=True)
    return char_key(ch) + char_key(ch, release=True)


def press_named(name: str, mods: int = 0) -> str:
    """Press then release of a named key (with modifiers held)."""
    return named_key(name, mods) + named_key(name, mods, release=True)


def mouse(x: int, y: int, *, button: int = 0, press: bool = True,
          motion: bool = False, held: bool = False) -> str:
    """One SGR pixel report. ``motion`` without ``held`` is a hover move."""
    if x < 0 or y < 0:
        raise ValueError("pointer coordinates must be non-negative")
    if motion:
        code = 32 + (button if held else 3)
        return f"\x1b[<{code};{int(x)};{int(y)}M"
    return f"\x1b[<{button};{int(x)};{int(y)}{'M' if press else 'm'}"


def parse_key(name: str) -> tuple[str, int]:
    """``ctrl+escape`` -> (``escape``, CTRL)."""
    parts = [part for part in name.lower().split("+") if part]
    if not parts:
        raise ValueError("empty key name")
    mods = 0
    for part in parts[:-1]:
        bit = {"shift": SHIFT, "alt": ALT, "ctrl": CTRL}.get(part)
        if bit is None:
            raise ValueError(f"unknown modifier {part!r}")
        mods |= bit
    return parts[-1], mods
