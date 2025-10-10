from __future__ import annotations

import pytest

from util.color import normalize_color, parse_hex_color_str, to_u8_rgba


def _approx_tuple(t, p=1e-6):
    return tuple(round(v, 6) for v in t)


def test_parse_hex_color_valid_variants() -> None:
    assert _approx_tuple(parse_hex_color_str("#112233")) == (
        round(0x11 / 255.0, 6),
        round(0x22 / 255.0, 6),
        round(0x33 / 255.0, 6),
        1.0,
    )
    assert _approx_tuple(parse_hex_color_str("0x112233CC")) == (
        round(0x11 / 255.0, 6),
        round(0x22 / 255.0, 6),
        round(0x33 / 255.0, 6),
        round(0xCC / 255.0, 6),
    )
    assert _approx_tuple(parse_hex_color_str("112233")) == (
        round(0x11 / 255.0, 6),
        round(0x22 / 255.0, 6),
        round(0x33 / 255.0, 6),
        1.0,
    )


def test_parse_hex_color_invalid() -> None:
    with pytest.raises(ValueError):
        parse_hex_color_str("#123")
    with pytest.raises(ValueError):
        parse_hex_color_str("not-a-color")


def test_normalize_color_from_tuple_01() -> None:
    rgba = normalize_color((0.1, 0.2, 0.3))
    assert _approx_tuple(rgba) == (0.1, 0.2, 0.3, 1.0)


def test_normalize_color_from_tuple_255() -> None:
    rgba = normalize_color((255, 128, 0, 64))
    assert _approx_tuple(rgba) == (
        round(255 / 255.0, 6),
        round(128 / 255.0, 6),
        round(0 / 255.0, 6),
        round(64 / 255.0, 6),
    )


def test_to_u8_rgba_from_hex_and_tuple() -> None:
    assert to_u8_rgba("#FF00FF80") == (255, 0, 255, 128)
    assert to_u8_rgba((0.0, 1.0, 0.5)) == (0, 255, 128, 255)
