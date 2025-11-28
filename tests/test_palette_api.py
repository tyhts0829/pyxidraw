from __future__ import annotations

"""palette API (`api.C`) の基本動作テスト。"""

import pytest

from api.palette import C
from palette import (  # type: ignore[import]
    ColorInput,
    PaletteStyle,
    PaletteType,
    generate_palette,
)
from util.palette_state import set_palette


def _make_sample_palette():
    """単純なサンプルパレットを生成する。"""
    base = ColorInput.from_srgb(1.0, 0.0, 0.0)
    return generate_palette(
        base_color=base,
        palette_type=PaletteType.ANALOGOUS,
        palette_style=PaletteStyle.SQUARE,
        n_colors=4,
    )


def test_C_len_and_getitem_basic():
    """C の長さとインデックスアクセスの基本挙動。"""
    set_palette(None)
    assert len(C) == 0
    with pytest.raises(IndexError):
        _ = C[0]

    pal = _make_sample_palette()
    set_palette(pal)
    assert len(C) == 4
    rgba = C[0]
    assert isinstance(rgba, tuple)
    assert len(rgba) == 4
    assert all(0.0 <= float(c) <= 1.0 for c in rgba)


def test_C_hex_and_export():
    """HEX エクスポートがパレットと同じ件数になることを確認する。"""
    pal = _make_sample_palette()
    set_palette(pal)

    hex_list = C.hex()
    assert len(hex_list) == len(pal.colors)
    assert all(isinstance(h, str) and h.startswith("#") for h in hex_list)

    exported = C.export("hex")
    assert len(exported) == len(pal.colors)
