from __future__ import annotations

"""engine.ui.palette.helpers の補助関数テスト。"""

from engine.ui.palette.helpers import build_palette_from_values  # type: ignore[import]


def _build(
    palette_type_value: str,
    n_colors_value: int,
) -> int:
    """簡易ヘルパ: Palette を生成し、色数を返す。"""
    pal = build_palette_from_values(
        base_color_value=None,
        palette_type_value=palette_type_value,
        palette_style_value="Square",
        n_colors_value=n_colors_value,
        L_value=60.0,
        C_value=0.1,
        h_value=0.0,
    )
    return len(pal.colors)


def test_build_palette_from_values_triadic_normalizes_n_colors() -> None:
    """TRIADIC では n_colors が常に 3 になる。"""
    assert _build("TRI", 2) == 3
    assert _build("TRI", 3) == 3
    assert _build("TRI", 5) == 3


def test_build_palette_from_values_tetradic_normalizes_n_colors() -> None:
    """TETRADIC では n_colors が常に 4 になる。"""
    assert _build("TET", 2) == 4
    assert _build("TET", 4) == 4
    assert _build("TET", 6) == 4


def test_build_palette_from_values_other_types_keep_n_colors() -> None:
    """TRI/TET 以外では n_colors がそのまま尊重される。"""
    assert _build("ANA", 2) == 2
    assert _build("COM", 5) == 5
    assert _build("SPL", 6) == 6
    assert _build("TAS", 3) == 3
