from __future__ import annotations

"""パレット設定値から Palette を生成する補助関数。

どこで: `engine.ui.palette`。
何を: GUI/スナップショットから渡される単純な値群を正規化し、`palette` ドメイン層の
      `generate_palette` を呼び出すための薄いアダプタを提供する。
なぜ: Parameter GUI とワーカー側の両方から同じロジックでパレットを構築するため。
"""

from typing import Any

from palette import (  # type: ignore[import]
    PALETTE_STYLE_OPTIONS,
    PALETTE_TYPE_OPTIONS,
    ColorInput,
    Palette,
    PaletteStyle,
    PaletteType,
    generate_palette,
)
from util.color import normalize_color as _norm  # type: ignore[import]
from util.utils import load_config as _load_cfg  # type: ignore[import]


def _default_base_color() -> tuple[float, float, float, float]:
    """設定ファイルからベースカラーの既定値を解決する。"""
    try:
        cfg = _load_cfg() or {}
    except Exception:
        cfg = {}
    canvas = cfg.get("canvas", {}) if isinstance(cfg, dict) else {}
    raw = canvas.get("line_color") or canvas.get("background_color") or (0.0, 0.0, 0.0, 1.0)
    try:
        r, g, b, a = _norm(raw)
        return float(r), float(g), float(b), float(a)
    except Exception:
        return 0.0, 0.0, 0.0, 1.0


_TYPE_BY_LABEL: dict[str, PaletteType] = {label: enum for label, enum in PALETTE_TYPE_OPTIONS}
_STYLE_BY_LABEL: dict[str, PaletteStyle] = {label: enum for label, enum in PALETTE_STYLE_OPTIONS}


def build_palette_from_values(
    *,
    base_color_value: Any | None,
    palette_type_value: Any | None,
    palette_style_value: Any | None,
    n_colors_value: Any | None,
    L_value: Any | None = None,
    C_value: Any | None = None,
    h_value: Any | None = None,
) -> Palette:
    """単純な値から Palette を構築する。

    - base_color_value は RGBA/HEX など `util.color.normalize_color` が扱える値を想定。
    - L/C/h が与えられている場合はそれを優先して OKLCH ベースでベースカラーを決める。
    - type/style は UI 上のラベル文字列を想定し、不明な値は既定値にフォールバックする。
    - n_colors は 1 以上の int に正規化し、失敗時は 4 を用いる。
    """
    # ベースカラー（L/C/h が揃っていればそれを優先）
    if L_value is not None and C_value is not None and h_value is not None:
        try:
            L = float(L_value)
        except Exception:
            L = 60.0
        try:
            C = float(C_value)
        except Exception:
            C = 0.1
        try:
            h = float(h_value)
        except Exception:
            h = 0.0
        # 範囲を軽くクランプ
        if L < 0.0:
            L = 0.0
        if L > 100.0:
            L = 100.0
        if C < 0.0:
            C = 0.0
        # 0.4 くらいまでに抑えておく（広げたければ後で調整）
        if C > 0.4:
            C = 0.4
        h = float(h % 360.0)
        base_input = ColorInput.from_oklch(L, C, h)
    else:
        # 旧来の RGBA/HEX 経路
        if base_color_value is None:
            rgba = _default_base_color()
        else:
            try:
                r, g, b, a = _norm(base_color_value)
                rgba = float(r), float(g), float(b), float(a)
            except Exception:
                rgba = _default_base_color()
        base_input = ColorInput.from_srgb(rgba[0], rgba[1], rgba[2])

    # パレット種別
    type_label = str(palette_type_value) if palette_type_value is not None else ""
    palette_type = _TYPE_BY_LABEL.get(type_label)
    if palette_type is None:
        palette_type = next(iter(_TYPE_BY_LABEL.values()))

    # スタイル
    style_label = str(palette_style_value) if palette_style_value is not None else ""
    palette_style = _STYLE_BY_LABEL.get(style_label)
    if palette_style is None:
        palette_style = next(iter(_STYLE_BY_LABEL.values()))

    # 色数
    try:
        n_colors = int(n_colors_value) if n_colors_value is not None else 4
    except Exception:
        n_colors = 4
    if n_colors <= 0:
        n_colors = 4

    return generate_palette(
        base_color=base_input,
        palette_type=palette_type,
        palette_style=palette_style,
        n_colors=n_colors,
    )


__all__ = ["build_palette_from_values"]
