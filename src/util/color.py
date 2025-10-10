"""
どこで: `util.color`。
何を: 色指定の正規化/変換（Hex, RGBA 0–1, RGBA 0–255）を一元化。
なぜ: API/GUI/HUD 全体で同一の受理仕様とエラーメッセージを提供するため。
"""

from __future__ import annotations

from typing import Sequence


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def parse_hex_color_str(s: str) -> tuple[float, float, float, float]:
    """Hex 文字列から RGBA(0–1) を返す。

    受理形式: "#RRGGBB", "#RRGGBBAA", "0xRRGGBB", "0xRRGGBBAA", "RRGGBB", "RRGGBBAA"。
    大文字/小文字は不問。
    """
    t = s.strip()
    if t.startswith("#"):
        t = t[1:]
    elif t.lower().startswith("0x"):
        t = t[2:]
    if len(t) not in (6, 8):
        raise ValueError(f"invalid hex color length: '{s}' (expected RRGGBB or RRGGBBAA)")
    try:
        r = int(t[0:2], 16)
        g = int(t[2:4], 16)
        b = int(t[4:6], 16)
        a = int(t[6:8], 16) if len(t) == 8 else 255
    except ValueError as e:
        raise ValueError(f"invalid hex color: '{s}'") from e
    return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)


def _as_sequence(value: object) -> Sequence[float | int] | None:
    if isinstance(value, (list, tuple)):
        return value  # type: ignore[return-value]
    return None


def normalize_color(value: object) -> tuple[float, float, float, float]:
    """色を RGBA(0–1) へ正規化する。

    - 受理: Hex 文字列, (r,g,b[,a]) （0–1 または 0–255）
    - 返値: (r,g,b,a) （0–1）
    """
    if isinstance(value, str):
        return parse_hex_color_str(value)
    seq = _as_sequence(value)
    if seq is None:
        raise ValueError(f"unsupported color type: {type(value)!r}")
    if len(seq) not in (3, 4):
        raise ValueError("color tuple/list must be length 3 or 4")
    # まず float 扱い（0–1）を試み、全要素が 0..1 ならそのまま
    try:
        fseq = [float(seq[0]), float(seq[1]), float(seq[2])]
        a = float(seq[3]) if len(seq) == 4 else 1.0
        if all(0.0 <= x <= 1.0 for x in fseq + [a]):
            r, g, b = fseq
            return (_clamp01(r), _clamp01(g), _clamp01(b), _clamp01(a))
    except Exception:
        pass
    # 次に 0–255 とみなし、整数丸め → 0–1 へスケール
    try:
        r = int(round(float(seq[0])))
        g = int(round(float(seq[1])))
        b = int(round(float(seq[2])))
        a = int(round(float(seq[3]))) if len(seq) == 4 else 255
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        a = max(0, min(255, a))
        return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
    except Exception as e:
        raise ValueError(f"invalid color tuple/list: {value!r}") from e


def to_u8_rgba(value: object) -> tuple[int, int, int, int]:
    """色を RGBA(0–255) へ変換する。"""
    r, g, b, a = normalize_color(value)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), int(round(a * 255)))


def to_u8_rgb(value: object) -> tuple[int, int, int]:
    """色を RGB(0–255) へ変換する。"""
    r, g, b, a = to_u8_rgba(value)
    return (r, g, b)


__all__ = [
    "parse_hex_color_str",
    "normalize_color",
    "to_u8_rgba",
    "to_u8_rgb",
]
