"""
0–1 正規化パラメータの共通変換ユーティリティ（提案5）
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple


def clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else x


def norm_to_range(x: float, lo: float, hi: float) -> float:
    x = clamp01(float(x))
    return lo + (hi - lo) * x


def norm_to_int(x: float, lo: int, hi: int) -> int:
    return int(round(norm_to_range(x, lo, hi)))


def norm_to_rad(x: float) -> float:
    """0..1 → 0..2π"""
    return float(x) * math.tau


def ensure_vec3(v: float | Iterable[float]) -> Tuple[float, float, float]:
    if isinstance(v, (int, float)):
        f = float(v)
        return (f, f, f)
    t = tuple(float(x) for x in v)
    if len(t) == 1:
        return (t[0], t[0], t[0])
    if len(t) != 3:
        raise ValueError("expected scalar, 1-tuple, or 3-tuple for vec3")
    return (t[0], t[1], t[2])


__all__ = [
    "clamp01",
    "norm_to_range",
    "norm_to_int",
    "norm_to_rad",
    "ensure_vec3",
]

