"""
0–1 正規化パラメータの共通変換ユーティリティ（提案5）

および、キャッシュ鍵向けのパラメータ正規化（ShapesAPI / Pipeline 共有）。
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Tuple

import numpy as np


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


def ensure_vec3(v: float | Iterable[float]) -> tuple[float, float, float]:
    if isinstance(v, (int, float)):
        f = float(v)
        return (f, f, f)
    t = tuple(float(x) for x in v)
    if len(t) == 1:
        return (t[0], t[0], t[0])
    if len(t) != 3:
        raise ValueError("vec3 には数値の単体、1要素タプル、または3要素タプルを指定してください")
    return (t[0], t[1], t[2])


__all__ = [
    "clamp01",
    "norm_to_range",
    "norm_to_int",
    "norm_to_rad",
    "ensure_vec3",
    "make_hashable_param",
    "params_to_tuple",
]


# ---- キャッシュ鍵向けのパラメータ正規化 ---------------------------------------


def _key_for_sorting_object_key(k: object) -> str:
    return f"{type(k).__name__}:{repr(k)}"


def make_hashable_param(obj: object) -> object:
    """キャッシュ鍵生成のためにハッシュ可能へ正規化する。

    - dict: キーを安定ソートし、(k, v) のタプル列に再帰変換。
    - list/tuple: 再帰的にタプル化。
    - numpy.ndarray: dtype=object は tolist() で列挙。それ以外は ("nd", shape, dtype, blake2b-128) へ。
    - set/frozenset: 要素を安定ソートしてタプル化。
    - bytes/bytearray: bytes 化。
    - numpy scalar: Python 組込みへ。
    - それ以外: ハッシュ可能ならそのまま、不可なら ("obj", qualname, id) にフォールバック。
    """
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: _key_for_sorting_object_key(kv[0]))
        return tuple((k, make_hashable_param(v)) for k, v in items)

    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable_param(x) for x in obj)

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "O":
            return ("nd_obj", tuple(make_hashable_param(x) for x in obj.tolist()))
        arr = np.ascontiguousarray(obj)
        h = hashlib.blake2b(digest_size=16)
        h.update(arr.view(np.uint8).tobytes())
        return ("nd", arr.shape, str(arr.dtype), h.digest())

    if isinstance(obj, (set, frozenset)):
        return (
            "set",
            tuple(sorted((make_hashable_param(x) for x in obj), key=_key_for_sorting_object_key)),
        )

    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)

    try:
        hash(obj)  # type: ignore[arg-type]
        return obj
    except Exception:
        cls_name = getattr(obj, "__class__", type(obj)).__qualname__
        return ("obj", cls_name, id(obj))


def params_to_tuple(params: dict[str, Any]) -> Tuple[Tuple[str, object], ...]:
    """パラメータ辞書を「順序安定・ハッシュ可能」なタプル列に正規化する。"""
    items = sorted(params.items(), key=lambda kv: _key_for_sorting_object_key(kv[0]))
    return tuple((k, make_hashable_param(v)) for k, v in items)
