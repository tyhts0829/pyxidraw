"""
どこで: `common` のパラメータ正規化ユーティリティ。
何を: 0–1 正規化の双方向変換と、キャッシュ鍵向けのパラメータ正規化（hashable 化）。
なぜ: `api.shapes`/`api.effects` が決定的かつ比較容易な形でパラメータを扱えるようにするため。
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Callable, Iterable, Mapping, Tuple

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
    "quantize_params",
    "signature_tuple",
    "params_signature",
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


# ---- 量子化（ParamSignature 用ユーティリティ） -----------------------------


def _env_quant_step(default_step: float | None) -> float:
    if default_step is not None:
        return float(default_step)
    env = os.getenv("PXD_PIPELINE_QUANT_STEP")
    try:
        return float(env) if env is not None else 1e-6
    except Exception:
        return 1e-6


def _quantize_scalar(value: Any, step: float) -> Any:
    # float（および np.floating）のみ量子化。int/bool はそのまま。
    if isinstance(value, (float, np.floating)):
        return round(float(value) / float(step)) * float(step)
    return value


def quantize_params(
    params: Mapping[str, Any],
    meta: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    default_step: float | None = None,
) -> dict[str, object]:
    """パラメータ辞書を `step` に基づいて量子化する。

    - 数値/ベクトル: `__param_meta__[name]['step']` を優先し、無ければ環境変数
      `PXD_PIPELINE_QUANT_STEP`、それも無ければ 1e-3。
    - それ以外: 値はそのまま。
    """
    step_default = _env_quant_step(default_step)
    meta = meta or {}

    out: dict[str, object] = {}
    changed = False
    for k, v in params.items():
        m = meta.get(k, {})
        step = m.get("step")
        if isinstance(v, (tuple, list)):
            # ベクトルの各成分を量子化
            if isinstance(step, (tuple, list)) and step:
                steps: list[float | None] = list(step)  # type: ignore[assignment]
            else:
                steps = [step if isinstance(step, (int, float)) else step_default] * len(v)
            result: list[Any] = []
            for idx, comp in enumerate(v):
                s = steps[idx if idx < len(steps) else -1]
                s_val = float(s) if isinstance(s, (int, float)) else step_default
                q = _quantize_scalar(comp, s_val)
                if not changed and q != comp:
                    # NaN の場合は同値扱い（両者が NaN なら変化なし）
                    try:
                        if not (
                            isinstance(q, float)
                            and isinstance(comp, float)
                            and np.isnan(q)
                            and np.isnan(comp)
                        ):
                            changed = True
                    except Exception:
                        changed = True
                result.append(q)
            out[k] = tuple(result)
            continue
        s = float(step) if isinstance(step, (int, float)) else step_default
        qv = _quantize_scalar(v, s)
        if not changed and qv != v:
            try:
                if not (
                    isinstance(qv, float) and isinstance(v, float) and np.isnan(qv) and np.isnan(v)
                ):
                    changed = True
            except Exception:
                changed = True
        out[k] = qv
    # 変化なしの場合は元の dict をそのまま返す（割り当て削減）
    if not changed and isinstance(params, dict):
        return params  # type: ignore[return-value]
    return out


def signature_tuple(
    params: Mapping[str, Any],
    meta: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    default_step: float | None = None,
) -> Tuple[Tuple[str, object], ...]:
    """量子化→ハッシュ可能化まで行い、署名タプルを返す。"""
    q = quantize_params(params, meta, default_step=default_step)
    return params_to_tuple(q)  # type: ignore[return-value]


def params_signature(
    fn: Callable[..., Any], params: Mapping[str, Any], *, default_step: float | None = None
) -> Tuple[Tuple[str, object], ...]:
    """関数の `__param_meta__` を考慮してパラメータ署名タプルを生成する。

    - float は量子化対象。既定 1e-6（`PXD_PIPELINE_QUANT_STEP` 上書き可）。
    - int/bool は非量子化でそのまま。
    - ベクトルは成分ごとに量子化。`meta['step']` 指定があれば優先。
    """
    meta = getattr(fn, "__param_meta__", {}) or {}
    return signature_tuple(params, meta, default_step=default_step)
