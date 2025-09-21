"""
どこで: `engine.ui.parameters` の値変換レイヤ。
何を: 0..1 正規化と実レンジの写像（normalize/denormalize）。`RangeHint` に基づき決定的に線形変換。
なぜ: GUI/自動化からの一貫した値スケーリングを提供し、効果/形状実装を単位系から解放するため。

補足:
- 入力は常に「正規化値」（0..1 外も許容）。本レイヤではクランプしない。
- `clamp_normalized()` は UI 表示用の補助（バー/トラックの描画域に収める）としてのみ使用する。
"""

from __future__ import annotations

from typing import Any

from .state import RangeHint, ValueType


def _coerce_float(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    raise TypeError(f"数値へ正規化できない値です: {value!r}")


def clamp_normalized(value: float, hint: RangeHint | None) -> float:
    if hint is None:
        return max(0.0, min(1.0, value))
    lo = float(hint.min_value)
    hi = float(hint.max_value)
    if lo == hi:
        return lo
    if lo < hi:
        return max(lo, min(hi, value))
    # min/max が逆転している場合は交換する（異常系ガード）
    return max(hi, min(lo, value))


def normalize_scalar(actual: Any, hint: RangeHint | None, *, value_type: ValueType) -> float:
    if value_type in {"enum", "vector"}:
        raise ValueError("normalize_scalar は enum / vector には使用できません")
    if hint is None:
        if value_type == "bool":
            return 1.0 if bool(actual) else 0.0
        return _coerce_float(actual)

    mapped_min = hint.mapped_min
    mapped_max = hint.mapped_max

    if mapped_min is None or mapped_max is None or mapped_min == mapped_max:
        if value_type == "bool":
            return 1.0 if bool(actual) else 0.0
        # マッピングが不十分な場合は、与えられた値をそのまま正規化値として扱う（クランプしない）
        return _coerce_float(actual)

    lo = float(mapped_min)
    hi = float(mapped_max)
    span = hi - lo
    if span == 0.0:
        return clamp_normalized(float(hint.min_value), hint)

    normalized = (float(_coerce_float(actual)) - lo) / span
    return float(normalized)


def denormalize_scalar(normalized: Any, hint: RangeHint | None, *, value_type: ValueType) -> Any:
    if value_type in {"enum", "vector"}:
        raise ValueError("denormalize_scalar は enum / vector には使用できません")
    if hint is None:
        if value_type == "bool":
            return bool(normalized)
        if value_type == "int":
            return int(round(float(normalized)))
        return float(normalized)

    # クランプせず、そのまま線形変換する
    normalized_value = _coerce_float(normalized)

    mapped_min = hint.mapped_min
    mapped_max = hint.mapped_max
    mapped_step = hint.mapped_step

    if mapped_min is None or mapped_max is None:
        if value_type == "bool":
            return normalized_value >= 0.5
        if value_type == "int":
            return int(round(normalized_value))
        return normalized_value

    lo = float(mapped_min)
    hi = float(mapped_max)
    actual = lo + (hi - lo) * normalized_value

    if mapped_step is not None:
        step = float(mapped_step)
        if step > 0:
            actual = lo + round((actual - lo) / step) * step

    if value_type == "bool":
        return actual >= (lo + hi) * 0.5
    if value_type == "int":
        return int(round(actual))
    return float(actual)


__all__ = ["clamp_normalized", "normalize_scalar", "denormalize_scalar"]
