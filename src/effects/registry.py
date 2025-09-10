"""
関数ベースのエフェクトレジストリ。

@effect デコレータで `Geometry -> Geometry` な関数のみを登録します。
クラス継承や互換ラッパは廃止しました（提案に基づく簡素化）。
エイリアスはサポートしません（明示名のみ）。
"""

from __future__ import annotations

from inspect import isfunction
from typing import Any, Callable

from common.base_registry import BaseRegistry
from engine.core.geometry import Geometry

EffectFn = Callable[[Geometry], Geometry]
_REGISTRY: dict[str, Callable[..., Geometry]] = {}


def _normalize_key(name: str) -> str:
    """キー正規化: shapes と同一ポリシー（Camel→snake, '-'→'_', lower）。"""
    return BaseRegistry._normalize_key(name)


def effect(arg: Any | None = None, /, name: str | None = None):
    """関数をエフェクトとして登録するデコレータ（エイリアス非対応）。

    使い方:
    - `@effect`            -> obj.__name__ で登録
    - `@effect()`          -> obj.__name__ で登録
    - `@effect("custom")` -> "custom" 名で登録
    """

    def _register(obj):
        if not isfunction(obj):
            raise TypeError("@effect は関数のみ登録可能です（クラス/インスタンスは不可）")
        key = _normalize_key(name or obj.__name__)
        _REGISTRY[key] = obj
        return obj

    # 直付け (@effect) の場合
    if callable(arg) and name is None:
        return _register(arg)

    # 引数付き (@effect() / @effect("name")) の場合
    return _register


def get_effect(name: str) -> Callable[..., Geometry]:
    key = _normalize_key(name)
    if key not in _REGISTRY:
        raise KeyError(f"effect '{name}' is not registered")
    return _REGISTRY[key]


def list_effects() -> list[str]:
    return sorted(_REGISTRY.keys())


def clear_registry() -> None:
    _REGISTRY.clear()
