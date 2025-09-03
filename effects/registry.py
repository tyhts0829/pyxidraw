"""
関数ベースのエフェクトレジストリ。

@effect デコレータで `Geometry -> Geometry` な関数を登録します。
（必要なら BaseEffect サブクラスもラップ対応します。）
エイリアスはサポートしません（明示名のみ）。
"""

from __future__ import annotations

from inspect import isclass
from typing import Any, Callable, Dict

from engine.core.geometry import Geometry


EffectFn = Callable[[Geometry], Geometry]
_REGISTRY: Dict[str, Callable[..., Geometry]] = {}


def _normalize_key(name: str) -> str:
    return name.replace("-", "_").lower()


def effect(arg: Any | None = None, /, name: str | None = None):
    """関数/クラスをエフェクトとして登録するデコレータ（エイリアス非対応）。

    使い方:
    - `@effect`                -> obj.__name__ で登録
    - `@effect()`              -> obj.__name__ で登録
    - `@effect("custom")`     -> "custom" 名で登録
    - 関数/クラスいずれも可（クラスはインスタンス化して __call__ を使うラッパ登録）
    """

    def _register(obj):
        key = _normalize_key(name or obj.__name__)
        if isclass(obj):
            def wrapped(g: Geometry, **params: Any) -> Geometry:
                return obj()(g, **params)  # type: ignore[misc]
            _REGISTRY[key] = wrapped
        else:
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
