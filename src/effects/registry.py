"""
エフェクトレジストリ（BaseRegistry 準拠、関数専用）。

- shapes と同一ポリシー（キー正規化・API）で管理するため、共通の `common.base_registry.BaseRegistry`
  を使用します。登録対象は関数（`Geometry -> Geometry`）のみに制限します。
- 公開 API は shapes と対称:
  - `effect`（デコレータ）
  - `get_effect(name)` / `list_effects()` / `is_effect_registered(name)` / `clear_registry()`
  - `get_registry()`（テスト・高度な用途向け）
"""

from __future__ import annotations

from inspect import isfunction
from typing import Any, Callable

from common.base_registry import BaseRegistry
from engine.core.geometry import Geometry

EffectFn = Callable[[Geometry], Geometry]

# 共通レジストリ
_effect_registry = BaseRegistry()


def effect(arg: Any | None = None, /, name: str | None = None):
    """エフェクト関数を登録するデコレータ。

    使い方:
    - `@effect`            -> 関数名から自動推論
    - `@effect()`          -> 関数名から自動推論
    - `@effect("custom")` -> 明示名で登録
    """

    def _wrap_register(obj: Any):
        if not isfunction(obj):
            raise TypeError("@effect は関数のみ登録可能です（クラス/インスタンスは不可）")
        # BaseRegistry に登録（キー正規化は内部で実施）
        return _effect_registry.register(name)(obj)

    # 直付け (@effect) の場合
    if callable(arg) and name is None:
        return _wrap_register(arg)

    # 引数付き (@effect() / @effect("name")) の場合
    return _wrap_register


def get_effect(name: str) -> Callable[..., Geometry]:
    return _effect_registry.get(name)


def list_effects() -> list[str]:
    return sorted(_effect_registry.list_all())


def is_effect_registered(name: str) -> bool:
    return _effect_registry.is_registered(name)


def clear_registry() -> None:
    _effect_registry.clear()


def get_registry() -> BaseRegistry:
    return _effect_registry
