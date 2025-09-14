"""
Effects レジストリ（関数専用）

概要:
- キー正規化（Camel→snake、小文字化、`-`→`_`）を行う共通基盤 `BaseRegistry` で管理する。
- 登録対象は `Geometry -> Geometry` の関数のみ（`@effect`）。
- 公開 API は Shapes 側と対称。

公開 API:
- `effect`（デコレータ）: 関数をレジストリに登録。
- `get_effect(name)` / `list_effects()` / `is_effect_registered(name)` / `clear_registry()`
- `get_registry()`: 読み取り専用ビュー（テスト/診断用）。
"""

from __future__ import annotations

from inspect import isfunction
from typing import Any, Callable, Mapping

from common.base_registry import BaseRegistry
from engine.core.geometry import Geometry

EffectFn = Callable[[Geometry], Geometry]

# 共通レジストリ
_effect_registry = BaseRegistry()


def effect(arg: Any | None = None, /, name: str | None = None):
    """エフェクト関数を登録するデコレータ。

    使用例:
    - `@effect`            → 関数名から自動推論
    - `@effect()`          → 同上
    - `@effect("custom")` → 明示名で登録

    例外:
    - TypeError: 関数以外を登録しようとした場合。
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
    """登録されたエフェクト関数を取得。

    例外:
    - KeyError: 未登録名の場合。
    """
    return _effect_registry.get(name)


def list_effects() -> list[str]:
    """登録済みエフェクト名をソートして返す。"""
    return sorted(_effect_registry.list_all())


def is_effect_registered(name: str) -> bool:
    """名前が登録済みかを返す。"""
    return _effect_registry.is_registered(name)


def clear_registry() -> None:
    """レジストリをクリア（テスト用途）。"""
    _effect_registry.clear()


def get_registry() -> Mapping[str, Any]:
    """読み取り専用ビューとしてレジストリ辞書を返す。"""
    return _effect_registry.registry
