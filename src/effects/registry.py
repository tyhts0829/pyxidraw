"""Effects レジストリ（関数専用）。

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

import inspect
from typing import Any, Callable, Mapping

from common.base_registry import BaseRegistry
from engine.core.geometry import Geometry

EffectFn = Callable[[Geometry], Geometry]

# 共通レジストリ
_effect_registry = BaseRegistry()


def effect(arg: Any | None = None, /, name: str | None = None):
    """エフェクト関数を登録するデコレータ。

    使用例:
    - `@effect` / `@effect()`                → 関数名から自動推論。
    - `@effect("custom")` / `@effect(name="custom")` → 明示名で登録。

    例外:
    - TypeError: 関数以外を登録しようとした場合。
    """

    def _register_checked(obj: Any, resolved_name: str | None = None):
        if not inspect.isfunction(obj):
            raise TypeError(
                f"@effect は関数のみ登録可能です（クラス/インスタンスは不可）: got {obj!r}"
            )
        return _effect_registry.register(resolved_name)(obj)

    # 直付け (@effect)
    if inspect.isfunction(arg) and name is None:
        return _register_checked(arg, None)

    # 位置引数で名前を渡した (@effect("name"))
    if isinstance(arg, str) and name is None:

        def _decorator(obj: Any):
            return _register_checked(obj, arg)

        return _decorator

    # name キーワード引数、または引数なし
    def _decorator(obj: Any):
        return _register_checked(obj, name)

    return _decorator


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


__all__ = [
    "effect",
    "get_effect",
    "list_effects",
    "is_effect_registered",
    "clear_registry",
    "get_registry",
]
