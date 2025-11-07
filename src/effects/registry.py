"""
どこで: `effects` のレジストリ層（関数専用）。
何を: `@effect` デコレータによる登録と取得/一覧/検査を提供（キーは正規化）。
なぜ: 加工ステージの拡張を一貫 API で管理し、`api.effects` から安全に解決するため。

公開 API 概要:
- `effect`（デコレータ）: 関数を登録
- `get_effect(name)` / `list_effects()` / `is_effect_registered(name)` / `clear_registry()`
- `get_registry()`: 読み取り専用ビュー（テスト/診断用）
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping

from common.base_registry import BaseRegistry

EffectFn = Callable[..., Any]

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

        # 元実装（純関数）をそのまま登録・返却
        orig_fn = obj
        return _effect_registry.register(resolved_name)(orig_fn)

    # 直付け (@effect)
    if inspect.isfunction(arg) and name is None:
        return _register_checked(arg, None)

    # 位置引数で名前を渡した (@effect("name"))
    if isinstance(arg, str) and name is None:

        def _decorator_named(obj: Any):
            return _register_checked(obj, arg)

        return _decorator_named

    # name キーワード引数、または引数なし
    def _decorator_generic(obj: Any):
        return _register_checked(obj, name)

    return _decorator_generic


def get_effect(name: str) -> Callable[..., Any]:
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


def unregister(name: str) -> None:
    """名前を指定して登録を解除（存在しない場合は無視）。"""
    _effect_registry.unregister(name)


def get_registry() -> Mapping[str, Any]:
    """読み取り専用ビューとしてレジストリ辞書を返す。"""
    return _effect_registry.registry


__all__ = [
    "effect",
    "get_effect",
    "list_effects",
    "is_effect_registered",
    "clear_registry",
    "unregister",
    "get_registry",
]
