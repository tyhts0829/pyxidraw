"""
どこで: `shapes` のレジストリ層（関数専用）。
何を: `@shape` デコレータで shape 関数を登録し、取得/一覧/検査を提供。
なぜ: 形状生成の拡張を一貫 API で管理し、`api.shapes` から安全に解決するため。

概要:
- Effect と対称の API（`@shape` / `get_shape` / `list_shapes` / `is_shape_registered`）。
- 登録対象は「関数」のみ（`Geometry` またはポリライン列を返す）。
- デコレータは名前省略可（`@shape` / `@shape()`）と明示名指定をサポート。
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping

from common.base_registry import BaseRegistry

ShapeFn = Callable[..., Any]

# 統一されたレジストリシステム
_shape_registry = BaseRegistry()


def shape(arg: Any | None = None, /, name: str | None = None):
    """シェイプ関数をレジストリに登録するデコレータ。

    使用例:
    - `@shape` / `@shape()`                      → 関数名から自動推論。
    - `@shape("custom")` / `@shape(name="custom")` → 明示名で登録。

    例外:
    - TypeError: 関数以外を登録しようとした場合。
    """

    def _register_checked(obj: Any, resolved_name: str | None = None):
        if not inspect.isfunction(obj):
            raise TypeError(f"@shape は関数のみ登録可能です: got {obj!r}")

        orig_fn = obj
        # 純関数をそのまま登録・返却
        return _shape_registry.register(resolved_name)(orig_fn)

    # 直付け (@shape)
    if inspect.isfunction(arg) and name is None:
        return _register_checked(arg, None)

    # 位置引数で名前を渡した (@shape("name"))
    if isinstance(arg, str) and name is None:

        def _decorator_named(obj: Any):
            return _register_checked(obj, arg)

        return _decorator_named

    # name キーワード引数、または引数なし
    def _decorator_generic(obj: Any):
        return _register_checked(obj, name)

    return _decorator_generic


def get_shape(name: str) -> ShapeFn:
    """登録されたシェイプ関数を取得。

    引数:
        name: シェイプ名

    返り値:
        シェイプ関数

    例外:
        KeyError: シェイプが登録されていない場合
    """
    return _shape_registry.get(name)


def list_shapes() -> list[str]:
    """登録されているシェイプの一覧を取得。

    返り値:
        シェイプ名のリスト
    """
    return sorted(_shape_registry.list_all())


def is_shape_registered(name: str) -> bool:
    """シェイプが登録されているかチェック。"""
    return _shape_registry.is_registered(name)


def clear_registry() -> None:
    """レジストリをクリア（テスト用）。"""
    _shape_registry.clear()


def unregister(name: str) -> None:
    """名前を指定して登録を解除（存在しない場合は無視）。"""
    _shape_registry.unregister(name)


def get_registry() -> Mapping[str, Any]:
    """読み取り専用ビューとしてレジストリ辞書を返す。"""
    return _shape_registry.registry


__all__ = [
    "shape",
    "get_shape",
    "list_shapes",
    "is_shape_registered",
    "clear_registry",
    "unregister",
    "get_registry",
]
