"""シェイプレジストリ — 統一レジストリと型安全な登録。

概要:
- effects と対称の API (`@shape` / `get_shape` / `list_shapes` / `is_shape_registered`)。
- 登録対象は `BaseShape` 派生クラスに限定（誤登録を早期検出）。
- デコレータは名前省略可（`@shape` / `@shape()`）に加え、
  明示名指定（`@shape("name")` / `@shape(name="name")`）の双方をサポート。
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping

from common.base_registry import BaseRegistry

from .base import BaseShape

# 統一されたレジストリシステム
_shape_registry = BaseRegistry()


def shape(arg: Any | None = None, /, name: str | None = None):
    """シェイプをレジストリに登録するデコレータ（型安全）。

    使用例:
    - `@shape` / `@shape()`                      → クラス名から自動推論（`Sphere` → `sphere`）。
    - `@shape("custom")` / `@shape(name="custom")` → 明示名で登録。

    制約:
    - 登録できるのは `BaseShape` の派生クラスのみ。
    """

    def _register_checked(obj: Any, resolved_name: str | None = None) -> Any:
        if not (inspect.isclass(obj) and issubclass(obj, BaseShape)):
            raise TypeError(f"@shape は BaseShape 派生クラスのみ登録可能です: got {obj!r}")
        # BaseRegistry に登録（キー正規化は内部で実施）
        return _shape_registry.register(resolved_name)(obj)

    # 直付け (@shape) の場合
    if inspect.isclass(arg) and issubclass(arg, BaseShape) and name is None:
        return _register_checked(arg, None)

    # 引数付き（@shape() / @shape("name") / @shape(name="name")）
    if isinstance(arg, str) and name is None:
        # 位置引数で名前が与えられたケース
        def _decorator(obj: Any) -> Any:
            return _register_checked(obj, arg)

        return _decorator

    # name キーワード引数、もしくは引数なし
    def _decorator(obj: Any) -> Any:
        return _register_checked(obj, name)

    return _decorator


def get_shape(name: str) -> type[BaseShape]:
    """登録されたシェイプクラスを取得。

    引数:
        name: シェイプ名

    返り値:
        シェイプクラス

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
