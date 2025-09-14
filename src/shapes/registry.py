"""
シェイプレジストリ — 統一レジストリと型安全な登録

概要:
- effects と対称の API（`@shape` / `get_shape` / `list_shapes` / `is_shape_registered`）。
- 登録対象は `BaseShape` 派生クラスに限定（誤登録を早期検出）。
- デコレータは名前省略可（`@shape` / `@shape()`）。
  エイリアスはサポートしない（明示名のみ）。
"""

from typing import Any, Mapping

from common.base_registry import BaseRegistry

from .base import BaseShape

# 統一されたレジストリシステム
_shape_registry = BaseRegistry()


def shape(arg: str | type[BaseShape] | None = None):  # type: ignore[name-defined]
    """シェイプをレジストリに登録するデコレータ（型安全）。

    使い方:
    - 省略名で登録: `@shape` または `@shape()` -> クラス名から推論（`Sphere` -> `sphere`）。
    - 明示名で登録: `@shape("sphere")`。

    制約:
    - 登録できるのは `BaseShape` の派生クラスのみ。
    """

    def _register_checked(obj: Any) -> Any:
        if not isinstance(obj, type) or not issubclass(obj, BaseShape):
            raise TypeError("@shape は BaseShape 派生クラスのみ登録可能です")
        # BaseRegistry に登録（キー正規化は内部で実施）
        return _shape_registry.register(None)(obj)

    # 直付け (@shape) の場合
    if isinstance(arg, type):
        return _register_checked(arg)

    # 引数付き (@shape() / @shape("name")) の場合
    name = arg if isinstance(arg, str) else None

    def _decorator(obj: Any) -> Any:
        if not isinstance(obj, type) or not issubclass(obj, BaseShape):
            raise TypeError("@shape は BaseShape 派生クラスのみ登録可能です")
        return _shape_registry.register(name)(obj)

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
    """シェイプが登録されているかチェック。

    引数:
        name: シェイプ名。

    返り値:
        登録されている場合は True。
    """
    return _shape_registry.is_registered(name)


def clear_registry():
    """レジストリをクリア（テスト用）。"""
    _shape_registry.clear()


def unregister(name: str) -> None:
    """名前を指定して登録を解除（存在しない場合は無視）。"""
    _shape_registry.unregister(name)


def get_registry() -> Mapping[str, Any]:
    """読み取り専用ビューとしてレジストリ辞書を返す。"""
    return _shape_registry.registry
