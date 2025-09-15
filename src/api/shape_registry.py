"""
形状レジストリ API（公開エントリポイント）

目的:
- `from api import shape` を唯一の公開経路とし、ユーザー拡張（@shape デコレータ）を簡潔に。
- ここでは shape デコレータを再輸出しない（破壊的変更）。
- 既存のレジストリ操作は薄い委譲として提供する。
"""

from __future__ import annotations

from shapes import registry as _shapes_registry
from shapes.base import BaseShape
from shapes.registry import get_shape, list_shapes


def get_shape_generator(name: str) -> type[BaseShape]:
    """登録済みシェイプ“クラス”を取得（薄い委譲）。

    Parameters
    ----------
    name : str
        シェイプ名。

    Returns
    -------
    type[BaseShape]
        登録済みのシェイプクラス。

    Raises
    ------
    ValueError
        指定名のシェイプが未登録の場合。
    """
    try:
        return get_shape(name)
    except KeyError:
        raise ValueError(f"未登録のシェイプ名です: {name}")


def list_registered_shapes() -> list[str]:
    """登録されているシェイプ名の一覧（昇順）。

    Returns
    -------
    list[str]
        登録名（snake_case）の昇順リスト。
    """
    return list_shapes()


def unregister_shape(name: str) -> None:
    """シェイプの登録を解除（主にテスト用）。

    未登録名は無視する。

    Parameters
    ----------
    name : str
        登録解除するシェイプ名。
    """
    _shapes_registry.unregister(name)


__all__ = [
    "get_shape_generator",
    "list_registered_shapes",
    "unregister_shape",
]
