"""
形状レジストリシステム - レジストリパターンによる拡張可能な形状管理。
`shapes.registry` の公開APIへ薄く委譲するエントリポイント。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Type, Union

from shapes.registry import shape, get_shape, list_shapes, is_shape_registered, clear_registry
from shapes import registry as _shapes_registry


register_shape = shape


def get_shape_generator(name: str):
    """登録された形状生成器を取得（薄い委譲 API）。"""
    try:
        return get_shape(name)
    except KeyError:
        raise ValueError(f"未登録のシェイプ名です: {name}")


def list_registered_shapes() -> List[str]:
    """登録されているすべての形状名を返す。"""
    return list_shapes()


def unregister_shape(name: str):
    """形状の登録を解除（主にテスト用）。"""
    try:
        _shapes_registry.get_registry().unregister(name)
    except Exception:
        # 存在しない場合は無視（安全側）
        return


__all__ = [
    "register_shape",
    "get_shape_generator",
    "list_registered_shapes",
    "unregister_shape",
]
