"""
形状レジストリシステム - レジストリパターンによる拡張可能な形状管理。
`shapes.registry` の公開APIへ薄く委譲するエントリポイント。
"""

from __future__ import annotations

from shapes import registry as _shapes_registry
from shapes.registry import get_shape, list_shapes, shape

register_shape = shape


def get_shape_generator(name: str):
    """登録された形状生成器を取得（薄い委譲 API）。"""
    try:
        return get_shape(name)
    except KeyError:
        raise ValueError(f"未登録のシェイプ名です: {name}")


def list_registered_shapes() -> list[str]:
    """登録されているすべての形状名を返す。"""
    return list_shapes()


def unregister_shape(name: str):
    """形状の登録を解除（主にテスト用）。未登録名は無視。"""
    _shapes_registry.unregister(name)


__all__ = [
    "register_shape",
    "get_shape_generator",
    "list_registered_shapes",
    "unregister_shape",
]
