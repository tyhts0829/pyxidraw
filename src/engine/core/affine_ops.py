"""
どこで: `engine.core` の軽量アフィン変換ユーティリティ。
何を: Geometry に対する translate/scale/rotate を小さな純関数として提供。
なぜ: LazyGeometry 本体から変換ロジックを分離し、責務を明確化するため。
"""

from __future__ import annotations

from engine.core.geometry import Geometry


def translate(g: Geometry, *, delta: tuple[float, float, float]) -> Geometry:
    dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
    return g.translate(dx, dy, dz)


def scale(
    g: Geometry,
    *,
    auto_center: bool = False,
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Geometry:
    sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
    cx, cy, cz = float(pivot[0]), float(pivot[1]), float(pivot[2])
    center = (cx, cy, cz)
    return g.scale(sx, sy, sz, center=center)


def rotate(
    g: Geometry,
    *,
    auto_center: bool = False,
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Geometry:
    rx, ry, rz = float(rotation[0]), float(rotation[1]), float(rotation[2])
    cx, cy, cz = float(pivot[0]), float(pivot[1]), float(pivot[2])
    center = (cx, cy, cz)
    return g.rotate(x=rx, y=ry, z=rz, center=center)


__all__ = ["translate", "scale", "rotate"]
