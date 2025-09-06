"""
エンジン層の変換ユーティリティ関数群（Geometry 統一版）。

基本方針:
- 個別の変換は `Geometry` のメソッド（`g.translate/scale/rotate`）の利用を推奨。
- ここでは「まとめて適用」の `transform_combined()` を主用途とし、
  個別関数は互換と補助のために提供する。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from .geometry import Geometry


def translate(g: Geometry, dx: float, dy: float, dz: float = 0) -> Geometry:
    """平行移動（薄いラッパ）: `Geometry.translate` に委譲。"""
    return g.translate(dx, dy, dz)


def scale_uniform(g: Geometry, factor: float, center=(0, 0, 0)) -> Geometry:
    """一様スケーリング（薄いラッパ）: `Geometry.scale` に委譲。"""
    return g.scale(factor, factor, factor, center)


def scale(g: Geometry, sx: float, sy: float, sz: float = 1.0, center=(0, 0, 0)) -> Geometry:
    """非一様スケーリング（薄いラッパ）: `Geometry.scale` に委譲。"""
    return g.scale(sx, sy, sz, center)


def rotate_z(g: Geometry, angle_rad: float, center=(0, 0, 0)) -> Geometry:
    """Z軸回りの回転（薄いラッパ）: `Geometry.rotate` に委譲。"""
    return g.rotate(z=angle_rad, center=center)


def rotate_x(g: Geometry, angle_rad: float, center=(0, 0, 0)) -> Geometry:
    """X軸回りの回転（薄いラッパ）: `Geometry.rotate` に委譲。"""
    return g.rotate(x=angle_rad, center=center)


def rotate_y(g: Geometry, angle_rad: float, center=(0, 0, 0)) -> Geometry:
    """Y軸回りの回転（薄いラッパ）: `Geometry.rotate` に委譲。"""
    return g.rotate(y=angle_rad, center=center)


def rotate_xyz(g: Geometry, rx: float, ry: float, rz: float, center=(0, 0, 0)) -> Geometry:
    """XYZ まとめ回転（薄いラッパ）: `Geometry.rotate` に委譲。"""
    return g.rotate(x=rx, y=ry, z=rz, center=center)


def transform_combined(
    g: Geometry,
    center=(0, 0, 0),
    scale_factors=(1, 1, 1),
    rotate_angles=(0, 0, 0),
) -> Geometry:
    """複合変換：スケール → 回転 → 移動を順次適用。

    引数:
        g: 変換対象の Geometry
        center: 最終的な中心位置
        scale_factors: (sx, sy, sz) スケール係数
        rotate_angles: (rx, ry, rz) 回転角度（ラジアン）

    返り値:
        変換後の新しい Geometry
    """
    result: Geometry = g
    
    # 1. スケール変換（原点中心）
    sx, sy, sz = scale_factors
    if sx != 1 or sy != 1 or sz != 1:
        result = scale(result, sx, sy, sz, center=(0, 0, 0))
    
    # 2. 回転変換（原点中心）
    rx, ry, rz = rotate_angles
    if rx != 0 or ry != 0 or rz != 0:
        result = rotate_xyz(result, rx, ry, rz, center=(0, 0, 0))
    
    # 3. 移動変換（最終位置へ）
    cx, cy, cz = center
    if cx != 0 or cy != 0 or cz != 0:
        result = translate(result, cx, cy, cz)

    return result
