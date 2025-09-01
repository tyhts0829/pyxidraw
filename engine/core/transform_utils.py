"""
エンジン層の変換ユーティリティ関数群。
API層への依存なし、純粋な数学的変換を提供。
"""

import numpy as np
from .geometry_data import GeometryData


def translate(g: GeometryData, dx: float, dy: float, dz: float = 0) -> GeometryData:
    """平行移動変換を適用。
    
    Args:
        g: 変換対象のGeometryData
        dx, dy, dz: 移動ベクトル
        
    Returns:
        変換後の新しいGeometryData
    """
    vec = np.array([dx, dy, dz], dtype=np.float32)
    new_coords = g.coords + vec
    return GeometryData(new_coords, g.offsets.copy())


def scale_uniform(g: GeometryData, factor: float, center=(0, 0, 0)) -> GeometryData:
    """一様スケーリング変換を適用。
    
    Args:
        g: 変換対象のGeometryData
        factor: スケール係数
        center: スケールの中心点
        
    Returns:
        変換後の新しいGeometryData
    """
    return scale(g, factor, factor, factor, center)


def scale(g: GeometryData, sx: float, sy: float, sz: float = 1.0, center=(0, 0, 0)) -> GeometryData:
    """非一様スケーリング変換を適用。
    
    Args:
        g: 変換対象のGeometryData
        sx, sy, sz: 各軸のスケール係数
        center: スケールの中心点
        
    Returns:
        変換後の新しいGeometryData
    """
    cx, cy, cz = center
    center_vec = np.array([cx, cy, cz], dtype=np.float32)
    scale_vec = np.array([sx, sy, sz], dtype=np.float32)
    
    # 中心を原点に移動 → スケール → 中心を戻す
    new_coords = g.coords.copy()
    new_coords -= center_vec
    new_coords *= scale_vec
    new_coords += center_vec
    
    return GeometryData(new_coords, g.offsets.copy())


def rotate_z(g: GeometryData, angle_rad: float, center=(0, 0, 0)) -> GeometryData:
    """Z軸周りの回転変換を適用。
    
    Args:
        g: 変換対象のGeometryData
        angle_rad: 回転角度（ラジアン）
        center: 回転の中心点
        
    Returns:
        変換後の新しいGeometryData
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    cx, cy, cz = center
    
    new_coords = g.coords.copy()
    # 中心を原点に移動
    new_coords[:, 0] -= cx
    new_coords[:, 1] -= cy
    new_coords[:, 2] -= cz
    
    # Z軸周りの回転行列を適用
    x_new = new_coords[:, 0] * c - new_coords[:, 1] * s
    y_new = new_coords[:, 0] * s + new_coords[:, 1] * c
    new_coords[:, 0] = x_new
    new_coords[:, 1] = y_new
    
    # 中心を戻す
    new_coords[:, 0] += cx
    new_coords[:, 1] += cy
    new_coords[:, 2] += cz
    
    return GeometryData(new_coords, g.offsets.copy())


def rotate_x(g: GeometryData, angle_rad: float, center=(0, 0, 0)) -> GeometryData:
    """X軸周りの回転変換を適用。"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    cx, cy, cz = center
    
    new_coords = g.coords.copy()
    new_coords[:, 0] -= cx
    new_coords[:, 1] -= cy
    new_coords[:, 2] -= cz
    
    # X軸周りの回転行列を適用
    y_new = new_coords[:, 1] * c - new_coords[:, 2] * s
    z_new = new_coords[:, 1] * s + new_coords[:, 2] * c
    new_coords[:, 1] = y_new
    new_coords[:, 2] = z_new
    
    new_coords[:, 0] += cx
    new_coords[:, 1] += cy
    new_coords[:, 2] += cz
    
    return GeometryData(new_coords, g.offsets.copy())


def rotate_y(g: GeometryData, angle_rad: float, center=(0, 0, 0)) -> GeometryData:
    """Y軸周りの回転変換を適用。"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    cx, cy, cz = center
    
    new_coords = g.coords.copy()
    new_coords[:, 0] -= cx
    new_coords[:, 1] -= cy
    new_coords[:, 2] -= cz
    
    # Y軸周りの回転行列を適用
    x_new = new_coords[:, 0] * c + new_coords[:, 2] * s
    z_new = -new_coords[:, 0] * s + new_coords[:, 2] * c
    new_coords[:, 0] = x_new
    new_coords[:, 2] = z_new
    
    new_coords[:, 0] += cx
    new_coords[:, 1] += cy
    new_coords[:, 2] += cz
    
    return GeometryData(new_coords, g.offsets.copy())


def rotate_xyz(g: GeometryData, rx: float, ry: float, rz: float, center=(0, 0, 0)) -> GeometryData:
    """XYZ軸周りの連続回転変換を適用。
    
    Args:
        g: 変換対象のGeometryData
        rx, ry, rz: 各軸の回転角度（ラジアン）
        center: 回転の中心点
        
    Returns:
        変換後の新しいGeometryData
    """
    result = g
    if rx != 0:
        result = rotate_x(result, rx, center)
    if ry != 0:
        result = rotate_y(result, ry, center)
    if rz != 0:
        result = rotate_z(result, rz, center)
    return result


def transform_combined(g: GeometryData, center=(0, 0, 0), scale_factors=(1, 1, 1), rotate_angles=(0, 0, 0)) -> GeometryData:
    """複合変換：スケール → 回転 → 移動を順次適用。
    
    Args:
        g: 変換対象のGeometryData
        center: 最終的な中心位置
        scale_factors: (sx, sy, sz) スケール係数
        rotate_angles: (rx, ry, rz) 回転角度（ラジアン）
        
    Returns:
        変換後の新しいGeometryData
    """
    result = g
    
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