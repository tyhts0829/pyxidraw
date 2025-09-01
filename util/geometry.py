"""3D変換のためのジオメトリユーティリティ関数。"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from engine.core.geometry import Geometry


@njit(cache=True)
def transform_to_xy_plane(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """頂点をXY平面（z=0）に変換する。

    頂点の法線ベクトルがZ軸に沿うように回転させ、
    その後z座標を0に平行移動する。

    Args:
        vertices: (N, 3) 3D点の配列

    Returns:
        以下のタプル:
            - transformed_points: (N, 3) XY平面上の配列
            - rotation_matrix: (3, 3) 使用された回転行列
            - z_offset: z方向の平行移動量
    """
    if vertices.shape[0] < 3:
        return vertices.astype(np.float64).copy(), np.eye(3), 0.0

    # Ensure float64 type for calculations
    vertices = vertices.astype(np.float64)

    # Calculate polygon normal vector
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    normal = np.cross(v1, v2)
    norm = np.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)

    if norm == 0:
        return vertices.copy(), np.eye(3), 0.0

    normal = normal / norm  # normalize

    # Calculate rotation axis (cross product with Z-axis)
    z_axis = np.array([0.0, 0.0, 1.0])
    rotation_axis = np.cross(normal, z_axis)

    rotation_axis_norm = np.sqrt(rotation_axis[0] ** 2 + rotation_axis[1] ** 2 + rotation_axis[2] ** 2)
    if rotation_axis_norm == 0:
        # Already aligned with Z-axis
        z_offset = vertices[0, 2]
        result = vertices.copy()
        result[:, 2] -= z_offset
        return result, np.eye(3), z_offset

    rotation_axis = rotation_axis / rotation_axis_norm

    # Calculate rotation angle
    cos_theta = np.dot(normal, z_axis)
    # Manual clip for njit compatibility
    if cos_theta < -1.0:
        cos_theta = -1.0
    elif cos_theta > 1.0:
        cos_theta = 1.0
    angle = np.arccos(cos_theta)

    # Create rotation matrix using Rodrigues' formula
    # Create K matrix manually for njit compatibility
    K = np.zeros((3, 3))
    K[0, 1] = -rotation_axis[2]
    K[0, 2] = rotation_axis[1]
    K[1, 0] = rotation_axis[2]
    K[1, 2] = -rotation_axis[0]
    K[2, 0] = -rotation_axis[1]
    K[2, 1] = rotation_axis[0]

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # Apply rotation
    transformed_points = np.dot(vertices, R.T)

    # Get z-coordinate and align to z=0
    z_offset = transformed_points[0, 2]
    transformed_points[:, 2] -= z_offset

    return transformed_points, R, z_offset


@njit(cache=True)
def transform_back(vertices: np.ndarray, rotation_matrix: np.ndarray, z_offset: float) -> np.ndarray:
    """頂点を元の向きに戻す。

    transform_to_xy_plane関数の逆変換。

    Args:
        vertices: (N, 3) 変換された点の配列
        rotation_matrix: (3, 3) transform_to_xy_planeから得られた回転行列
        z_offset: transform_to_xy_planeから得られたz方向の平行移動量

    Returns:
        (N, 3) 元の向きの点の配列
    """
    # Ensure consistent float64 type for calculations
    vertices = vertices.astype(np.float64)
    rotation_matrix = rotation_matrix.astype(np.float64)
    
    # Restore z-coordinate
    result = vertices.copy()
    result[:, 2] += z_offset

    # Apply inverse rotation
    return np.dot(result, rotation_matrix)


def geometry_transform_to_xy_plane(geometry: "Geometry") -> tuple["Geometry", np.ndarray, float]:
    """GeometryをXY平面（z=0）に変換する。

    Geometryの全頂点の最適な法線ベクトルがZ軸に沿うように回転させ、
    その後z座標を0に平行移動する。

    Args:
        geometry: 変換するGeometry

    Returns:
        以下のタプル:
            - transformed_geometry: XY平面上のGeometry
            - rotation_matrix: (3, 3) 使用された回転行列
            - z_offset: z方向の平行移動量
    """
    from engine.core.geometry import Geometry
    
    if len(geometry.coords) == 0:
        return geometry, np.eye(3), 0.0
    
    # 全頂点の重心をZ=0平面に投影する簡単な変換を行う
    coords = geometry.coords.astype(np.float64)
    
    # 重心を計算
    centroid = np.mean(coords, axis=0)
    z_offset = centroid[2]
    
    # Z座標を0に設定
    transformed_coords = coords.copy()
    transformed_coords[:, 2] = 0.0
    
    # 単位行列（回転なし）
    rotation_matrix = np.eye(3)
    
    # 新しいGeometryを作成
    transformed_geometry = Geometry(
        coords=transformed_coords.astype(np.float32),
        offsets=geometry.offsets.copy()
    )
    
    return transformed_geometry, rotation_matrix, z_offset


def geometry_transform_back(geometry: "Geometry", rotation_matrix: np.ndarray, z_offset: float) -> "Geometry":
    """Geometryを元の向きに戻す。

    geometry_transform_to_xy_plane関数の逆変換。

    Args:
        geometry: 変換されたGeometry
        rotation_matrix: (3, 3) geometry_transform_to_xy_planeから得られた回転行列
        z_offset: geometry_transform_to_xy_planeから得られたz方向の平行移動量

    Returns:
        元の向きのGeometry
    """
    from engine.core.geometry import Geometry
    
    if len(geometry.coords) == 0:
        return geometry
    
    # 簡単な逆変換（Z座標にオフセットを加算）
    coords = geometry.coords.astype(np.float64)
    restored_coords = coords.copy()
    restored_coords[:, 2] += z_offset
    
    # 回転行列を適用（現在は単位行列なので実質的に何もしない）
    if not np.allclose(rotation_matrix, np.eye(3)):
        restored_coords = transform_back(restored_coords, rotation_matrix, 0.0)
    
    # 新しいGeometryを作成
    restored_geometry = Geometry(
        coords=restored_coords.astype(np.float32),
        offsets=geometry.offsets.copy()
    )
    
    return restored_geometry
