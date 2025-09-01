from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from util.geometry import transform_back, transform_to_xy_plane

from .base import BaseEffect
from .registry import effect


@effect
class Filling(BaseEffect):
    """閉じた形状をハッチングパターンで塗りつぶします。"""

    # 塗りつぶし線の最大密度（density=1.0のときの線間隔の係数）
    MAX_FILL_LINES = 100  # density=1.0のときに最大100本の線を生成

    def apply(
        self, 
        coords: np.ndarray, 
        offsets: np.ndarray,
        pattern: str = "lines",
        density: float = 0.5,
        angle: float = 0.0,
        **params: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """塗りつぶしエフェクトを適用します。

        Args:
            coords: 入力座標配列
            offsets: 入力オフセット配列（閉じた形状を形成する必要がある）
            pattern: 塗りつぶしパターン ("lines", "cross", "dots") - デフォルト "lines"
            density: 塗りつぶし密度 (0.0-1.0) - デフォルト 0.5
            angle: パターンの角度（ラジアン） - デフォルト 0.0
            **params: 追加パラメータ

        Returns:
            (new_coords, new_offsets): 元の形状と塗りつぶし線を含む座標配列とオフセット配列
        """

        if density <= 0:
            return coords.copy(), offsets.copy()

        filled_results = []
        
        # 既存の線を追加
        for i in range(len(offsets) - 1):
            vertices = coords[offsets[i]:offsets[i+1]]
            if len(vertices) < 3:
                filled_results.append(vertices)
                continue

            # 元の形状を追加
            filled_results.append(vertices)

            # 塗りつぶし線を生成
            if pattern == "lines":
                fill_lines = self._generate_line_fill(vertices, density, angle)
            elif pattern == "cross":
                fill_lines = self._generate_cross_fill(vertices, density, angle)
            elif pattern == "dots":
                fill_lines = self._generate_dot_fill(vertices, density)
            else:
                fill_lines = self._generate_line_fill(vertices, density, angle)

            filled_results.extend(fill_lines)

        # 結果を純粋なnumpy配列として構築
        if not filled_results:
            return coords.copy(), offsets.copy()
        
        # 全線を連結
        all_coords = np.vstack(filled_results)
        
        # 新しいオフセット配列を構築
        new_offsets = [0]
        current_offset = 0
        for line in filled_results:
            current_offset += len(line)
            new_offsets.append(current_offset)
        
        return all_coords, np.array(new_offsets, dtype=np.int32)

    def _generate_line_fill(self, vertices: np.ndarray, density: float, angle: float = 0.0) -> list[np.ndarray]:
        """平行線塗りつぶしパターンを生成します。"""
        # Transform to XY plane for easier processing
        vertices_2d, rotation_matrix, z_offset = transform_to_xy_plane(vertices)

        # Get 2D coords
        coords_2d = vertices_2d[:, :2]

        # Apply rotation to polygon if angle is specified
        if angle != 0.0:
            cos_a, sin_a = np.cos(-angle), np.sin(-angle)  # Inverse rotation for polygon
            center = np.mean(coords_2d, axis=0)
            coords_2d_centered = coords_2d - center
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            coords_2d = coords_2d_centered @ rot_matrix.T + center

        # Calculate bounding box
        _, min_y = np.min(coords_2d, axis=0)
        _, max_y = np.max(coords_2d, axis=0)

        # Calculate spacing based on density (inversed: 0=few lines, 1=many lines)
        # density=1.0 -> MAX_FILL_LINES lines, density=0.0 -> very few lines
        if density <= 0:
            return []

        # Calculate spacing: smaller spacing = more lines
        # At density=1.0, we want MAX_FILL_LINES lines in the bounding box
        # At density=0.1, we want fewer lines
        num_lines = max(2, int(self.MAX_FILL_LINES * density))
        spacing = (max_y - min_y) / num_lines
        if spacing <= 0:
            return []

        # Generate horizontal lines
        y_values = np.arange(min_y, max_y, spacing)
        fill_lines = []

        # Use batch processing for better performance
        intersection_results = generate_line_intersections_batch(coords_2d, y_values)

        for y, intersections in intersection_results:
            # Sort intersections and create line segments
            intersections_sorted = np.sort(intersections)
            for i in range(0, len(intersections_sorted) - 1, 2):
                if i + 1 < len(intersections_sorted):
                    x1, x2 = intersections_sorted[i], intersections_sorted[i + 1]
                    line_2d = np.array([[x1, y], [x2, y]])

                    # Apply forward rotation if needed
                    if angle != 0.0:
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        center = np.mean(vertices_2d[:, :2], axis=0)  # Use original center
                        line_2d_centered = line_2d - center
                        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                        line_2d = line_2d_centered @ rot_matrix.T + center

                    # Convert back to 3D
                    line_3d = np.hstack([line_2d, np.zeros((2, 1))])

                    # Transform back to original orientation
                    line_final = transform_back(line_3d, rotation_matrix, z_offset)
                    fill_lines.append(line_final)

        return fill_lines

    def _generate_cross_fill(self, vertices: np.ndarray, density: float, angle: float = 0.0) -> list[np.ndarray]:
        """クロスハッチ塗りつぶしパターンを生成します。"""
        lines1 = self._generate_line_fill(vertices, density, angle)
        lines2 = self._generate_line_fill(vertices, density, angle + np.pi / 2)
        return lines1 + lines2

    def _generate_dot_fill(self, vertices: np.ndarray, density: float) -> list[np.ndarray]:
        """ドット塗りつぶしパターンを生成します。"""
        # Transform to XY plane
        vertices_2d, rotation_matrix, z_offset = transform_to_xy_plane(vertices)
        coords_2d = vertices_2d[:, :2]

        # Calculate bounding box
        min_x, min_y = np.min(coords_2d, axis=0)
        max_x, max_y = np.max(coords_2d, axis=0)

        # Calculate spacing (inversed: 0=few dots, 1=many dots)
        if density <= 0:
            return []

        # Calculate spacing for dot grid
        # At density=1.0, we want many dots (MAX_FILL_LINES x MAX_FILL_LINES grid)
        # At density=0.1, we want fewer dots
        grid_size = max(2, int(self.MAX_FILL_LINES * density))
        spacing = min(max_x - min_x, max_y - min_y) / grid_size
        if spacing <= 0:
            return []

        # Pre-calculate grid points for better performance
        x_values = np.arange(min_x, max_x + spacing, spacing)
        y_values = np.arange(min_y, max_y + spacing, spacing)

        # Use batch processing for finding dots
        dot_points = find_dots_in_polygon(coords_2d, x_values, y_values)

        dots = []
        for i in range(len(dot_points)):
            # Create a small dot (just a point for now)
            dot_3d = np.array([[dot_points[i, 0], dot_points[i, 1], 0]])
            dot_final = transform_back(dot_3d, rotation_matrix, z_offset)
            dots.append(dot_final)

        return dots

    def _find_line_intersections(self, polygon: np.ndarray, y: float) -> list[float]:
        """水平線とポリゴンエッジの交点を検索します。"""
        intersections_array = find_line_intersections_njit(polygon, y)
        # Convert back to list and remove invalid values (-1)
        return [x for x in intersections_array if x != -1]

    def _point_in_polygon(self, polygon: np.ndarray, point: list[float]) -> bool:
        """レイキャスティングアルゴリズムを使用して点がポリゴン内部にあるかをチェックします。"""
        x, y = point
        return point_in_polygon_njit(polygon, x, y)


# Numba-compiled functions for performance
@njit(cache=True)
def find_line_intersections_njit(polygon: np.ndarray, y: float) -> np.ndarray:
    """水平線とポリゴンエッジの交点を検索します（Numba最適化版）。"""
    n = len(polygon)
    intersections = np.full(n, -1.0)  # Pre-allocate with invalid values
    count = 0

    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        # Check if line segment crosses the horizontal line
        if (p1[1] <= y < p2[1]) or (p2[1] <= y < p1[1]):
            # Calculate intersection x-coordinate
            if p2[1] != p1[1]:  # Avoid division by zero
                x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                intersections[count] = x
                count += 1

    return intersections[:count]


@njit(cache=True)
def point_in_polygon_njit(polygon: np.ndarray, x: float, y: float) -> bool:
    """レイキャスティングアルゴリズムを使用して点がポリゴン内部にあるかをチェックします（Numba最適化版）。"""
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0, 0], polygon[0, 1]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n, 0], polygon[i % n, 1]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@njit(cache=True)
def generate_line_intersections_batch(polygon: np.ndarray, y_values: np.ndarray) -> list:
    """複数のy値に対して交点を一括計算（Numba最適化版）。"""
    results = []
    for y in y_values:
        intersections = find_line_intersections_njit(polygon, y)
        if len(intersections) >= 2:
            results.append((y, intersections))
    return results


@njit(cache=True)
def find_dots_in_polygon(polygon: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """ポリゴン内部のグリッド点を高速に検索（Numba最適化版）。"""
    # Pre-allocate result array
    max_points = len(x_values) * len(y_values)
    points = np.empty((max_points, 2))
    count = 0

    for y in y_values:
        for x in x_values:
            if point_in_polygon_njit(polygon, x, y):
                points[count, 0] = x
                points[count, 1] = y
                count += 1

    return points[:count]
