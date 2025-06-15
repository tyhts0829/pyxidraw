from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect
from util.geometry import transform_to_xy_plane, transform_back


class Filling(BaseEffect):
    """閉じた形状をハッチングパターンで塗りつぶします。"""
    
    def apply(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """塗りつぶしエフェクトを適用します。
        
        Args:
            vertices_list: 入力頂点配列（閉じた形状を形成する必要がある）
            pattern: 塗りつぶしパターン ("lines", "cross", "dots") - デフォルト "lines"
            density: 塗りつぶし密度 (0.0-1.0) - デフォルト 0.1
            angle: パターンの角度（ラジアン） - デフォルト 0.0
            **params: 追加パラメータ
            
        Returns:
            元の形状と塗りつぶし線を含む塗りつぶし頂点配列
        """
        pattern = params.get('pattern', 'lines')
        density = params.get('density', 0.1)
        angle = params.get('angle', 0.0)
        
        if density <= 0:
            return vertices_list.copy()
        
        filled_results = []
        
        for vertices in vertices_list:
            if len(vertices) < 3:
                filled_results.append(vertices)
                continue
                
            # Add original shape
            filled_results.append(vertices)
            
            # Generate fill lines
            if pattern == "lines":
                fill_lines = self._generate_line_fill(vertices, density, angle)
            elif pattern == "cross":
                fill_lines = self._generate_cross_fill(vertices, density, angle)
            elif pattern == "dots":
                fill_lines = self._generate_dot_fill(vertices, density)
            else:
                fill_lines = self._generate_line_fill(vertices, density, angle)
            
            filled_results.extend(fill_lines)
        
        return filled_results
    
    def _generate_line_fill(self, vertices: np.ndarray, density: float, angle: float = 0.0) -> list[np.ndarray]:
        """平行線塗りつぶしパターンを生成します。"""
        # Transform to XY plane for easier processing
        vertices_2d, rotation_matrix, z_offset = transform_to_xy_plane(vertices)
        
        # Get 2D coordinates
        coords_2d = vertices_2d[:, :2]
        
        # Calculate bounding box
        min_x, min_y = np.min(coords_2d, axis=0)
        max_x, max_y = np.max(coords_2d, axis=0)
        
        # Calculate spacing based on density
        spacing = (max_y - min_y) * density / 10.0
        if spacing <= 0:
            return []
        
        # Generate horizontal lines
        y_values = np.arange(min_y, max_y, spacing)
        fill_lines = []
        
        for y in y_values:
            intersections = self._find_line_intersections(coords_2d, y)
            if len(intersections) >= 2:
                # Sort intersections and create line segments
                intersections.sort()
                for i in range(0, len(intersections) - 1, 2):
                    if i + 1 < len(intersections):
                        x1, x2 = intersections[i], intersections[i + 1]
                        line_2d = np.array([[x1, y], [x2, y]])
                        
                        # Convert back to 3D
                        line_3d = np.hstack([line_2d, np.zeros((2, 1))])
                        
                        # Apply rotation if needed
                        if angle != 0.0:
                            cos_a, sin_a = np.cos(angle), np.sin(angle)
                            center = np.mean(coords_2d, axis=0)
                            line_2d_centered = line_2d - center
                            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                            line_2d_rotated = line_2d_centered @ rot_matrix.T + center
                            line_3d[:, :2] = line_2d_rotated
                        
                        # Transform back to original orientation
                        line_final = transform_back(line_3d, rotation_matrix, z_offset)
                        fill_lines.append(line_final)
        
        return fill_lines
    
    def _generate_cross_fill(self, vertices: np.ndarray, density: float, angle: float = 0.0) -> list[np.ndarray]:
        """クロスハッチ塗りつぶしパターンを生成します。"""
        lines1 = self._generate_line_fill(vertices, density, angle)
        lines2 = self._generate_line_fill(vertices, density, angle + np.pi/2)
        return lines1 + lines2
    
    def _generate_dot_fill(self, vertices: np.ndarray, density: float) -> list[np.ndarray]:
        """ドット塗りつぶしパターンを生成します。"""
        # Transform to XY plane
        vertices_2d, rotation_matrix, z_offset = transform_to_xy_plane(vertices)
        coords_2d = vertices_2d[:, :2]
        
        # Calculate bounding box
        min_x, min_y = np.min(coords_2d, axis=0)
        max_x, max_y = np.max(coords_2d, axis=0)
        
        # Calculate spacing
        spacing = min(max_x - min_x, max_y - min_y) * density / 5.0
        if spacing <= 0:
            return []
        
        dots = []
        y = min_y
        while y <= max_y:
            x = min_x
            while x <= max_x:
                if self._point_in_polygon(coords_2d, [x, y]):
                    # Create a small dot (just a point for now)
                    dot_3d = np.array([[x, y, 0]])
                    dot_final = transform_back(dot_3d, rotation_matrix, z_offset)
                    dots.append(dot_final)
                x += spacing
            y += spacing
        
        return dots
    
    def _find_line_intersections(self, polygon: np.ndarray, y: float) -> list[float]:
        """水平線とポリゴンエッジの交点を検索します。"""
        intersections = []
        n = len(polygon)
        
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            
            # Check if line segment crosses the horizontal line
            if (p1[1] <= y < p2[1]) or (p2[1] <= y < p1[1]):
                # Calculate intersection x-coordinate
                if p2[1] != p1[1]:  # Avoid division by zero
                    x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                    intersections.append(x)
        
        return intersections
    
    def _point_in_polygon(self, polygon: np.ndarray, point: list[float]) -> bool:
        """レイキャスティングアルゴリズムを使用して点がポリゴン内部にあるかをチェックします。"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside