from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from util.geometry import transform_back, transform_to_xy_plane

from .base import BaseEffect
from .registry import effect


@effect
class Buffer(BaseEffect):
    """Shapelyを使用した高精度なバッファー/オフセット処理を行います。"""

    def apply(
        self,
        coords: np.ndarray,
        offsets: np.ndarray,
        distance: float = 0.5,
        join_style: float = 0.5,
        resolution: float = 0.5,
        **params: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """バッファーエフェクトを適用します。

        Shapelyライブラリを使用して高精度なバッファー処理を実行します。

        Args:
            coords: 入力座標配列
            offsets: 入力オフセット配列
            distance: バッファー距離 - デフォルト 0.5 (0.0-1.0レンジで25倍される)
            join_style: 角の接合スタイル（0.0-1.0でround/mitre/bevelを選択） - デフォルト 0.5
            resolution: バッファーの解像度（0.0-1.0で1-10の整数値に変換） - デフォルト 0.5
            **params: 追加パラメータ

        Returns:
            (buffered_coords, buffered_offsets): バッファー化された座標配列とオフセット配列
        """

        # 0.0-1.0 レンジを実際の距離値にスケール（25倍）
        actual_distance = distance * 25.0

        if actual_distance == 0:
            return coords.copy(), offsets.copy()

        # join_styleを文字列に変換
        join_style_str = self._determine_join_style(join_style)

        # 解像度を整数に変換（1-10）
        resolution_int = max(1, min(10, int(resolution * 10)))

        # 既存の線を取得
        vertices_list = []
        for i in range(len(offsets) - 1):
            vertices = coords[offsets[i] : offsets[i + 1]]
            if len(vertices) >= 2:
                vertices_list.append(vertices)

        # バッファー処理を実行
        new_vertices_list = self._buffer(vertices_list, actual_distance, join_style_str, resolution_int)

        # 空の配列や無効な配列を除外
        filtered_vertices_list = []
        for vertices in new_vertices_list:
            if vertices is not None and len(vertices) > 0 and isinstance(vertices, np.ndarray):
                # データ型を確実にfloat32に変換
                vertices = vertices.astype(np.float32)
                filtered_vertices_list.append(vertices)

        # 有効な頂点がない場合は元の配列を返す
        if not filtered_vertices_list:
            return coords.copy(), offsets.copy()

        # 結果を純粋なnumpy配列として構築
        all_coords = np.vstack(filtered_vertices_list)

        # 新しいオフセット配列を構築
        new_offsets = [0]
        current_offset = 0
        for line in filtered_vertices_list:
            current_offset += len(line)
            new_offsets.append(current_offset)

        return all_coords, np.array(new_offsets, dtype=np.int32)

    def _buffer(
        self,
        vertices_list: list[np.ndarray],
        distance: float,
        join_style: str,
        resolution: int,
    ) -> list[np.ndarray]:
        """Shapelyを使用してバッファー処理を実行します。"""
        if distance == 0:
            return vertices_list

        new_vertices_list = []

        for vertices in vertices_list:
            # 曲線を閉じる（始点と終点が近い場合）
            vertices = self._close_curve(vertices, 1e-3)

            # XY平面に変換
            vertices_on_xy, rotation_matrix, z_offset = transform_to_xy_plane(vertices)

            # ShapelyのLineStringを作成
            line = LineString(vertices_on_xy[:, :2])

            # バッファー処理
            buffered_line = line.buffer(distance, join_style=join_style, resolution=resolution)  # type: ignore

            if buffered_line.is_empty:
                continue

            # 結果を処理
            if isinstance(buffered_line, (LineString, MultiLineString)):
                new_vertices_list = self._extract_vertices_from_line(
                    new_vertices_list, buffered_line, rotation_matrix, z_offset
                )
            elif isinstance(buffered_line, (Polygon, MultiPolygon)):
                new_vertices_list = self._extract_vertices_from_polygon(
                    new_vertices_list, buffered_line, rotation_matrix, z_offset
                )

        # スケールを元に戻す（バッファーによる拡大を補正）
        scale_factor = 1 / (1 + distance * 2 / 25.0)  # 距離スケールを考慮
        new_vertices_list = self._scaling(new_vertices_list, scale_factor)

        # デバッグ: 戻り値の型チェック
        for i, item in enumerate(new_vertices_list):
            if not isinstance(item, np.ndarray):
                print(f"Error: new_vertices_list[{i}] is not np.ndarray, type: {type(item)}, value: {item}")

        return new_vertices_list

    def _extract_vertices_from_polygon(
        self, new_vertices_list: list, buffered_line: BaseGeometry, rotation_matrix: np.ndarray, z_offset: float
    ) -> list:
        """Polygonから頂点を抽出します。"""
        if isinstance(buffered_line, Polygon):
            polygons = [buffered_line]
        else:  # MultiPolygon
            from shapely.geometry import MultiPolygon

            polygons = buffered_line.geoms if isinstance(buffered_line, MultiPolygon) else []

        for poly in polygons:
            coords = np.array(poly.exterior.coords)
            new_vertices = np.hstack([coords, np.zeros((len(coords), 1))])

            # 元の3D空間に戻す
            restored_vertices = transform_back(new_vertices, rotation_matrix, z_offset)
            new_vertices_list.append(restored_vertices)

        return new_vertices_list

    def _extract_vertices_from_line(
        self, new_vertices_list: list, buffered_line: BaseGeometry, rotation_matrix: np.ndarray, z_offset: float
    ) -> list:
        """LineStringから頂点を抽出します。"""
        if isinstance(buffered_line, LineString):
            lines = [buffered_line]
        else:  # MultiLineString
            from shapely.geometry import MultiLineString

            lines = buffered_line.geoms if isinstance(buffered_line, MultiLineString) else []

        for line in lines:
            coords = np.array(line.coords)
            new_vertices = np.hstack([coords, np.zeros((len(coords), 1))])

            # 元の3D空間に戻す
            restored_vertices = transform_back(new_vertices, rotation_matrix, z_offset)
            new_vertices_list.append(restored_vertices)

        return new_vertices_list

    def _determine_join_style(self, join_style: float) -> str:
        """
        join_styleが0.0-1.0の値の場合、0.0〜0.33なら"round"、0.33〜0.67なら"mitre"、0.67〜1.0なら"bevel"
        """
        if 0.0 <= join_style < 0.33:
            return "mitre"
        elif 0.33 <= join_style < 0.67:
            return "round"
        elif 0.67 <= join_style <= 1.0:
            return "bevel"
        else:
            return "round"  # デフォルト

    def _close_curve(self, points: np.ndarray, threshold: float) -> np.ndarray:
        """始点と終点が近い場合、曲線を閉じます。"""
        if len(points) < 2:
            return points

        # 始点と終点の座標を取得
        start = points[0]
        end = points[-1]

        # 始点と終点の距離を計算
        dist = np.linalg.norm(start - end)

        # 距離がthreshold以下なら、終点を削除し、始点を終点として追加
        if dist <= threshold:
            points_copy = points[:-1]  # 終点を削除
            points = np.vstack([points_copy, start])  # 始点を終点として追加

        return points

    def _scaling(self, vertices_list: list[np.ndarray], scale_factor: float) -> list[np.ndarray]:
        """頂点リストに一様スケーリングを適用します。"""
        scaled_vertices_list = []
        for vertices in vertices_list:
            if len(vertices) == 0:
                scaled_vertices_list.append(vertices)
                continue

            # 重心を計算
            centroid = np.mean(vertices, axis=0)

            # 重心を中心にスケーリング
            scaled_vertices = (vertices - centroid) * scale_factor + centroid
            scaled_vertices_list.append(scaled_vertices)

        return scaled_vertices_list
