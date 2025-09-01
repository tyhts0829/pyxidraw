"""
GeometryAPI - メソッドチェーン対応の高レベルジオメトリAPI。
GeometryDataをラップし、直感的な変換メソッドを提供。
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from engine.core import transform_utils
from engine.core.geometry_data import GeometryData


class GeometryAPI:
    """GeometryDataのメソッドチェーン対応ラッパークラス。"""

    def __init__(self, data: GeometryData):
        """
        Args:
            data: ラップするGeometryData
        """
        self._data = data

    @property
    def data(self) -> GeometryData:
        """内部のGeometryDataを取得。"""
        return self._data

    @property
    def coords(self) -> np.ndarray:
        """頂点座標配列を取得。"""
        return self._data.coords

    @property
    def offsets(self) -> np.ndarray:
        """オフセット配列を取得。"""
        return self._data.offsets

    @property
    def guid(self) -> str:
        """GUIDを取得。"""
        return str(self._data.guid)

    def copy(self) -> "GeometryAPI":
        """深いコピーを作成。"""
        return GeometryAPI(self._data.copy())

    # === 変形メソッド ===

    def size(self, sx: float, sy: Optional[float] = None, sz: Optional[float] = None) -> "GeometryAPI":
        """拡大縮小変換。

        Args:
            sx: X軸スケール係数
            sy: Y軸スケール係数（Noneの場合はsxと同じ）
            sz: Z軸スケール係数（Noneの場合はsxと同じ）

        Returns:
            変換後の新しいGeometryAPI
        """
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx

        new_data = transform_utils.scale(self._data, sx, sy, sz)
        return GeometryAPI(new_data)

    def at(self, x: float, y: float, z: float = 0) -> "GeometryAPI":
        """平行移動変換。

        Args:
            x, y, z: 移動先座標

        Returns:
            変換後の新しいGeometryAPI
        """
        new_data = transform_utils.translate(self._data, x, y, z)
        return GeometryAPI(new_data)

    def rotate(self, x: float = 0, y: float = 0, z: float = 0, center=(0, 0, 0)) -> "GeometryAPI":
        """回転変換。

        Args:
            x, y, z: 各軸の回転角度（ラジアン）
            center: 回転中心

        Returns:
            変換後の新しいGeometryAPI
        """
        new_data = transform_utils.rotate_xyz(self._data, x, y, z, center)
        return GeometryAPI(new_data)

    def scale_at(self, factor: float, center=(0, 0, 0)) -> "GeometryAPI":
        """指定中心点でのスケーリング変換。

        Args:
            factor: スケール係数
            center: スケール中心点

        Returns:
            変換後の新しいGeometryAPI
        """
        new_data = transform_utils.scale_uniform(self._data, factor, center)
        return GeometryAPI(new_data)

    def translate(self, dx: float, dy: float, dz: float = 0) -> "GeometryAPI":
        """相対的な平行移動変換。

        Args:
            dx, dy, dz: 移動ベクトル

        Returns:
            変換後の新しいGeometryAPI
        """
        new_data = transform_utils.translate(self._data, dx, dy, dz)
        return GeometryAPI(new_data)

    # === 便利メソッド ===

    def spin(self, angle: float, center=(0, 0, 0)) -> "GeometryAPI":
        """Z軸周りの回転（degrees）。

        Args:
            angle: 回転角度（度）
            center: 回転中心

        Returns:
            変換後の新しいGeometryAPI
        """
        angle_rad = math.radians(angle)
        new_data = transform_utils.rotate_z(self._data, angle_rad, center)
        return GeometryAPI(new_data)

    def move(self, dx: float, dy: float, dz: float = 0) -> "GeometryAPI":
        """translateの別名。"""
        return self.translate(dx, dy, dz)

    def grow(self, factor: float) -> "GeometryAPI":
        """原点中心の拡大縮小。"""
        return self.size(factor)

    # === 静的メソッド ===

    @staticmethod
    def from_lines(lines: Sequence[Union[np.ndarray, List]]) -> "GeometryAPI":
        """線分リストからGeometryAPIを作成。

        Args:
            lines: 線分のリスト。各要素は(N, 2)または(N, 3)の配列

        Returns:
            作成されたGeometryAPI
        """
        # numpy配列のリストに変換
        np_lines = []
        for line in lines:
            if isinstance(line, (list, tuple)):
                line = np.array(line, dtype=np.float32)
            np_lines.append(line)

        data = GeometryData.from_lines(np_lines)
        return GeometryAPI(data)

    @staticmethod
    def empty() -> "GeometryAPI":
        """空のGeometryAPIを作成。"""
        empty_coords = np.empty((0, 3), dtype=np.float32)
        empty_offsets = np.array([0], dtype=np.int32)
        data = GeometryData(empty_coords, empty_offsets)
        return GeometryAPI(data)

    # === 結合・操作 ===

    def concat(self, other: "GeometryAPI") -> "GeometryAPI":
        """他のGeometryAPIと結合。

        Args:
            other: 結合するGeometryAPI

        Returns:
            結合されたGeometryAPI
        """
        new_data = self._data.concat(other._data)
        return GeometryAPI(new_data)

    def __add__(self, other: "GeometryAPI") -> "GeometryAPI":
        """+ 演算子で結合。"""
        return self.concat(other)

    # === 情報取得 ===

    def is_empty(self) -> bool:
        """空かどうかを判定。"""
        return self._data.is_empty()

    def num_points(self) -> int:
        """総頂点数を取得。"""
        return self._data.num_points

    def num_lines(self) -> int:
        """線分数を取得。"""
        return len(self._data)

    def bounds(self) -> tuple:
        """バウンディングボックスを取得。

        Returns:
            ((min_x, min_y, min_z), (max_x, max_y, max_z))
        """
        if self.is_empty():
            return ((0, 0, 0), (0, 0, 0))

        coords = self.coords
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        return tuple(min_coords), tuple(max_coords)

    def center(self) -> tuple:
        """中心点を取得。"""
        if self.is_empty():
            return (0, 0, 0)

        bounds_min, bounds_max = self.bounds()
        center_x = (bounds_min[0] + bounds_max[0]) / 2
        center_y = (bounds_min[1] + bounds_max[1]) / 2
        center_z = (bounds_min[2] + bounds_max[2]) / 2
        return (center_x, center_y, center_z)

    # === 変換チェーン用 ===

    def transform(self, func: Callable[["GeometryAPI"], "GeometryAPI"]) -> "GeometryAPI":
        """カスタム変換関数を適用。

        Args:
            func: GeometryAPIを受け取りGeometryAPIを返す関数

        Returns:
            変換後のGeometryAPI
        """
        return func(self)

    # === エフェクトメソッド（仕様書のapply例に対応） ===

    def noise(self, intensity: float = 0.1) -> "GeometryAPI":
        """ノイズエフェクトを適用（apply内での使用のため）。

        Args:
            intensity: ノイズ強度

        Returns:
            ノイズを適用したGeometryAPI
        """
        # E.add(self).noise(intensity).result()を内部で実行
        from . import E

        return E.add(self).noise(intensity).result()

    def __repr__(self) -> str:
        """文字列表現。"""
        return f"GeometryAPI(points={self.num_points()}, lines={self.num_lines()}, guid={self.guid[:8]}...)"

    def __str__(self) -> str:
        """簡潔な文字列表現。"""
        return f"Geometry({self.num_points()}pts, {self.num_lines()}lines)"
