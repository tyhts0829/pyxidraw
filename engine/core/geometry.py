"""
このライブラリの中核となるデータ構造（純粋データラッパー）
coords   : float32 ndarray  (N, 3)   # 全頂点を 1 本の連続メモリで保持
offsets  : int32   ndarray  (M+1,)   # 各線の開始 index（最後に N を追加）
lines[i] = coords[offsets[i] : offsets[i+1]]
"""

import numpy as np
from .geometry_data import GeometryData


class Geometry:
    """純粋なデータラッパー。内部ではGeometryDataを保持。"""
    
    def __init__(self, coords: np.ndarray, offsets: np.ndarray):
        self._data = GeometryData(coords, offsets)
    
    # プロパティとして提供
    @property
    def coords(self) -> np.ndarray:
        return self._data.coords
    
    @property
    def offsets(self) -> np.ndarray:
        return self._data.offsets

    # ── ファクトリ ───────────────────
    @classmethod
    def from_lines(cls, lines: list[np.ndarray]) -> "Geometry":
        """任意の線群を list[array] から構築。"""
        data = GeometryData.from_lines(lines)
        return cls.from_data(data)
    
    @classmethod
    def from_data(cls, data: GeometryData) -> "Geometry":
        """GeometryDataから薄いファクトリメソッド"""
        obj = cls.__new__(cls)
        obj._data = data
        return obj

    def map(self, fn) -> "Geometry":
        """座標配列に任意の関数 fn: (N,3)->(N,3) を適用"""
        new_coords = fn(self.coords)
        new_data = GeometryData(new_coords, self.offsets)
        return self.from_data(new_data)

    def as_arrays(self, *, copy: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """coords, offsets を返す。copy=True ならディープコピー。

        Args:
            copy: True なら独立したコピーを返す。False なら zero-copy view を返す。

        Returns:
            tuple[np.ndarray, np.ndarray]: (coords, offsets) のタプル

        Note:
            デフォルトの copy=False では、元の Geometry と同じメモリを共有する
            ビューを返すため、O(1) で取得でき、巨大データでも高速です。
            取得したビューを変更すると元の Geometry も変更されます。
        """
        return self._data.as_arrays(copy=copy)

    def __add__(self, other: "Geometry"):
        """他のGeometryと連結"""
        new_data = self._data.concat(other._data)
        return self.from_data(new_data)
    
    def __len__(self) -> int:
        """線分の数を返す"""
        return len(self._data)