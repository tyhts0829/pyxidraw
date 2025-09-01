"""
純粋なデータコンテナとして機能するGeometryDataクラス。
API層やエフェクト層への依存を一切持たない。
"""

from __future__ import annotations
import numpy as np
import uuid
import hashlib
from typing import Tuple


class GeometryData:
    """頂点列とオフセット列だけを保持する純粋データコンテナ。"""
    
    __slots__ = ("coords", "offsets", "guid")
    
    def __init__(self, coords: np.ndarray, offsets: np.ndarray):
        """
        Args:
            coords: 頂点座標配列 shape: (N, 3), dtype=float32
            offsets: 線分オフセット配列 shape: (M+1,), dtype=int32
        """
        self.coords = coords.astype(np.float32, copy=False)
        self.offsets = offsets.astype(np.int32, copy=False)
        self.guid = uuid.uuid4()
    
    def copy(self) -> "GeometryData":
        """深いコピーを作成する。"""
        new_geom = GeometryData(self.coords.copy(), self.offsets.copy())
        new_geom.guid = self.guid  # 同じGUIDを保持
        return new_geom
    
    def concat(self, other: "GeometryData") -> "GeometryData":
        """別のGeometryDataと連結した新インスタンスを返す。"""
        if len(other.coords) == 0:
            return self.copy()
        if len(self.coords) == 0:
            return other.copy()
            
        offset_shift = self.coords.shape[0]
        new_coords = np.vstack([self.coords, other.coords])
        # 他方のオフセットを調整して連結
        adjusted_other_offsets = other.offsets[1:] + offset_shift
        new_offsets = np.hstack([self.offsets, adjusted_other_offsets])
        return GeometryData(new_coords, new_offsets)
    
    def as_arrays(self, *, copy: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """coords, offsetsをタプルで返す。
        
        Args:
            copy: Trueなら独立したコピーを返す
            
        Returns:
            tuple[np.ndarray, np.ndarray]: (coords, offsets)
        """
        if copy:
            return self.coords.copy(), self.offsets.copy()
        return self.coords, self.offsets
    
    def __len__(self) -> int:
        """線分の数を返す。"""
        return len(self.offsets) - 1 if len(self.offsets) > 0 else 0
    
    @property
    def num_points(self) -> int:
        """総頂点数を返す。"""
        return len(self.coords)
    
    @classmethod
    def from_lines(cls, lines: list[np.ndarray]) -> "GeometryData":
        """任意の線群を list[array] から構築。"""
        if not lines:
            return cls(np.empty((0, 3), dtype=np.float32), np.array([0], dtype=np.int32))
            
        offsets = np.empty(len(lines) + 1, np.int32)
        offsets[0] = 0
        coords = []
        
        for i, line in enumerate(lines, 1):
            line_array = np.asarray(line, dtype=np.float32)
            if line_array.ndim == 1:
                line_array = line_array.reshape(-1, 3)
            elif line_array.shape[1] != 3:
                # 2D座標の場合、z=0を追加
                if line_array.shape[1] == 2:
                    zeros = np.zeros((len(line_array), 1), dtype=np.float32)
                    line_array = np.hstack([line_array, zeros])
                else:
                    raise ValueError(f"Invalid coordinate shape: {line_array.shape}")
            
            offsets[i] = offsets[i - 1] + len(line_array)
            coords.append(line_array)
        
        if coords:
            all_coords = np.concatenate(coords, axis=0)
        else:
            all_coords = np.empty((0, 3), dtype=np.float32)
            
        return cls(all_coords, offsets)
    
    def is_empty(self) -> bool:
        """データが空かどうかを判定。"""
        return len(self.coords) == 0
    
    def get_hash(self) -> str:
        """GUIDベースのハッシュ値を返す（キャッシュキー用）。"""
        return hashlib.md5(str(self.guid).encode()).hexdigest()
    
    def __hash__(self) -> int:
        """ハッシュ値を返す（set, dictキー用）。"""
        return hash(self.guid)