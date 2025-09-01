from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
from common.cacheable_base import LRUCacheable


class BaseEffect(LRUCacheable, ABC):
    """すべてのエフェクトのベースクラス。キャッシング機能付きの変換処理を担当します。"""
    
    def __init__(self, maxsize: int = 128):
        super().__init__(maxsize=maxsize)
    
    @abstractmethod
    def apply(self, coords: np.ndarray, offsets: np.ndarray, **params: Any) -> tuple[np.ndarray, np.ndarray]:
        """座標とオフセット配列にエフェクトを適用します。
        
        Args:
            coords: 入力座標配列
            offsets: 入力オフセット配列
            **params: エフェクト固有のパラメータ
            
        Returns:
            (new_coords, new_offsets): 変換された座標配列とオフセット配列
        """
        pass
    
    def _execute(self, coords: np.ndarray, offsets: np.ndarray, **params: Any) -> tuple[np.ndarray, np.ndarray]:
        """実際の処理を実行（キャッシング用）"""
        return self.apply(coords, offsets, **params)
    
    def __call__(self, geometry_or_coords: Any, *maybe_offsets: Any, **params: Any):
        """Geometry もしくは (coords, offsets) に対して適用し、呼び出し元に合わせて返却。

        - Geometry を受け取った場合: Geometry を返す
        - (coords, offsets) を受け取った場合: (new_coords, new_offsets) を返す
        """
        # Geometry ライクか判定（duck typing）
        if hasattr(geometry_or_coords, "as_arrays") and callable(geometry_or_coords.as_arrays):
            geometry = geometry_or_coords
            coords, offsets = geometry.as_arrays(copy=False)
            # numpy.ndarray はハッシュ不可のため、ここではLRUを使わず直接実行
            new_coords, new_offsets = self._execute(coords, offsets, **params)

            from engine.core.geometry_data import GeometryData
            from engine.core.geometry import Geometry
            return Geometry.from_data(GeometryData(new_coords, new_offsets))

        # それ以外は (coords, offsets) として処理
        if not maybe_offsets:
            raise TypeError("Effect.__call__ expects (Geometry) or (coords, offsets)")
        coords = geometry_or_coords
        offsets = maybe_offsets[0]
        return super().__call__(coords, offsets, **params)
