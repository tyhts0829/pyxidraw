from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


class Desolve(BaseEffect):
    """線を個別のセグメントに分解/分割します。"""
    
    def apply(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """分解エフェクトを適用します。
        
        連続したポリラインを個別の線セグメントに分割します。
        
        Args:
            vertices_list: 入力頂点配列
            factor: 分解率 (0.0-1.0) - ランダム性を制御（基本実装では使用しない）
            seed: 再現可能性のためのランダムシード（基本実装では使用しない）
            **params: 追加パラメータ
            
        Returns:
            個別の線セグメントのリスト（各セグメントは2個の頂点を持つ）
        """
        factor = params.get('factor', 0.5)
        seed = params.get('seed', None)
        
        new_vertices_list = []
        
        for vertices in vertices_list:
            dissolved_segments = self._desolve_core(vertices, factor, seed)
            new_vertices_list.extend(dissolved_segments)
        
        return new_vertices_list
    
    def _desolve_core(self, vertices: np.ndarray, factor: float, seed: int | None) -> list[np.ndarray]:
        """ポリラインを個別の線セグメントに分割します。
        
        Args:
            vertices: ポリラインを形成する(N, 3)の頂点配列
            factor: 分解率（現在未使用、将来の拡張用）
            seed: ランダムシード（現在未使用、将来の拡張用）
            
        Returns:
            (2, 3)配列のリスト、各配列は1つの線セグメントを表す
        """
        # If vertices has 2 or fewer points, return as is
        if len(vertices) <= 2:
            return [vertices]
        
        # Break polyline into individual segments
        segments = []
        for i in range(len(vertices) - 1):
            segment = vertices[i:i+2].copy()
            segments.append(segment)
        
        return segments