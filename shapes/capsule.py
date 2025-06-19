from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from api.effects import scaling
from .base import BaseShape


class Capsule(BaseShape):
    """trimeshを使用したカプセル（回転スタジアム）形状生成器。"""
    
    _cached_unit_capsules: dict[tuple[int, int], list[np.ndarray]] = {}
    
    @classmethod
    def _generate_unit_capsule(cls, count: tuple[int, int]) -> list[np.ndarray]:
        """指定されたcount値でユニットカプセル（半径=0.5、高さ=1.0）を生成してキャッシュする。"""
        if count in cls._cached_unit_capsules:
            return cls._cached_unit_capsules[count]
        
        # 高さ=1.0、半径=0.5のtrimeshカプセルを作成
        mesh = trimesh.creation.capsule(height=1.0, radius=0.5, count=count)
        
        # ワイヤーフレームエッジを線として抽出
        vertices_list = []
        
        # メッシュエッジを取得して線分に変換
        edges = mesh.edges_unique
        for edge in edges:
            # このエッジの頂点を取得
            v1 = mesh.vertices[edge[0]]
            v2 = mesh.vertices[edge[1]]
            # 線分を作成
            line = np.array([v1, v2], dtype=np.float32)
            vertices_list.append(line)
        
        cls._cached_unit_capsules[count] = vertices_list
        return vertices_list
    
    def generate(self, radius: float = 0.2, height: float = 0.4,
                segments: int = 32, latitude_segments: int = 16, **params: Any) -> list[np.ndarray]:
        """カプセル形状を生成する。
        
        Args:
            radius: 半球の半径
            height: 円柱部分の高さ
            segments: 経度方向のセグメント数（周方向の分割数）
            latitude_segments: 緯度方向のセグメント数（半球の分割数）
            **params: 追加パラメータ（無視される）
            
        Returns:
            カプセル線の頂点配列リスト
        """
        # count値を設定（[経度, 緯度]）
        count = (segments, latitude_segments)
        
        # キャッシュからユニットカプセルを取得
        unit_capsule = self._generate_unit_capsule(count)
        
        # スケーリング係数を計算
        # ユニットカプセルは半径=0.5、高さ=1.0
        scale_xy = radius / 0.5  # 半径のスケール
        scale_z = height / 1.0   # 高さのスケール
        
        # api.effectsを使用してスケーリングを適用
        scaled_capsule = scaling(
            unit_capsule,
            scale_x=scale_xy,
            scale_y=scale_xy,
            scale_z=scale_z
        )
        
        return scaled_capsule