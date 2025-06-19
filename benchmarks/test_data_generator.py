#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テストデータ生成モジュール

ベンチマーク用のテスト形状データを生成する責務を持つクラス。
"""

from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

# Type aliases
Vertices = NDArray[np.float32]  # Single array of shape (N, 3)
VerticesList = List[NDArray[np.float32]]  # List of vertex arrays


class TestDataGenerator:
    """ベンチマーク用テストデータ生成クラス"""

    def generate_test_shapes(self) -> Dict[str, VerticesList]:
        """テスト用の形状データセットを生成"""
        return {
            "small": [self.create_rectangle(1, 1)],
            "medium": [self.create_polygon(20)],
            "large": self.create_large_shape()
        }

    @staticmethod
    def create_rectangle(width: float, height: float) -> Vertices:
        """シンプルな長方形を作成（3D座標）"""
        hw, hh = width / 2, height / 2
        return np.array([
            [-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], 
            [-hw, hh, 0.0], [-hw, -hh, 0.0]
        ], dtype=np.float32)

    @staticmethod
    def create_polygon(n: int) -> Vertices:
        """n辺の正多角形を作成（3D座標）"""
        angles = np.linspace(0, 2 * np.pi, n + 1)
        return np.column_stack([
            np.cos(angles),
            np.sin(angles),
            np.zeros_like(angles)
        ]).astype(np.float32)

    @staticmethod
    def create_circle(radius: float, segments: int = 64) -> Vertices:
        """円を作成（3D座標）"""
        angles = np.linspace(0, 2 * np.pi, segments + 1)
        return np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros_like(angles)
        ]).astype(np.float32)

    def create_large_shape(self) -> VerticesList:
        """ベンチマーク用の大きく複雑な形状を作成（複数の3D配列のリスト）"""
        return [self.create_circle(1.0 + i * 0.1) for i in range(10)]