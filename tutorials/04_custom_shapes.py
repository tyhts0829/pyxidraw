#!/usr/bin/env python3
"""
チュートリアル 04: カスタム形状の作成

独自の形状を定義して、形状レジストリに登録する方法を学びます。
BaseShapeクラスの継承とレジストリパターンの使用。
"""

import os
import logging
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
try:
    while REPO_ROOT in sys.path:
        sys.path.remove(REPO_ROOT)
except ValueError:
    pass
sys.path.insert(0, REPO_ROOT)

import numpy as np
from api import G, E
from api.runner import run_sketch
from engine.core.geometry import Geometry
from shapes.base import BaseShape
from shapes.registry import shape
from util.constants import CANVAS_SIZES
from common.logging import setup_default_logging


@shape
class Star(BaseShape):
    """
    星形のカスタム形状クラス
    """
    
    def __init__(self, points=5, inner_radius=0.5):
        """
        Args:
            points: 星の頂点数
            inner_radius: 内側の半径の比率（0-1）
        """
        super().__init__()
        self.points = points
        self.inner_radius = inner_radius
    
    def generate(self):
        """
        星形の頂点と線を生成
        
        Returns:
            Geometry: 星形のジオメトリ
        """
        vertices = []
        polylines: list[np.ndarray] = []
        
        # 角度の計算
        angle_step = 2 * np.pi / (self.points * 2)
        
        # 頂点を生成（外側と内側を交互に）
        for i in range(self.points * 2):
            angle = i * angle_step - np.pi / 2  # 上から開始
            
            if i % 2 == 0:
                # 外側の頂点
                radius = 1.0
            else:
                # 内側の頂点
                radius = self.inner_radius
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0.0
            
            vertices.append([x, y, z])
        
        # 輪郭線（閉ループ）
        outline = []
        for i in range(self.points * 2):
            outline.append(vertices[i])
        outline.append(vertices[0])
        polylines.append(np.array(outline, dtype=np.float32))
        
        # 中心から各頂点への線も追加（オプション）
        # 中心から放射線
        center = np.array([[0, 0, 0]], dtype=np.float32)
        for i in range(0, self.points * 2, 2):
            polylines.append(np.vstack([center, np.array([vertices[i]], dtype=np.float32)]))

        return Geometry.from_lines(polylines)


@shape
class Spiral(BaseShape):
    def generate(self, turns=3, points_per_turn=20, height=1.0):
        """螺旋形状を生成"""
        vertices = []
        lines = []

        total_points = turns * points_per_turn

        for i in range(total_points):
            # パラメータ t を 0 から 1 に正規化
            t = i / (total_points - 1)

            # 螺旋の計算
            angle = 2 * np.pi * turns * t
            radius = 0.5 * (1 - t * 0.5)  # 徐々に半径を小さく

            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = height * t - height / 2  # 高さ方向に移動

            vertices.append([x, y, z])

            # 前の点とつなぐ
            if i > 0:
                lines.append([i - 1, i])

        # ラインチェーンに変換
        polyline = np.array(vertices, dtype=np.float32)
        return Geometry.from_lines([polyline])


def draw(t, cc):
    """
    カスタム形状を使用した描画
    """
    combined = G.empty()
    
    # 1. カスタム星形を生成
    star5 = G.star(points=5, inner_radius=0.4)\
        .scale(80, 80, 80)\
        .translate(150, 150, 0)
    combined = combined + star5
    
    # 2. 8頂点の星
    star8 = G.star(points=8, inner_radius=0.6)\
        .scale(60, 60, 60)\
        .translate(250, 150, 0)
    combined = combined + star8
    
    # 3. 螺旋形状
    spiral = G.spiral(turns=4, points_per_turn=30, height=100)\
        .scale(100, 100, 100)\
        .translate(200, 250, 0)
    spiral = (
        E.pipeline
        .rotation(rotate=(30/360.0, (t * 0.5)/360.0, 0.0))
        .build()
    )(spiral)
    combined = combined + spiral
    
    return combined


def main():
    """メイン実行関数"""
    setup_default_logging()
    logger = logging.getLogger(__name__)
    logger.info("=== チュートリアル 04: カスタム形状の作成 ===")
    logger.info("独自の形状を定義して使用します：")
    logger.info("- Star: クラスベースのカスタム形状")
    logger.info("- Spiral: クラスベースのカスタム形状")
    logger.info("カスタム形状は G.star() や G.spiral() として利用可能")
    logger.info("終了するには Ctrl+C を押してください")
    
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    headless = args.headless or os.environ.get("PYXIDRAW_HEADLESS") == "1"

    if headless:
        g = draw(0, {})
        c, o = g.as_arrays()
        logger.info("Headless OK: points=%d, lines=%d", c.shape[0], max(0, o.shape[0]-1))
    else:
        run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_300"]) 


if __name__ == "__main__":
    main()
