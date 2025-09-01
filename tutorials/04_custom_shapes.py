#!/usr/bin/env python3
"""
チュートリアル 04: カスタム形状の作成

独自の形状を定義して、形状レジストリに登録する方法を学びます。
BaseShapeクラスの継承とレジストリパターンの使用。
"""

import numpy as np
from api import G
from api.runner import run_sketch
from api.geometry_api import GeometryAPI
from shapes.base import BaseShape
from shapes.registry import shape
from util.constants import CANVAS_SIZES


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
            GeometryAPI: 星形のジオメトリ
        """
        vertices = []
        lines = []
        
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
        
        # 線を生成（星の輪郭）
        for i in range(self.points * 2):
            next_i = (i + 1) % (self.points * 2)
            lines.append([i, next_i])
        
        # 中心から各頂点への線も追加（オプション）
        center_index = len(vertices)
        vertices.append([0, 0, 0])  # 中心点
        
        for i in range(0, self.points * 2, 2):  # 外側の頂点のみ
            lines.append([center_index, i])
        
        # NumPy配列に変換
        vertices = np.array(vertices, dtype=np.float32)
        lines = np.array(lines, dtype=np.int32)
        
        # GeometryAPIオブジェクトを作成
        return GeometryAPI.from_vertices_and_faces(vertices, lines)


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

        vertices = np.array(vertices, dtype=np.float32)
        lines = np.array(lines, dtype=np.int32)

        return GeometryAPI.from_vertices_and_faces(vertices, lines)


def draw(t, cc):
    """
    カスタム形状を使用した描画
    """
    combined = G.empty()
    
    # 1. カスタム星形を生成
    star5 = G.star(points=5, inner_radius=0.4)\
        .size(80, 80, 80)\
        .at(150, 150, 0)
    combined = combined.add(star5)
    
    # 2. 8頂点の星
    star8 = G.star(points=8, inner_radius=0.6)\
        .size(60, 60, 60)\
        .at(250, 150, 0)
    combined = combined.add(star8)
    
    # 3. 螺旋形状
    spiral = G.spiral(turns=4, points_per_turn=30, height=100)\
        .size(100, 100, 100)\
        .at(200, 250, 0)
    
    # 螺旋に回転を適用
    from api import E
    spiral = E.add(spiral).rotate(x=30, y=t * 0.5).result()
    combined = combined.add(spiral)
    
    return combined


def main():
    """メイン実行関数"""
    print("=== チュートリアル 04: カスタム形状の作成 ===")
    print("独自の形状を定義して使用します：")
    print("- Star: クラスベースのカスタム形状")
    print("- Spiral: クラスベースのカスタム形状")
    print("\nカスタム形状は G.star() や G.spiral() として利用可能")
    print("\n終了するには Ctrl+C を押してください")
    
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    headless = args.headless or os.environ.get("PYXIDRAW_HEADLESS") == "1"

    if headless:
        g = draw(0, {})
        print("Headless OK: points=", len(g.coords), "lines=", len(g.offsets) - 1)
    else:
        import arc
        arc.start(midi=False)
        run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_400"])
        arc.stop()


if __name__ == "__main__":
    main()
