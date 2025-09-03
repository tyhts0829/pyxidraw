#!/usr/bin/env python3
"""
チュートリアル 02: 複数の形状の組み合わせ

複数の形状を組み合わせて、より複雑な構成を作ります。
異なる形状タイプの使い方を学びます。
"""

import os
import logging
import sys

# Ensure repository root import precedence
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
try:
    while REPO_ROOT in sys.path:
        sys.path.remove(REPO_ROOT)
except ValueError:
    pass
sys.path.insert(0, REPO_ROOT)

from api import G
from api.runner import run_sketch
from util.constants import CANVAS_SIZES
from common.logging import setup_default_logging


def draw(t, cc):
    """
    複数の形状を組み合わせて描画
    
    Args:
        t: 時間（フレーム番号）
        cc: MIDIコントローラーの値
    
    Returns:
        Geometry: 組み合わせた形状
    """
    # 空のジオメトリから開始
    combined = G.empty()
    
    # 1. 中央に球体
    sphere = G.sphere(subdivisions=0.4).scale(80, 80, 80).translate(200, 200, 0)
    combined = combined + sphere
    
    # 2. 正多面体（立方体）を左上に配置
    cube = G.polyhedron(polygon_type="cube").scale(60, 60, 60).translate(100, 100, 0)
    combined = combined + cube
    
    # 3. トーラス（ドーナツ型）を右上に配置
    torus = G.torus(
        major_radius=40,  # 大きい半径
        minor_radius=15,  # 小さい半径
        major_segments=16,
        minor_segments=32
    ).translate(300, 100, 0)
    combined = combined + torus
    
    # 4. 多角形（六角形）を下部に配置
    hexagon = G.polygon(n_sides=6).scale(70, 70, 70).translate(200, 300, 0)
    combined = combined + hexagon
    
    # 5. 円錐を左下に配置
    cone = G.cone(
        base_radius=40,
        height=80,
        radial_segments=20
    ).translate(100, 300, 0)
    combined = combined + cone
    
    # 6. シリンダーを右下に配置
    cylinder = G.cylinder(
        radius=30,
        height=80,
        radial_segments=16
    ).translate(300, 300, 0)
    combined = combined + cylinder
    
    return combined


def main():
    """メイン実行関数"""
    setup_default_logging()
    logger = logging.getLogger(__name__)
    logger.info("=== チュートリアル 02: 複数の形状の組み合わせ ===")
    logger.info("様々な基本形状を組み合わせて表示します：")
    logger.info("- 球体（中央）")
    logger.info("- 立方体（左上）")
    logger.info("- トーラス（右上）")
    logger.info("- 六角形（下中央）")
    logger.info("- 円錐（左下）")
    logger.info("- シリンダー（右下）")
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
