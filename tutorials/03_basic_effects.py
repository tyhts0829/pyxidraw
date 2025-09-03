#!/usr/bin/env python3
"""
チュートリアル 03: 基本的なエフェクトの適用

形状にエフェクトを適用して、変形や装飾を行います。
エフェクトチェーンの基本的な使い方を学びます。
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

from api import E, G
from api.runner import run_sketch
from util.constants import CANVAS_SIZES
from common.logging import setup_default_logging


def draw(t, cc):
    """
    エフェクトを適用した形状を生成
    
    Args:
        t: 時間（フレーム番号）
        cc: MIDIコントローラーの値
    
    Returns:
        Geometry: エフェクトを適用した形状
    """
    # 時間による動的な値を計算
    time_factor = t * 0.01
    
    # 基本となる立方体を生成
    base_shape = G.polyhedron(polygon_type="cube").scale(100, 100, 100).translate(200, 200, 0)
    
    # パイプラインで適用（ノイズ→回転→スケール）
    import math
    scale_factor = 1.0 + 0.2 * math.sin(time_factor)
    pipeline = (
        E.pipeline
        .noise(intensity=0.2)
        .rotation(rotate=((time_factor * 30)/360.0, (time_factor * 45)/360.0, 0.0))
        .scaling(scale=(scale_factor, scale_factor, scale_factor))
        .build()
    )
    return pipeline(base_shape)


def draw_comparison(t, cc):
    """
    エフェクト適用前後の比較表示
    """
    # 左側：エフェクトなし
    original = G.polyhedron(polygon_type="cube").scale(80, 80, 80).translate(150, 200, 0)
    
    # 右側：エフェクトあり
    effected = G.polyhedron(polygon_type="cube").scale(80, 80, 80).translate(250, 200, 0)
    effected = (
        E.pipeline
        .noise(intensity=0.3)
        .rotation(rotate=(45/360.0, 45/360.0, 0.0))
        .build()
    )(effected)
    
    # 両方を組み合わせて返す
    return original + effected


def main():
    """メイン実行関数"""
    setup_default_logging()
    logger = logging.getLogger(__name__)
    logger.info("=== チュートリアル 03: 基本的なエフェクトの適用 ===")
    logger.info("立方体に様々なエフェクトを適用します：")
    logger.info("- ノイズ：形状に揺らぎを追加")
    logger.info("- 回転：時間とともに回転")
    logger.info("- スケール：脈動するような拡大縮小")
    logger.info("- 細分化：滑らかな表面に変換")
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
        # 比較表示も可能（コメントを外して実行）
        # run_sketch(draw_comparison, canvas_size=CANVAS_SIZES["SQUARE_300"]) 


if __name__ == "__main__":
    main()
