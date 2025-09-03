#!/usr/bin/env python3
"""
チュートリアル 01: 基本的な形状の生成

PyxiDrawの基本的な使い方を学びます。
シンプルな形状を生成して表示します。
"""
import os
import logging
import sys

# Ensure repository root is importable with highest precedence
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
    基本的な形状を生成する関数

    Args:
        t: 時間（フレーム番号）
        cc: MIDIコントローラーの値（辞書形式）

    Returns:
        Geometry: 描画する形状
    """
    # 1. 球体を生成
    # subdivisions: 細分化レベル（0-1の範囲）
    sphere = G.sphere(subdivisions=0.3)
    # 2. スケール（x, y, z）
    sphere = sphere.scale(100, 100, 100)
    # 3. 平行移動（x, y, z）
    sphere = sphere.translate(200, 200, 0)

    return sphere


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    setup_default_logging()
    logger = logging.getLogger(__name__)

    if args.headless or os.environ.get("PYXIDRAW_HEADLESS") == "1":
        g = draw(0, {})
        c, o = g.as_arrays()
        logger.info("Headless OK: points=%d, lines=%d", c.shape[0], max(0, o.shape[0] - 1))
    else:
        run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_300"])


if __name__ == "__main__":
    main()
