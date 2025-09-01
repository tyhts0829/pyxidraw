#!/usr/bin/env python3
"""
チュートリアル 01: 基本的な形状の生成

PyxiDrawの基本的な使い方を学びます。
シンプルな形状を生成して表示します。
"""
import os
import sys

# Ensure repository root is importable when running from tutorials/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from api import G
from api.runner import run_sketch
from util.constants import CANVAS_SIZES


def draw(t, cc):
    """
    基本的な形状を生成する関数

    Args:
        t: 時間（フレーム番号）
        cc: MIDIコントローラーの値（辞書形式）

    Returns:
        GeometryAPI: 描画する形状
    """
    # 1. 球体を生成
    # subdivisions: 細分化レベル（0-1の範囲）
    sphere = G.sphere(subdivisions=0.3)

    # 2. サイズを設定（x, y, z）
    sphere = sphere.size(100, 100, 100)

    # 3. 位置を設定（x, y, z）
    sphere = sphere.at(200, 200, 0)

    return sphere


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.headless or os.environ.get("PYXIDRAW_HEADLESS") == "1":
        g = draw(0, {})
        print("Headless OK: points=", len(g.coords), "lines=", len(g.offsets) - 1)
    else:
        run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_400"])


if __name__ == "__main__":
    main()
