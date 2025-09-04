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

from api import G, run
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
    setup_default_logging()
    logger = logging.getLogger(__name__)
    # headless モード:
    #   - ウィンドウや OpenGL を起動せず、描画関数 `draw(t, cc)` だけを1回実行して
    #     Geometry の基本統計（頂点数・ライン数）をログに出します。
    #   - CI/サーバー（ディスプレイなし）や高速スモークチェック向けです。
    #   - 有効化方法: `PYXIDRAW_HEADLESS=1 python tutorials/01_basic_shapes.py`
    #   - 注意: 見た目（レンダリング品質やシェーダ）は検証しません。時間依存の確認は
    #           任意に `t` を変えて `draw(t, {})` を複数回呼び出してください。
    headless = os.environ.get("PYXIDRAW_HEADLESS") == "1"

    if headless:
        # ウィンドウを開かず、t=0・MIDI入力なしで一度だけ実行
        g = draw(0, {})
        c, o = g.as_arrays()
        logger.info("Headless OK: points=%d, lines=%d", c.shape[0], max(0, o.shape[0] - 1))
    else:
        # 通常モード: pyglet/m moderngl によるリアルタイム描画を起動
        run(draw, canvas_size=CANVAS_SIZES["SQUARE_300"]) 


if __name__ == "__main__":
    main()
