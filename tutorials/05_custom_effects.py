#!/usr/bin/env python3
"""
チュートリアル 05: カスタムエフェクトの作成

独自のエフェクトを定義して、エフェクトチェーンに組み込む方法を学びます。
デコレータパターンとエフェクト登録の実装。
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
from api import G, E, run
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES
from effects.registry import effect
from common.logging import setup_default_logging


# wave / explode / twist は標準エフェクトとして effects パッケージに実装済み。
# このチュートリアルでは gradient のみ独自に登録します。


# カスタムエフェクト4: カラーグラデーション（頂点色は保持しないためデモ用にz偏移）
@effect()
def gradient(g: Geometry, *, color_start=[1, 0, 0], color_end=[0, 0, 1], axis: str = "y") -> Geometry:
    """
    形状にカラーグラデーションを適用
    
    Args:
        g: 入力ジオメトリ
        color_start: 開始色 [R, G, B]
        color_end: 終了色 [R, G, B]
        axis: グラデーションの方向
    
    Returns:
        Geometry: グラデーションに応じて z を微小変化（デモ用途）
    """
    coords, offsets = g.as_arrays(copy=False)
    
    # 軸のマッピング
    axis_map = {"x": 0, "y": 1, "z": 2}
    grad_axis = axis_map.get(axis, 1)
    
    # 範囲を取得
    min_val = np.min(coords[:, grad_axis])
    max_val = np.max(coords[:, grad_axis])
    range_val = max_val - min_val
    
    # カラー配列を初期化
    colors = np.zeros((len(coords), 3), dtype=np.float32)
    
    if range_val > 0:
        for i in range(len(coords)):
            # 位置を0-1に正規化
            t = (coords[i][grad_axis] - min_val) / range_val
            
            # 線形補間でカラーを計算
            colors[i] = (1 - t) * np.array(color_start) + t * np.array(color_end)
    
    # 色は保持しない実装のため、グラデーション強度で z を微小に変化させる例
    z_shift = (colors[:, 0] * 0.1).astype(np.float32)
    out = coords.copy()
    out[:, 2] += z_shift
    return Geometry(out, offsets.copy())


def draw(t, cc):
    """
    カスタムエフェクトを使用した描画
    """
    # 時間パラメータ
    time_factor = t * 0.01
    
    # ベースとなる形状（グリッド）
    base = G.grid(width=10, height=10).scale(150, 150, 150).translate(200, 200, 0)
    wave_amp = 0.1 + 0.05 * np.sin(time_factor)
    twist_angle = 30 * np.sin(time_factor * 0.5)
    explode_factor = 0.1 * (1 + np.sin(time_factor * 2))
    pipeline = (
        E.pipeline
        .wave(amplitude=wave_amp, frequency=3.0, axis="z")
        .twist(angle=twist_angle, axis="y")
        .explode(factor=explode_factor)
        .gradient(color_start=[1, 0, 0], color_end=[0, 0, 1], axis="y")
        .rotation(rotate=(0.0, (time_factor*20)/360.0, 0.0))
        .build()
    )
    return pipeline(base)


def draw_comparison(t, cc):
    """
    エフェクトの個別比較
    """
    combined = G.empty()
    
    # 元の形状
    original = G.polyhedron("cube").scale(50, 50, 50).translate(100, 100, 0)
    combined = combined + original
    
    # 波エフェクト
    wave = G.polyhedron("cube").scale(50, 50, 50).translate(200, 100, 0)
    wave = (E.pipeline.ripple(amplitude=0.2, frequency=2).build())(wave)
    combined = combined + wave
    
    # ツイストエフェクト
    twist = G.polyhedron("cube").scale(50, 50, 50).translate(300, 100, 0)
    twist = (E.pipeline.twist(angle=45).build())(twist)
    combined = combined + twist
    
    # 爆発エフェクト
    explode = G.polyhedron("cube").scale(50, 50, 50).translate(150, 200, 0)
    explode = (E.pipeline.explode(factor=0.3).build())(explode)
    combined = combined + explode
    
    # 複合エフェクト
    complex_fx = G.polyhedron("cube").scale(50, 50, 50).translate(250, 200, 0)
    complex_fx = (E.pipeline.ripple(amplitude=0.1).twist(angle=30).gradient().build())(complex_fx)
    combined = combined + complex_fx
    
    return combined


def main():
    """メイン実行関数"""
    setup_default_logging()
    logger = logging.getLogger(__name__)
    logger.info("=== チュートリアル 05: カスタムエフェクトの作成 ===")
    logger.info("独自のエフェクトを定義して適用します：")
    logger.info("- wave: 波状変形エフェクト")
    logger.info("- explode: 爆発エフェクト")
    logger.info("- twist: ねじれエフェクト")
    logger.info("- gradient: カラーグラデーション")
    logger.info("エフェクトは E.pipeline.ripple().twist() のように組み立て可能")
    logger.info("終了するには Ctrl+C を押してください")
    
    headless = os.environ.get("PYXIDRAW_HEADLESS") == "1"

    if headless:
        g = draw(0, {})
        c, o = g.as_arrays()
        logger.info("Headless OK: points=%d, lines=%d", c.shape[0], max(0, o.shape[0]-1))
    else:
        run(draw, canvas_size=CANVAS_SIZES["SQUARE_300"]) 
        # run(draw_comparison, canvas_size=CANVAS_SIZES["SQUARE_300"]) 


if __name__ == "__main__":
    main()
