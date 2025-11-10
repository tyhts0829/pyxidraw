from __future__ import annotations

import os

from api import E, G, lfo, run
from engine.core.geometry import Geometry

# PXD_DEBUG_PREFIX_CACHEをTrueに
os.environ["PXD_DEBUG_GLOBAL"] = "1"
CANVAS_SIZE = 400

osc = lfo(wave="sine", freq=0.1, octaves=4, persistence=0.5, lacunarity=2.0)


def draw(t: float) -> Geometry:
    """clip エフェクトの最小使用例。

    - マスク: 閉曲線（ここでは 64 辺の多角形）
    - 対象: グリッド線
    - 内側のみ保持し、マスク自体も描画に含める
    """

    # 1) マスク（リング）と対象（グリッド）を用意（同一平面・同一座標系）
    mask = G.text().scale(20, 20, 1.0)
    p1 = E.pipeline.affine().translate()
    mask = p1(mask)

    # 2) 対象（グリッド）
    grid = G.grid().scale(200, 200, 1.0)

    # 3) clip を適用（内側保持 + マスクも描画に含める）
    # p2 = E.pipeline.affine().displace()
    p2 = E.pipeline.affine().subdivide().displace().clip(outline=[mask])
    clipped_grid = p2(grid)
    return clipped_grid


if __name__ == "__main__":
    run(
        draw,
        canvas_size=(CANVAS_SIZE, CANVAS_SIZE),
        render_scale=4.5,
        use_midi=True,
        use_parameter_gui=True,
        workers=4,
        line_thickness=0.001,
        # show_hud=False,
    )
