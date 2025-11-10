from __future__ import annotations


from api import E, G, lfo, run
from engine.core.geometry import Geometry

# PXD_DEBUG_PREFIX_CACHEをTrueに
CANVAS_SIZE = 400

osc = lfo(wave="sine", freq=0.1, octaves=4, persistence=0.5, lacunarity=2.0)


def draw(t: float) -> Geometry:
    """clip エフェクトの最小使用例。

    - マスク: 閉曲線（ここでは 64 辺の多角形）
    - 対象: グリッド線
    - 内側のみ保持し、マスク自体も描画に含める
    """

    # 1) マスク（リング）と対象（グリッド）を用意（同一平面・同一座標系）
    ring_inner = G.polygon()
    p1 = E.pipeline.scale().affine().displace()
    ring_inner = p1(ring_inner)

    # 2) 対象（グリッド）
    ring_outer = G.polygon()
    p2 = E.pipeline.scale().affine().displace().fill().clip(outline=ring_inner)
    return p2(ring_outer)


if __name__ == "__main__":
    run(
        draw,
        canvas_size="A5",
        render_scale=10,
        use_midi=True,
        use_parameter_gui=True,
        workers=4,
        line_thickness=0.001,
        # show_hud=False,
    )
