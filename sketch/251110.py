from __future__ import annotations

from api import E, G, lfo, run

# PXD_DEBUG_PREFIX_CACHEをTrueに
CANVAS_SIZE = 400

osc = lfo(wave="sine", freq=0.1, octaves=4, persistence=0.5, lacunarity=2.0)


def draw(t: float):
    """clip エフェクトの最小使用例。

    - マスク: 閉曲線（ここでは 64 辺の多角形）
    - 対象: グリッド線
    - 内側のみ保持し、マスク自体も描画に含める
    """

    # 1) マスク（リング）と対象（グリッド）を用意（同一平面・同一座標系）
    mask = G.polygon()
    p1 = E.pipeline.scale(scale=(50, 50, 1.0)).affine().subdivide().displace()
    mask = p1(mask)

    # 2) 対象（グリッド）
    ring = G.polygon()
    p2 = (
        E.pipeline.scale(scale=(80, 80, 1.0))
        .affine()
        .displace()
        .scale(scale=(1, 1, 0))
        .fill()
        .clip(outline=mask)
    )

    dot = G.polygon()
    p3 = E.pipeline.scale(scale=(10, 10, 1.0)).affine().fill()
    return p2(ring), p3(dot)


if __name__ == "__main__":
    run(
        draw,
        canvas_size="A5",
        render_scale=6,
        use_midi=True,
        use_parameter_gui=True,
        workers=4,
        line_thickness=0.001,
        # show_hud=False,
    )
