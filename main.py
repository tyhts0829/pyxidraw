from __future__ import annotations

from api import E, G, lfo, run

CANVAS_SIZE = 400

osc = lfo(wave="sine", freq=0.1, octaves=4, persistence=0.5, lacunarity=2.0)


def draw(t: float):
    """デモ描画関数（MIDI は `api.cc` で制御）。"""
    base = G.text(em_size_mm=150).translate(CANVAS_SIZE // 2, CANVAS_SIZE // 2, 0)
    pipe = (
        E.affine()
        .fill()
        .subdivide()
        .displace(t_sec=osc(t * 0.01), spatial_freq=(osc(t * 0.05) + 1) * 0.02)
        .mirror()
        .rotate()
    )
    txt = G.text().scale(sx=10)
    e = E.affine().fill()

    geo = G.polygon().scale(50)
    p = E.affine()
    return pipe(base) + e(txt), p(geo)
    # return L(geometry=pipe(base)), L(geometry=e(txt))


if __name__ == "__main__":
    run(draw, canvas_size=(CANVAS_SIZE, CANVAS_SIZE), use_parameter_gui=True)
