from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

# src/ レイアウトからシンプルに import できるよう、`python main.py` 実行時にパスを補助
sys.path.insert(0, str((Path(__file__).resolve().parent / "src")))

from api import E, G, run  # type: ignore  # after sys.path tweak
from engine.core.geometry import Geometry  # type: ignore
from engine.ui.parameters.cc_binding import CC  # CCBinding を使用

CANVAS_SIZE = 400


def draw(t: float, cc: Mapping[int, float]) -> Geometry:
    # CCBinding を用いて scale.z を CC#2 にバインド（midi_override 優先で GUI の後も CC 主導）
    pipe = E.pipeline.affine(scale=(1, 1, CC(2))).extrude()
    poly = G.polyhedron().scale(200 * cc[1]).translate(CANVAS_SIZE // 2, CANVAS_SIZE // 2, 0)
    """デモ描画関数（MIDI 未接続でも安全に動作）。"""
    out = pipe(poly)
    # print(pipe.cache_info())
    # print(G.cache_info())
    # print(G.cache_info())
    return out


if __name__ == "__main__":
    run(
        draw,
        canvas_size=(CANVAS_SIZE, CANVAS_SIZE),
        render_scale=3,
        use_midi=True,
        use_parameter_gui=True,
        workers=6,
        line_thickness=0.001,
    )
