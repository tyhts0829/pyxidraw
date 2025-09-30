from __future__ import annotations

import sys
from pathlib import Path

# src/ レイアウトからシンプルに import できるよう、`python main.py` 実行時にパスを補助
sys.path.insert(0, str((Path(__file__).resolve().parent / "src")))

from api import E, G, cc, run  # type: ignore  # after sys.path tweak
from engine.core.geometry import Geometry  # type: ignore

CANVAS_SIZE = 400


def draw(t: float) -> Geometry:
    """デモ描画関数（MIDI は `api.cc` で制御）。"""
    # 破壊的変更に伴い、新しい引数へ移行
    # - font_size → em_size_mm（1em 高さ[mm]）
    # - align → text_align
    base = G.text(
        text="this is mukai syutoku",
        em_size_mm=20,
        text_align="center",
    ).translate(CANVAS_SIZE // 2, CANVAS_SIZE // 2, 0)
    pipe = E.pipeline.fill(angle_rad=cc[1] * 3.14, density=cc[2] * 50)
    return pipe(base)


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
