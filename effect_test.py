import arc

from api import E, G
from api.runner import run_sketch
from util.constants import CANVAS_SIZES


def draw(t, cc):
    """複数のエフェクトをテストするスケッチ"""

    array = G.polyhedron(12).size(100, 100, 100).at(100, 100, 0).rotate(cc[5], cc[5], cc[5])
    array = (
        E.add(array)
        .array(
            n_duplicates=cc[1], offset=(cc[2], cc[2], cc[2]), rotate=(cc[3], cc[3], cc[3]), scale=(cc[4], cc[4], cc[4])
        )
        .result()
    )
    return array


if __name__ == "__main__":
    arc.start(midi=False)  # MIDIを無効化してテスト
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=8, background=(1, 1, 1, 1))
    arc.stop()
