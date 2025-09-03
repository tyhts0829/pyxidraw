import arc

from api import E, G, run
from util.constants import CANVAS_SIZES


def draw(t, cc):
    """複数のエフェクトをテストするスケッチ（新API）"""

    # 形状生成と基本変換（Geometry メソッド）
    g = G.polyhedron(polygon_type=12).scale(100, 100, 100).translate(100, 100, 0)

    # 回転は正規化 0..1 → 0..2π（effects.rotation 内で変換）
    pipeline = (
        E.pipeline
        .rotation(center=(100, 100, 0), rotate=(cc[5], cc[5], cc[5]))
        .array(
            n_duplicates=cc[1],
            offset=(cc[2], cc[2], cc[2]),
            rotate=(cc[3], cc[3], cc[3]),
            scale=(cc[4], cc[4], cc[4]),
        )
        .build()
    )

    return pipeline(g)


if __name__ == "__main__":
    arc.start(midi=False)  # MIDIを無効化してテスト
    run(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=8, background=(1, 1, 1, 1))
    arc.stop()
