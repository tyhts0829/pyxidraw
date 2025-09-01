import arc

from api import E, G
from api.runner import run_sketch
from util.constants import CANVAS_SIZES


def draw(t, cc):
    """PyxiDraw次期API仕様の全機能デモ"""
    # デフォルト値を設定
    sphere_subdivisions = cc.get(1, 0.5)
    noise_intensity = cc.get(2, 0.3)
    rotation_val = cc.get(3, 0.1)
    sphere_type = cc.get(4, 0.5)

    sphere = (
        G.sphere(subdivisions=sphere_subdivisions, sphere_type=sphere_type)
        .size(80, 80, 80)
        .at(100, 100, 0)
        .rotate(x=rotation_val, y=rotation_val, z=rotation_val, center=(100, 100, 0))
    )
    sphere = (
        E.add(sphere)
        .noise(intensity=noise_intensity, time=t)
        .rotation(x=rotation_val, y=rotation_val, z=rotation_val, center=(100, 100, 0))
        .result()
    )
    return sphere


if __name__ == "__main__":
    # arc.start(midi=True)
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=6, background=(1, 1, 1, 1))
    # arc.stop()
