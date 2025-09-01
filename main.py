import arc
import numpy as np

from api import E, G, GeometryAPI
from api.runner import run_sketch
from util.constants import CANVAS_SIZES


# カスタムエフェクト登録のデモ（名前省略: 関数名が登録名に）
@E.register()
def swirl(g, strength=1.0):
    """スワール効果 - 各頂点を個別に回転"""
    coords = g.coords.copy()
    # 各頂点のX座標に基づいて個別に回転角度を計算し、Z軸回転を適用
    for i in range(len(coords)):
        angle = coords[i, 0] * strength * 0.01
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x, y = coords[i, 0], coords[i, 1]
        coords[i, 0] = x * cos_a - y * sin_a
        coords[i, 1] = x * sin_a + y * cos_a
    return GeometryAPI.from_lines([coords[g.offsets[i] : g.offsets[i + 1]] for i in range(len(g.offsets) - 1)])


@E.register()
def wave(g, amplitude=10.0, frequency=0.1):
    """波効果 - 追加のカスタムエフェクト"""
    coords = g.coords.copy()
    coords[:, 2] += np.sin(coords[:, 0] * frequency) * amplitude
    return GeometryAPI.from_lines([coords[g.offsets[i] : g.offsets[i + 1]] for i in range(len(g.offsets) - 1)])


def draw(t, cc):
    """PyxiDraw次期API仕様の全機能デモ"""

    # === 1. 形状チェーン機能のデモ ===
    # 基本的な形状生成
    sphere = G.sphere(subdivisions=cc[1], sphere_type=cc[2]).size(80, 80, 80).at(50, 50, 0)

    polygon = G.polygon(n_sides=int(cc[3] * 8 + 3)).size(60, 60, 60).at(150, 50, 0)

    grid = G.grid(divisions=int(cc[4] * 10 + 5)).size(40, 40, 40).at(250, 50, 0)

    # === 2. from_lines機能のデモ ===
    # 手作業で線分を作成
    lines = [
        np.array([[0, 0, 0], [20, 0, 0]], dtype=np.float32),
        np.array([[20, 0, 0], [20, 20, 0]], dtype=np.float32),
        np.array([[20, 20, 0], [0, 20, 0]], dtype=np.float32),
        np.array([[0, 20, 0], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [20, 20, 0]], dtype=np.float32),  # 対角線
        np.array([[20, 0, 0], [0, 20, 0]], dtype=np.float32),  # 対角線
    ]
    custom_shape = G.from_lines(lines).at(50, 150, 0)

    # === 3. エフェクトチェーン機能のデモ ===
    # 標準エフェクト
    sphere_with_effects = E.add(sphere).noise(intensity=cc[5] * 0.5).filling(density=cc[6] * 0.8).result()

    # カスタムエフェクト
    polygon_with_swirl = E.add(polygon).swirl(strength=cc[7] * 2.0).wave(amplitude=cc[8] * 20.0).result()

    # === 4. apply機能のデモ（ワンショット効果） ===
    grid_with_apply = (
        E.add(grid)
        .apply(lambda g: g.rotate(z=cc[9] * 45))
        .apply(lambda g: g.spin(cc[10] * 180))
        .noise(intensity=0.1)
        .result()
    )

    # === 5. 形状結合のデモ ===
    # 複数の形状を結合
    combined_shape = sphere_with_effects + polygon_with_swirl + grid_with_apply + custom_shape

    # === 6. 複雑なチェーンのデモ ===
    complex_demo = G.torus().size(30, 30, 30).at(150, 150, 0).rotate(x=t * 30, y=t * 20)

    complex_demo = E.add(complex_demo).apply(lambda g: g.spin(t * 60)).swirl(strength=1.5).noise(intensity=0.2).result()

    # 最終的な形状を結合して返す
    final_result = combined_shape + complex_demo

    return final_result


if __name__ == "__main__":
    arc.start(midi=False)  # MIDIを無効化してテスト
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=8, background=(1, 1, 1, 1))
    arc.stop()
