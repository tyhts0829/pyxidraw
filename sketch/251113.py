from __future__ import annotations

"""
LazyGeometry の + 演算子挙動確認用スケッチ。

目的:
- Lazy 同士の `+` が遅延のまま結合され、描画結果として期待どおり連結されることを目視確認する。
- 実行時に `a + b + c` の多項結合を行い、パフォーマンス劣化が無いことを動作感で確認。
"""

from api import E, G, run
from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry


def draw(t: float) -> Geometry | LazyGeometry:
    # Lazy 生成（すべて LazyGeometry）
    a = G.polygon().scale(120, 120, 1).rotate(z=0.6 * t).translate(140, 140, 0)
    b = G.grid(nx=10, ny=10).scale(160, 160, 1).rotate(z=-0.3 * t).translate(140, 140, 0)
    c = G.polygon().scale(90, 90, 1).rotate(z=0.9 * t).translate(140, 140, 0)

    # Lazy + Lazy のみを使用（実体化しない）
    combo = a + b + c
    p0 = E.pipeline.translate().subdivide().displace()
    combo = p0(combo)
    return combo  # LazyGeometry を返しても Renderer 側で実体化される


if __name__ == "__main__":
    run(
        draw,
        canvas_size="A5",
        render_scale=6,
        use_midi=False,
        use_parameter_gui=True,
        workers=4,
        line_thickness=0.001,
    )
