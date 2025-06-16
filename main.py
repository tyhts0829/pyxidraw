import math

import arc
import numpy as np

from api import effects, shapes
from api.runner import run_sketch


def draw(t, cc) -> list[np.ndarray]:
    # Demonstrate new shape and effect system

    # Use polygon shape with number of sides controlled by MIDI
    n_sides = 3 + int(cc[1] * 10)  # 3-13 sides
    poly = shapes.polygon(n_sides=n_sides, center=(100, 100, 0), scale=(100, 100, 100))
    polyh = shapes.polyhedron(polygon_type="dodeca", center=(100, 150, 0), scale=(80, 80, 80), rotate=(cc[2], 0, 0))
    polyh = effects.filling(polyh, density=cc[3], angle=cc[4])
    # polyとpolyhを組み合わせて描画
    ret = []
    ret.extend(poly)
    ret.extend(polyh)
    return ret


def draw_complex_demo(t, cc) -> list[np.ndarray]:
    """Complex demo showcasing multiple shapes and effects."""
    all_vertices = []

    # Central polygon
    poly = shapes.polygon(n_sides=6)
    poly_bold = effects.boldify(poly, offset=0.5)
    poly_scaled = effects.scaling(poly_bold, uniform_scale=0.3)
    all_vertices.extend(poly_scaled)

    # Surrounding spheres with rotation
    for i in range(4):
        angle = i * math.pi / 2 + t * 0.2
        sphere = shapes.sphere(subdivisions=0.3)
        sphere_scaled = effects.scaling(sphere, uniform_scale=0.1)
        sphere_translated = effects.translation(
            sphere_scaled, offset_x=0.4 * math.cos(angle), offset_y=0.4 * math.sin(angle)
        )
        all_vertices.extend(sphere_translated)

    # Grid background
    grid = shapes.grid(n_divisions=(0.2, 0.2))
    grid_scaled = effects.scaling(grid, uniform_scale=1.5)
    grid_wobbled = effects.wobble(grid_scaled, amplitude=0.02, frequency=3.0)
    all_vertices.extend(grid_wobbled)

    return all_vertices


if __name__ == "__main__":
    arc.start(midi=True)

    # Choose which demo to run
    # Simple demo using MIDI controls
    run_sketch(draw, canvas_size="A4", render_scale=4, background=(1, 1, 1, 1))

    # Complex demo (uncomment to use)
    # run_sketch(draw_complex_demo, canvas_size="A4", render_scale=4, background=(1, 1, 1, 1))

    arc.stop()
