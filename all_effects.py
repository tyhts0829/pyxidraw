import math

import arc
import numpy as np

from api import effects, shapes
from api.runner import run_sketch


def draw(t, cc) -> list[np.ndarray]:
    """Display all available effects in a grid layout."""
    all_vertices = []

    # Grid configuration (A4 LANDSCAPE: 297mm x 210mm)
    # Origin is top-left, x+ is right, y+ is down
    cols = 4
    rows = 4
    canvas_width = 297
    canvas_height = 210
    # MIDI control for grid spacing (cc[3] for X, cc[4] for Y)
    spacing_x = 60 + cc[3] * 20  # 50-70mm range
    spacing_y = 40 + cc[4] * 20  # 40-60mm range

    # Calculate grid dimensions and center it
    grid_width = (cols - 1) * spacing_x
    grid_height = (rows - 1) * spacing_y
    start_x = (canvas_width - grid_width) / 2
    start_y = (canvas_height - grid_height) / 2

    # Base shape for effects demonstration
    def base_shape(x, y):
        return shapes.polyhedron(
            polygon_type=12,
            scale=(20, 20, 20),
            rotate=(2, 2, 2),
            center=(x, y, 0),
        )

    # Draw each effect in grid (4x4 = 16 effects total)
    effect_index = 0

    # Row 1: Basic transformations
    # Boldify
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = base_shape(x, y)
    shape = effects.boldify(shape)
    all_vertices.extend(shape)

    label = shapes.text(text="boldify", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    effect_index += 1

    # Rotation
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = base_shape(x, y)
    shape = effects.rotation(shape, center=(x, y, 0), rotate=(t * 0.5, t * 0.6, t * 0.7))
    all_vertices.extend(shape)

    label = shapes.text(text="rotation", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    effect_index += 1

    # Scaling
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = base_shape(x, y)
    shape = effects.scaling(shape, center=(x, y, 0), scale=(1 + 0.2 * math.sin(t), 1 + 0.2 * math.cos(t), 1))
    all_vertices.extend(shape)

    label = shapes.text(text="scaling", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    effect_index += 1

    # Row 2: Line modifications
    # Dashify
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = base_shape(x, y)
    shape = effects.dashify(shape, dash_length=0.08, gap_length=0.04)
    all_vertices.extend(shape)

    label = shapes.text(text="dashify", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    effect_index += 1

    # Noise
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = base_shape(x, y)
    shape = effects.noise(shape, intensity=1, frequency=0.5, t=t * 0.1)
    all_vertices.extend(shape)

    label = shapes.text(text="noise", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    effect_index += 1

    # Subdivision
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = base_shape(x, y)
    shape = effects.subdivision(shape, subdivisions=2, smoothing=0.3)
    all_vertices.extend(shape)

    label = shapes.text(text="subdivision", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    effect_index += 1

    # Wobble
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = base_shape(x, y)
    shape = effects.wobble(shape, amplitude=1, frequency=10, phase=t)
    all_vertices.extend(shape)

    label = shapes.text(text="wobble", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    effect_index += 1

    # Row 3: Advanced effects
    # Array
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = [shapes.polygon(n_sides=4, scale=(6, 6, 6), center=(x, y, 0))[0]]
    shape = effects.array(shape)
    all_vertices.extend(shape)

    label = shapes.text(text="array", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    effect_index += 1

    # Extrude
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = [shapes.polygon(n_sides=3, scale=(10, 10, 10), center=(x, y, 0))[0]]
    shape = effects.extrude(shape, direction=(0, 0, 1), distance=10)
    all_vertices.extend(shape)

    label = shapes.text(text="extrude", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    effect_index += 1

    # Filling
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = base_shape(x, y)
    shape = effects.filling(shape, pattern="lines", density=0.15, angle=math.pi / 4)
    all_vertices.extend(shape)

    label = shapes.text(text="filling", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    effect_index += 1

    # Trimming
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = base_shape(x, y)
    trim_start = 0.2 + 0.3 * math.sin(t)
    trim_end = 0.8 + 0.2 * math.cos(t * 0.7)
    shape = effects.trimming(shape, start_param=trim_start, end_param=trim_end)
    all_vertices.extend(shape)

    label = shapes.text(text="trimming", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    effect_index += 1

    # Row 4: Complex effects
    # Webify
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = base_shape(x, y)
    shape = effects.webify(shape, connection_probability=0.005, max_distance=0.001)
    all_vertices.extend(shape)

    label = shapes.text(text="webify", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    effect_index += 1

    # Desolve
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = base_shape(x, y)
    dissolve_factor = 0.3 + 0.3 * math.sin(t * 0.5)
    shape = effects.desolve(shape, factor=dissolve_factor, seed=123)
    all_vertices.extend(shape)

    label = shapes.text(text="desolve", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    effect_index += 1

    # Collapse
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = base_shape(x, y)
    collapse_factor = 0.5 + 0.4 * math.sin(t * 0.8)
    shape = effects.collapse(shape, center=(x, y, 0), factor=collapse_factor)
    all_vertices.extend(shape)

    label = shapes.text(text="collapse", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    effect_index += 1

    # Buffer
    row = effect_index // cols
    col = effect_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = [shapes.polygon(n_sides=3, scale=(12, 12, 12), center=(x, y, 0))[0]]
    buffer_distance = 3 + 2 * math.sin(t)
    shape = effects.buffer(shape, distance=buffer_distance, join_style="round")
    all_vertices.extend(shape)

    label = shapes.text(text="buffer", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)

    return all_vertices


if __name__ == "__main__":
    arc.start(midi=True)

    # Run the sketch
    run_sketch(draw, canvas_size="A4_LANDSCAPE", render_scale=8, background=(1, 1, 1, 1))

    arc.stop()
