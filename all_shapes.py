import math

import arc
import numpy as np

from api import shapes
from api.runner import run_sketch


def draw(t, cc) -> list[np.ndarray]:
    """Display all available shapes in a grid layout."""
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

    # Draw each shape in grid (4x4 = 16 shapes total)
    shape_index = 0

    # Row 1: Polygons
    for n_sides in [3, 4, 5, 6]:
        row = shape_index // cols
        col = shape_index % cols
        x = start_x + col * spacing_x
        y = start_y + row * spacing_y

        shape = shapes.polygon(
            n_sides=n_sides, scale=(25, 25, 25), rotate=(0, 0, t * 0.5 + shape_index * 0.1), center=(x, y, 0)
        )
        all_vertices.extend(shape)

        label = shapes.text(text=f"{n_sides}-gon", size=8, center=(x, y + 15, 0))
        all_vertices.extend(label)
        shape_index += 1

    # Row 2: 3D shapes
    shape_types = ["tetrahedron", "cube", "octahedron"]
    for shape_type in shape_types:
        row = shape_index // cols
        col = shape_index % cols
        x = start_x + col * spacing_x
        y = start_y + row * spacing_y

        shape = shapes.polyhedron(
            polygon_type=shape_type,
            scale=(25, 25, 25),
            rotate=(t * 0.3 + shape_index * 0.05, t * 0.4 + shape_index * 0.07, t * 0.5 + shape_index * 0.1),
            center=(x, y, 0),
        )
        all_vertices.extend(shape)

        label = shapes.text(text=shape_type, size=8, center=(x, y + 15, 0))
        all_vertices.extend(label)
        shape_index += 1

    # Add sphere to complete row 2
    row = shape_index // cols
    col = shape_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = shapes.sphere(
        subdivisions=0.3,
        scale=(25, 25, 25),
        rotate=(t * 0.3 + shape_index * 0.05, t * 0.4 + shape_index * 0.07, t * 0.5 + shape_index * 0.1),
        center=(x, y, 0),
    )
    all_vertices.extend(shape)

    label = shapes.text(text="sphere", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    shape_index += 1

    # Row 3: More 3D shapes
    # Torus
    row = shape_index // cols
    col = shape_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = shapes.torus(
        major_radius=10,
        minor_radius=4,
        rotate=(t * 0.3 + shape_index * 0.05, t * 0.4 + shape_index * 0.07, t * 0.5 + shape_index * 0.1),
        center=(x, y, 0),
    )
    all_vertices.extend(shape)

    label = shapes.text(text="torus", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    shape_index += 1

    # Cylinder
    row = shape_index // cols
    col = shape_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = shapes.cylinder(
        radius=15,
        height=30,
        rotate=(t * 0.3 + shape_index * 0.05, t * 0.4 + shape_index * 0.07, t * 0.5 + shape_index * 0.1),
        center=(x, y, 0),
    )
    all_vertices.extend(shape)

    label = shapes.text(text="cylinder", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    shape_index += 1

    # Cone
    row = shape_index // cols
    col = shape_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = shapes.cone(
        radius=20,
        height=30,
        rotate=(t * 0.3 + shape_index * 0.05, t * 0.4 + shape_index * 0.07, t * 0.5 + shape_index * 0.1),
        center=(x, y, 0),
    )
    all_vertices.extend(shape)

    label = shapes.text(text="cone", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    shape_index += 1

    # Capsule
    row = shape_index // cols
    col = shape_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = shapes.capsule(
        radius=12,
        height=25,
        segments=int(5 * math.sin(t)) + 32,
        latitude_segments=int(5 * math.sin(t + 0.5)) + 16,
        rotate=(t * 0.3 + shape_index * 0.05, t * 0.4 + shape_index * 0.07, t * 0.5 + shape_index * 0.1),
        center=(x, y, 0),
    )
    all_vertices.extend(shape)

    label = shapes.text(text="capsule", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    shape_index += 1

    # Row 4: Special shapes
    # Grid
    row = shape_index // cols
    col = shape_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = shapes.grid(
        n_divisions=(1, 1), scale=(25, 25, 25), rotate=(0, 0, t * 0.5 + shape_index * 0.1), center=(x, y, 0)
    )
    all_vertices.extend(shape)

    label = shapes.text(text="grid", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    shape_index += 1

    # Lissajous
    row = shape_index // cols
    col = shape_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = shapes.lissajous(
        freq_x=3, freq_y=2, points=500, scale=(25, 25, 25), rotate=(0, 0, t * 0.5 + shape_index * 0.1), center=(x, y, 0)
    )
    all_vertices.extend(shape)

    label = shapes.text(text="lissajous", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    shape_index += 1

    # Text
    row = shape_index // cols
    col = shape_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = shapes.text(text="HI", size=25, rotate=(0, 0, t * 0.5 + shape_index * 0.1), center=(x, y, 0))
    all_vertices.extend(shape)

    label = shapes.text(text="text", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)
    shape_index += 1

    # Asemic glyph
    row = shape_index // cols
    col = shape_index % cols
    x = start_x + col * spacing_x
    y = start_y + row * spacing_y

    shape = shapes.asemic_glyph(
        complexity=5, seed=42, scale=(25, 25, 25), rotate=(0, 0, t * 0.5 + shape_index * 0.1), center=(x, y, 0)
    )
    all_vertices.extend(shape)

    label = shapes.text(text="asemic", size=8, center=(x, y + 15, 0))
    all_vertices.extend(label)

    return all_vertices


if __name__ == "__main__":
    arc.start(midi=True)

    # Run the sketch
    run_sketch(draw, canvas_size="A4_LANDSCAPE", render_scale=4, background=(1, 1, 1, 1))

    arc.stop()
