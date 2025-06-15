import math

import arc
import numpy as np

from api import effects, shapes
from api.runner import run_sketch


def draw(t, cc) -> list[np.ndarray]:
    """Display all available shapes in a grid layout."""
    all_vertices = []

    # Grid configuration (A4 LANDSCAPE: 297mm x 210mm)
    # Origin is top-left, x+ is right, y+ is down
    cols = 4
    rows = 4
    # MIDI control for grid spacing (cc[3] for X, cc[4] for Y)
    spacing_x = 50 + cc[3] * 20  # 50-70mm range
    spacing_y = 50 + cc[4] * 20  # 40-60mm range
    margin = 30  # margin from edges
    start_x = margin  # Start from left margin
    start_y = margin  # Start from top margin

    # List of shapes to display (scaled to fit in 70mm grid cells)
    shape_funcs = [
        # Row 1: Polygons
        lambda: shapes.polygon(n_sides=3, scale=(25, 25, 25)),
        lambda: shapes.polygon(n_sides=4, scale=(25, 25, 25)),
        lambda: shapes.polygon(n_sides=5, scale=(25, 25, 25)),
        lambda: shapes.polygon(n_sides=6, scale=(25, 25, 25)),
        # Row 2: 3D shapes
        lambda: shapes.sphere(subdivisions=0.3, scale=(25, 25, 25)),
        lambda: shapes.polyhedron(polygon_type="tetrahedron", scale=(25, 25, 25)),
        lambda: shapes.polyhedron(polygon_type="cube", scale=(25, 25, 25)),
        lambda: shapes.polyhedron(polygon_type="octahedron", scale=(25, 25, 25)),
        # Row 3: More 3D shapes
        lambda: shapes.torus(major_radius=20, minor_radius=8),
        lambda: shapes.cylinder(radius=15, height=30),
        lambda: shapes.cone(radius=20, height=30),
        lambda: shapes.capsule(radius=12, height=25),
        # Row 4: Special shapes
        lambda: shapes.grid(n_divisions=(3, 3), scale=(25, 25, 25)),
        lambda: shapes.lissajous(freq_x=3, freq_y=2, points=500, scale=(25, 25, 25)),
        lambda: shapes.text(text="HI", size=25),
        lambda: shapes.asemic_glyph(complexity=5, seed=42, scale=(25, 25, 25)),
    ]

    # Optional: Add attractor as a separate larger shape
    if cc[1] > 0.5:  # Toggle with MIDI control
        attractor = shapes.attractor(attractor_type="lorenz", points=5000, dt=0.005, scale=(2, 2, 2))
        for vertices in attractor:
            vertices[:, 0] += 240  # Move to right side
            vertices[:, 1] += 105  # Move to center vertically
        all_vertices.extend(attractor)

    # Draw each shape in grid
    for i, shape_func in enumerate(shape_funcs):
        row = i // cols
        col = i % cols
        x = start_x + col * spacing_x
        y = start_y + row * spacing_y

        # Create shape with center at grid position
        shape = shape_func()

        # Use center parameter to position the shape
        for vertices in shape:
            vertices[:, 0] += x  # Move X to grid position
            vertices[:, 1] += y  # Move Y to grid position

        # Optional: Add rotation animation
        if cc[2] > 0.1:  # Control rotation with MIDI
            shape = effects.rotation(shape, rotate_z=t * 0.5 + i * 0.1)

        all_vertices.extend(shape)

    return all_vertices


if __name__ == "__main__":
    arc.start(midi=True)

    # Run the sketch
    run_sketch(draw, canvas_size="A4_LANDSCAPE", render_scale=4, background=(1, 1, 1, 1))

    arc.stop()
