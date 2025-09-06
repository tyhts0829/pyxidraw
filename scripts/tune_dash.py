"""
Headless tuner for the `dash` effect.

Generates a PNG snapshot of a single cube placed at the canvas center
with the `dash` effect applied, so we can judge good default params.

Usage (from repo root):
    python scripts/tune_dash.py --out screenshots/dash_default.png
    python scripts/tune_dash.py --dash_length 6 --gap_length 3 --out screenshots/dash_6_3.png

Notes:
    - Uses Matplotlib Agg backend (no window / OpenGL required).
    - Coordinates are in millimeters; `--scale` controls px/mm.
    - Y axis is flipped to match on-screen orientation (origin top-left).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Ensure Matplotlib uses a writable config dir to silence warnings in some envs
os.environ.setdefault("MPLCONFIGDIR", os.path.join("screenshots", ".mplconfig"))

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import numpy as np

# Add repo root to sys.path to allow `from api import ...`
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    if str(REPO_ROOT) in sys.path:
        sys.path.remove(str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT))
    os.makedirs('screenshots', exist_ok=True)
    with open('screenshots/tune_dash_debug.txt', 'w') as f:
        f.write('sys.path\n' + '\n'.join(sys.path) + '\n')
        f.write(f'REPO_ROOT={REPO_ROOT}\n')
except Exception:
    pass

from api import E, G  # noqa: E402
from util.constants import CANVAS_SIZES  # noqa: E402


def build_geometry(dash_length: float, gap_length: float) -> "Geometry":
    """Create a centered cube and apply dash(dash_length, gap_length)."""
    canvas_w, canvas_h = CANVAS_SIZES["SQUARE_300"]
    cx, cy = canvas_w / 2.0, canvas_h / 2.0

    size = min(canvas_w, canvas_h) * 0.5

    base = G.polyhedron(polygon_type="cube")
    cube = base.scale(size, size, size).translate(cx, cy, 0.0)

    pipeline = (
        E.pipeline
        .dash(dash_length=dash_length, gap_length=gap_length)
        .build()
    )
    return pipeline(cube)


def render_geometry_to_png(
    geom: "Geometry",
    out_path: str,
    *,
    render_scale: int = 6,
    bgcolor=(1.0, 1.0, 1.0, 1.0),
    linecolor=(0, 0, 0),
    linewidth: float = 1.2,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    canvas_w, canvas_h = CANVAS_SIZES["SQUARE_300"]
    width_px = int(canvas_w * render_scale)
    height_px = int(canvas_h * render_scale)

    coords, offsets = geom.as_arrays(copy=False)

    # mm -> pixels, flip Y for conventional image coordinates
    x = coords[:, 0] * render_scale
    y = (canvas_h - coords[:, 1]) * render_scale

    dpi = 100
    fig_w_in = width_px / dpi
    fig_h_in = height_px / dpi
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed
    ax.set_xlim(0, width_px)
    ax.set_ylim(height_px, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)

    # Draw each polyline
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e - s < 2:
            continue
        ax.plot(x[s:e], y[s:e], color=linecolor, linewidth=linewidth, solid_capstyle='round')

    fig.savefig(out_path, dpi=dpi, facecolor=bgcolor)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dash_length", type=float, default=6.0, help="dash length in mm (>0)")
    p.add_argument("--gap_length", type=float, default=3.0, help="gap length in mm (>=0)")
    p.add_argument("--out", default="screenshots/dash.png")
    p.add_argument("--scale", type=int, default=6)
    args = p.parse_args()

    geom = build_geometry(args.dash_length, args.gap_length)
    out = render_geometry_to_png(geom, args.out, render_scale=args.scale)
    print(out)


if __name__ == "__main__":
    main()

