"""
Headless tuner for the `offset` effect (Shapely-based buffer).

Generates a PNG snapshot of a single cube placed at the canvas center
with the `offset` effect applied, to help decide sensible defaults.

Usage (from repo root):
    python scripts/tune_offset.py --out screenshots/offset_default.png
    python scripts/tune_offset.py --distance 0.2 --join round --segments 12 --out screenshots/offset_round.png
    python scripts/tune_offset.py --distance-mm 5.0 --join bevel --segments 16 --out screenshots/offset_bevel.png

Notes:
    - Requires Shapely. If Shapely is not available, the effect will likely be a no-op.
    - Uses Matplotlib Agg backend (no window / OpenGL required).
    - Coordinates are in millimeters; `--scale` controls px/mm.
    - Y axis is flipped to match on-screen orientation (origin top-left).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("screenshots", ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    if str(REPO_ROOT) in sys.path:
        sys.path.remove(str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT))
    os.makedirs('screenshots', exist_ok=True)
    with open('screenshots/tune_offset_debug.txt', 'w') as f:
        f.write('sys.path\n' + '\n'.join(sys.path) + '\n')
        f.write(f'REPO_ROOT={REPO_ROOT}\n')
except Exception:
    pass

from api import E, G  # noqa: E402
from util.constants import CANVAS_SIZES  # noqa: E402


def build_geometry(distance: float | None, distance_mm: float | None, join: str, segments: int):
    canvas_w, canvas_h = CANVAS_SIZES["SQUARE_300"]
    cx, cy = canvas_w / 2.0, canvas_h / 2.0

    size = min(canvas_w, canvas_h) * 0.5
    base = G.polyhedron(polygon_type="cube")
    cube = base.scale(size, size, size).translate(cx, cy, 0.0)

    kwargs = {"join": join, "segments_per_circle": int(segments)}
    if distance_mm is not None:
        kwargs["distance_mm"] = float(distance_mm)
    elif distance is not None:
        kwargs["distance"] = float(distance)

    pipeline = (E.pipeline.offset(**kwargs).build())
    return pipeline(cube)


def render_geometry_to_png(geom, out_path: str, *, render_scale: int = 6):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    canvas_w, canvas_h = CANVAS_SIZES["SQUARE_300"]
    width_px = int(canvas_w * render_scale)
    height_px = int(canvas_h * render_scale)
    coords, offsets = geom.as_arrays(copy=False)

    x = coords[:, 0] * render_scale
    y = (canvas_h - coords[:, 1]) * render_scale

    dpi = 100
    fig_w_in = width_px / dpi
    fig_h_in = height_px / dpi
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_px)
    ax.set_ylim(height_px, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e - s < 2:
            continue
        ax.plot(x[s:e], y[s:e], color=(0, 0, 0), linewidth=1.2, solid_capstyle='round')

    fig.savefig(out_path, dpi=dpi, facecolor=(1, 1, 1, 1))
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distance", type=float, default=0.2, help="normalized distance 0..1 (mapped to mm)")
    p.add_argument("--distance-mm", type=float, default=None, dest="distance_mm", help="absolute distance in mm (overrides --distance)")
    p.add_argument("--join", choices=["mitre", "round", "bevel"], default="round")
    p.add_argument("--segments", type=int, default=12, help="segments per circle (>=1)")
    p.add_argument("--out", default="screenshots/offset.png")
    p.add_argument("--scale", type=int, default=6)
    args = p.parse_args()

    geom = build_geometry(args.distance, args.distance_mm, args.join, args.segments)
    out = render_geometry_to_png(geom, args.out, render_scale=args.scale)
    print(out)


if __name__ == "__main__":
    main()

