"""
Headless tuner for the `repeat` effect.

Generates a PNG snapshot of a single cube placed at the canvas center
with the `repeat` effect applied, to help decide sensible defaults.

Usage (from repo root):
    python scripts/tune_repeat.py --out screenshots/repeat_default.png
    python scripts/tune_repeat.py --count 4 --offset 12 0 0 --scale 1 1 1 --out screenshots/repeat_4x.png
    python scripts/tune_repeat.py --count 5 --offset 8 0 0 --angles 0 0 0.1745 --out screenshots/repeat_rot.png

Notes:
    - Uses Matplotlib Agg backend (no window / OpenGL required).
    - Coordinates are in millimeters; `--scale-px` controls px/mm.
    - Y axis is flipped to match on-screen orientation (origin top-left).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

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
    with open('screenshots/tune_repeat_debug.txt', 'w') as f:
        f.write('sys.path\n' + '\n'.join(sys.path) + '\n')
        f.write(f'REPO_ROOT={REPO_ROOT}\n')
except Exception:
    pass

from api import E, G  # noqa: E402
from util.constants import CANVAS_SIZES  # noqa: E402


def _vec3_arg(vals: Sequence[float], default: float) -> tuple[float, float, float]:
    if len(vals) == 0:
        v = float(default)
        return (v, v, v)
    if len(vals) == 1:
        v = float(vals[0])
        return (v, v, v)
    if len(vals) == 3:
        return (float(vals[0]), float(vals[1]), float(vals[2]))
    raise ValueError("vec3 args must be 0/1/3 values")


def build_geometry(count: int, offset: tuple[float, float, float], angles: tuple[float, float, float], scale: tuple[float, float, float], pivot: tuple[float, float, float]):
    canvas_w, canvas_h = CANVAS_SIZES["SQUARE_300"]
    cx, cy = canvas_w / 2.0, canvas_h / 2.0

    size = min(canvas_w, canvas_h) * 0.5
    base = G.polyhedron(polygon_type="cube")
    cube = base.scale(size, size, size).translate(cx, cy, 0.0)

    pipeline = (
        E.pipeline
        .repeat(count=count, offset=offset, angles_rad_step=angles, scale=scale, pivot=pivot)
        .build()
    )
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
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--offset", type=float, nargs='*', default=[12.0], help="vec3 mm; 0/1/3 values")
    p.add_argument("--angles", type=float, nargs='*', default=[0.0], help="vec3 radians; 0/1/3 values")
    p.add_argument("--scale", type=float, nargs='*', default=[1.0], help="vec3; 0/1/3 values")
    p.add_argument("--pivot", type=float, nargs='*', default=[0.0], help="vec3; 0/1/3 values")
    p.add_argument("--out", default="screenshots/repeat.png")
    p.add_argument("--scale-px", type=int, default=6)
    args = p.parse_args()

    offset = _vec3_arg(args.offset, 12.0)
    angles = _vec3_arg(args.angles, 0.0)
    scale = _vec3_arg(args.scale, 1.0)
    pivot = _vec3_arg(args.pivot, 0.0)

    geom = build_geometry(args.count, offset, angles, scale, pivot)
    out = render_geometry_to_png(geom, args.out, render_scale=args.scale_px)
    print(out)


if __name__ == "__main__":
    main()

