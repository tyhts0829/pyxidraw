"""
Headless tuner for the `affine` effect (scale -> rotate around a pivot).

Places a single cube at canvas center, applies affine with configurable
pivot/angles/scale, and saves a PNG for quick visual comparison.

Usage:
  python scripts/tune_affine.py --out screenshots/affine_default.png
  python scripts/tune_affine.py --angle 0.35 --scale 1.1 0.85 1.0 \
      --out screenshots/affine_rot035_s110_085.png
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
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from api import E, G  # noqa: E402
from util.constants import CANVAS_SIZES  # noqa: E402


def build_geometry(angle_z: float, scale_vec: tuple[float, float, float]):
    cw, ch = CANVAS_SIZES["SQUARE_300"]
    cx, cy = cw / 2.0, ch / 2.0
    size = min(cw, ch) * 0.55
    base = G.polyhedron(polygon_type="cube")
    cube = base.scale(size, size, size).translate(cx, cy, 0.0)
    pipe = (
        E.pipeline
        .affine(pivot=(cx, cy, 0.0), angles_rad=(0.0, 0.0, angle_z), scale=scale_vec)
        .build()
    )
    return pipe(cube)


def render_geometry_to_png(geom, out_path: str, *, render_scale: int = 6):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    cw, ch = CANVAS_SIZES["SQUARE_300"]
    width_px = int(cw * render_scale)
    height_px = int(ch * render_scale)
    coords, offsets = geom.as_arrays(copy=False)
    x = coords[:, 0] * render_scale
    y = (ch - coords[:, 1]) * render_scale
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_px)
    ax.set_ylim(height_px, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e - s >= 2:
            ax.plot(x[s:e], y[s:e], color=(0, 0, 0), linewidth=1.2, solid_capstyle='round')
    fig.savefig(out_path, dpi=dpi, facecolor=(1, 1, 1, 1))
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--angle", type=float, default=0.35, help="Z angle [rad]")
    p.add_argument("--scale", nargs=3, type=float, default=(1.1, 0.85, 1.0))
    p.add_argument("--out", default="screenshots/affine.png")
    p.add_argument("--scale_px", type=int, default=6)
    args = p.parse_args()

    g = build_geometry(args.angle, tuple(args.scale))
    print(render_geometry_to_png(g, args.out, render_scale=args.scale_px))


if __name__ == "__main__":
    main()

