"""
Headless tuner for the `fill` effect.

Places a single cube at the canvas center and applies `fill` with
configurable params, exporting a PNG snapshot for visual inspection.

Usage (from repo root):
    python scripts/tune_fill.py --out screenshots/fill_default.png
    python scripts/tune_fill.py --mode lines --density 0.35 --angle 0.7854 \
        --out screenshots/fill_lines_d035_a45.png
    python scripts/tune_fill.py --mode cross --density 0.35 --out screenshots/fill_cross.png
    python scripts/tune_fill.py --mode dots  --density 0.35 --out screenshots/fill_dots.png

Notes:
    - Uses Matplotlib Agg backend (headless).
    - Coordinates are millimeters; `--scale` controls px/mm.
    - Y axis is flipped to match on-screen orientation (origin top-left).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Ensure Matplotlib uses a writable config dir
os.environ.setdefault("MPLCONFIGDIR", os.path.join("screenshots", ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root tops sys.path (sandbox-safe)
REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    if str(REPO_ROOT) in sys.path:
        sys.path.remove(str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT))
except Exception:
    pass

from api import E, G  # noqa: E402
from util.constants import CANVAS_SIZES  # noqa: E402


def build_geometry(mode: str, density: float, angle: float):
    canvas_w, canvas_h = CANVAS_SIZES["SQUARE_300"]
    cx, cy = canvas_w / 2.0, canvas_h / 2.0

    size = min(canvas_w, canvas_h) * 0.55
    base = G.polyhedron(polygon_type="cube")
    cube = base.scale(size, size, size).translate(cx, cy, 0.0)

    pipe = (E.pipeline.fill(mode=mode, density=density, angle_rad=angle).build())
    return pipe(cube)


def render_geometry_to_png(
    geom,
    out_path: str,
    *,
    render_scale: int = 6,
    bgcolor=(1.0, 1.0, 1.0, 1.0),
    linecolor=(0, 0, 0),
    linewidth: float = 1.0,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    canvas_w, canvas_h = CANVAS_SIZES["SQUARE_300"]
    width_px = int(canvas_w * render_scale)
    height_px = int(canvas_h * render_scale)

    coords, offsets = geom.as_arrays(copy=False)
    x = coords[:, 0] * render_scale
    y = (canvas_h - coords[:, 1]) * render_scale

    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_px)
    ax.set_ylim(height_px, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)

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
    p.add_argument("--mode", choices=["lines", "cross", "dots"], default="lines")
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--angle", type=float, default=0.0, help="radians")
    p.add_argument("--out", default="screenshots/fill.png")
    p.add_argument("--scale", type=int, default=6)
    args = p.parse_args()

    g = build_geometry(args.mode, args.density, args.angle)
    out = render_geometry_to_png(g, args.out, render_scale=args.scale)
    print(out)


if __name__ == "__main__":
    main()
