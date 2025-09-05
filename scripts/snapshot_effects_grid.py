"""
Headless snapshot generator for effects_grid_cc_explicit.

Renders the Geometry returned by `draw(t, cc)` to a PNG using Matplotlib (Agg),
without opening a window or requiring an OpenGL context.

Usage (from repo root):
    python scripts/snapshot_effects_grid.py --preset empty --out screenshots/grid_empty.png
    python scripts/snapshot_effects_grid.py --preset all07 --out screenshots/grid_all07.png

Notes:
    - Coordinates are in millimeters; we scale to pixels with `--scale` (default 6).
    - Y axis is flipped to match on-screen rendering (origin at top-left in output image).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

# Ensure Matplotlib uses a writable config dir to silence warnings in some envs
os.environ.setdefault("MPLCONFIGDIR", os.path.join("screenshots", ".mplconfig"))

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path
import sys

# Force repo root to top of sys.path to avoid picking up similarly-named modules
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.constants import CANVAS_SIZES
import effects_grid_cc_explicit as grid


def build_cc_map(preset: str) -> Dict[int, float]:
    if preset == "empty":
        return {}
    if preset == "all07":
        return {spec.cc: 0.7 for spec in grid.EFFECTS}
    if preset == "half":
        # Alternate 0.35 / 0.7 for some variety
        m: Dict[int, float] = {}
        for i, spec in enumerate(grid.EFFECTS):
            m[spec.cc] = 0.35 if (i % 2 == 0) else 0.7
        return m
    raise ValueError(f"Unknown preset: {preset}")


def render_geometry_to_png(
    out_path: str,
    *,
    preset: str = "empty",
    render_scale: int = 6,
    t_sec: float = 0.0,
    bgcolor=(1.0, 1.0, 1.0, 1.0),
    linecolor=(0, 0, 0),
    linewidth: float = 1.0,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    canvas_w, canvas_h = CANVAS_SIZES["SQUARE_300"]
    width_px = int(canvas_w * render_scale)
    height_px = int(canvas_h * render_scale)

    cc = build_cc_map(preset)
    geom = grid.draw(t_sec, cc)
    coords, offsets = geom.as_arrays(copy=False)

    # mm -> pixels, flip Y for conventional image coordinates
    x = coords[:, 0] * render_scale
    y = (canvas_h - coords[:, 1]) * render_scale

    # Matplotlib figure setup
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
    p.add_argument("--preset", choices=["empty", "all07", "half"], default="empty")
    p.add_argument("--out", default="screenshots/grid.png")
    p.add_argument("--scale", type=int, default=6)
    p.add_argument("--t", type=float, default=0.0, help="time in seconds")
    args = p.parse_args()

    out = render_geometry_to_png(
        args.out,
        preset=args.preset,
        render_scale=args.scale,
        t_sec=args.t,
    )
    print(out)


if __name__ == "__main__":
    main()
