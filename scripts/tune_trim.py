"""
Headless tuner for the `trim` effect.

Places a centered cube and trims each polyline to [start_param, end_param]
in normalized arc-length, exporting a PNG for quick inspection.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("screenshots", ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from api import E, G  # noqa: E402
from util.constants import CANVAS_SIZES  # noqa: E402


def build_geometry(start_param: float, end_param: float, *, subdivisions: float = 0.6, sphere_type: float = 0.1):
    cw, ch = CANVAS_SIZES["SQUARE_300"]
    cx, cy = cw / 2.0, ch / 2.0
    size = min(cw, ch) * 0.55
    # Use sphere to make trimmed segments more visually appealing
    base = G.sphere(subdivisions=subdivisions, sphere_type=sphere_type)
    cube = base.scale(size, size, size).translate(cx, cy, 0.0)
    pipe = (
        E.pipeline
        .trim(start_param=start_param, end_param=end_param)
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
    p.add_argument("--start", type=float, default=0.1)
    p.add_argument("--end", type=float, default=0.9)
    p.add_argument("--out", default="screenshots/trim.png")
    p.add_argument("--subdiv", type=float, default=0.6)
    p.add_argument("--stype", type=float, default=0.1, help="0..1 (sphere style)")
    p.add_argument("--scale_px", type=int, default=6)
    args = p.parse_args()

    g = build_geometry(args.start, args.end, subdivisions=args.subdiv, sphere_type=args.stype)
    print(render_geometry_to_png(g, args.out, render_scale=args.scale_px))


if __name__ == "__main__":
    main()
