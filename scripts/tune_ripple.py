"""
Headless tuner for the `ripple` effect.

Generates a PNG snapshot of a single cube placed at the canvas center
with the `ripple` effect applied (optionally preceded by subdivide),
so we can judge sensible default params.

Usage (from repo root):
    python scripts/tune_ripple.py --out screenshots/ripple_default.png
    python scripts/tune_ripple.py --amplitude 1.5 --frequency 0.03 --out screenshots/ripple_1p5_f003.png
    python scripts/tune_ripple.py --frequency 0.03 0.00 0.00 --out screenshots/ripple_xaxis.png

Notes:
    - Uses Matplotlib Agg backend (no window / OpenGL required).
    - Coordinates are in millimeters; `--scale` controls px/mm.
    - Y axis is flipped to match on-screen orientation (origin top-left).
    - For low-vertex lines, `--subdivisions`>0 improves curvature visibility.
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
    with open('screenshots/tune_ripple_debug.txt', 'w') as f:
        f.write('sys.path\n' + '\n'.join(sys.path) + '\n')
        f.write(f'REPO_ROOT={REPO_ROOT}\n')
except Exception:
    pass

from api import E, G  # noqa: E402
from util.constants import CANVAS_SIZES  # noqa: E402


def _normalize_frequency_arg(freq: Sequence[float]) -> float | tuple[float, float, float]:
    if len(freq) == 0:
        return 0.03
    if len(freq) == 1:
        return float(freq[0])
    if len(freq) == 3:
        return (float(freq[0]), float(freq[1]), float(freq[2]))
    raise ValueError("--frequency はスカラー（1つ）または3要素で指定してください")


def build_geometry(amplitude: float, frequency: float | tuple[float, float, float], phase: float, subdivisions: float) -> "Geometry":
    """Create a centered cube and apply [subdivide]->ripple(amplitude, frequency, phase)."""
    canvas_w, canvas_h = CANVAS_SIZES["SQUARE_300"]
    cx, cy = canvas_w / 2.0, canvas_h / 2.0

    size = min(canvas_w, canvas_h) * 0.5

    base = G.polyhedron(polygon_type="cube")
    cube = base.scale(size, size, size).translate(cx, cy, 0.0)

    pb = E.pipeline
    if subdivisions > 0:
        pb = pb.subdivide(subdivisions=subdivisions)
    pipeline = (
        pb
        .ripple(amplitude=amplitude, frequency=frequency, phase=phase)
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
    p.add_argument("--amplitude", type=float, default=1.5, help="displacement amplitude in mm (>=0)")
    p.add_argument("--frequency", type=float, nargs="*", default=[0.03], help="spatial frequency (cycles per unit). Provide one value for all axes, or three for (fx fy fz)")
    p.add_argument("--phase", type=float, default=0.0, help="phase in radians")
    p.add_argument("--out", default="screenshots/ripple.png")
    p.add_argument("--subdivisions", type=float, default=0.7, help="0..1 normalized; increases line vertex density before ripple")
    p.add_argument("--scale", type=int, default=6)
    args = p.parse_args()

    freq = _normalize_frequency_arg(args.frequency)
    geom = build_geometry(args.amplitude, freq, args.phase, args.subdivisions)
    out = render_geometry_to_png(geom, args.out, render_scale=args.scale)
    print(out)


if __name__ == "__main__":
    main()

