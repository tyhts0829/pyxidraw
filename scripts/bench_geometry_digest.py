#!/usr/bin/env python3
"""
Geometry digest micro-benchmark.

Measures the overhead of pipeline hashing with and without Geometry.digest.

Usage:
  python scripts/bench_geometry_digest.py --points 200000 --iters 200
  PXD_DISABLE_GEOMETRY_DIGEST=1 python scripts/bench_geometry_digest.py

Notes:
  - When PXD_DISABLE_GEOMETRY_DIGEST=1, api.pipeline falls back to per-call
    hashing of coords/offsets, which is slower for large geometries.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from api import E
from engine.core.geometry import Geometry


def make_geometry(points: int, num_lines: int = 1) -> Geometry:
    pts_per_line = points // num_lines
    coords_list = []
    offsets = [0]
    for i in range(num_lines):
        # simple polyline along x
        x = np.linspace(0, 100, pts_per_line, dtype=np.float32)
        y = np.full_like(x, i * 2, dtype=np.float32)
        z = np.zeros_like(x, dtype=np.float32)
        coords_list.append(np.stack([x, y, z], axis=1))
        offsets.append(offsets[-1] + pts_per_line)
    coords = np.vstack(coords_list)
    return Geometry(coords, np.asarray(offsets, dtype=np.int32))


def bench(points: int, iters: int) -> float:
    g = make_geometry(points)
    p = E.pipeline.cache(maxsize=8).build()  # identity pipeline with cache
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = p(g)
    t1 = time.perf_counter()
    return t1 - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=200_000)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    disabled = os.getenv("PXD_DISABLE_GEOMETRY_DIGEST") in ("1", "true", "TRUE", "True")
    sec = bench(args.points, args.iters)
    print(
        f"digest={'disabled' if disabled else 'enabled'} points={args.points} iters={args.iters} time={sec:.3f}s "
        f"({sec/args.iters*1000:.2f} ms/call)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

