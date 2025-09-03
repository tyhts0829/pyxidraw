from __future__ import annotations

import os
import logging
from typing import Mapping

import numpy as np

from api import E, G, run
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES
from common.logging import setup_default_logging


def draw(t: float, cc: Mapping[int, float]) -> Geometry:
    # 基本形状
    sphere = G.sphere(subdivisions=cc.get(1, 0.5), sphere_type=cc.get(2, 0.5))
    sphere = sphere.scale(80, 80, 80).translate(50, 50, 0)

    polygon = G.polygon(n_sides=int(cc.get(3, 0.5) * 8 + 3)).scale(60, 60, 60).translate(150, 50, 0)

    grid = G.grid(divisions=int(cc.get(4, 0.5) * 10 + 5)).scale(40, 40, 40).translate(250, 50, 0)

    # from_lines デモ
    lines = [
        np.array([[0, 0, 0], [20, 0, 0]], dtype=np.float32),
        np.array([[20, 0, 0], [20, 20, 0]], dtype=np.float32),
        np.array([[20, 20, 0], [0, 20, 0]], dtype=np.float32),
        np.array([[0, 20, 0], [0, 0, 0]], dtype=np.float32),
    ]
    custom_shape = G.from_lines(lines).translate(50, 150, 0)

    # 新パイプライン（関数エフェクト）
    sphere2 = (
        E.pipeline
        .noise(intensity=cc.get(5, 0.3))
        .filling(density=cc.get(6, 0.6))
        .build()
    )(sphere)

    # 複合（単純に足し合わせ）
    combined = sphere2 + polygon + grid + custom_shape

    # 回転デモ
    rx = cc.get(7, 0.0)
    ry = cc.get(8, 0.0)
    rz = cc.get(9, 0.0)
    combined = (
        E.pipeline
        .rotation(center=(150, 100, 0), rotate=(rx, ry, rz))
        .build()
    )(combined)

    return combined


def _parse_canvas(size_str: str):
    if isinstance(size_str, str) and "x" in size_str.lower():
        w, h = size_str.lower().split("x")
        return int(w), int(h)
    return CANVAS_SIZES.get(size_str.upper(), CANVAS_SIZES["SQUARE_200"])


if __name__ == "__main__":
    import argparse

    setup_default_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="PyxiDraw demo launcher (API-first, thin CLI)")
    parser.add_argument("--size", default="SQUARE_200", help="キャンバスサイズキーまたは 'WxH'（例: 300x300）")
    parser.add_argument("--scale", type=int, default=8, help="レンダリング拡大率")
    parser.add_argument("--midi", action="store_true", help="MIDI を有効化する")
    args = parser.parse_args()

    canvas = _parse_canvas(args.size)
    use_midi = args.midi or os.environ.get("PYXIDRAW_USE_MIDI") == "1"

    logger.info("Launching demo: size=%s, scale=%d, midi=%s", str(canvas), args.scale, use_midi)
    run(draw, canvas_size=canvas, render_scale=args.scale, background=(1, 1, 1, 1), use_midi=use_midi)
