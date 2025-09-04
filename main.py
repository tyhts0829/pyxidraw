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
        .displace(intensity=cc.get(5, 0.3))
        .fill(density=cc.get(6, 0.6))
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
        .rotate(pivot=(150, 100, 0), angles_rad=(rx * 2 * np.pi, ry * 2 * np.pi, rz * 2 * np.pi))
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--midi", dest="midi", action="store_true", help="MIDI を有効化する（既定: 有効）")
    group.add_argument("--no-midi", dest="no_midi", action="store_true", help="MIDI を無効化する（ヘッドレス向け）")
    parser.add_argument("--midi-strict", action="store_true", help="MIDI 初期化に失敗したら終了する（公演/本番向け）")
    args = parser.parse_args()

    canvas = _parse_canvas(args.size)
    # 既定は True。環境変数/設定/CLI で上書き可能。
    cfg_enabled_default = None
    cfg_strict_default = None
    try:
        from util.utils import load_config  # noqa: WPS433

        cfg = load_config() or {}
        midi_cfg = cfg.get("midi", {}) if isinstance(cfg, dict) else {}
        if isinstance(midi_cfg, dict):
            if "enabled_default" in midi_cfg:
                cfg_enabled_default = bool(midi_cfg.get("enabled_default"))
            if "strict_default" in midi_cfg:
                cfg_strict_default = bool(midi_cfg.get("strict_default"))
    except Exception:
        pass

    # use_midi 決定
    env = os.environ.get("PYXIDRAW_USE_MIDI")
    if env is not None:
        use_midi = env == "1" or env.lower() in ("true", "on", "yes")
    elif cfg_enabled_default is not None:
        use_midi = cfg_enabled_default
    else:
        use_midi = True
    if getattr(args, "midi", False):
        use_midi = True
    if getattr(args, "no_midi", False):
        use_midi = False

    # midi_strict 決定
    env_strict = os.environ.get("PYXIDRAW_MIDI_STRICT")
    if getattr(args, "midi_strict", False):
        midi_strict_eff = True
    elif env_strict is not None:
        midi_strict_eff = env_strict == "1" or env_strict.lower() in ("true", "on", "yes")
    elif cfg_strict_default is not None:
        midi_strict_eff = cfg_strict_default
    else:
        midi_strict_eff = False

    logger.info("Launching demo: size=%s, scale=%d, midi=%s strict=%s", str(canvas), args.scale, use_midi, midi_strict_eff)
    run(
        draw,
        canvas_size=canvas,
        render_scale=args.scale,
        background=(1, 1, 1, 1),
        use_midi=use_midi,
        midi_strict=midi_strict_eff,
    )
