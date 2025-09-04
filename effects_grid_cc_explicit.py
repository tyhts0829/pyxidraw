"""
エフェクト・カタログ（明示版）

- 各セルに立方体を配置し、個別に指定したエフェクトを適用
- “主要パラメータ1つ” を CC でコントロール（固定割当）
- CC1/2/3 は全セルのビュー回転（Z 回りは CC3）

起動:
    python effects_grid_cc_explicit.py

備考:
- 依存の有無（例: shapely）によって一部エフェクトはフォールバック（元形状表示）
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, Mapping

from api import E, G, run
from common.logging import setup_default_logging
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES

logger = logging.getLogger(__name__)


# ---- CC 定義 ------------------------------------------------------------
CC_RX = 1  # 全体回転 X (0..1 -> 0..2π)
CC_RY = 2  # 全体回転 Y
CC_RZ = 3  # 全体回転 Z


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _grid_layout(n_items: int, canvas_w: float, canvas_h: float) -> tuple[int, int, float, float, float]:
    cols = int(math.ceil(math.sqrt(n_items))) if n_items > 0 else 1
    rows = int(math.ceil(n_items / cols)) if cols > 0 else 1
    cell_w = canvas_w / cols
    cell_h = canvas_h / rows
    cell_size = min(cell_w, cell_h) * 0.8
    return cols, rows, cell_w, cell_h, cell_size


# ---- エフェクト仕様（明示的に1つずつ） ---------------------------------
ApplyFn = Callable[[Geometry, float, float, float, float], Geometry]


@dataclass(frozen=True)
class EffectSpec:
    name: str
    cc: int
    apply: ApplyFn  # (g, cx, cy, cell_size, v_norm) -> Geometry


def _apply_translate(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    dx = (v - 0.5) * (s * 0.25)
    return (E.pipeline.translate(delta=(dx, 0.0, 0.0)).build())(g)


def _apply_rotate(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    ang = v * 2.0 * math.pi
    return (E.pipeline.rotate(pivot=(cx, cy, 0.0), angles_rad=(0.0, 0.0, ang)).build())(g)


def _apply_scale(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    k = 1.0 + (v - 0.5) * 1.5  # 約 0.25..1.75
    return (E.pipeline.scale(pivot=(cx, cy, 0.0), scale=(k, k, k)).build())(g)


def _apply_displace(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    amp = v * 5.0  # mm
    return (E.pipeline.displace(amplitude_mm=amp, spatial_freq=(0.5, 0.5, 0.5), t_sec=0.0).build())(g)


def _apply_fill(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    return (E.pipeline.fill(mode="lines", density=v, angle_rad=0.0).build())(g)


def _apply_repeat(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    count = int(round(v * 8))
    off = (s * 0.25, 0.0, 0.0)
    return (E.pipeline.repeat(count=count, offset=off, angles_rad_step=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), pivot=(cx, cy, 0.0)).build())(g)


def _apply_subdivide(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    return (E.pipeline.subdivide(subdivisions=v).build())(g)


def _apply_offset(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    try:
        return (E.pipeline.offset(distance=v, join="round", segments_per_circle=8).build())(g)
    except Exception as e:  # shapely 未導入など
        logger.debug("offset failed: %s", e)
        return g


def _apply_affine(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    ang = v * 2.0 * math.pi
    return (E.pipeline.affine(pivot=(cx, cy, 0.0), angles_rad=(0.0, 0.0, ang), scale=(1.0, 1.0, 1.0)).build())(g)


def _apply_extrude(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    return (E.pipeline.extrude(direction=(0.0, 0.0, 1.0), distance=v, scale=0.5, subdivisions=0.0, center_mode="origin").build())(g)


def _apply_collapse(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    return (E.pipeline.collapse(intensity=v * 5.0, subdivisions=0.3).build())(g)


def _apply_dash(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    dash_len = max(0.5, v * 10.0)
    return (E.pipeline.dash(dash_length=dash_len, gap_length=2.0).build())(g)


def _apply_ripple(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    return (E.pipeline.ripple(amplitude=v * 5.0, frequency=(0.1, 0.1, 0.1), phase=0.0).build())(g)


def _apply_wobble(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    return (E.pipeline.wobble(amplitude=v * 5.0, frequency=(0.1, 0.1, 0.1), phase=0.0).build())(g)


def _apply_explode(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    return (E.pipeline.explode(factor=v).build())(g)


def _apply_twist(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    return (E.pipeline.twist(angle=v * 180.0, axis="y").build())(g)


def _apply_trim(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    return (E.pipeline.trim(start_param=0.0, end_param=v).build())(g)


def _apply_weave(g: Geometry, cx: float, cy: float, s: float, v: float) -> Geometry:
    try:
        return (E.pipeline.weave(num_candidate_lines=0.2, relaxation_iterations=0.3, step=v).build())(g)
    except Exception as e:
        logger.debug("weave failed: %s", e)
        return g


EFFECTS: list[EffectSpec] = [
    EffectSpec("translate.delta.x", 20, _apply_translate),
    EffectSpec("rotate.angles_rad.z", 21, _apply_rotate),
    EffectSpec("scale.uniform", 22, _apply_scale),
    EffectSpec("displace.amplitude_mm", 23, _apply_displace),
    EffectSpec("fill.density", 24, _apply_fill),
    EffectSpec("repeat.count", 25, _apply_repeat),
    EffectSpec("subdivide.subdivisions", 26, _apply_subdivide),
    EffectSpec("offset.distance", 27, _apply_offset),
    EffectSpec("affine.rotateZ", 28, _apply_affine),
    EffectSpec("extrude.distance", 29, _apply_extrude),
    EffectSpec("collapse.intensity", 30, _apply_collapse),
    EffectSpec("dash.dash_length", 31, _apply_dash),
    EffectSpec("ripple.amplitude", 32, _apply_ripple),
    EffectSpec("wobble.amplitude", 33, _apply_wobble),
    EffectSpec("explode.factor", 34, _apply_explode),
    EffectSpec("twist.angle", 35, _apply_twist),
    EffectSpec("trim.end_param", 36, _apply_trim),
    EffectSpec("weave.step", 37, _apply_weave),
]


def _log_mapping() -> None:
    lines = ["[Effects CC Mapping (explicit)]"]
    lines.append(f"View Rotate CC: X={CC_RX}, Y={CC_RY}, Z={CC_RZ}")
    for e in EFFECTS:
        lines.append(f"CC {e.cc:>3}: {e.name}")
    logger.info("\n" + "\n".join(lines))


def draw(t: float, cc: Mapping[int, float]) -> Geometry:
    canvas_w, canvas_h = CANVAS_SIZES["SQUARE_300"]

    if not hasattr(draw, "_logged"):
        _log_mapping()
        setattr(draw, "_logged", True)

    cols, rows, cell_w, cell_h, cell_size = _grid_layout(len(EFFECTS), canvas_w, canvas_h)

    # ビュー回転（0..1 -> 0..2π）
    rx = _clamp01(cc.get(CC_RX, 0.0)) * 2.0 * math.pi
    ry = _clamp01(cc.get(CC_RY, 0.0)) * 2.0 * math.pi
    rz = _clamp01(cc.get(CC_RZ, 0.0)) * 2.0 * math.pi

    combined: Geometry | None = None

    for idx, spec in enumerate(EFFECTS):
        col = idx % cols
        row = idx // cols
        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h

        # 立方体（unit系をセルに合わせてスケール→配置）
        base = G.polyhedron(polygon_type="cube")
        g = base.scale(cell_size, cell_size, cell_size).translate(cx, cy, 0.0)

        # 各セルのプレビュー回転
        if rx or ry or rz:
            g = (E.pipeline.rotate(pivot=(cx, cy, 0.0), angles_rad=(rx, ry, rz)).build())(g)

        # エフェクト固有の CC を 0..1 で取得
        v = _clamp01(cc.get(spec.cc, 0.0))

        # 明示適用
        try:
            g2 = spec.apply(g, cx, cy, cell_size, v)
        except Exception as e:
            logger.debug("effect '%s' failed: %s", spec.name, e)
            g2 = g

        combined = g2 if combined is None else (combined + g2)

    return combined if combined is not None else G.empty()


if __name__ == "__main__":
    setup_default_logging()
    CANVAS = CANVAS_SIZES["SQUARE_300"]
    SCALE = 6
    env = os.environ.get("PYXIDRAW_USE_MIDI")
    USE_MIDI = True if env is None else (env == "1" or env.lower() in ("true", "on", "yes"))
    run(draw, canvas_size=CANVAS, render_scale=SCALE, background=(1, 1, 1, 1), use_midi=USE_MIDI)

