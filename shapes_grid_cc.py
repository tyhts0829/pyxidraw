"""
MIDI CC で各シェイプを回転させつつ、登録済みシェイプをグリッド配置して表示。

- CC 1: rotate X (0.0–1.0 → 0–2π)
- CC 2: rotate Y (0.0–1.0 → 0–2π)
- CC 3: rotate Z (0.0–1.0 → 0–2π)

使い方（APIファースト、CLIなし）:
    from api import run
    run(draw, canvas_size=(300,300), render_scale=6, use_midi=False)

備考:
- 破壊的変更後のアーキテクチャに準拠（Geometry 統一 / E.pipeline）。
- MIDI 未接続環境ではデフォルトで無効。環境変数 `PYXIDRAW_USE_MIDI=1` で有効化可能。
"""

from __future__ import annotations

import logging
import math
import os
from typing import Mapping

from api import E, G, run
from common.logging import setup_default_logging
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES

logger = logging.getLogger(__name__)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _grid_layout(n_items: int, canvas_w: float, canvas_h: float) -> tuple[int, int, float, float, float]:
    """Compute grid columns/rows and cell metrics.

    Returns: (cols, rows, cell_w, cell_h, cell_size)
    """
    cols = int(math.ceil(math.sqrt(n_items))) if n_items > 0 else 1
    rows = int(math.ceil(n_items / cols)) if cols > 0 else 1
    cell_w = canvas_w / cols
    cell_h = canvas_h / rows
    # Use smaller dimension as the base, with margin
    cell_size = min(cell_w, cell_h) * 0.8
    return cols, rows, cell_w, cell_h, cell_size


def draw(t: float, cc: Mapping[int, float]) -> Geometry:
    logger.debug("CC: %s", cc)
    # Canvas setup
    canvas_w, canvas_h = CANVAS_SIZES["SQUARE_300"]

    # Rotation from MIDI CC (normalized 0..1)
    rx = _clamp01(cc.get(1, 0.0))
    ry = _clamp01(cc.get(2, 0.0))
    rz = _clamp01(cc.get(3, 0.0))

    # List all registered shapes via factory
    shape_names = sorted(G.list_shapes())
    if not shape_names:
        return G.empty()

    cols, rows, cell_w, cell_h, cell_size = _grid_layout(len(shape_names), canvas_w, canvas_h)

    combined: Geometry | None = None
    for idx, name in enumerate(shape_names):
        # Cell center
        col = idx % cols
        row = idx // cols
        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h

        # Build shape with defaults, scale to cell, place at center
        try:
            shape_fn = getattr(G, name)
        except AttributeError:
            # 未公開/未登録シェイプはスキップ
            continue

        # 既定引数なしで生成できないシェイプはスキップ
        try:
            base = shape_fn()
        except TypeError:
            # 必須引数がある場合
            continue
        except Exception:
            # 生成に失敗した場合
            continue

        g = base.scale(cell_size, cell_size, cell_size).translate(cx, cy, 0)

        # Rotate around each shape's own center using effect (0..1 → tau)
        rotated = (E.pipeline.rotation(center=(cx, cy, 0), rotate=(rx, ry, rz)).build())(g)

        combined = rotated if combined is None else (combined + rotated)

    return combined if combined is not None else G.empty()


if __name__ == "__main__":
    setup_default_logging()
    # デフォルト定数（チューニングしやすく）
    CANVAS = CANVAS_SIZES["SQUARE_300"]
    SCALE = 6
    USE_MIDI = os.environ.get("PYXIDRAW_USE_MIDI") == "1"
    run(draw, canvas_size=CANVAS, render_scale=SCALE, background=(1, 1, 1, 1), use_midi=USE_MIDI)
