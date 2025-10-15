"""
どこで: `api.sketch_runner.utils`（純粋関数/小ヘルパ）。
何を: FPS/キャンバス解決・投影行列・HUDメトリクス取得（spawn安全）を提供。
なぜ: `api.sketch` を薄く保ち、テスト容易性と再利用性を上げるため。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from util.constants import CANVAS_SIZES


def resolve_fps(requested_fps: int | None, *, default: int = 60) -> int:
    """FPS を解決して 1 以上の int を返す。

    - 明示指定があればそれを優先（数値化できない/<=0 は既定へ）。
    - それ以外は設定ファイルから読み取り、失敗時は既定値。
    """
    if requested_fps is not None:
        try:
            v = int(requested_fps)
            return max(1, v)
        except Exception:
            return max(1, int(default))
    try:
        from util.utils import load_config

        cfg = load_config() or {}
        ccfg = cfg.get("canvas_controller", {}) if isinstance(cfg, dict) else {}
        v = int(ccfg.get("fps", default))
        return max(1, v)
    except Exception:
        return max(1, int(default))


def resolve_canvas_size(canvas_size: str | tuple[int, int]) -> tuple[int, int]:
    """キャンバス [mm] を解決する。

    - 文字列: `CANVAS_SIZES` のキー（大文字/小文字は無視）
    - タプル: `(width_mm, height_mm)` をそのまま（正であることを検証）
    - それ以外/未知キーは `ValueError`
    """
    if isinstance(canvas_size, str):
        key = canvas_size.upper()
        if key not in CANVAS_SIZES:
            allowed = ", ".join(sorted(CANVAS_SIZES.keys()))
            raise ValueError(f"invalid canvas_size: {canvas_size}; allowed={allowed}")
        w, h = CANVAS_SIZES[key]
        return int(w), int(h)
    try:
        w, h = int(canvas_size[0]), int(canvas_size[1])  # type: ignore[index]
        if w <= 0 or h <= 0:
            raise ValueError(f"canvas_size must be positive, got: {(w, h)}")
        return w, h
    except Exception as e:  # noqa: BLE001 - 入力検証のため簡潔に
        raise ValueError(f"invalid canvas_size tuple: {canvas_size}") from e


def build_projection(canvas_width: float, canvas_height: float) -> "np.ndarray":
    """キャンバス mm を基準とする正射影行列（ModernGL 用の転置済み）を返す。"""
    proj = np.array(
        [
            [2 / canvas_width, 0, 0, -1],
            [0, -2 / canvas_height, 0, 1],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype="f4",
    ).T
    return proj


def hud_metrics_snapshot() -> dict[str, dict[str, int]]:
    """shape/effect のキャッシュ累計を取得する（HUD 用差分の材）。

    multiprocessing の spawn 方式でもシリアライズ可能なトップレベル関数として定義する。
    """
    try:
        from api.shapes import ShapesAPI as _ShapesAPI

        s_info: dict[str, Any] = _ShapesAPI.cache_info()
    except Exception:
        s_info = {"hits": 0, "misses": 0}
    try:
        from api.effects import global_cache_counters as _effects_counters

        e_info: dict[str, Any] = _effects_counters()
    except Exception:
        e_info = {"compiled": 0, "enabled": 0, "hits": 0, "misses": 0}
    return {
        "shape": {
            "hits": int(s_info.get("hits", 0)),
            "misses": int(s_info.get("misses", 0)),
        },
        "effect": {
            "compiled": int(e_info.get("compiled", 0)),
            "enabled": int(e_info.get("enabled", 0)),
            "hits": int(e_info.get("hits", 0)),
            "misses": int(e_info.get("misses", 0)),
        },
    }


__all__ = [
    "resolve_fps",
    "resolve_canvas_size",
    "build_projection",
    "hud_metrics_snapshot",
]
