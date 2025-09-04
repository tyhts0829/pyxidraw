"""
エフェクトのカタログ表示（CCで主要パラメータを個別操作）。

- 各セルに立方体（`G.polyhedron(polygon_type="cube")`）を配置
- レジストリ登録済みエフェクトを網羅的に適用（1セル=1エフェクト）
- 主要パラメータを1つ選んで CC に割当（自動選択＋ヒューリスティック）
- 参考: CC1,2,3 は全体プレビュー用の回転（各セルの中心回り）

起動例:
    python effects_grid_cc.py

環境変数:
    PYXIDRAW_USE_MIDI=0  # MIDIなしのヘッドレスにも対応（api.runの既定を踏襲）
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from inspect import signature, Parameter
from typing import Callable, Dict, Mapping, Tuple

from api import E, G, run
from common.logging import setup_default_logging
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES

# 明示 import でレジストリ初期化を確実化（api.pipeline内でも副作用importされるが保険）
import effects as _effects  # noqa: F401
from effects.registry import get_effect, list_effects


logger = logging.getLogger(__name__)


# ---- 定数 -------------------------------------------------------------
VIEW_CC_RX = 1
VIEW_CC_RY = 2
VIEW_CC_RZ = 3
EFFECT_CC_BASE = 20  # 各エフェクトに順番に割り当て


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _grid_layout(n_items: int, canvas_w: float, canvas_h: float) -> tuple[int, int, float, float, float]:
    cols = int(math.ceil(math.sqrt(n_items))) if n_items > 0 else 1
    rows = int(math.ceil(n_items / cols)) if cols > 0 else 1
    cell_w = canvas_w / cols
    cell_h = canvas_h / rows
    cell_size = min(cell_w, cell_h) * 0.8
    return cols, rows, cell_w, cell_h, cell_size


@dataclass
class ParamBinding:
    effect_name: str
    param_name: str
    cc_num: int
    # 正規化 0..1 を実パラメータに写像
    map_fn: Callable[[float, float], object]  # (v_norm, cell_size) -> value
    # エフェクト毎の追加引数（例: pivotなど）を注入したい場合に使用
    extra_kw: Dict[str, object]


def _default_mapper_for(param: str) -> Callable[[float, float], object]:
    """汎用ヒューリスティック。パラメータ名でレンジを決める。"""
    p = param.lower()

    if p in ("amplitude_mm",):
        return lambda v, _s: float(v) * 5.0  # 0..5 mm
    if p in ("amplitude", "intensity"):
        return lambda v, _s: float(v) * 5.0  # 0..5 mm 相当
    if p in ("distance_mm",):
        return lambda v, _s: float(v) * 15.0  # 0..15 mm
    if p in ("distance", "factor", "density", "subdivisions", "step", "num_candidate_lines", "relaxation_iterations"):
        return lambda v, _s: float(v)  # 規格化そのまま
    if p in ("end_param", "t_sec"):
        return lambda v, _s: float(v)
    if p in ("dash_length",):
        return lambda v, _s: max(0.5, float(v) * 10.0)  # mm
    if p in ("gap_length",):
        return lambda v, _s: max(0.25, float(v) * 6.0)  # mm
    if p in ("angle",):
        return lambda v, _s: float(v) * 180.0  # 度
    if p in ("angles_rad",):
        return lambda v, _s: (0.0, 0.0, float(v) * 2.0 * math.pi)  # Zのみ
    if p in ("scale",):
        return lambda v, _s: (1.0 + (float(v) - 0.5) * 1.5,) * 3  # 約 0.25..1.75
    if p in ("count",):
        return lambda v, _s: int(round(float(v) * 10))  # 0..10
    if p in ("delta",):
        return lambda v, s: ((float(v) - 0.5) * (s * 0.25), 0.0, 0.0)  # セル内移動

    # デフォルト: 0..1 のまま
    return lambda v, _s: float(v)


def _pick_primary_param(effect_name: str) -> str | None:
    """主要パラメータ候補を1つ選ぶ（メタ情報優先→関数シグネチャ）。"""
    fn = get_effect(effect_name)
    meta = getattr(fn, "__param_meta__", {}) or {}
    # 優先候補（経験則）
    preferred = [
        "amplitude_mm", "amplitude", "distance_mm", "distance", "factor", "density",
        "subdivisions", "angle", "angles_rad", "scale", "count", "dash_length",
        "end_param", "step", "num_candidate_lines", "relaxation_iterations", "delta",
    ]
    for k in preferred:
        if k in meta:
            t = meta[k].get("type") if isinstance(meta[k], dict) else None
            if t in (None, "number", "integer", "vec3"):
                return k

    # メタが無い/合わない場合はシグネチャから最初の keyword を拝借
    try:
        sig = signature(fn)
        for p in sig.parameters.values():
            if p.name == "g":
                continue
            if p.kind in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
                return p.name
    except Exception:
        pass
    return None


def _build_bindings(effect_names: list[str]) -> list[ParamBinding]:
    """エフェクト一覧から CC バインディングを生成。"""
    bindings: list[ParamBinding] = []
    for i, name in enumerate(effect_names):
        param = _pick_primary_param(name)
        if not param:
            logger.debug("skip effect (no suitable param): %s", name)
            continue

        map_fn = _default_mapper_for(param)
        cc_num = EFFECT_CC_BASE + i

        extra: Dict[str, object] = {}
        # 幾何がセル外へ飛ばないように中心ピボットを補完する系
        if name in ("rotate", "scale", "affine"):
            # pivot は実行時に (cx, cy, 0) を注入するので sentinel を置いておく
            extra["__inject_pivot__"] = True
        if name == "repeat":
            # 配列配置の見やすさ向上
            extra.setdefault("offset", (8.0, 0.0, 0.0))
        if name == "dash":
            # ダッシュ長を動かすときのギャップの既定
            extra.setdefault("gap_length", 2.0)
        if name == "displace":
            # 時間で揺れるタイプは t を渡す
            extra.setdefault("__inject_time__", True)

        bindings.append(ParamBinding(name, param, cc_num, map_fn, extra))
    return bindings


def _log_mapping(bindings: list[ParamBinding]) -> None:
    lines = ["[Effects CC Mapping]"]
    lines.append(f"View Rotate (rx, ry, rz) = CC {VIEW_CC_RX}, {VIEW_CC_RY}, {VIEW_CC_RZ}")
    for b in bindings:
        lines.append(f"CC {b.cc_num:>3}: {b.effect_name}.{b.param_name}")
    logger.info("\n" + "\n".join(lines))


def draw(t: float, cc: Mapping[int, float]) -> Geometry:
    canvas_w, canvas_h = CANVAS_SIZES["SQUARE_300"]

    # 一度だけバインディングを構築（関数属性にキャッシュ）
    if not hasattr(draw, "_bindings"):
        names = list_effects()
        # 表示性のため軽くソート（名前順）
        names = sorted(names)
        bindings = _build_bindings(names)
        setattr(draw, "_bindings", bindings)
        _log_mapping(bindings)

    bindings: list[ParamBinding] = getattr(draw, "_bindings")  # type: ignore[assignment]

    # グリッド算出（エフェクト数ベース）
    cols, rows, cell_w, cell_h, cell_size = _grid_layout(len(bindings), canvas_w, canvas_h)

    # グローバルプレビュー回転（各セル中心回り）
    rx = _clamp01(cc.get(VIEW_CC_RX, 0.0)) * 2 * math.pi
    ry = _clamp01(cc.get(VIEW_CC_RY, 0.0)) * 2 * math.pi
    rz = _clamp01(cc.get(VIEW_CC_RZ, 0.0)) * 2 * math.pi

    out: Geometry | None = None

    for idx, bind in enumerate(bindings):
        col = idx % cols
        row = idx // cols
        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h

        # 立方体のベース（半径0.5スケールなのでセルサイズでスケール）
        try:
            base = G.polyhedron(polygon_type="cube")
        except Exception:
            base = G.polyhedron(polygon_type="hexahedron")

        g = base.scale(cell_size, cell_size, cell_size).translate(cx, cy, 0.0)

        # プレビュー用の視点回転（セル中心回り）
        if rx or ry or rz:
            g = (E.pipeline.rotate(pivot=(cx, cy, 0.0), angles_rad=(rx, ry, rz)).build())(g)

        # CC 値（0..1）
        v_norm = _clamp01(cc.get(bind.cc_num, 0.0))
        param_value = bind.map_fn(v_norm, cell_size)

        # エキストラの注入（pivot, tなど）
        kw = dict(bind.extra_kw)
        if kw.pop("__inject_pivot__", False):
            kw["pivot"] = (cx, cy, 0.0)
        if kw.pop("__inject_time__", False):
            # displace 等の揺らぎ系
            kw.setdefault("t_sec", float(t))

        # メインパラメータ
        kw[bind.param_name] = param_value

        # パイプラインで単発エフェクト適用
        try:
            g2 = getattr(E.pipeline, bind.effect_name)(**kw).build()(g)
        except Exception as e:
            # 万一の失敗はフォールバックとして元ジオメトリを表示
            logger.debug("effect failed: %s(%s): %s", bind.effect_name, kw, e)
            g2 = g

        out = g2 if out is None else (out + g2)

    return out if out is not None else G.empty()


if __name__ == "__main__":
    setup_default_logging()
    CANVAS = CANVAS_SIZES["SQUARE_300"]
    SCALE = 6
    env = os.environ.get("PYXIDRAW_USE_MIDI")
    USE_MIDI = True if env is None else (env == "1" or env.lower() in ("true", "on", "yes"))
    run(draw, canvas_size=CANVAS, render_scale=SCALE, background=(1, 1, 1, 1), use_midi=USE_MIDI)

