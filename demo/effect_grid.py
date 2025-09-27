"""
どこで: demo/effect_grid.py（デモ用スクリプト）
何を: 同一の参照形状を格子状に並べ、各セルに異なるエフェクトを 1 つだけ適用して一覧表示する。
なぜ: 全エフェクトの“見た目の効き”を一目で比較できるようにするため。

起動: `python demo/effect_grid.py`

注意:
- 依存は既存のランタイムに準拠（`api.sketch.run` を使用）。
- ラベル描画は `shapes.text` 依存（環境によりフォント/numba の有無で描画不可の場合は自動スキップ）。
- 各エフェクトのパラメータは「関数のデフォルト引数」をそのまま使用する。
"""

from __future__ import annotations

import sys
from math import ceil
from pathlib import Path
from typing import Any

# src/ を import パスへ追加（main.py と同様の簡易ブート）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import inspect

# 登録の副作用を有効にする
import effects  # noqa: F401  # register all effects
from api import G, run  # type: ignore  # after sys.path tweak
from effects.registry import get_effect, list_effects
from engine.core.geometry import Geometry

# === 調整可能な定数 =======================================================
# 参照形状（全セルで同じ形状を使う）
REFERENCE_SHAPE = "polyhedron"
# 4 = icosahedron（_TYPE_ORDER の 0..4 に対応）
REFERENCE_SHAPE_PARAMS: dict[str, Any] = {"polygon_index": 4}

# レイアウト
CELL_SIZE = (100.0, 100.0)  # (w, h)
COLUMNS = 5  # 1 行あたりのセル数（列数）
PADDING = 10.0  # セル内の余白（内側マージン）—フィット時に上下左右から差し引く
GAP = 10.0  # セル間の間隔（外側ギャップ）—レイアウト時にセル間へ加算
LINE_THICKNESS = 0.0006  # 描画線の太さ（スクリーン座標に対する比率）
EDGE_MARGIN = 30.0  # ウィンドウ外枠の余白（上下左右, px 相当）

# ラベルはシンプルに固定サイズで描画


# === 小ユーティリティ =====================================================
def _bbox(g: Geometry) -> tuple[float, float, float, float]:
    if g.is_empty:
        return (0.0, 0.0, 0.0, 0.0)
    c, _ = g.as_arrays(copy=False)
    min_x = float(c[:, 0].min())
    max_x = float(c[:, 0].max())
    min_y = float(c[:, 1].min())
    max_y = float(c[:, 1].max())
    return (min_x, min_y, max_x, max_y)


def _fit_into(g: Geometry, *, width: float, height: float, center: tuple[float, float]) -> Geometry:
    """ジオメトリを等方スケールで矩形内にフィットさせ、中心へ配置する。"""
    if g.is_empty:
        return g
    min_x, min_y, max_x, max_y = _bbox(g)
    w = max(max_x - min_x, 1e-6)
    h = max(max_y - min_y, 1e-6)
    scale = min(width / w, height / h)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    gx, gy = center
    return g.scale(scale, center=(cx, cy, 0.0)).translate(gx - cx, gy - cy, 0.0)


def _shorten(text: str, *, max_len: int = 22) -> str:
    if len(text) <= max_len:
        return text
    head = max_len // 2 - 1
    tail = max_len - head - 1
    return f"{text[:head]}…{text[-tail:]}"


def _build_params(name: str, fn: Any) -> dict[str, Any]:
    """エフェクト関数のデフォルト引数だけを抽出して辞書化する。"""
    try:
        sig = inspect.signature(fn)
    except Exception:
        return {}

    params: dict[str, Any] = {}
    for p in sig.parameters.values():
        # エフェクトの最初の引数 `g` はスキップ
        if p.name == "g":
            continue
        # 位置/キーワードで受けられるパラメータのみ対象
        if p.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            continue
        if p.default is not inspect._empty:  # type: ignore[attr-defined]
            params[p.name] = p.default
    return params


def _initialize_grid() -> None:
    """セルごとの静的ジオメトリ（回転前）とラベルを構築してキャッシュ。"""
    global _CELL_GEOMS, _CELL_CENTERS, _LABELS_GEO
    names = list_effects()
    names.sort()

    cell_w, cell_h = CELL_SIZE
    inner_w = max(1.0, cell_w - 2 * PADDING)
    inner_h = max(1.0, cell_h - 2 * PADDING)

    # 参照形状（セル内初期フィット）
    shape_name = REFERENCE_SHAPE
    shape_params = dict(REFERENCE_SHAPE_PARAMS)
    base = getattr(G, shape_name)(**shape_params)
    base = _fit_into(base, width=inner_w, height=inner_h, center=(inner_w * 0.5, inner_h * 0.5))

    cols = max(1, int(COLUMNS))
    _CELL_GEOMS = []
    _CELL_CENTERS = []
    labels = G.empty()

    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        ox = EDGE_MARGIN + c * (cell_w + GAP)
        oy = EDGE_MARGIN + r * (cell_h + GAP)
        cx = ox + PADDING + inner_w * 0.5
        cy = oy + PADDING + inner_h * 0.5

        # エフェクト適用（再フィットなし・はみ出し許容）。
        # 参照形状 `base` はセル内の内枠中心 (inner_w*0.5, inner_h*0.5) に配置済み。
        # エフェクトはその中心を基準に作用するため、結果をセル中心 (cx, cy) へ平行移動のみ行う。
        try:
            fn = get_effect(name)
            params = _build_params(name, fn)
            effected = fn(base, **params).translate(cx - inner_w * 0.5, cy - inner_h * 0.5, 0.0)
        except Exception:
            effected = G.empty()

        _CELL_GEOMS.append(effected)
        _CELL_CENTERS.append((cx, cy))

        # ラベル（静的）
        try:
            label = _shorten(name)
            label_geo = G.text(text=label, font_size=20).translate(
                ox + PADDING, oy + PADDING * 0.9, 0.0
            )
            labels = labels.concat(label_geo)
        except Exception:
            pass

    _LABELS_GEO = labels


# ---- draw --------------------------------------------------------------- #
_CELL_GEOMS: list[Geometry] | None = None
_CELL_CENTERS: list[tuple[float, float]] | None = None
_LABELS_GEO: Geometry | None = None
ANGULAR_SPEED = 0.6  # [rad/s]


def draw(t: float) -> Geometry:
    global _CELL_GEOMS, _CELL_CENTERS, _LABELS_GEO
    if _CELL_GEOMS is None:
        _initialize_grid()
    assert _CELL_GEOMS is not None and _CELL_CENTERS is not None

    theta = float(t) * ANGULAR_SPEED
    out = G.empty()
    for g, (cx, cy) in zip(_CELL_GEOMS, _CELL_CENTERS):
        if g.is_empty:
            continue
        out = out.concat(g.rotate(x=theta, y=theta, z=theta, center=(cx, cy, 0.0)))

    if _LABELS_GEO is not None and not _LABELS_GEO.is_empty:
        out = out.concat(_LABELS_GEO)

    return out


if __name__ == "__main__":
    names = list_effects()
    cols = max(1, int(COLUMNS))
    rows = int(ceil(len(names) / cols)) if names else 1
    cell_w, cell_h = CELL_SIZE
    width = int(2 * EDGE_MARGIN + cols * cell_w + (cols - 1) * GAP)
    height = int(2 * EDGE_MARGIN + rows * cell_h + (rows - 1) * GAP)

    run(
        draw,
        canvas_size=(width, height),
        render_scale=2.5,
        use_midi=False,
        use_parameter_gui=False,
        workers=1,
        line_thickness=LINE_THICKNESS,
    )
