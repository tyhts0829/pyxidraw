"""
どこで: demo/effect_grid.py（デモ用スクリプト）
何を: 同一参照形状を格子に並べ、各セルへ 1 つのエフェクトを適用して一覧表示する。
なぜ: 全エフェクトの見た目を一目で比較するため。

起動: `python demo/effect_grid.py`

注意:
- 依存は既存ランタイムに準拠（`api.sketch.run`）。
- ラベル描画は `shapes.text` 依存（フォント/numba 不在時は自動スキップ）。
- 各エフェクトのパラメータは関数のデフォルト引数を使用する。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
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
# 描画から除外するエフェクト関数名（定数）
EFFECT_TRANSLATE: str = "translate"
EFFECT_SUBDIVIDE: str = "subdivide"
EFFECT_AFFINE: str = "affine"
EFFECT_ROTATE: str = "rotate"
EXCLUDED_EFFECTS: set[str] = {
    EFFECT_TRANSLATE,
    EFFECT_SUBDIVIDE,
    EFFECT_AFFINE,
    EFFECT_ROTATE,
}

# 参照形状（全セルで同じ形状を使う）
REFERENCE_SHAPE: str = "polyhedron"
# 4 = icosahedron（_TYPE_ORDER の 0..4 に対応）
REFERENCE_SHAPE_PARAMS: dict[str, Any] = {"polygon_index": 3}

# レイアウト
CELL_SIZE: tuple[float, float] = (100.0, 100.0)  # (w, h)
COLUMNS: int = 5  # 1 行あたりのセル数（列数）
PADDING: float = 10.0  # セル内の余白（内側マージン）
GAP: float = 10.0  # セル間の間隔（外側ギャップ）
LINE_THICKNESS: float = 0.0006  # 描画線の太さ（スクリーン座標比）
EDGE_MARGIN: float = 30.0  # ウィンドウ外枠の余白（上下左右, px 相当）
LABEL_FONT_SIZE: int = 12  # ラベル描画のフォントサイズ（1em 高さ[mm]）
# ラベル塗りつぶし（ハッチ）設定（エフェクトは既定値を使用）
LABEL_USE_FILL: bool = True
SUBDIVIDE_TARGET = [
    "displace",
    "twist",
    "wobble",
]  # これらのエフェクトには事前に subdivide を挟む
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


def effect_default_params(name: str, fn: Any) -> dict[str, Any]:
    """エフェクト関数のデフォルト引数のみを抽出して辞書にする。"""
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
        if p.default is not inspect.Parameter.empty:
            params[p.name] = p.default
    return params


def _make_label_geo(text: str, origin: tuple[float, float]) -> Geometry:
    """ラベル用の Geometry を生成する。環境依存エラーは呼び出し側で握りつぶす。"""
    x0, y0 = origin
    label = _shorten(text)
    g = G.text(text=label, em_size_mm=float(LABEL_FONT_SIZE), text_align="left").translate(
        x0, y0, 0.0
    )
    if LABEL_USE_FILL:
        try:
            fill_fn = get_effect("fill")
            # 既定パラメータでハッチング（lines, angle=45°, density=35）
            g = fill_fn(g, angle_rad=0, density=50)
        except Exception:
            # フォント/numba 不在などの環境差は無視してアウトラインのみ描画
            pass
    return g


@dataclass
class GridCache:
    cell_geoms: list[Geometry]
    cell_centers: list[tuple[float, float]]
    labels_geo: Geometry


def _initialize_grid() -> None:
    """セルの静的ジオメトリ（回転前）とラベルを構築してキャッシュする。"""
    global _CACHE
    names = [n for n in list_effects() if n not in EXCLUDED_EFFECTS]
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
    cell_geoms: list[Geometry] = []
    cell_centers: list[tuple[float, float]] = []
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
            params = effect_default_params(name, fn)

            # 必要に応じて事前に subdivide を挟む
            src = base
            if name in SUBDIVIDE_TARGET:
                subdivide_fn = get_effect(EFFECT_SUBDIVIDE)
                # subdivide はデフォルトパラメータで実行（重さは effect 側で制御）
                src = subdivide_fn(src)

            effected = fn(src, **params).translate(cx - inner_w * 0.5, cy - inner_h * 0.5, 0.0)
        except Exception:
            effected = G.empty()

        cell_geoms.append(effected)
        cell_centers.append((cx, cy))

        # ラベル（静的）
        try:
            label_geo = _make_label_geo(name, (ox + PADDING, oy + PADDING * 0.9))
            labels = labels.concat(label_geo)
        except Exception:
            pass

    _CACHE = GridCache(cell_geoms=cell_geoms, cell_centers=cell_centers, labels_geo=labels)


# ---- draw --------------------------------------------------------------- #
_CACHE: GridCache | None = None
ANGULAR_SPEED: float = 0.6  # [rad/s]


def draw(t: float) -> Geometry:
    """各セルのジオメトリを回転させ、ラベルを重ねて返す。"""
    global _CACHE
    if _CACHE is None:
        _initialize_grid()
    assert _CACHE is not None

    theta = float(t) * ANGULAR_SPEED
    out = G.empty()
    for g, (cx, cy) in zip(_CACHE.cell_geoms, _CACHE.cell_centers):
        if g.is_empty:
            continue
        out = out.concat(g.rotate(x=theta, y=theta, z=theta, center=(cx, cy, 0.0)))

    if not _CACHE.labels_geo.is_empty:
        out = out.concat(_CACHE.labels_geo)

    return out


if __name__ == "__main__":
    names = [n for n in list_effects() if n not in EXCLUDED_EFFECTS]
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
