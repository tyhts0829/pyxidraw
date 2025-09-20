"""
どこで: demo/shape_grid.py（デモ用スクリプト）
何を: 登録済みの shape を格子状に並べ、各セルに 1 つずつ描画して一覧表示する。
なぜ: 全 shape の“素の見た目”を一目で比較できるようにするため。

起動: `python demo/shape_grid.py`

注意:
- 依存は既存のランタイムに準拠（`api.run` を使用）。
- ラベル描画は `shapes.text` 依存（環境によりフォント/numba の有無で描画不可の場合は自動スキップ）。
"""

from __future__ import annotations

import sys
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

# src/ を import パスへ追加（main.py と同様の簡易ブート）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 登録の副作用を有効にする
import shapes  # noqa: F401  # register all shapes
from api import G, run  # type: ignore  # after sys.path tweak
from engine.core.geometry import Geometry
from engine.ui.parameters.introspection import FunctionIntrospector
from engine.ui.parameters.normalization import denormalize_scalar
from engine.ui.parameters.state import ParameterLayoutConfig, RangeHint
from shapes.registry import get_shape, list_shapes

# === レイアウト/描画定数 ===================================================
CELL_SIZE = (100.0, 100.0)  # (w, h)
COLUMNS = 5  # 1 行あたりのセル数（列数）
PADDING = 10.0  # セル内の余白（内側マージン）—フィット時に上下左右から差し引く
GAP = 10.0  # セル間の間隔（外側ギャップ）—レイアウト時にセル間へ加算
LINE_THICKNESS = 0.0006  # 描画線の太さ（スクリーン座標に対する比率）
EDGE_MARGIN = 30.0  # ウィンドウ外枠の余白（上下左右, px 相当）

# パラメータ解決
NORMALIZED_DEFAULT = 0.5  # 0..1 の既定値
_INTROSPECTOR = FunctionIntrospector()
_LAYOUT_CFG = ParameterLayoutConfig()


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


def _value_type_of(meta_type: str | None, default: Any) -> str:
    mt = (meta_type or "").lower()
    if mt in {"number", "float"}:
        return "float"
    if mt in {"integer", "int"}:
        return "int"
    if mt == "bool":
        return "bool"
    if mt in {"vec2", "vec3", "vector"}:
        return "vector"
    # 推定
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int) and not isinstance(default, bool):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, Sequence) and len(default) in (2, 3):
        return "vector"
    return "float"


def _as_tuple3(value: Any, *, fill: float = 0.0) -> tuple[float, float, float]:
    if isinstance(value, Sequence) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    if isinstance(value, Sequence) and len(value) == 2:
        return (float(value[0]), float(value[1]), fill)
    v = float(value) if isinstance(value, (int, float)) else fill
    return (v, v, v)


def _denorm_scalar(
    default: Any, *, name: str, value_type: str, meta: Mapping[str, Any] | None
) -> Any:
    # 正規化レンジは常に 0..1
    if (
        meta
        and isinstance(meta.get("min"), (int, float))
        and isinstance(meta.get("max"), (int, float))
    ):
        hint = RangeHint(0.0, 1.0, mapped_min=float(meta["min"]), mapped_max=float(meta["max"]))
        return denormalize_scalar(NORMALIZED_DEFAULT, hint, value_type=value_type)  # type: ignore[arg-type]
    # ヒューリスティック（デフォルト値まわり）
    rng = _LAYOUT_CFG.derive_range(name=name, value_type=value_type, default_value=default)  # type: ignore[arg-type]
    hint = RangeHint(0.0, 1.0, mapped_min=float(rng.min_value), mapped_max=float(rng.max_value))
    return denormalize_scalar(NORMALIZED_DEFAULT, hint, value_type=value_type)  # type: ignore[arg-type]


def _denorm_vector(
    default: Any, *, name: str, meta: Mapping[str, Any] | None
) -> tuple[float, float, float]:
    # meta にベクトル min/max があれば成分ごとに適用
    if meta and isinstance(meta.get("min"), Sequence) and isinstance(meta.get("max"), Sequence):
        mins = _as_tuple3(meta["min"], fill=0.0)
        maxs = _as_tuple3(meta["max"], fill=1.0)
        out = []
        for lo, hi in zip(mins, maxs):
            hint = RangeHint(0.0, 1.0, mapped_min=lo, mapped_max=hi)
            out.append(float(denormalize_scalar(NORMALIZED_DEFAULT, hint, value_type="float")))
        return (out[0], out[1], out[2])
    # ヒューリスティック（各成分同一レンジ）
    base = _as_tuple3(default, fill=0.0)
    rng = _LAYOUT_CFG.derive_range(name=name, value_type="vector", default_value=base[0])
    hint = RangeHint(0.0, 1.0, mapped_min=float(rng.min_value), mapped_max=float(rng.max_value))
    v = float(denormalize_scalar(NORMALIZED_DEFAULT, hint, value_type="float"))
    return (v, v, v)


def _build_params(name: str, fn: Any) -> dict[str, Any]:
    """shape 関数のパラメータを 0..1 正規化から実レンジへ決定的に構築。"""
    info = _INTROSPECTOR.resolve(kind="shape", name=name, fn=fn)
    meta = info.param_meta
    try:
        import inspect

        sig = inspect.signature(fn)
    except Exception:
        return {}

    params: dict[str, Any] = {}
    for p in sig.parameters.values():
        # 位置/キーワードで受けられるパラメータのみ対象
        if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            continue
        # 無名 *args はスキップ
        if p.kind == p.VAR_POSITIONAL:
            continue
        # meta 情報
        m = meta.get(p.name)
        m_type = None if m is None else str(m.get("type", ""))
        value_type = _value_type_of(m_type, p.default)

        try:
            if value_type == "vector":
                params[p.name] = _denorm_vector(p.default, name=p.name, meta=m)
            elif value_type == "int":
                v = _denorm_scalar(p.default, name=p.name, value_type="int", meta=m)
                params[p.name] = int(round(float(v)))
            elif value_type == "bool":
                v = _denorm_scalar(p.default, name=p.name, value_type="bool", meta=m)
                params[p.name] = bool(v)
            elif value_type == "float":
                params[p.name] = float(
                    _denorm_scalar(p.default, name=p.name, value_type="float", meta=m)
                )
            else:
                # 未知型はデフォルトを尊重
                if p.default is not p.empty:
                    params[p.name] = p.default
        except Exception:
            # 変換に失敗した場合はデフォルトにフォールバック
            if p.default is not p.empty:
                params[p.name] = p.default

    return params


# ---- grid 構築 ----------------------------------------------------------- #
_CELL_GEOMS: list[Geometry] | None = None
_CELL_CENTERS: list[tuple[float, float]] | None = None
_LABELS_GEO: Geometry | None = None
ANGULAR_SPEED = 0.6  # [rad/s]


def _initialize_grid() -> None:
    """セルごとの静的ジオメトリ（回転前）とラベルを構築してキャッシュ。"""
    global _CELL_GEOMS, _CELL_CENTERS, _LABELS_GEO

    names = list_shapes()
    names.sort()

    cell_w, cell_h = CELL_SIZE
    inner_w = max(1.0, cell_w - 2 * PADDING)
    inner_h = max(1.0, cell_h - 2 * PADDING)

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

        # shape 生成（セル中心へフィット）
        try:
            fn = get_shape(name)
            params = _build_params(name, fn)
            g = fn(**params)
            if not isinstance(g, Geometry):
                g = Geometry.from_lines(g)
            g = _fit_into(g, width=inner_w, height=inner_h, center=(cx, cy))
        except Exception:
            g = G.empty()

        _CELL_GEOMS.append(g)
        _CELL_CENTERS.append((cx, cy))

        # ラベル（静的）
        try:
            label = _shorten(name)
            label_geo = G.text(text=label, font_size=0.2).translate(
                ox + PADDING, oy + PADDING * 0.9, 0.0
            )
            labels = labels.concat(label_geo)
        except Exception:
            pass

    _LABELS_GEO = labels


# ---- draw ---------------------------------------------------------------- #
def draw(t: float, _cc: Mapping[int, float]) -> Geometry:
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
    names = list_shapes()
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
