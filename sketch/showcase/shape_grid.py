"""
どこで: demo/shape_grid.py（デモ用スクリプト）
何を: 登録済みの shape を格子状に並べ、各セルに 1 つずつ描画して一覧表示する。
なぜ: 全 shape の“素の見た目”を一目で比較できるようにするため。

起動: `python demo/shape_grid.py`

注意:
- 依存は既存のランタイムに準拠（`api.run` を使用）。
- ラベル描画は `shapes.text` 依存（環境によりフォント/numba の有無で描画不可の場合は自動スキップ）。
 - 前提: 事前に `pip install -e .[dev]` を実行して import パスを解決する。
"""

from __future__ import annotations

from math import ceil
from typing import Any, Mapping, Sequence

import effects  # noqa: F401  # register all effects (for fill)

# 登録の副作用を有効にする
import shapes  # noqa: F401  # register all shapes
from api import G, run
from effects.registry import get_effect  # for labeling fill
from engine.core.geometry import Geometry
from engine.ui.parameters.introspection import FunctionIntrospector
from engine.ui.parameters.state import ParameterLayoutConfig
from shapes.registry import get_shape, list_shapes

# === レイアウト/描画定数 ===================================================
CELL_SIZE = (100.0, 100.0)  # (w, h)
COLUMNS = 5  # 1 行あたりのセル数（列数）
PADDING = 10.0  # セル内の余白（内側マージン）—フィット時に上下左右から差し引く
GAP = 10.0  # セル間の間隔（外側ギャップ）—レイアウト時にセル間へ加算
LINE_THICKNESS = 0.0006  # 描画線の太さ（スクリーン座標に対する比率）
EDGE_MARGIN = 30.0  # ウィンドウ外枠の余白（上下左右, px 相当）
LABEL_FONT_SIZE_MM = 10.0  # ラベルのフォントサイズ（1em 高さ[mm]）
FAIL_MARK_SIZE = 8.0  # 失敗印（×）の一辺サイズ

# パラメータ代表値の比率（レンジ内の中間値を使う）
DEFAULT_RATIO = 0.5
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


def _value_type_of(meta: Mapping[str, Any] | None, default: Any) -> str:
    """値型を推定する。

    優先順位:
    - __param_meta__ の type が vec2/vec3/vector → vector
    - __param_meta__ の min/max がシーケンス → vector
    - 既定値がシーケンス長 2/3/4 → vector
    - 上記以外は default の型から推定（bool/int/float）
    """
    mt = str(meta.get("type", "")).lower() if isinstance(meta, Mapping) else ""
    if mt in {"vec2", "vec3", "vector"}:
        return "vector"
    if mt in {"string", "str"}:
        return "string"
    if isinstance(meta, Mapping) and "choices" in meta:
        return "enum"
    if isinstance(meta, Mapping):
        mn, mx = meta.get("min"), meta.get("max")
        if isinstance(mn, Sequence) and isinstance(mx, Sequence) and len(mn) == len(mx):
            if len(mn) in (2, 3, 4):
                return "vector"
    # 文字列は Sequence 判定から除外
    if (
        isinstance(default, Sequence)
        and not isinstance(default, (str, bytes, bytearray))
        and len(default) in (2, 3, 4)
    ):
        return "vector"
    if isinstance(default, str):
        return "string"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int) and not isinstance(default, bool):
        return "int"
    if isinstance(default, float):
        return "float"
    return "float"


def _as_tuple(value: Any, dim: int, *, fill: float = 0.0) -> tuple[float, ...]:
    if isinstance(value, Sequence) and len(value) >= dim:
        return tuple(float(value[i]) for i in range(dim))
    if isinstance(value, Sequence):
        vals = [float(value[i]) for i in range(len(value))]
        while len(vals) < dim:
            vals.append(fill)
        return tuple(vals)
    v = float(value) if isinstance(value, (int, float)) else fill
    return tuple(v for _ in range(dim))


def _denorm_scalar(
    default: Any, *, name: str, value_type: str, meta: Mapping[str, Any] | None
) -> Any:
    # 実レンジヒント（min/max/step）のみ使用。中央比率で代表値を決める。
    if (
        meta
        and isinstance(meta.get("min"), (int, float))
        and isinstance(meta.get("max"), (int, float))
    ):
        lo = float(meta["min"])
        hi = float(meta["max"])
    else:
        rng = _LAYOUT_CFG.derive_range(name=name, value_type=value_type, default_value=default)  # type: ignore[arg-type]
        lo = float(rng.min_value)
        hi = float(rng.max_value)
    v = lo + (hi - lo) * DEFAULT_RATIO
    if value_type == "int":
        return int(round(v))
    if value_type == "bool":
        return v >= (lo + hi) * 0.5
    return float(v)


def _vector_dim(meta: Mapping[str, Any] | None, default: Any) -> int:
    if isinstance(default, Sequence) and len(default) in (2, 3, 4):
        return len(default)
    if isinstance(meta, Mapping):
        mn, mx = meta.get("min"), meta.get("max")
        if isinstance(mn, Sequence) and isinstance(mx, Sequence) and len(mn) == len(mx):
            if len(mn) in (2, 3, 4):
                return len(mn)
    return 3


def _denorm_vector(default: Any, *, name: str, meta: Mapping[str, Any] | None) -> tuple[float, ...]:
    dim = _vector_dim(meta, default)
    # meta にベクトル min/max があれば成分ごとに適用
    if (
        isinstance(meta, Mapping)
        and isinstance(meta.get("min"), Sequence)
        and isinstance(meta.get("max"), Sequence)
    ):
        mins = _as_tuple(meta["min"], dim, fill=0.0)
        maxs = _as_tuple(meta["max"], dim, fill=1.0)
        out = []
        for lo, hi in zip(mins, maxs):
            v = float(lo) + (float(hi) - float(lo)) * DEFAULT_RATIO
            out.append(float(v))
        return tuple(out)
    # ヒューリスティック（各成分同一レンジ）
    base = _as_tuple(default, dim, fill=0.0)
    rng = _LAYOUT_CFG.derive_range(name=name, value_type="vector", default_value=base[0])
    lo, hi = float(rng.min_value), float(rng.max_value)
    v = lo + (hi - lo) * DEFAULT_RATIO
    return tuple(v for _ in range(dim))


def _build_params(name: str, fn: Any) -> dict[str, Any]:
    """shape 関数のパラメータをヒューリスティックに決定的構築（実値ベース）。"""
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
        value_type = _value_type_of(m, p.default)

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
            elif value_type in {"string", "enum"}:
                # 文字列/列挙はデフォルト値を尊重。デフォルトが無い場合は choices の先頭を使う
                if p.default is not p.empty:
                    params[p.name] = p.default
                elif (
                    isinstance(m, Mapping)
                    and isinstance(m.get("choices"), Sequence)
                    and m["choices"]
                ):
                    params[p.name] = m["choices"][0]
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


def _make_fail_mark(ox: float, oy: float, *, cell_h: float) -> Geometry:
    """セル左上に小さな × 印（2 本線）を描くジオメトリを返す。"""
    # 左上に配置（内側 PADDING 分だけ余白）
    x0 = ox + PADDING
    y0 = oy + cell_h - PADDING - FAIL_MARK_SIZE
    p1 = (x0, y0, 0.0)
    p2 = (x0 + FAIL_MARK_SIZE, y0 + FAIL_MARK_SIZE, 0.0)
    p3 = (x0 + FAIL_MARK_SIZE, y0, 0.0)
    p4 = (x0, y0 + FAIL_MARK_SIZE, 0.0)
    return Geometry.from_lines([[p1, p2], [p3, p4]])


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
            # 失敗印を追加
            labels = labels.concat(_make_fail_mark(ox, oy, cell_h=cell_h))

        _CELL_GEOMS.append(g)
        _CELL_CENTERS.append((cx, cy))

        # ラベル（静的）
        try:
            label = _shorten(name)
            label_geo = G.text(text=label, em_size_mm=LABEL_FONT_SIZE_MM).translate(
                ox + PADDING, oy + PADDING * 0.9, 0.0
            )
            # ラベルにハッチフィル（効果が使えない環境では無視）
            try:
                fill_fn = get_effect("fill")
                label_geo = fill_fn(label_geo, angle_rad=0, density=50)
            except Exception:
                pass
            labels = labels.concat(label_geo)
        except Exception:
            pass

        # 生成自体は成功したが空だった場合も印を付ける
        if g.is_empty:
            labels = labels.concat(_make_fail_mark(ox, oy, cell_h=cell_h))

    _LABELS_GEO = labels


# ---- draw ---------------------------------------------------------------- #
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
