from __future__ import annotations

"""
どこで: `effects.clip`（Geometry→Geometry の純関数エフェクト）
何を: 同一平面内の閉曲線（リング）で定義される領域をマスクとして、対象ジオメトリを内側/外側でクリップする。
なぜ: `effects.fill` と同様の共平面推定→XY 整列→偶奇規則に基づき、回転/スケールに対して安定なクリップを実現するため。

非目標:
- 3D で複数平面に跨る厳密クリップ（非共平面は安全側 no-op）。
- 線のスタイル変更（別エフェクト `style`）。

実装メモ:
- Shapely が利用可能なら `Polygon` の `symmetric_difference` で偶奇領域を構成し、
  `LineString` の `intersection`/`difference` でクリップする。
- Shapely が利用不可の場合は簡易フォールバック（線分をマスクリングで分割→中点の偶奇判定）。
"""

from typing import Any, Iterable, Sequence, Tuple, cast

import numpy as np

from common.types import ObjectRef as _ObjectRef
from engine.core.geometry import Geometry
from util.geom3d_frame import choose_coplanar_frame
from util.geom3d_ops import transform_back
from util.polygon_grouping import point_in_polygon_njit as _point_in_polygon

from .registry import effect


def _ensure_closed(loop: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if loop.shape[0] == 0:
        return loop
    p0 = loop[0]
    p1 = loop[-1]
    if float(np.linalg.norm(p0 - p1)) <= eps:
        return loop
    return np.vstack([loop, p0])


def _collect_mask_rings_from_outline(
    outline: Geometry | Sequence[Geometry], *, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """outline（単体/配列）から閉路リングのみを抽出し、配列に連結して返す。

    Returns
    -------
    coords, offsets, rings3d
        - coords/offsets: 連結済み頂点配列（float32）とオフセット（int32）
        - rings3d: 元の 3D 座標（各リングの配列、閉路済み）
    """

    # ObjectRef/LazyGeometry を許容し、Geometry へ正規化
    def _unwrap(x: Any) -> Geometry | None:
        if isinstance(x, _ObjectRef):
            obj = x.unwrap()
        else:
            obj = x
        if isinstance(obj, Geometry):
            return obj
        return None

    if isinstance(outline, Geometry) or isinstance(outline, _ObjectRef):
        g0 = _unwrap(outline)
        outlines: list[Geometry] = [g0] if g0 is not None else []
    else:
        outlines = []
        try:
            for x in outline:  # type: ignore[assignment]
                gx = _unwrap(x)
                if gx is not None:
                    outlines.append(gx)
        except Exception:
            pass
    all_coords: list[np.ndarray] = []
    all_offsets: list[np.ndarray] = []
    rings3d: list[np.ndarray] = []
    coord_shift = 0
    for g in outlines:
        c, o = g.as_arrays(copy=False)
        for i in range(len(o) - 1):
            s, e = int(o[i]), int(o[i + 1])
            ring = c[s:e]
            if ring.shape[0] < 3:
                continue
            ring_closed = _ensure_closed(ring, eps)
            # 閉路でなければ除外
            if ring_closed.shape[0] < 3:
                continue
            all_coords.append(ring_closed.astype(np.float32, copy=False))
            # offsets は個々のリングを独立に扱う
            if all_offsets:
                prev = all_offsets[-1][-1]
            else:
                prev = 0
            n = ring_closed.shape[0]
            all_offsets.append(np.array([prev, prev + n], dtype=np.int32))
            rings3d.append(ring_closed.astype(np.float32, copy=False))
            coord_shift += n
    if not all_coords:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.array([0], dtype=np.int32),
            [],
        )
    coords = np.vstack(all_coords).astype(np.float32, copy=False)
    # all_offsets は各リングの [start, end] の 2 要素配列。これを累積して 1 本の offsets へ。
    offs = [0]
    s = 0
    for oc in all_coords:
        s += oc.shape[0]
        offs.append(s)
    offsets = np.asarray(offs, dtype=np.int32)
    return coords, offsets, rings3d


def _to_lines_from_shapely(geom) -> list[np.ndarray]:  # type: ignore[no-untyped-def]
    """Shapely Geometry から LineString 系を抽出して ndarray の配列へ。Z=0 を付与。"""
    arrs: list[np.ndarray] = []
    try:
        if geom.is_empty:
            return arrs
    except Exception:
        return arrs
    gtype = getattr(geom, "geom_type", "")
    if gtype == "LineString":
        coords = np.asarray(geom.coords, dtype=np.float32)
        if coords.shape[0] >= 2:
            arrs.append(np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=np.float32)]))
        return arrs
    for g in getattr(geom, "geoms", []):  # MultiLineString / GeometryCollection
        arrs.extend(_to_lines_from_shapely(g))
    return arrs


def _segment_intersections_with_polygon_edges(
    a: np.ndarray, b: np.ndarray, poly2d: np.ndarray
) -> list[Tuple[float, float]]:
    """線分 AB と多角形の各辺の交点（パラメータ t, 点 x）を返す（t∈[0,1]）。

    戻り値は (t, x) のリスト。数値安定性のため重複 t は後段でユニーク化する想定。
    """
    out: list[Tuple[float, float]] = []
    n = poly2d.shape[0]
    for i in range(n - 1):
        c = poly2d[i]
        d = poly2d[i + 1]
        # 2D 直線同士の交点（パラメトリック）
        den = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0])
        if abs(den) < 1e-12:
            continue
        t = ((c[0] - a[0]) * (d[1] - c[1]) - (c[1] - a[1]) * (d[0] - c[0])) / den
        u = ((c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])) / den
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            x = a[0] + t * (b[0] - a[0])
            out.append((float(t), float(x)))
    return out


def _clip_lines_without_shapely(
    target_xy: np.ndarray,
    target_offs: np.ndarray,
    mask_rings_xy: list[np.ndarray],
    *,
    draw_inside: bool,
    draw_outside: bool,
) -> list[np.ndarray]:
    """Shapely 不在時の簡易クリップ。

    各対象ポリラインをセグメントに分割し、交点で刻んで偶奇判定により採否を決定する。
    返り値は XY（Z=0）の線分配列。
    """
    lines_out: list[np.ndarray] = []
    if not mask_rings_xy:
        return []
    for i in range(len(target_offs) - 1):
        s, e = int(target_offs[i]), int(target_offs[i + 1])
        poly = target_xy[s:e, :2]
        if poly.shape[0] < 2:
            continue
        for j in range(poly.shape[0] - 1):
            a = poly[j]
            b = poly[j + 1]
            ts = [0.0, 1.0]
            xs = [a[0], b[0]]  # 参考用（ソート安定化目的のみ）
            # 全マスクリングの各辺と交差
            for ring in mask_rings_xy:
                inters = _segment_intersections_with_polygon_edges(a, b, ring)
                for t, x in inters:
                    if 0.0 < t < 1.0:
                        ts.append(t)
                        xs.append(x)
            # ソート→ユニーク化
            order = np.argsort(np.asarray(ts))
            ts_sorted = np.asarray(ts)[order]
            # 刻んで偶奇判定
            for k in range(len(ts_sorted) - 1):
                t0 = float(ts_sorted[k])
                t1 = float(ts_sorted[k + 1])
                if t1 - t0 <= 1e-9:
                    continue
                m = (t0 + t1) * 0.5
                mx = a[0] + m * (b[0] - a[0])
                my = a[1] + m * (b[1] - a[1])
                # 偶奇（複数リングの XOR）
                cnt = 0
                for ring in mask_rings_xy:
                    if _point_in_polygon(ring, float(mx), float(my)):
                        cnt += 1
                inside = (cnt % 2) == 1
                if (inside and draw_inside) or ((not inside) and draw_outside):
                    p0 = np.array(
                        [a[0] + t0 * (b[0] - a[0]), a[1] + t0 * (b[1] - a[1])], dtype=np.float32
                    )
                    p1 = np.array(
                        [a[0] + t1 * (b[0] - a[0]), a[1] + t1 * (b[1] - a[1])], dtype=np.float32
                    )
                    seg2 = np.vstack([p0, p1]).astype(np.float32)
                    seg3 = np.hstack([seg2, np.zeros((2, 1), dtype=np.float32)])
                    lines_out.append(seg3)
    return lines_out


def _aabb2d(poly2d: np.ndarray) -> tuple[float, float, float, float]:
    minx = float(np.min(poly2d[:, 0]))
    maxx = float(np.max(poly2d[:, 0]))
    miny = float(np.min(poly2d[:, 1]))
    maxy = float(np.max(poly2d[:, 1]))
    return minx, miny, maxx, maxy


def _aabb_overlap(
    ax0: float, ay0: float, ax1: float, ay1: float, bx0: float, by0: float, bx1: float, by1: float
) -> bool:
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _clip_lines_project_xy(
    target_coords: np.ndarray,
    target_offs: np.ndarray,
    mask_rings_3d: list[np.ndarray],
    *,
    draw_inside: bool,
    draw_outside: bool,
) -> list[np.ndarray]:
    """非共平面でも XY 投影で確実に切るフォールバック。

    - マスクは 3D→XY 投影し偶奇規則で採否。
    - 対象は各セグメントを XY で分割し、3D は線形補間で復元。
    """
    if target_coords.size == 0 or target_offs.size <= 1:
        return []
    # 投影マスクと AABB
    rings_xy: list[np.ndarray] = []
    rings_aabb: list[tuple[float, float, float, float]] = []
    for ring3d in mask_rings_3d:
        if ring3d.shape[0] < 3:
            continue
        r2 = ring3d[:, :2].astype(np.float32, copy=False)
        r2h = _ensure_closed(np.hstack([r2, np.zeros((r2.shape[0], 1), dtype=np.float32)]))
        r = r2h[:, :2]
        if r.shape[0] < 3:
            continue
        rings_xy.append(r)
        rings_aabb.append(_aabb2d(r))
    if not rings_xy:
        return []

    out_lines: list[np.ndarray] = []
    eps = 1e-9
    for i in range(len(target_offs) - 1):
        s, e = int(target_offs[i]), int(target_offs[i + 1])
        if e - s < 2:
            continue
        for j in range(s, e - 1):
            A3 = target_coords[j]
            B3 = target_coords[j + 1]
            A2 = A3[:2]
            B2 = B3[:2]
            seg_minx = float(min(A2[0], B2[0]))
            seg_maxx = float(max(A2[0], B2[0]))
            seg_miny = float(min(A2[1], B2[1]))
            seg_maxy = float(max(A2[1], B2[1]))
            ts: list[float] = [0.0, 1.0]
            # 交点収集（AABB で早期除外）
            for r_xy, aabb in zip(rings_xy, rings_aabb):
                if not _aabb_overlap(seg_minx, seg_miny, seg_maxx, seg_maxy, *aabb):
                    continue
                inters = _segment_intersections_with_polygon_edges(A2, B2, r_xy)
                for t, _x in inters:
                    if eps < t < 1.0 - eps:
                        ts.append(float(t))
            if len(ts) <= 2:
                # セグメント全体の中点で判定
                m = 0.5
                mx = float(A2[0] + m * (B2[0] - A2[0]))
                my = float(A2[1] + m * (B2[1] - A2[1]))
                cnt = 0
                for r_xy in rings_xy:
                    if _point_in_polygon(r_xy, mx, my):
                        cnt += 1
                inside = (cnt % 2) == 1
                if (inside and draw_inside) or ((not inside) and draw_outside):
                    out_lines.append(np.vstack([A3, B3]).astype(np.float32))
                continue
            # ソート＆ユニーク化
            ts_sorted = sorted(ts)
            uniq: list[float] = []
            for t in ts_sorted:
                if not uniq or abs(t - uniq[-1]) > eps:
                    uniq.append(t)
            for k in range(len(uniq) - 1):
                t0 = float(uniq[k])
                t1 = float(uniq[k + 1])
                if t1 - t0 <= eps:
                    continue
                m = (t0 + t1) * 0.5
                mx = float(A2[0] + m * (B2[0] - A2[0]))
                my = float(A2[1] + m * (B2[1] - A2[1]))
                cnt = 0
                for r_xy in rings_xy:
                    if _point_in_polygon(r_xy, mx, my):
                        cnt += 1
                inside = (cnt % 2) == 1
                if (inside and draw_inside) or ((not inside) and draw_outside):
                    P0 = A3 + (B3 - A3) * t0
                    P1 = A3 + (B3 - A3) * t1
                    out_lines.append(np.vstack([P0, P1]).astype(np.float32))
    return out_lines


def _numpy_lines_to_geometry(lines: Iterable[np.ndarray]) -> Geometry:
    arrs = [np.asarray(l, dtype=np.float32) for l in lines if l is not None and len(l) > 0]
    if not arrs:
        return Geometry.from_lines([])
    return Geometry.from_lines(arrs)


@effect()
def clip(
    g: Geometry,
    *,
    outline: Geometry | Sequence[Geometry],
    draw_outline: bool = False,
    draw_inside: bool = True,
    draw_outside: bool = False,
    use_projection_fallback: bool = True,
    projection_use_world_xy: bool = True,
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-4,
) -> Geometry:
    """閉曲線マスクで対象をクリップ（純関数）。

    Parameters
    ----------
    g : Geometry
        対象ジオメトリ（クリップされる側）。
    outline : Geometry | Sequence[Geometry]
        マスク輪郭（閉曲線）。複数指定可。偶奇規則で内外を決定。
    draw_outline : bool, default False
        出力にマスク輪郭も含める。
    draw_inside : bool, default True
        マスク内側を出力に含める。
    draw_outside : bool, default False
        マスク外側を出力に含める。
    use_projection_fallback : bool, default True
        共平面でない場合に XY 投影フォールバックでクリップする。
    projection_use_world_xy : bool, default True
        True のときワールド XY へ投影（将来拡張用スイッチ）。
    eps_abs, eps_rel : float, default 1e-5, 1e-4
        共平面判定の絶対/相対許容誤差。
    """
    if not (draw_inside or draw_outside):
        # 何も描かない指定は no-op とする
        return Geometry(g.coords.copy(), g.offsets.copy())

    tgt_coords, tgt_offs = g.as_arrays(copy=False)
    mask_coords, mask_offs, rings3d = _collect_mask_rings_from_outline(outline)
    if len(rings3d) == 0:
        # 有効な閉曲線が無い場合は安全側で no-op（落とさない）
        return Geometry(g.coords.copy(), g.offsets.copy())

    # 連結して共平面判定/整列
    if tgt_coords.size == 0:
        comb_coords = mask_coords
        comb_offs = mask_offs
    elif mask_coords.size == 0:
        comb_coords = tgt_coords
        comb_offs = tgt_offs
    else:
        comb_coords = np.vstack([tgt_coords, mask_coords]).astype(np.float32, copy=False)
        comb_offs = np.hstack([tgt_offs, (mask_offs[1:] + tgt_coords.shape[0]).astype(np.int32)])
        comb_offs = comb_offs.astype(np.int32, copy=False)

    planar, v2d_all, R_all, z_all, _ref_h = choose_coplanar_frame(
        comb_coords, comb_offs, eps_abs=float(eps_abs), eps_rel=float(eps_rel)
    )

    # 非共平面: 投影フォールバック or no-op
    if not planar:
        if use_projection_fallback and projection_use_world_xy:
            # XY 投影でクリップし、3D は元線分で復元
            out_lines_3d = _clip_lines_project_xy(
                tgt_coords,
                tgt_offs,
                rings3d,
                draw_inside=draw_inside,
                draw_outside=draw_outside,
            )
            out_geo = _numpy_lines_to_geometry(out_lines_3d)
            if draw_outline:
                out_geo = out_geo.concat(_numpy_lines_to_geometry(rings3d))
            return out_geo
        # フォールバック無効: no-op + 輪郭連結のみ
        out = Geometry(g.coords.copy(), g.offsets.copy())
        if draw_outline:
            out = out.concat(_numpy_lines_to_geometry(rings3d))
        return out

    # XY 座標に分離
    n_t = tgt_coords.shape[0]
    tgt_xy_all = v2d_all[:n_t]
    mask_xy_all = v2d_all[n_t:]

    # Mask region（Shapely 優先）
    shapely_ok = False
    try:
        from shapely.geometry import LineString, Polygon  # type: ignore

        shapely_ok = True
    except Exception:
        shapely_ok = False

    if shapely_ok:
        from shapely.geometry import Polygon  # type: ignore

        region_obj = None
        for i in range(len(mask_offs) - 1):
            s, e = int(mask_offs[i]), int(mask_offs[i + 1])
            ring = mask_xy_all[s:e, :2].astype(np.float32, copy=False)
            ring = ring if ring.shape[0] < 2 or not np.allclose(ring[0], ring[-1]) else ring[:-1]
            if ring.shape[0] < 3:
                continue
            try:
                pg = Polygon(ring)
                if not pg.is_valid:
                    pg = pg.buffer(0)
            except Exception:
                continue
            region_obj = pg if region_obj is None else region_obj.symmetric_difference(pg)
        if region_obj is None or region_obj.is_empty:
            out = Geometry(g.coords.copy(), g.offsets.copy())
            if draw_outline:
                out = out.concat(_numpy_lines_to_geometry(rings3d))
            return out

        # 対象ラインをクリップ
        out_lines_xy: list[np.ndarray] = []
        for i in range(len(tgt_offs) - 1):
            s, e = int(tgt_offs[i]), int(tgt_offs[i + 1])
            pl = tgt_xy_all[s:e, :2].astype(np.float32, copy=False)
            if pl.shape[0] < 2:
                continue
            try:
                ls = LineString(pl)
            except Exception:
                continue
            if draw_inside:
                try:
                    gi = ls.intersection(region_obj)
                    out_lines_xy.extend(_to_lines_from_shapely(gi))
                except Exception:
                    pass
            if draw_outside:
                try:
                    go = ls.difference(region_obj)
                    out_lines_xy.extend(_to_lines_from_shapely(go))
                except Exception:
                    pass

        # XY → 3D
        out_lines = [transform_back(arr.astype(np.float32), R_all, z_all) for arr in out_lines_xy]
        out_geo = _numpy_lines_to_geometry(out_lines)
        if draw_outline:
            out_geo = out_geo.concat(_numpy_lines_to_geometry(rings3d))
        return out_geo

    # ---- フォールバック（Shapely 無） -------------------------------------
    # マスクリング（XY）を収集
    mask_rings_xy: list[np.ndarray] = []
    for i in range(len(mask_offs) - 1):
        s, e = int(mask_offs[i]), int(mask_offs[i + 1])
        ring = mask_xy_all[s:e, :2].astype(np.float32, copy=False)
        ring = _ensure_closed(np.hstack([ring, np.zeros((ring.shape[0], 1), dtype=np.float32)]))
        mask_rings_xy.append(ring[:, :2])

    lines_xy = _clip_lines_without_shapely(
        tgt_xy_all.astype(np.float32, copy=False),
        tgt_offs,
        mask_rings_xy,
        draw_inside=draw_inside,
        draw_outside=draw_outside,
    )
    lines_3d = [transform_back(l.astype(np.float32), R_all, z_all) for l in lines_xy]
    out_geo = _numpy_lines_to_geometry(lines_3d)
    if draw_outline:
        out_geo = out_geo.concat(_numpy_lines_to_geometry(rings3d))
    return out_geo


# UI RangeHint（量子化粒度は step を設定。outline は GUI 対象外）
cast(Any, clip).__param_meta__ = {
    "draw_outline": {"type": "boolean"},
    "draw_inside": {"type": "boolean"},
    "draw_outside": {"type": "boolean"},
    "use_projection_fallback": {"type": "boolean"},
    "projection_use_world_xy": {"type": "boolean"},
    "eps_abs": {"type": "number", "min": 1e-7, "max": 1e-2, "step": 1e-6},
    "eps_rel": {"type": "number", "min": 1e-7, "max": 1e-2, "step": 1e-6},
}
