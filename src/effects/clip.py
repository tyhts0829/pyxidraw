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

import hashlib
from collections import OrderedDict
from typing import Any, Iterable, Sequence, Tuple, cast

import numpy as np
from numba import njit  # type: ignore[attr-defined]

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


def _numpy_lines_to_geometry(lines: Iterable[np.ndarray]) -> Geometry:
    arrs = [np.asarray(l, dtype=np.float32) for l in lines if l is not None and len(l) > 0]
    if not arrs:
        return Geometry.from_lines([])
    return Geometry.from_lines(arrs)


def _prepare_projection_mask(rings3d: list[np.ndarray]) -> dict[str, Any]:
    """投影系マスクの前処理（XY 投影 + AABB + 簡易グリッド）。"""
    rings_xy: list[np.ndarray] = []
    rings_aabb: list[tuple[float, float, float, float]] = []
    union_bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]
    total_edges = 0
    coords_concat: list[np.ndarray] = []
    offsets_list: list[int] = [0]
    for ring3d in rings3d:
        if ring3d.shape[0] < 3:
            continue
        r2 = ring3d[:, :2].astype(np.float32, copy=False)
        r2h = _ensure_closed(np.hstack([r2, np.zeros((r2.shape[0], 1), dtype=np.float32)]))
        r = r2h[:, :2]
        if r.shape[0] < 3:
            continue
        rings_xy.append(r)
        coords_concat.append(r)
        offsets_list.append(offsets_list[-1] + r.shape[0])
        aabb = _aabb2d(r)
        rings_aabb.append(aabb)
        union_bounds[0] = min(union_bounds[0], aabb[0])
        union_bounds[1] = min(union_bounds[1], aabb[1])
        union_bounds[2] = max(union_bounds[2], aabb[2])
        union_bounds[3] = max(union_bounds[3], aabb[3])
        total_edges += max(0, r.shape[0] - 1)
    if not rings_xy:
        return {"rings_xy": [], "rings_aabb": [], "union": (0, 0, 0, 0), "grid": {}}
    nx = max(8, min(128, int(np.sqrt(max(1, total_edges)))))
    ny = nx
    minx, miny, maxx, maxy = union_bounds
    width = max(1e-9, float(maxx - minx))
    height = max(1e-9, float(maxy - miny))
    csx = width / float(nx)
    csy = height / float(ny)
    grid: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for ri, r in enumerate(rings_xy):
        m = r.shape[0]
        for ei in range(m - 1):
            p = r[ei]
            q = r[ei + 1]
            x0 = float(min(p[0], q[0]))
            x1 = float(max(p[0], q[0]))
            y0 = float(min(p[1], q[1]))
            y1 = float(max(p[1], q[1]))
            ix0 = int((x0 - minx) / csx)
            iy0 = int((y0 - miny) / csy)
            ix1 = int((x1 - minx) / csx)
            iy1 = int((y1 - miny) / csy)
            if ix0 < 0:
                ix0 = 0
            if iy0 < 0:
                iy0 = 0
            if ix1 >= nx:
                ix1 = nx - 1
            if iy1 >= ny:
                iy1 = ny - 1
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    grid.setdefault((ix, iy), []).append((ri, ei))
    coords2d = (
        np.vstack(coords_concat).astype(np.float32, copy=False)
        if coords_concat
        else np.zeros((0, 2), dtype=np.float32)
    )
    offsets = np.asarray(offsets_list, dtype=np.int32)
    return {
        "rings_xy": rings_xy,
        "rings_aabb": rings_aabb,
        "union": (
            float(union_bounds[0]),
            float(union_bounds[1]),
            float(union_bounds[2]),
            float(union_bounds[3]),
        ),
        "grid": grid,
        "minx": float(minx),
        "miny": float(miny),
        "csx": float(csx),
        "csy": float(csy),
        "nx": int(nx),
        "ny": int(ny),
        "coords2d": coords2d,
        "offsets": offsets,
    }


def _segment_intersections_with_candidates(
    A2: np.ndarray,
    B2: np.ndarray,
    rings_xy: list[np.ndarray],
    candidates: list[tuple[int, int]],
) -> list[float]:
    out: list[float] = []
    ax, ay = float(A2[0]), float(A2[1])
    bx, by = float(B2[0]), float(B2[1])
    for ri, ei in candidates:
        r = rings_xy[ri]
        p = r[ei]
        q = r[ei + 1]
        den = (bx - ax) * (q[1] - p[1]) - (by - ay) * (q[0] - p[0])
        if abs(den) < 1e-12:
            continue
        t = ((p[0] - ax) * (q[1] - p[1]) - (p[1] - ay) * (q[0] - p[0])) / den
        u = ((p[0] - ax) * (by - ay) - (p[1] - ay) * (bx - ax)) / den
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            out.append(float(t))
    return out


@njit(cache=True)
def _njit_intersections_with_candidates(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    coords2d: np.ndarray,
    offsets: np.ndarray,
    cand_pairs: np.ndarray,
) -> np.ndarray:
    max_n = cand_pairs.shape[0]
    out = np.empty(max_n, dtype=np.float32)
    count = 0
    for k in range(max_n):
        ri = int(cand_pairs[k, 0])
        ei = int(cand_pairs[k, 1])
        s = int(offsets[ri])
        p0 = coords2d[s + ei]
        p1 = coords2d[s + ei + 1]
        den = (bx - ax) * (p1[1] - p0[1]) - (by - ay) * (p1[0] - p0[0])
        if abs(den) < 1e-12:
            continue
        t = ((p0[0] - ax) * (p1[1] - p0[1]) - (p0[1] - ay) * (p1[0] - p0[0])) / den
        u = ((p0[0] - ax) * (by - ay) - (p0[1] - ay) * (bx - ax)) / den
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            out[count] = np.float32(t)
            count += 1
    return out[:count]


@njit(cache=True)
def _njit_evenodd_point(x: float, y: float, coords2d: np.ndarray, offsets: np.ndarray) -> int:
    inside = False
    nr = offsets.shape[0] - 1
    for i in range(nr):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        # レイキャスト（半開区間）
        j = e - 1
        for k in range(s, e):
            xi = coords2d[k, 0]
            yi = coords2d[k, 1]
            xj = coords2d[j, 0]
            yj = coords2d[j, 1]
            cond = ((yi > y) != (yj > y)) and (x <= (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            if cond:
                inside = not inside
            j = k
    return 1 if inside else 0


def _clip_lines_project_xy_prepared(
    target_coords: np.ndarray,
    target_offs: np.ndarray,
    prep: dict[str, Any],
    *,
    draw_inside: bool,
    draw_outside: bool,
) -> list[np.ndarray]:
    rings_xy: list[np.ndarray] = prep.get("rings_xy", [])
    if not rings_xy:
        return []
    rings_aabb = prep.get("rings_aabb", [])
    union_bounds = prep.get("union", (0.0, 0.0, 0.0, 0.0))
    grid = prep.get("grid", {})
    minx = float(prep.get("minx", 0.0))
    miny = float(prep.get("miny", 0.0))
    csx = float(prep.get("csx", 1.0))
    csy = float(prep.get("csy", 1.0))
    nx = int(prep.get("nx", 1))
    ny = int(prep.get("ny", 1))
    coords2d = prep.get("coords2d")
    offsets = prep.get("offsets")
    out_lines: list[np.ndarray] = []
    eps = 1e-9
    for i in range(len(target_offs) - 1):
        s, e = int(target_offs[i]), int(target_offs[i + 1])
        if e - s < 2:
            continue
        poly2 = target_coords[s:e, :2]
        pminx, pminy, pmaxx, pmaxy = _aabb2d(poly2)
        if not _aabb_overlap(pminx, pminy, pmaxx, pmaxy, *union_bounds):
            if draw_outside and not draw_inside:
                for j in range(s, e - 1):
                    out_lines.append(
                        np.vstack([target_coords[j], target_coords[j + 1]]).astype(np.float32)
                    )
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
            ix0 = int((seg_minx - minx) / csx)
            iy0 = int((seg_miny - miny) / csy)
            ix1 = int((seg_maxx - minx) / csx)
            iy1 = int((seg_maxy - miny) / csy)
            if ix0 < 0:
                ix0 = 0
            if iy0 < 0:
                iy0 = 0
            if ix1 >= nx:
                ix1 = nx - 1
            if iy1 >= ny:
                iy1 = ny - 1
            cand: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    for ri, ei in grid.get((ix, iy), []):
                        key = (ri, ei)
                        if key not in seen:
                            seen.add(key)
                            aabb = rings_aabb[ri]
                            if _aabb_overlap(seg_minx, seg_miny, seg_maxx, seg_maxy, *aabb):
                                cand.append(key)
            if cand:
                if isinstance(coords2d, np.ndarray) and isinstance(offsets, np.ndarray):
                    try:
                        cand_arr = np.asarray(cand, dtype=np.int32)
                        inters = _njit_intersections_with_candidates(
                            float(A2[0]),
                            float(A2[1]),
                            float(B2[0]),
                            float(B2[1]),
                            coords2d.astype(np.float32, copy=False),
                            offsets.astype(np.int32, copy=False),
                            cand_arr,
                        )
                        for t in inters:
                            if eps < t < 1.0 - eps:
                                ts.append(float(t))
                    except Exception:
                        inters = _segment_intersections_with_candidates(A2, B2, rings_xy, cand)
                        for t in inters:
                            if eps < t < 1.0 - eps:
                                ts.append(float(t))
                else:
                    inters = _segment_intersections_with_candidates(A2, B2, rings_xy, cand)
                    for t in inters:
                        if eps < t < 1.0 - eps:
                            ts.append(float(t))
            if len(ts) <= 2:
                m = 0.5
                mx = float(A2[0] + m * (B2[0] - A2[0]))
                my = float(A2[1] + m * (B2[1] - A2[1]))
                if isinstance(coords2d, np.ndarray) and isinstance(offsets, np.ndarray):
                    try:
                        inside = bool(_njit_evenodd_point(float(mx), float(my), coords2d, offsets))
                    except Exception:
                        cnt = 0
                        for r_xy in rings_xy:
                            if _point_in_polygon(r_xy, mx, my):
                                cnt += 1
                        inside = (cnt % 2) == 1
                else:
                    cnt = 0
                    for r_xy in rings_xy:
                        if _point_in_polygon(r_xy, mx, my):
                            cnt += 1
                    inside = (cnt % 2) == 1
                if (inside and draw_inside) or ((not inside) and draw_outside):
                    out_lines.append(np.vstack([A3, B3]).astype(np.float32))
                continue
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
                if isinstance(coords2d, np.ndarray) and isinstance(offsets, np.ndarray):
                    try:
                        inside = bool(_njit_evenodd_point(float(mx), float(my), coords2d, offsets))
                    except Exception:
                        cnt = 0
                        for r_xy in rings_xy:
                            if _point_in_polygon(r_xy, mx, my):
                                cnt += 1
                        inside = (cnt % 2) == 1
                else:
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


# ── マスク前処理キャッシュ（内容ベース LRU） ─────────────────────────────
try:
    from common.settings import get as _get_settings  # 設定優先

    _s = _get_settings()
    _MASK_CACHE_MAXSIZE = int(getattr(_s, "CLIP_MASK_CACHE_MAXSIZE", 64) or 64)
except Exception:
    import os as _os

    try:
        _MASK_CACHE_MAXSIZE = int(_os.getenv("PXD_CLIP_MASK_CACHE_MAXSIZE", "64"))
    except Exception:
        _MASK_CACHE_MAXSIZE = 64

_MASK_CACHE: "OrderedDict[tuple[bytes, str], dict[str, Any]]" = OrderedDict()
_MASK_CACHE_HITS = 0
_MASK_CACHE_MISSES = 0


def _mask_digest_from_rings(rings3d: list[np.ndarray]) -> bytes:
    """閉路リング群の内容から blake2b-128 ダイジェストを生成。"""
    h = hashlib.blake2b(digest_size=16)
    for r in rings3d:
        try:
            a = np.ascontiguousarray(r.astype(np.float32, copy=False))
            h.update(a.view(np.uint8).tobytes())
            h.update(np.int64(a.shape[0]).tobytes())
        except Exception:
            # 失敗時は型情報のみ
            h.update(b"ring")
    return h.digest()


def _mask_cache_get(key: tuple[bytes, str]) -> dict[str, Any] | None:
    global _MASK_CACHE_HITS
    try:
        v = _MASK_CACHE.pop(key)
    except KeyError:
        return None
    _MASK_CACHE[key] = v
    _MASK_CACHE_HITS += 1
    return v


def _mask_cache_put(key: tuple[bytes, str], value: dict[str, Any]) -> None:
    global _MASK_CACHE_MISSES
    _MASK_CACHE[key] = value
    _MASK_CACHE_MISSES += 1
    if _MASK_CACHE_MAXSIZE is not None and _MASK_CACHE_MAXSIZE > 0:
        while len(_MASK_CACHE) > _MASK_CACHE_MAXSIZE:
            _MASK_CACHE.popitem(last=False)


# 結果キャッシュ/量子化 digest/モード安定化
try:
    from common.settings import get as _get_settings  # type: ignore

    _s2 = _get_settings()
    _RESULT_CACHE_MAXSIZE = int(getattr(_s2, "CLIP_RESULT_CACHE_MAXSIZE", 16) or 16)
    _RESULT_CACHE_MAX_VERTS = int(getattr(_s2, "CLIP_RESULT_CACHE_MAX_VERTS", 200_000) or 200_000)
    _DIGEST_STEP_DEFAULT = float(getattr(_s2, "CLIP_DIGEST_STEP", 1e-4))
    _MODE_STABLE_ENABLED = bool(getattr(_s2, "CLIP_MODE_STABLE", True))
except Exception:
    import os as _os

    try:
        _RESULT_CACHE_MAXSIZE = int(_os.getenv("PXD_CLIP_RESULT_CACHE_MAXSIZE", "16"))
    except Exception:
        _RESULT_CACHE_MAXSIZE = 16
    try:
        _RESULT_CACHE_MAX_VERTS = int(_os.getenv("PXD_CLIP_RESULT_CACHE_MAX_VERTS", "200000"))
    except Exception:
        _RESULT_CACHE_MAX_VERTS = 200_000
    try:
        _DIGEST_STEP_DEFAULT = float(_os.getenv("PXD_CLIP_DIGEST_STEP", "1e-4"))
    except Exception:
        _DIGEST_STEP_DEFAULT = 1e-4
    try:
        _MODE_STABLE_ENABLED = bool(int(_os.getenv("PXD_CLIP_MODE_STABLE", "1")))
    except Exception:
        _MODE_STABLE_ENABLED = True

_RESULT_CACHE: "OrderedDict[tuple[Any, ...], Geometry]" = OrderedDict()
_MODE_PIN: dict[bytes, str] = {}


def _quantized_digest_for_coords(coords: np.ndarray, offsets: np.ndarray, step: float) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    if coords.size:
        q = np.rint(coords.astype(np.float64, copy=False) / float(step)).astype(
            np.int64, copy=False
        )
        h.update(q.tobytes())
    if offsets.size:
        h.update(offsets.astype(np.int32, copy=False).tobytes())
    return h.digest()


def _quantized_digest_for_rings(rings3d: list[np.ndarray], step: float) -> bytes:
    if not rings3d:
        return b"\x00" * 16
    coords_list: list[np.ndarray] = []
    offs = [0]
    s = 0
    for r in rings3d:
        a = np.asarray(r, dtype=np.float32)
        coords_list.append(a)
        s += a.shape[0]
        offs.append(s)
    coords = np.vstack(coords_list).astype(np.float32, copy=False)
    offsets = np.asarray(offs, dtype=np.int32)
    return _quantized_digest_for_coords(coords, offsets, step)


def _outline_digest_quantized(outline: Geometry | Sequence[Geometry], step: float) -> bytes:
    h = hashlib.blake2b(digest_size=16)

    def _one(g: Geometry) -> None:
        c, o = g.as_arrays(copy=False)
        dg = _quantized_digest_for_coords(c, o, step)
        h.update(dg)

    if isinstance(outline, Geometry):
        _one(outline)
    elif isinstance(outline, _ObjectRef):
        obj = outline.unwrap()
        if isinstance(obj, Geometry):
            _one(obj)
    else:
        try:
            for x in outline:  # type: ignore[assignment]
                if isinstance(x, _ObjectRef):
                    obj = x.unwrap()
                    if isinstance(obj, Geometry):
                        _one(obj)
                elif isinstance(x, Geometry):
                    _one(x)
        except Exception:
            pass
    return h.digest()


_MASK_RINGS_CACHE: "OrderedDict[bytes, list[np.ndarray]]" = OrderedDict()
try:
    _MASK_RINGS_CACHE_MAXSIZE = int(_os.getenv("PXD_CLIP_MASK_RINGS_CACHE_MAXSIZE", "32"))  # type: ignore[name-defined]
except Exception:
    _MASK_RINGS_CACHE_MAXSIZE = 32


def _collect_mask_rings_cached(
    outline: Geometry | Sequence[Geometry], *, step: float
) -> list[np.ndarray]:
    d = _outline_digest_quantized(outline, step)
    v = _MASK_RINGS_CACHE.get(d)
    if v is not None:
        _ = _MASK_RINGS_CACHE.pop(d)
        _MASK_RINGS_CACHE[d] = v
        return v
    # fallback: build once and cache
    _coords, _offs, rings = _collect_mask_rings_from_outline(outline)
    _MASK_RINGS_CACHE[d] = rings
    if _MASK_RINGS_CACHE_MAXSIZE is not None and _MASK_RINGS_CACHE_MAXSIZE > 0:
        while len(_MASK_RINGS_CACHE) > _MASK_RINGS_CACHE_MAXSIZE:
            _MASK_RINGS_CACHE.popitem(last=False)
    return rings


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
    # マスク収集（収集済みの再利用を試みる）
    rings3d = _collect_mask_rings_cached(outline, step=float(_DIGEST_STEP_DEFAULT))
    # coords/offsets は必要時のみ構築
    if rings3d:
        _m_coords = np.vstack([r for r in rings3d]).astype(np.float32, copy=False)
        _offs = [0]
        s_acc = 0
        for r in rings3d:
            s_acc += r.shape[0]
            _offs.append(s_acc)
        mask_coords = _m_coords
        mask_offs = np.asarray(_offs, dtype=np.int32)
    else:
        mask_coords = np.zeros((0, 3), dtype=np.float32)
        mask_offs = np.asarray([0], dtype=np.int32)
    if len(rings3d) == 0:
        # 有効な閉曲線が無い場合は安全側で no-op（落とさない）
        return Geometry(g.coords.copy(), g.offsets.copy())

    # 結果キャッシュのための digest（量子化ステップ）
    digest_step = float(_DIGEST_STEP_DEFAULT)
    mask_digest_q = _quantized_digest_for_rings(rings3d, digest_step)
    tgt_digest_q = _quantized_digest_for_coords(tgt_coords, tgt_offs, digest_step)

    # モード安定化（弱ピン止め）: proj ピンの場合は choose_coplanar_frame をスキップ
    last_mode = _MODE_PIN.get(mask_digest_q) if _MODE_STABLE_ENABLED else None
    if last_mode == "proj":
        mode = "proj"
        res_key = (
            "clip-result",
            mode,
            mask_digest_q,
            tgt_digest_q,
            bool(draw_inside),
            bool(draw_outside),
            bool(draw_outline),
            bool(use_projection_fallback),
            bool(projection_use_world_xy),
        )
        if _RESULT_CACHE_MAXSIZE != 0:
            try:
                out_cached = _RESULT_CACHE.pop(res_key)
                _RESULT_CACHE[res_key] = out_cached
                return out_cached
            except KeyError:
                pass
        if use_projection_fallback and projection_use_world_xy:
            digest = _mask_digest_from_rings(rings3d)
            cache_key = (digest, "proj")
            prep = _mask_cache_get(cache_key)
            if prep is None:
                prep = _prepare_projection_mask(rings3d)
                _mask_cache_put(cache_key, prep)
            out_lines_3d = _clip_lines_project_xy_prepared(
                tgt_coords, tgt_offs, prep, draw_inside=draw_inside, draw_outside=draw_outside
            )
            out_geo = _numpy_lines_to_geometry(out_lines_3d)
            if draw_outline:
                out_geo = out_geo.concat(_numpy_lines_to_geometry(rings3d))
            if _RESULT_CACHE_MAXSIZE != 0 and out_geo.n_vertices <= _RESULT_CACHE_MAX_VERTS:
                _RESULT_CACHE[res_key] = out_geo
                if _RESULT_CACHE_MAXSIZE is not None and _RESULT_CACHE_MAXSIZE > 0:
                    while len(_RESULT_CACHE) > _RESULT_CACHE_MAXSIZE:
                        _RESULT_CACHE.popitem(last=False)
            return out_geo
        out = Geometry(g.coords.copy(), g.offsets.copy())
        if draw_outline:
            out = out.concat(_numpy_lines_to_geometry(rings3d))
        if _RESULT_CACHE_MAXSIZE != 0 and out.n_vertices <= _RESULT_CACHE_MAX_VERTS:
            _RESULT_CACHE[res_key] = out
            if _RESULT_CACHE_MAXSIZE is not None and _RESULT_CACHE_MAXSIZE > 0:
                while len(_RESULT_CACHE) > _RESULT_CACHE_MAXSIZE:
                    _RESULT_CACHE.popitem(last=False)
        return out

    # ここで初めて共平面判定/整列（必要ケースのみ）
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

    # 最終モード決定とピン更新
    mode = "planar" if planar else "proj"
    if _MODE_STABLE_ENABLED and last_mode not in ("planar", "proj"):
        _MODE_PIN[mask_digest_q] = mode

    # 結果 LRU（ヒット時は即返す）
    res_key = (
        "clip-result",
        mode,
        mask_digest_q,
        tgt_digest_q,
        bool(draw_inside),
        bool(draw_outside),
        bool(draw_outline),
        bool(use_projection_fallback),
        bool(projection_use_world_xy),
    )
    if _RESULT_CACHE_MAXSIZE != 0:
        try:
            out_cached = _RESULT_CACHE.pop(res_key)
            _RESULT_CACHE[res_key] = out_cached
            return out_cached
        except KeyError:
            pass

    # 非共平面 or 安定化モードが proj の場合: 投影フォールバック or no-op
    if not planar or mode == "proj":
        if use_projection_fallback and projection_use_world_xy:
            # マスク前処理（内容ダイジェストで LRU）
            digest = _mask_digest_from_rings(rings3d)
            cache_key = (digest, "proj")
            prep = _mask_cache_get(cache_key)
            if prep is None:
                prep = _prepare_projection_mask(rings3d)
                _mask_cache_put(cache_key, prep)
            # XY 投影でクリップし、3D は元線分で復元
            out_lines_3d = _clip_lines_project_xy_prepared(
                tgt_coords, tgt_offs, prep, draw_inside=draw_inside, draw_outside=draw_outside
            )
            out_geo = _numpy_lines_to_geometry(out_lines_3d)
            if draw_outline:
                out_geo = out_geo.concat(_numpy_lines_to_geometry(rings3d))
            # 結果を保存
            if _RESULT_CACHE_MAXSIZE != 0 and out_geo.n_vertices <= _RESULT_CACHE_MAX_VERTS:
                _RESULT_CACHE[res_key] = out_geo
                if _RESULT_CACHE_MAXSIZE is not None and _RESULT_CACHE_MAXSIZE > 0:
                    while len(_RESULT_CACHE) > _RESULT_CACHE_MAXSIZE:
                        _RESULT_CACHE.popitem(last=False)
            return out_geo
        # フォールバック無効: no-op + 輪郭連結のみ
        out = Geometry(g.coords.copy(), g.offsets.copy())
        if draw_outline:
            out = out.concat(_numpy_lines_to_geometry(rings3d))
        # 結果保存（no-op）
        if _RESULT_CACHE_MAXSIZE != 0 and out.n_vertices <= _RESULT_CACHE_MAX_VERTS:
            _RESULT_CACHE[res_key] = out
            if _RESULT_CACHE_MAXSIZE is not None and _RESULT_CACHE_MAXSIZE > 0:
                while len(_RESULT_CACHE) > _RESULT_CACHE_MAXSIZE:
                    _RESULT_CACHE.popitem(last=False)
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

    if shapely_ok and (mode == "planar"):
        from shapely.geometry import Polygon  # type: ignore

        # マスク領域（偶奇合成）を内容ダイジェストで LRU
        digest = _mask_digest_from_rings(rings3d)
        cache_key = (digest, "planar")
        cached = _mask_cache_get(cache_key)
        prepared_region = None
        if cached is not None:
            region_obj = cached.get("region")
            region_bounds = cached.get("bounds", None)
            prepared_region = cached.get("prepared")
        else:
            region_obj = None
            for i in range(len(mask_offs) - 1):
                s, e = int(mask_offs[i]), int(mask_offs[i + 1])
                ring = mask_xy_all[s:e, :2].astype(np.float32, copy=False)
                ring = (
                    ring if ring.shape[0] < 2 or not np.allclose(ring[0], ring[-1]) else ring[:-1]
                )
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
            region_bounds = getattr(region_obj, "bounds", None)
            try:
                from shapely.prepared import prep as _prep  # type: ignore

                prepared_region = _prep(region_obj)
            except Exception:
                prepared_region = None
            _mask_cache_put(
                cache_key,
                {"region": region_obj, "bounds": region_bounds, "prepared": prepared_region},
            )

        if region_obj is None or getattr(region_obj, "is_empty", False):
            out = Geometry(g.coords.copy(), g.offsets.copy())
            if draw_outline:
                out = out.concat(_numpy_lines_to_geometry(rings3d))
            return out

        # 対象ラインをクリップ（prepared 早期分岐 + バルク処理）
        out_lines_xy: list[np.ndarray] = []
        bulk_lines: list[np.ndarray] = []
        for i in range(len(tgt_offs) - 1):
            s, e = int(tgt_offs[i]), int(tgt_offs[i + 1])
            pl = tgt_xy_all[s:e, :2].astype(np.float32, copy=False)
            if pl.shape[0] < 2:
                continue
            try:
                ls = LineString(pl)
            except Exception:
                continue
            # 早期除外（bounds ベース）
            if region_bounds is not None:
                pminx, pminy, pmaxx, pmaxy = _aabb2d(pl)
                rb = region_bounds
                if not _aabb_overlap(
                    pminx,
                    pminy,
                    pmaxx,
                    pmaxy,
                    float(rb[0]),
                    float(rb[1]),
                    float(rb[2]),
                    float(rb[3]),
                ):
                    if draw_outside and not draw_inside:
                        out_lines_xy.append(pl)
                    # inside のみならスキップ
                    continue
            # prepared による完全内外の早期分岐
            if prepared_region is not None:
                try:
                    if prepared_region.disjoint(ls):
                        if draw_outside and not draw_inside:
                            out_lines_xy.append(pl)
                        continue
                    if prepared_region.contains(ls):
                        if draw_inside and not draw_outside:
                            out_lines_xy.append(pl)
                        continue
                except Exception:
                    pass
            # バルクへ回す
            bulk_lines.append(pl)

        if bulk_lines:
            try:
                from shapely.geometry import MultiLineString  # type: ignore

                mls = MultiLineString(bulk_lines)
                if draw_inside:
                    gi = mls.intersection(region_obj)
                    out_lines_xy.extend(_to_lines_from_shapely(gi))
                if draw_outside:
                    go = mls.difference(region_obj)
                    out_lines_xy.extend(_to_lines_from_shapely(go))
            except Exception:
                # フォールバック: per-line
                for pl in bulk_lines:
                    try:
                        ls = LineString(pl)
                        if draw_inside:
                            gi = ls.intersection(region_obj)
                            out_lines_xy.extend(_to_lines_from_shapely(gi))
                        if draw_outside:
                            go = ls.difference(region_obj)
                            out_lines_xy.extend(_to_lines_from_shapely(go))
                    except Exception:
                        continue

        # XY → 3D
        out_lines = [transform_back(arr.astype(np.float32), R_all, z_all) for arr in out_lines_xy]
        out_geo = _numpy_lines_to_geometry(out_lines)
        if draw_outline:
            out_geo = out_geo.concat(_numpy_lines_to_geometry(rings3d))
        # 結果保存
        if _RESULT_CACHE_MAXSIZE != 0 and out_geo.n_vertices <= _RESULT_CACHE_MAX_VERTS:
            _RESULT_CACHE[res_key] = out_geo
            if _RESULT_CACHE_MAXSIZE is not None and _RESULT_CACHE_MAXSIZE > 0:
                while len(_RESULT_CACHE) > _RESULT_CACHE_MAXSIZE:
                    _RESULT_CACHE.popitem(last=False)
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
    # 結果保存
    if _RESULT_CACHE_MAXSIZE != 0 and out_geo.n_vertices <= _RESULT_CACHE_MAX_VERTS:
        _RESULT_CACHE[res_key] = out_geo
        if _RESULT_CACHE_MAXSIZE is not None and _RESULT_CACHE_MAXSIZE > 0:
            while len(_RESULT_CACHE) > _RESULT_CACHE_MAXSIZE:
                _RESULT_CACHE.popitem(last=False)
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
