"""
どこで: `effects`
何を: 真の3D放射状ミラー（球面くさび、大円境界）
なぜ: 3D空間において指定軸と中心を基準に方位等分の鏡映を提供するため。

v1 仕様（概要）:
- くさびは回転軸 `axis` を含む 2 枚の平面（境界）で構成（方位角の間隔 Δφ=π/n_azimuth）。
- ソース領域は基準くさび内（2 半空間の AND）にクリップした部分線のみ。
- 複製は 2n（非反転 + 境界1反転 を n 回の回転で複製）。`mirror_equator=True` で赤道反転も加え最大 4n。
- 数値安定: EPS=1e-6, INCLUDE_BOUNDARY=True 固定。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from engine.core.geometry import Geometry

from .registry import effect

# 固定許容誤差と境界ポリシー
EPS: float = 1e-6
INCLUDE_BOUNDARY: bool = True


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def _rotate_around_axis(
    points: np.ndarray, axis: np.ndarray, angle: float, center: np.ndarray
) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    k = _unit(axis)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    # ロドリゲスの回転公式: v' = v c + (k×v) s + k (k·v)(1-c)
    p = points.copy()
    p[:, 0] -= center[0]
    p[:, 1] -= center[1]
    p[:, 2] -= center[2]
    v = p
    kv = np.cross(k, v)
    kdotv = np.dot(v, k)
    v_rot = v * c + kv * s + np.outer(kdotv, k) * (1.0 - c)
    v_rot[:, 0] += center[0]
    v_rot[:, 1] += center[1]
    v_rot[:, 2] += center[2]
    return v_rot.astype(np.float32)


def _reflect_across_plane(points: np.ndarray, normal: np.ndarray, center: np.ndarray) -> np.ndarray:
    n = _unit(normal)
    p = points.copy()
    p[:, 0] -= center[0]
    p[:, 1] -= center[1]
    p[:, 2] -= center[2]
    proj = np.dot(p, n).reshape(-1, 1)  # (N,1)
    p_ref = p - 2.0 * proj * n
    p_ref[:, 0] += center[0]
    p_ref[:, 1] += center[1]
    p_ref[:, 2] += center[2]
    return p_ref.astype(np.float32)


def _clip_polyline_halfspace_3d(
    vertices: np.ndarray, *, normal: np.ndarray, center: np.ndarray
) -> list[np.ndarray]:
    """3D 半空間でクリップ（内側: n·(p-c) >= -EPS）。"""
    nrm = _unit(normal)
    n = vertices.shape[0]
    if n == 0:
        return []
    if n == 1:
        s = float(np.dot(vertices[0] - center, nrm))
        ok = s >= (-EPS if INCLUDE_BOUNDARY else EPS)
        return [vertices.copy()] if ok else []

    out: list[np.ndarray] = []
    cur: list[np.ndarray] = []
    a = vertices[0].astype(np.float32)
    sA = float(np.dot(a - center, nrm))
    inA = sA >= (-EPS if INCLUDE_BOUNDARY else EPS)
    if inA:
        cur.append(a)
    for i in range(1, n):
        b = vertices[i].astype(np.float32)
        sB = float(np.dot(b - center, nrm))
        inB = sB >= (-EPS if INCLUDE_BOUNDARY else EPS)
        if inA and inB:
            cur.append(b)
        elif inA and not inB:
            denom = sA - sB
            t = 0.0 if abs(denom) < 1e-20 else sA / (sA - sB)
            t = min(max(t, 0.0), 1.0)
            p = a + (b - a) * np.float32(t)
            if len(cur) == 0 or not np.allclose(cur[-1], p, atol=EPS):
                cur.append(p)
            if len(cur) >= 1:
                out.append(np.vstack(cur).astype(np.float32))
            cur = []
        elif (not inA) and inB:
            denom = sA - sB
            t = 0.0 if abs(denom) < 1e-20 else sA / (sA - sB)
            t = min(max(t, 0.0), 1.0)
            p = a + (b - a) * np.float32(t)
            cur = [p, b]
        else:
            pass
        a, sA, inA = b, sB, inB
    if cur:
        out.append(np.vstack(cur).astype(np.float32))
    return out


def _basis_perp_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = _unit(axis)
    # 任意の基準ベクトル（軸とほぼ平行を避ける）
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(a, ref))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    b0 = _unit(np.cross(a, ref))
    b1 = _unit(np.cross(a, b0))
    return b0, b1


def _compute_azimuth_plane_normals(
    n_azimuth: int, axis: np.ndarray, phi0: float
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.pi / float(n_azimuth)
    b0, b1 = _basis_perp_axis(axis)
    u0 = np.cos(phi0) * b0 + np.sin(phi0) * b1
    u1 = np.cos(phi0 + delta) * b0 + np.sin(phi0 + delta) * b1
    n0 = _unit(np.cross(axis, u0))
    n1 = _unit(np.cross(axis, u1))
    return n0, n1


def _equator_normal(axis: np.ndarray) -> np.ndarray:
    """赤道面（軸に垂直な平面）の法線。軸そのものを単位化して返す。"""
    return _unit(axis)


def _clip_polyline_wedge(
    vertices: np.ndarray,
    *,
    n0: np.ndarray,
    n1: np.ndarray,
    center: np.ndarray,
) -> list[np.ndarray]:
    """2 枚の境界面（n0 と n1 の大円）で挟まれたくさび領域でクリップ。"""
    pieces = _clip_polyline_halfspace_3d(vertices, normal=n0, center=center)
    res: list[np.ndarray] = []
    for p in pieces:
        res.extend(_clip_polyline_halfspace_3d(p, normal=-n1, center=center))
    return res


def _dedup_lines(lines: Iterable[np.ndarray]) -> list[np.ndarray]:
    seen: set[tuple] = set()
    out: list[np.ndarray] = []
    inv = 1.0 / EPS if EPS > 0 else 1e6
    for ln in lines:
        if ln.shape[0] == 0:
            continue
        q = np.rint(ln * inv).astype(np.int64)
        key = (q.shape[0],) + tuple(q.flatten().tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append(ln.astype(np.float32))
    return out


@effect(name="mirror3d")
def mirror3d(
    g: Geometry,
    *,
    n_azimuth: int = 1,
    cx: float = 0.0,
    cy: float = 0.0,
    cz: float = 0.0,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    phi0_deg: float = 0.0,
    mirror_equator: bool = False,
    source_side: bool | Sequence[bool] = True,
) -> Geometry:
    """球面くさびベースの 3D 放射状ミラー。

    Parameters
    ----------
    n_azimuth : int, default 1
        方位の等分数（Δφ=π/n_azimuth）。
    cx, cy, cz : float, default 0.0
        中心座標（平面は必ずこの点を通る）。
    axis : tuple[float,float,float], default (0,0,1)
        くさびの回転軸（内部で正規化）。
    phi0_deg : float, default 0.0
        くさび開始角（軸回り）。
    mirror_equator : bool, default False
        軸に垂直な赤道面での反転も追加する。
    source_side : bool | Sequence[bool], default True
        赤道面のソース半空間（mirror_equator=True のときのみ使用）。True=正側。
    """
    if n_azimuth < 1:
        raise ValueError("n_azimuth は 1 以上の整数である必要があります。")

    center = np.array([cx, cy, cz], dtype=np.float32)
    ax = _unit(np.asarray(axis, dtype=np.float32))
    phi0 = float(np.deg2rad(phi0_deg))
    n0, n1 = _compute_azimuth_plane_normals(int(n_azimuth), ax, phi0)

    coords, offsets = g.as_arrays(copy=False)
    src_lines: list[np.ndarray] = []
    # くさび内にクリップ（H0: n0·(p-c)>=0, H1: (-n1)·(p-c)>=0）
    for i in range(len(offsets) - 1):
        v = coords[offsets[i] : offsets[i + 1]]
        if v.shape[0] == 0:
            continue
        pieces = _clip_polyline_halfspace_3d(v, normal=n0, center=center)
        tmp: list[np.ndarray] = []
        for p in pieces:
            tmp.extend(_clip_polyline_halfspace_3d(p, normal=-n1, center=center))
        # mirror_equator の場合は、さらに赤道半空間で片側に限定（ソース抽出のみに適用）
        if mirror_equator:
            sign = 1 if (isinstance(source_side, bool) and source_side) else 1
            if not isinstance(source_side, bool) and len(source_side) > 0:
                sign = 1 if bool(source_side[0]) else -1
            eq_n = ax * float(sign)
            tmp2: list[np.ndarray] = []
            for p in tmp:
                tmp2.extend(_clip_polyline_halfspace_3d(p, normal=eq_n, center=center))
            tmp = tmp2
        for p in tmp:
            if p.shape[0] >= 1:
                src_lines.append(p.astype(np.float32))

    # 複製: 2n = 回転 n 回 × {非反転, 境界1反転}
    out_lines: list[np.ndarray] = []
    step = 2.0 * np.pi / float(n_azimuth)
    for p in src_lines:
        for m in range(int(n_azimuth)):
            out_lines.append(_rotate_around_axis(p, ax, m * step, center))
        pref = _reflect_across_plane(p, n0, center)
        for m in range(int(n_azimuth)):
            out_lines.append(_rotate_around_axis(pref, ax, m * step, center))

    # 赤道反転を追加
    if mirror_equator:
        eq_n = ax
        extra: list[np.ndarray] = []
        for ln in out_lines:
            extra.append(_reflect_across_plane(ln, eq_n, center))
        out_lines.extend(extra)

    uniq = _dedup_lines(out_lines)
    if not uniq:
        return Geometry(coords.copy(), offsets.copy())
    all_coords = np.vstack(uniq).astype(np.float32)
    new_offsets = np.zeros(len(uniq) + 1, dtype=np.int32)
    acc = 0
    for i, ln in enumerate(uniq, start=1):
        acc += ln.shape[0]
        new_offsets[i] = acc
    return Geometry(all_coords, new_offsets)


__all__ = ["mirror3d"]


mirror3d.__param_meta__ = {
    "n_azimuth": {"min": 1, "max": 64, "step": 1},
    "cx": {"min": -10000.0, "max": 10000.0, "step": 0.1},
    "cy": {"min": -10000.0, "max": 10000.0, "step": 0.1},
    "cz": {"min": -10000.0, "max": 10000.0, "step": 0.1},
    "phi0_deg": {"min": -180.0, "max": 180.0, "step": 1.0},
}
