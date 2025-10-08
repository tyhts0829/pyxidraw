from __future__ import annotations

"""
概要（アルゴリズム要約）
- 各ポリラインを XY 平面へ射影（場合により回転を伴う）
- 指定した本数の方向（角度）でハッチ線を生成
- 生成要素を元の 3D 姿勢に戻して合成

密度は線の本数スケールとして扱い、最大で 200 本を生成します。
"""


from typing import Iterable

import numpy as np
from numba import njit  # type: ignore[attr-defined]

from engine.core.geometry import Geometry
from util.geom3d_ops import transform_back, transform_to_xy_plane

from .registry import effect

# 塗りつぶし線の最大密度（最大本数）
MAX_FILL_LINES = 200
NONPLANAR_EPS_ABS = 1e-5
NONPLANAR_EPS_REL = 1e-4


def _generate_line_fill(
    vertices: np.ndarray, density: float, angle: float = 0.0
) -> list[np.ndarray]:
    """平行線塗りつぶしパターンを生成します。"""
    # 処理を簡素化するため XY 平面へ射影
    vertices_2d, rotation_matrix, z_offset = transform_to_xy_plane(vertices)

    # 2D 座標（未回転）
    coords_2d = vertices_2d[:, :2]

    # 角度が指定されている場合はポリゴンを回転（交点計算用の作業座標）
    if angle != 0.0:
        # 逆回転（作業座標へ）
        cos_inv, sin_inv = np.cos(-angle), np.sin(-angle)
        center_inv = np.mean(coords_2d, axis=0)
        coords_2d_centered = coords_2d - center_inv
        rot2_inv = np.array([[cos_inv, -sin_inv], [sin_inv, cos_inv]])
        work_2d = coords_2d_centered @ rot2_inv.T + center_inv
        # 正回転（生成した水平線を元角度へ戻す）を事前計算
        cos_fwd, sin_fwd = np.cos(angle), np.sin(angle)
        center_fwd = np.mean(vertices_2d[:, :2], axis=0)
        rot2_fwd = np.array([[cos_fwd, -sin_fwd], [sin_fwd, cos_fwd]])
    else:
        work_2d = coords_2d

    # 参照間隔用の“未回転”高さ（角度に依存しない間隔を実現）
    _min_y_ref = float(np.min(coords_2d[:, 1]))
    _max_y_ref = float(np.max(coords_2d[:, 1]))
    ref_height = _max_y_ref - _min_y_ref
    if ref_height <= 0:
        return []

    # 実際のスキャン範囲は作業座標（角度考慮）
    min_y = float(np.min(work_2d[:, 1]))
    max_y = float(np.max(work_2d[:, 1]))

    # 密度に基づいて線間隔を算出（逆数的: 0=疎, 1=密）
    # density=1.0 で MAX_FILL_LINES 本、density=0.0 でごく少数
    if density <= 0:
        return []

    # 間隔計算: 間隔が小さいほど線が多い
    # density=1.0 → バウンディングボックス内に MAX_FILL_LINES 本を想定
    # density=0.1 → より少ない本数
    num_lines = int(round(density))
    if num_lines < 2:
        num_lines = 2
    if num_lines > MAX_FILL_LINES:
        num_lines = MAX_FILL_LINES
    # 間隔は“未回転高さ/本数”で固定（角度に依らない）
    spacing = ref_height / num_lines
    if spacing <= 0:
        return []

    # 水平ラインを生成
    y_values = np.arange(min_y, max_y, spacing)
    fill_lines = []

    # 性能向上のためバッチ処理で交点計算（作業座標 = 角度考慮後）
    intersection_results = generate_line_intersections_batch(work_2d, y_values)

    for y, intersections in intersection_results:
        # 交点をソートし線分を生成
        intersections_sorted = np.sort(intersections)
        for i in range(0, len(intersections_sorted) - 1, 2):
            if i + 1 < len(intersections_sorted):
                x1, x2 = intersections_sorted[i], intersections_sorted[i + 1]
                line_2d = np.array([[x1, y], [x2, y]], dtype=np.float32)

                # 必要に応じて正方向の回転を適用
                if angle != 0.0:
                    line_2d = (line_2d - center_fwd) @ rot2_fwd.T + center_fwd

                # 3D に戻す
                line_3d = np.hstack([line_2d, np.zeros((2, 1), dtype=np.float32)])

                # 元の姿勢に戻す
                line_final = transform_back(line_3d, rotation_matrix, z_offset)
                fill_lines.append(line_final)

    return fill_lines


# 旧 cross/dots パターンは angle_sets による複方向ハッチへ統合


# ── 偶奇規則ベースの多輪郭塗り（平面XY向け） ─────────────────────────────────
def _is_planar_xy(coords: np.ndarray, eps: float = 1e-6) -> bool:
    """全頂点の z が一定（XY 平面上）かの簡易判定。"""
    if coords.size == 0:
        return False
    z = coords[:, 2]
    return float(np.max(z) - np.min(z)) <= float(eps)


def _is_polygon_planar(
    vertices: np.ndarray, *, eps_abs: float = NONPLANAR_EPS_ABS, eps_rel: float = NONPLANAR_EPS_REL
) -> bool:
    """単一ポリラインが“ほぼ平面”かを簡易判定する。

    - `transform_to_xy_plane` で先頭3点が張る平面へ整列し、z 残差の最大値を評価する。
    - 閾値は絶対/相対の最大値（相対は元座標のバウンディングボックス対角でスケール不変）。
    """
    if vertices.shape[0] < 3:
        return False
    v = np.asarray(vertices, dtype=np.float32)
    v2d, _R, _z = transform_to_xy_plane(v)
    z = v2d[:, 2]
    z_span = float(np.max(np.abs(z))) if z.size else 0.0
    # 元座標の対角長
    mins = np.min(v, axis=0)
    maxs = np.max(v, axis=0)
    diag = float(np.sqrt(np.sum((maxs - mins) ** 2)))
    threshold = max(float(eps_abs), float(eps_rel) * diag)
    return z_span <= threshold


def _generate_line_fill_evenodd_multi(
    coords: np.ndarray, offsets: np.ndarray, density: float, angle: float = 0.0
) -> list[np.ndarray]:
    """複数輪郭を偶奇規則でまとめてハッチング（XY 平面前提）。"""
    if density <= 0 or offsets.size <= 1:
        return []

    coords_2d = coords[:, :2].astype(np.float32, copy=False)

    # 角度回転のための中心（全体）
    center = np.mean(coords_2d, axis=0)
    work_2d = coords_2d
    rot2_fwd: np.ndarray | None = None
    if angle != 0.0:
        # 逆回転（作業座標へ）
        cos_inv, sin_inv = np.cos(-angle), np.sin(-angle)
        rot_inv = np.array([[cos_inv, -sin_inv], [sin_inv, cos_inv]], dtype=np.float32)
        work_2d = (coords_2d - center) @ rot_inv.T + center
        # 正回転（線分を戻す）
        cos_fwd, sin_fwd = np.cos(angle), np.sin(angle)
        rot2_fwd = np.array([[cos_fwd, -sin_fwd], [sin_fwd, cos_fwd]], dtype=np.float32)

    # 参照間隔用の“未回転”高さ（角度に依存しない間隔を実現）
    ref_min_y = float(np.min(coords_2d[:, 1]))
    ref_max_y = float(np.max(coords_2d[:, 1]))
    ref_height = ref_max_y - ref_min_y
    if ref_height <= 0:
        return []

    # 実スキャン範囲は作業座標系（角度考慮）
    min_y = float(np.min(work_2d[:, 1]))
    max_y = float(np.max(work_2d[:, 1]))

    num_lines = int(round(float(density)))
    if num_lines < 2:
        num_lines = 2
    if num_lines > MAX_FILL_LINES:
        num_lines = MAX_FILL_LINES
    spacing = ref_height / float(num_lines)  # 角度に依らない一定間隔
    if spacing <= 0:
        return []

    y_values = np.arange(min_y, max_y, spacing, dtype=np.float32)
    out_lines: list[np.ndarray] = []
    z0 = float(coords[0, 2])

    # スキャンライン毎に全輪郭から交点を収集し偶奇規則で区間を作成
    for y in y_values:
        # 交点収集
        intersections_all = []  # list[float]
        for i in range(len(offsets) - 1):
            if offsets[i + 1] - offsets[i] < 2:
                continue
            poly = work_2d[offsets[i] : offsets[i + 1]]
            xs = find_line_intersections_njit(poly, float(y))
            if len(xs) > 0:
                intersections_all.extend(xs.tolist())

        if len(intersections_all) < 2:
            continue

        xs_sorted = np.sort(np.asarray(intersections_all, dtype=np.float32))
        # ペアで塗り区間を生成（偶奇）
        for j in range(0, xs_sorted.size - 1, 2):
            x1 = float(xs_sorted[j])
            x2 = float(xs_sorted[j + 1])
            if x2 - x1 <= 1e-9:
                continue
            seg2d = np.array([[x1, y], [x2, y]], dtype=np.float32)
            if angle != 0.0 and rot2_fwd is not None:
                # 元の角度へ戻す（前計算した行列を使用）
                seg2d = (seg2d - center) @ rot2_fwd.T + center
            # 3D に（一定 z）
            seg3d = np.hstack([seg2d, np.full((2, 1), z0, dtype=np.float32)])
            out_lines.append(seg3d)

    return out_lines


## 旧 cross/dots パターン関連の補助関数は廃止


## 以前の Python 実装ヘルパは NumPy/Numba 実装へ統合済みのため削除


@effect()
def fill(
    g: Geometry,
    *,
    angle_sets: int | list[int] | tuple[int, ...] = 1,
    angle_rad: float | list[float] | tuple[float, ...] = 0.7853981633974483,  # pi/4 ≈ 45°
    density: float | list[float] | tuple[float, ...] = 35.0,
    remove_boundary: bool = False,
) -> Geometry:
    """閉じた形状をハッチングで塗りつぶし（純関数）。

    Parameters
    ----------
    g : Geometry
        入力ジオメトリ。各行が 1 本のポリライン。
    angle_sets : int | list[int] | tuple[int, ...], default 1
        ハッチ方向の本数（1=単方向, 2=90°クロス, 3=60°間隔, ...）。
        配列指定時は図形（グループ）ごとに順番適用し、長さに応じてサイクルする。
    angle_rad : float | list[float] | tuple[float, ...], default pi/4
        ハッチ角（ラジアン）。配列指定時は図形（グループ）ごとに順番適用し、長さに応じてサイクルする。
        XY 共平面では外環＋穴のグループ単位で適用。
    density : float | list[float] | tuple[float, ...], default 35.0
        ハッチ密度（本数のスケール）。配列指定時は図形（グループ）ごとに順番適用し、長さに応じてサイクルする。
        0 以下はその図形では no-op。最大は内部定数（200）。
    remove_boundary : bool, default False
        元の閉じた輪郭線を出力から除去する。False で輪郭を残す。

    Notes
    -----
    間隔は未回転の参照高さに基づき角度に依存せず一定だが、スキャン範囲は角度を考慮した
    作業座標で決まるため、角度により実生成本数は多少変動する。
    非XY共平面の入力では輪郭ごとの個別処理を行い、このとき各ポリラインが十分に平面で
    ないと判定された場合は塗りをスキップして元の境界のみを返す。
    """
    coords, offsets = g.as_arrays(copy=False)

    def _as_float_seq(x: float | Iterable[float]) -> list[float]:
        if isinstance(x, (int, float, np.floating)):
            return [float(x)]
        if isinstance(x, (list, tuple)):
            return [float(v) for v in x]
        # その他 Iterable は受け取らない（仕様上 list/tuple のみ）
        raise TypeError("angle_rad/density は float または list/tuple[float] を指定してください")

    density_seq = _as_float_seq(density)  # type: ignore[arg-type]
    angle_seq = _as_float_seq(angle_rad)  # type: ignore[arg-type]

    def _as_int_seq(x: int | Iterable[int]) -> list[int]:
        if isinstance(x, (int, np.integer)):
            return [int(x)]
        if isinstance(x, (list, tuple)):
            return [int(v) for v in x]
        raise TypeError("angle_sets は int または list/tuple[int] を指定してください")

    angle_sets_seq = _as_int_seq(angle_sets)  # type: ignore[arg-type]

    if offsets.size <= 1:
        return Geometry(coords.copy(), offsets.copy())

    if all(d <= 0.0 for d in density_seq):
        return Geometry(coords.copy(), offsets.copy())

    # angle_sets は図形（グループ）ごとに決定する（配列時はサイクル）

    # 平面XYなら複数輪郭を偶奇規則で一括処理（穴を保持）
    if _is_planar_xy(coords):
        results: list[np.ndarray] = []

        # 1) 境界保持（必要時）
        if not remove_boundary:
            for i in range(len(offsets) - 1):
                results.append(coords[offsets[i] : offsets[i + 1]].copy())

        # 2) 外環＋穴のグループ化（入力順の外環ごとにグループを構成）
        groups = _build_evenodd_groups(coords, offsets)

        # 3) グループ単位で density/angle/angle_sets を順番適用し、方向を合成
        for gi, ring_indices in enumerate(groups):
            d = density_seq[gi % len(density_seq)]
            if d <= 0.0:
                continue
            d = max(0.0, min(MAX_FILL_LINES, float(d)))
            # グループの頂点配列にまとめ直し
            lines: list[np.ndarray] = []
            for idx in ring_indices:
                s, e = int(offsets[idx]), int(offsets[idx + 1])
                lines.append(coords[s:e])
            if not lines:
                continue
            g_coords = np.concatenate(lines, axis=0)
            g_offsets = np.zeros(len(lines) + 1, dtype=np.int32)
            acc = 0
            for i, ln in enumerate(lines):
                acc += ln.shape[0]
                g_offsets[i + 1] = acc

            base_ang = angle_seq[gi % len(angle_seq)]
            k_i = angle_sets_seq[gi % len(angle_sets_seq)]
            k_i = int(k_i) if int(k_i) > 0 else 1
            for i in range(k_i):
                ang_i = float(base_ang) + (np.pi / k_i) * i
                results.extend(_generate_line_fill_evenodd_multi(g_coords, g_offsets, d, ang_i))

        return Geometry.from_lines(results)

    # 非平面は従来通りポリゴン個別に処理
    filled_results: list[np.ndarray] = []
    for i in range(len(offsets) - 1):
        vertices = coords[offsets[i] : offsets[i + 1]]
        d = density_seq[i % len(density_seq)]
        if d <= 0.0:
            # 塗り線無し、境界だけ保持
            if not remove_boundary:
                filled_results.append(vertices)
            continue
        base_ang = angle_seq[i % len(angle_seq)]
        k_i = angle_sets_seq[i % len(angle_sets_seq)]
        k_i = int(k_i) if int(k_i) > 0 else 1
        filled_results.extend(
            _fill_single_polygon(
                vertices,
                angle_sets=k_i,
                density=max(0.0, min(MAX_FILL_LINES, float(d))),
                angle=float(base_ang),
                remove_boundary=remove_boundary,
            )
        )

    if not filled_results:
        return Geometry(coords.copy(), offsets.copy())

    return Geometry.from_lines(filled_results)


# UI 表示のためのメタ情報（RangeHint 構築に使用）
fill.__param_meta__ = {
    "angle_sets": {"type": "integer", "min": 1, "max": 6, "step": 1},
    "density": {"type": "number", "min": 0.0, "max": MAX_FILL_LINES, "step": 1.0},
    "angle_rad": {"type": "number", "min": 0.0, "max": 2 * np.pi},
    "remove_boundary": {"type": "boolean"},
}


def _fill_single_polygon(
    vertices: np.ndarray,
    *,
    angle_sets: int,
    density: float,
    angle: float,
    remove_boundary: bool,
) -> list[np.ndarray]:
    """単一ポリゴンに対して塗りつぶし線/ドットを生成し、元の輪郭と合わせて返す。"""
    if len(vertices) < 3:
        return [vertices]
    # 平面性が不足する場合は塗りをスキップ（境界のみ返す）
    if not _is_polygon_planar(vertices):
        return [vertices]

    out: list[np.ndarray] = [] if remove_boundary else [vertices]
    k = int(angle_sets) if int(angle_sets) > 0 else 1
    for i in range(k):
        ang_i = angle + (np.pi / k) * i
        out.extend(_generate_line_fill(vertices, density, ang_i))
    return out


# 後方互換クラスは廃止（関数APIのみ）


# 高速化のための Numba コンパイル関数群
@njit(cache=True)
def find_line_intersections_njit(polygon: np.ndarray, y: float) -> np.ndarray:
    """水平線とポリゴンエッジの交点を検索します（Numba最適化版）。"""
    n = len(polygon)
    intersections = np.full(n, -1.0)  # 無効値で事前確保
    count = 0

    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        # 線分が水平線と交差するか判定
        if (p1[1] <= y < p2[1]) or (p2[1] <= y < p1[1]):
            # 交点の x 座標を計算
            if p2[1] != p1[1]:  # 0 除算回避
                x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                intersections[count] = x
                count += 1

    return intersections[:count]


@njit(cache=True)
def point_in_polygon_njit(polygon: np.ndarray, x: float, y: float) -> bool:
    """レイキャスティングアルゴリズムを使用して点がポリゴン内部にあるかをチェックします（Numba最適化版）。"""
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0, 0], polygon[0, 1]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n, 0], polygon[i % n, 1]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@njit(cache=True)
def generate_line_intersections_batch(polygon: np.ndarray, y_values: np.ndarray) -> list:
    """複数のy値に対して交点を一括計算（Numba最適化版）。"""
    results = []
    for y in y_values:
        intersections = find_line_intersections_njit(polygon, y)
        if len(intersections) >= 2:
            results.append((y, intersections))
    return results


# ── グルーピング補助（XY 共平面、偶奇規則の外環＋穴をグループ化） ────────────────
def _polygon_area_signed_2d(poly2d: np.ndarray) -> float:
    x = poly2d[:, 0]
    y = poly2d[:, 1]
    # shoelace（閉路前提、最終辺は自動で %n ）
    s = 0.0
    n = poly2d.shape[0]
    for i in range(n):
        j = (i + 1) % n
        s += float(x[i] * y[j] - x[j] * y[i])
    return 0.5 * s


def _polygon_centroid_2d(poly2d: np.ndarray) -> tuple[float, float]:
    a = _polygon_area_signed_2d(poly2d)
    if abs(a) < 1e-12:
        c = np.mean(poly2d, axis=0)
        return float(c[0]), float(c[1])
    cx = 0.0
    cy = 0.0
    n = poly2d.shape[0]
    for i in range(n):
        j = (i + 1) % n
        cross = float(poly2d[i, 0] * poly2d[j, 1] - poly2d[j, 0] * poly2d[i, 1])
        cx += (poly2d[i, 0] + poly2d[j, 0]) * cross
        cy += (poly2d[i, 1] + poly2d[j, 1]) * cross
    f = 1.0 / (6.0 * a)
    return float(cx * f), float(cy * f)


def _build_evenodd_groups(coords: np.ndarray, offsets: np.ndarray) -> list[list[int]]:
    n = len(offsets) - 1
    if n <= 0:
        return []
    polys2d: list[np.ndarray] = []
    centroids: list[tuple[float, float]] = []
    areas: list[float] = []
    for i in range(n):
        s, e = int(offsets[i]), int(offsets[i + 1])
        p2d = coords[s:e, :2].astype(np.float32, copy=False)
        polys2d.append(p2d)
        centroids.append(_polygon_centroid_2d(p2d))
        areas.append(abs(_polygon_area_signed_2d(p2d)))

    containers: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        cx, cy = centroids[i]
        for j in range(n):
            if i == j:
                continue
            if point_in_polygon_njit(polys2d[j], float(cx), float(cy)):
                containers[i].append(j)

    is_outer = [(len(containers[i]) % 2) == 0 for i in range(n)]
    outer_indices = [i for i in range(n) if is_outer[i]]
    groups: dict[int, list[int]] = {oi: [oi] for oi in outer_indices}

    for i in range(n):
        if is_outer[i]:
            continue
        cands = [j for j in outer_indices if j in containers[i]]
        if cands:
            # 最も内側の外環（面積が最小）を親にする
            j_best = min(cands, key=lambda j: areas[j])
            groups.setdefault(j_best, []).append(i)
        else:
            # 外環が見つからない稀なケースは単独グループに
            groups.setdefault(i, []).append(i)

    # 入力順に外環を並べ、グループのリングも入力順に並べる
    ordered: list[list[int]] = []
    for oi in outer_indices:
        ring_is = groups.get(oi, [oi])
        ordered.append(sorted(ring_is))
    # 孤立グループ（外環が穴扱いになった等）の救済
    for k, v in groups.items():
        if k not in outer_indices:
            ordered.append(sorted(v))
    return ordered


## 旧 dots 用の交点探索は削除
