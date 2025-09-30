from __future__ import annotations

"""
概要（アルゴリズム要約）
- 各ポリラインを XY 平面へ射影（場合により回転を伴う）
- パターン（lines/cross/dots）に応じて塗りつぶし要素を生成
- 生成要素を元の 3D 姿勢に戻して合成

密度は線/ドットの本数として扱い、最大で 100 本（グリッドでは 100×100）を生成します。
"""


import numpy as np
from numba import njit  # type: ignore[attr-defined]

from engine.core.geometry import Geometry
from util.geom3d_ops import transform_back, transform_to_xy_plane

from .registry import effect

# 塗りつぶし線の最大密度（最大本数）
MAX_FILL_LINES = 100


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


def _generate_cross_fill(
    vertices: np.ndarray, density: float, angle: float = 0.0
) -> list[np.ndarray]:
    """クロスハッチ塗りつぶしパターンを生成します。"""
    lines1 = _generate_line_fill(vertices, density, angle)
    lines2 = _generate_line_fill(vertices, density, angle + np.pi / 2)
    return lines1 + lines2


def _generate_dot_fill(vertices: np.ndarray, density: float) -> list[np.ndarray]:
    """ドット塗りつぶしパターンを生成します。

    実装メモ:
    - レンダラは線分のみ描画するため「点」は表示されない。
      そのため各グリッド点を小さな「クロス（＋）」の2線分として表現する。
    - クロスのサイズはグリッド間隔に比例（視認性と過密回避のバランス）。
    """
    # Transform to XY plane
    vertices_2d, rotation_matrix, z_offset = transform_to_xy_plane(vertices)
    coords_2d = vertices_2d[:, :2]

    # Calculate bounding box
    min_x, min_y = np.min(coords_2d, axis=0)
    max_x, max_y = np.max(coords_2d, axis=0)

    # Calculate spacing (inversed: 0=few dots, 1=many dots)
    if density <= 0:
        return []

    # Calculate spacing for dot grid
    # At density=1.0, we want many dots (MAX_FILL_LINES x MAX_FILL_LINES grid)
    # At density=0.1, we want fewer dots
    grid_size = int(round(density))
    if grid_size < 2:
        grid_size = 2
    if grid_size > MAX_FILL_LINES:
        grid_size = MAX_FILL_LINES
    spacing = min(max_x - min_x, max_y - min_y) / grid_size
    if spacing <= 0:
        return []

    # Pre-calculate grid points for better performance
    x_values = np.arange(min_x, max_x + spacing, spacing)
    y_values = np.arange(min_y, max_y + spacing, spacing)

    # Use batch processing for finding dots (centers)
    centers = find_dots_in_polygon(coords_2d, x_values, y_values)

    # Represent a dot as a small cross of two short segments
    r = float(spacing) * 0.18  # size ratio tuned for clarity
    if r <= 0:
        return []

    out_lines: list[np.ndarray] = []
    for i in range(len(centers)):
        cx, cy = float(centers[i, 0]), float(centers[i, 1])
        # Horizontal tiny segment
        seg_h_2d = np.array([[cx - r, cy], [cx + r, cy]], dtype=np.float32)
        seg_h_3d = np.hstack([seg_h_2d, np.zeros((2, 1), dtype=np.float32)])
        out_lines.append(transform_back(seg_h_3d, rotation_matrix, z_offset))
        # Vertical tiny segment
        seg_v_2d = np.array([[cx, cy - r], [cx, cy + r]], dtype=np.float32)
        seg_v_3d = np.hstack([seg_v_2d, np.zeros((2, 1), dtype=np.float32)])
        out_lines.append(transform_back(seg_v_3d, rotation_matrix, z_offset))

    return out_lines


# ── 偶奇規則ベースの多輪郭塗り（平面XY向け） ─────────────────────────────────
def _is_planar_xy(coords: np.ndarray, eps: float = 1e-6) -> bool:
    """全頂点の z が一定（XY 平面上）かの簡易判定。"""
    if coords.size == 0:
        return False
    z = coords[:, 2]
    return float(np.max(z) - np.min(z)) <= float(eps)


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


def _generate_cross_fill_evenodd_multi(
    coords: np.ndarray, offsets: np.ndarray, density: float, angle: float = 0.0
) -> list[np.ndarray]:
    """偶奇規則のクロスハッチ（XY 平面前提）。"""
    lines1 = _generate_line_fill_evenodd_multi(coords, offsets, density, angle)
    lines2 = _generate_line_fill_evenodd_multi(coords, offsets, density, angle + np.pi / 2)
    return lines1 + lines2


@njit(cache=True)
def _point_in_polylines_evenodd_njit(
    coords_2d: np.ndarray, offsets: np.ndarray, x: float, y: float
) -> bool:
    """複数ポリライン（2D）の偶奇規則内外判定。"""
    inside = False
    for i in range(len(offsets) - 1):
        start = offsets[i]
        end = offsets[i + 1]
        if end - start < 3:
            continue
        poly = coords_2d[start:end]
        if point_in_polygon_njit(poly, x, y):
            inside = not inside
    return inside


def _generate_dot_fill_evenodd_multi(
    coords: np.ndarray, offsets: np.ndarray, density: float
) -> list[np.ndarray]:
    """偶奇規則のドット（XY 平面前提、点は小さな十字）。"""
    if density <= 0 or offsets.size <= 1:
        return []

    coords_2d = coords[:, :2].astype(np.float32, copy=False)
    min_xy = np.min(coords_2d, axis=0)
    max_xy = np.max(coords_2d, axis=0)
    min_x, min_y = float(min_xy[0]), float(min_xy[1])
    max_x, max_y = float(max_xy[0]), float(max_xy[1])

    grid_size = int(round(float(density)))
    if grid_size < 2:
        grid_size = 2
    if grid_size > MAX_FILL_LINES:
        grid_size = MAX_FILL_LINES
    spacing = min(max_x - min_x, max_y - min_y) / float(grid_size)
    if spacing <= 0:
        return []

    x_values = np.arange(min_x, max_x + spacing, spacing, dtype=np.float32)
    y_values = np.arange(min_y, max_y + spacing, spacing, dtype=np.float32)

    r = float(spacing) * 0.18
    if r <= 0:
        return []

    z0 = float(coords[0, 2])
    offsets_i32 = offsets.astype(np.int32, copy=False)
    out_lines: list[np.ndarray] = []
    for yy in y_values:
        for xx in x_values:
            if _point_in_polylines_evenodd_njit(coords_2d, offsets_i32, float(xx), float(yy)):
                # 小十字
                seg_h_2d = np.array([[xx - r, yy], [xx + r, yy]], dtype=np.float32)
                seg_v_2d = np.array([[xx, yy - r], [xx, yy + r]], dtype=np.float32)
                seg_h_3d = np.hstack([seg_h_2d, np.full((2, 1), z0, dtype=np.float32)])
                seg_v_3d = np.hstack([seg_v_2d, np.full((2, 1), z0, dtype=np.float32)])
                out_lines.append(seg_h_3d)
                out_lines.append(seg_v_3d)

    return out_lines


## 以前の Python 実装ヘルパは NumPy/Numba 実装へ統合済みのため削除


@effect()
def fill(
    g: Geometry,
    *,
    mode: str = "lines",
    angle_rad: float = 0.7853981633974483,  # pi/4 ≈ 45°
    density: float = 35.0,
) -> Geometry:
    """閉じた形状をハッチング/ドットで塗りつぶし（純関数）。

    Parameters
    ----------
    g : Geometry
        入力ジオメトリ。各行が 1 本のポリライン。
    mode : str, default 'lines'
        'lines'|'cross'|'dots'。0 本（density<=0）で no-op。
    angle_rad : float, default pi/4
        ハッチ角（ラジアン）。XY 共平面では偶奇規則で内外を判定。
    density : float, default 35.0
        本数/グリッド密度。0 で no-op。最大は内部定数（100）。

    Notes
    -----
    間隔は未回転の参照高さに基づき角度に依存せず一定だが、スキャン範囲は角度を考慮した
    作業座標で決まるため、角度により実生成本数は多少変動する。
    """
    coords, offsets = g.as_arrays(copy=False)
    if density <= 0 or offsets.size <= 1:
        return Geometry(coords.copy(), offsets.copy())

    density = max(0.0, min(MAX_FILL_LINES, float(density)))

    pat = mode or "lines"
    ang = float(angle_rad)

    # 平面XYなら複数輪郭を偶奇規則で一括処理（穴を保持）
    if _is_planar_xy(coords):
        # まず元の輪郭を残す
        results: list[np.ndarray] = []
        for i in range(len(offsets) - 1):
            results.append(coords[offsets[i] : offsets[i + 1]].copy())

        if pat == "lines":
            results.extend(_generate_line_fill_evenodd_multi(coords, offsets, density, ang))
        elif pat == "cross":
            results.extend(_generate_cross_fill_evenodd_multi(coords, offsets, density, ang))
        elif pat == "dots":
            results.extend(_generate_dot_fill_evenodd_multi(coords, offsets, density))
        else:
            results.extend(_generate_line_fill_evenodd_multi(coords, offsets, density, ang))

        return Geometry.from_lines(results)

    # 非平面は従来通りポリゴン個別に処理
    filled_results: list[np.ndarray] = []
    for i in range(len(offsets) - 1):
        vertices = coords[offsets[i] : offsets[i + 1]]
        filled_results.extend(
            _fill_single_polygon(vertices, pattern=pat, density=density, angle=ang)
        )

    if not filled_results:
        return Geometry(coords.copy(), offsets.copy())

    return Geometry.from_lines(filled_results)


# UI 表示のためのメタ情報（RangeHint 構築に使用）
fill.__param_meta__ = {
    "mode": {"choices": ["lines", "cross", "dots"]},
    "density": {"type": "number", "min": 0.0, "max": MAX_FILL_LINES, "step": 1.0},
    "angle_rad": {"type": "number", "min": 0.0, "max": 2 * np.pi},
}


def _fill_single_polygon(
    vertices: np.ndarray,
    *,
    pattern: str,
    density: float,
    angle: float,
) -> list[np.ndarray]:
    """単一ポリゴンに対して塗りつぶし線/ドットを生成し、元の輪郭と合わせて返す。"""
    if len(vertices) < 3:
        return [vertices]

    out: list[np.ndarray] = [vertices]
    if pattern == "lines":
        fill_lines = _generate_line_fill(vertices, density, angle)
    elif pattern == "cross":
        fill_lines = _generate_cross_fill(vertices, density, angle)
    elif pattern == "dots":
        fill_lines = _generate_dot_fill(vertices, density)
    else:
        fill_lines = _generate_line_fill(vertices, density, angle)
    out.extend(fill_lines)
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


@njit(cache=True)
def find_dots_in_polygon(
    polygon: np.ndarray, x_values: np.ndarray, y_values: np.ndarray
) -> np.ndarray:
    """ポリゴン内部のグリッド点を高速に検索（Numba最適化版）。"""
    # 結果配列を事前確保
    max_points = len(x_values) * len(y_values)
    points = np.empty((max_points, 2))
    count = 0

    for y in y_values:
        for x in x_values:
            if point_in_polygon_njit(polygon, x, y):
                points[count, 0] = x
                points[count, 1] = y
                count += 1

    return points[:count]
