from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from .base import BaseShape

# 型エイリアス
Point3D = tuple[float, float, float]
Region = tuple[float, float, float, float]


def distance(p: Point3D, q: Point3D) -> float:
    """2次元ユークリッド距離を計算"""
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def relative_neighborhood_graph(nodes: list[Point3D]) -> tuple[list[tuple[int, int]], dict]:
    """
    RNG (Relative Neighborhood Graph) を構築する。

    さらに、2つのノード間の距離が MIN_DISTANCE 未満の場合はエッジを引かないようにする。

    Args:
        nodes: [(x, y, z), ...] の点のリスト

    Returns:
        edges: (i, j) のタプルのリスト
        adjacency: 各ノード番号と隣接ノード番号のリストの辞書
    """
    MIN_DISTANCE = 0.1  # ノード間の距離がこれより小さい場合はエッジを引かない
    n = len(nodes)
    edges: list[tuple[int, int]] = []
    adjacency = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            dij = distance(nodes[i], nodes[j])
            if dij < MIN_DISTANCE:
                # ノード同士が近すぎる場合はエッジを引かない
                continue
            edge_valid = True
            for k in range(n):
                if k == i or k == j:
                    continue
                if distance(nodes[i], nodes[k]) < dij and distance(nodes[j], nodes[k]) < dij:
                    edge_valid = False
                    break
            if edge_valid:
                edges.append((i, j))
                adjacency[i].append(j)
                adjacency[j].append(i)
    return edges, adjacency


def random_walk_strokes(nodes: list[Point3D], adjacency: dict) -> list[list[int]]:
    """
    RNG上でランダムウォークによりストロークを生成する。
    各ストロークはランダムな出発点から2〜4ステップのウォークで得られ、
    一度使用されたエッジは削除される。

    Returns:
        各ストロークを構成するノードのインデックスリストのリスト
    """
    strokes: list[list[int]] = []
    n = len(nodes)
    # 隣接リストのコピー
    adj = {i: list(neighbors) for i, neighbors in adjacency.items()}
    while True:
        candidates = [i for i in range(n) if adj[i]]
        if not candidates:
            break
        start = random.choice(candidates)
        stroke = [start]
        current = start
        steps = random.randint(2, 4)
        for _ in range(steps):
            if not adj[current]:
                break
            next_node = random.choice(adj[current])
            adj[current].remove(next_node)
            if current in adj[next_node]:
                adj[next_node].remove(current)
            stroke.append(next_node)
            current = next_node
        strokes.append(stroke)
    return strokes


def snap_stroke(original: list[Point3D]) -> list[Point3D]:
    """
    各セグメントの方向を60度刻みにスナップする。

    Args:
        original: 元の頂点列

    Returns:
        スナップ後の頂点列
    """
    snapped = [original[0]]
    for point in original[1:]:
        last = snapped[-1]
        dx = point[0] - last[0]
        dy = point[1] - last[1]
        L = math.sqrt(dx**2 + dy**2)
        theta_deg = math.degrees(math.atan2(dy, dx))
        snapped_theta_deg = round(theta_deg / 60.0) * 60.0
        snapped_theta = math.radians(snapped_theta_deg)
        new_point = (
            last[0] + L * math.cos(snapped_theta),
            last[1] + L * math.sin(snapped_theta),
            0,
        )
        snapped.append(new_point)
    return snapped


def smooth_polyline(polyline: list[Point3D], smoothing_radius: float) -> list[Point3D]:
    """
    quadratic Bézier曲線により各内部コーナーを補間し、なめらかなポリラインを生成する。

    Args:
        polyline: 頂点列
        smoothing_radius: 補間用の最大オフセット距離

    Returns:
        補間後の頂点列
    """
    if len(polyline) < 3:
        return polyline
    new_points = [polyline[0]]
    for i in range(1, len(polyline) - 1):
        A = polyline[i - 1]
        B = polyline[i]
        C = polyline[i + 1]
        dAB = distance(A, B)
        dBC = distance(B, C)
        d = min(smoothing_radius, dAB / 2, dBC / 2)
        uBA = ((B[0] - A[0]) / dAB, (B[1] - A[1]) / dAB, (B[2] - A[2]) / dAB) if dAB != 0 else (0, 0, 0)
        uBC = ((C[0] - B[0]) / dBC, (C[1] - B[1]) / dBC, (C[2] - B[2]) / dBC) if dBC != 0 else (0, 0, 0)
        A_prime = (B[0] - uBA[0] * d, B[1] - uBA[1] * d, B[2] - uBA[2] * d)
        C_prime = (B[0] + uBC[0] * d, B[1] + uBC[1] * d, B[2] + uBC[2] * d)
        if distance(new_points[-1], A_prime) > 0.1:
            new_points.append(A_prime)
        num_arc_points = 5
        for t in np.linspace(0, 1, num_arc_points + 2)[1:-1]:
            x = (1 - t) ** 2 * A_prime[0] + 2 * (1 - t) * t * B[0] + t**2 * C_prime[0]
            y = (1 - t) ** 2 * A_prime[1] + 2 * (1 - t) * t * B[1] + t**2 * C_prime[1]
            z = (1 - t) ** 2 * A_prime[2] + 2 * (1 - t) * t * B[2] + t**2 * C_prime[2]
            new_points.append((x, y, z))
        new_points.append(C_prime)
    new_points.append(polyline[-1])
    return new_points


def generate_nodes(region: Region, cell_margin: float, placement_mode: str) -> list[Point3D]:
    """
    指定された領域と余白、配置モードに応じてノードを生成する。

    Args:
        region: (x0, y0, x1, y1) の領域
        cell_margin: 余白のサイズ
        placement_mode: "grid", "hexagon", "poisson", "spiral", "radial", "concentric" のいずれか

    Returns:
        生成されたノードのリスト [(x, y, 0), ...]
    """
    x0, y0, x1, y1 = region
    nodes: list[Point3D] = []

    if placement_mode == "grid":
        n = 2
        rand_int = random.randint(0, 1)
        xs = np.linspace(x0 + cell_margin, x1 - cell_margin, n + rand_int)
        ys = np.linspace(y0 + cell_margin, y1 - cell_margin, n + rand_int)
        for y in ys:
            for x in xs:
                nodes.append((float(x), float(y), 0.0))

    elif placement_mode == "hexagon":
        num_cols = 3
        num_rows = 3
        spacing_x = (x1 - x0 - 2 * cell_margin) / (num_cols - 1) if num_cols > 1 else 0
        spacing_y = (y1 - y0 - 2 * cell_margin) / (num_rows - 1) if num_rows > 1 else 0
        for row in range(num_rows):
            for col in range(num_cols):
                offset = (spacing_x / 2) if (row % 2 == 1) else 0
                x = x0 + cell_margin + col * spacing_x + offset
                y = y0 + cell_margin + row * spacing_y * 0.866
                nodes.append((float(x), float(y), 0.0))

    elif placement_mode == "poisson":
        # ポアソンディスクサンプリングによるノード配置
        x_min = x0 + cell_margin
        x_max = x1 - cell_margin
        y_min = y0 + cell_margin
        y_max = y1 - cell_margin
        r = min(x_max - x_min, y_max - y_min) / 8.0  # rを大きくするとノード数が減る
        k = 30  # 試行回数
        sample_points = []
        active_list = []
        p0 = (random.uniform(x_min, x_max), random.uniform(y_min, y_max))
        sample_points.append(p0)
        active_list.append(p0)
        while active_list:
            idx = random.randint(0, len(active_list) - 1)
            point = active_list[idx]
            found = False
            for _ in range(k):
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(r, 2 * r)
                candidate = (point[0] + radius * math.cos(angle), point[1] + radius * math.sin(angle))
                if candidate[0] < x_min or candidate[0] > x_max or candidate[1] < y_min or candidate[1] > y_max:
                    continue
                valid = True
                for p in sample_points:
                    if math.hypot(candidate[0] - p[0], candidate[1] - p[1]) < r:
                        valid = False
                        break
                if valid:
                    sample_points.append(candidate)
                    active_list.append(candidate)
                    found = True
                    break
            if not found:
                active_list.pop(idx)
        nodes = [(float(p[0]), float(p[1]), 0.0) for p in sample_points]

    elif placement_mode == "spiral":
        # スパイラル配置: 領域の中心から螺旋状にノードを配置
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        # 利用可能な最大半径（cell_marginを引いておく）
        max_radius = min(x1 - x0, y1 - y0) / 2 - cell_margin
        num_nodes = 12  # 配置するノード数
        delta_angle = 2 * math.pi / 12  # 1ステップあたりの角度（例：72度）
        for i in range(num_nodes):
            angle = i * delta_angle
            # 半径は0からmax_radiusへ線形に増加
            radius = max_radius * (i / (num_nodes - 1))
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            nodes.append((x, y, 0))

    elif placement_mode == "radial":
        # 放射状配置: 中心から複数の直線上にノードを配置
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        max_radius = min(x1 - x0, y1 - y0) / 2 - cell_margin
        num_rays = 3  # 放射する直線の数
        nodes_per_ray = 3  # 各直線上に配置するノード数
        for ray in range(num_rays):
            angle = ray * (2 * math.pi / num_rays)
            for i in range(1, nodes_per_ray + 1):
                r = max_radius * (i / (nodes_per_ray + 1))
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                nodes.append((float(x), float(y), 0.0))

    elif placement_mode == "concentric":
        # 同心円配置: 中心を起点として、複数の円周上にノードを配置
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        max_radius = min(x1 - x0, y1 - y0) / 2 - cell_margin
        num_circles = 1  # 円の数
        nodes_per_circle = 5  # 各円周上のノード数
        for circle in range(1, num_circles + 1):
            r = max_radius * (circle / num_circles)
            for j in range(nodes_per_circle):
                angle = j * (2 * math.pi / nodes_per_circle)
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                nodes.append((float(x), float(y), 0.0))
        # 中心ノードも追加
        nodes.append((float(cx), float(cy), 0.0))

    else:
        # デフォルトは grid
        n = 2
        rand_int = random.randint(0, 1)
        xs = np.linspace(x0 + cell_margin, x1 - cell_margin, n + rand_int)
        ys = np.linspace(y0 + cell_margin, y1 - cell_margin, n + rand_int)
        for y in ys:
            for x in xs:
                nodes.append((float(x), float(y), 0.0))
    return nodes


def add_diacritic(
    vertices_list: list[np.ndarray],
    nodes: list[Point3D],
    used_nodes: set,
    diacritic_probability: float,
    diacritic_radius: float,
) -> None:
    """
    使用されたノードの中から、一定の確率でディアクリティカル（アクセント記号）を追加する。
    追加するディアクリティカルは、"circle", "tilde", "grave", "umlaut", "acute",
    "circumflex", "caron", "cedilla" のいずれかの形状となる。

    Args:
        vertices_list: 既存の頂点リストに追加する
        nodes: すべてのノードのリスト
        used_nodes: ストローク生成で使用されたノードのインデックス集合
        diacritic_probability: ディアクリティカルを追加する確率
        diacritic_radius: ディアクリティカルのサイズ
    """
    drawn_diacritic = False
    for i in used_nodes:
        if random.random() < diacritic_probability and not drawn_diacritic:
            offset_x = random.uniform(-diacritic_radius, diacritic_radius)
            offset_y = random.uniform(-diacritic_radius, diacritic_radius)
            diacritic_center = (nodes[i][0] + offset_x, nodes[i][1] + offset_y, 0)
            diacritic_type = random.choice(
                ["circle", "tilde", "grave", "umlaut", "acute", "circumflex", "caron", "cedilla"]
            )
            if diacritic_type == "circle":
                # 円形アクセント（20角形で近似）
                n_sides = 20
                polygon_points = []
                for j in range(n_sides):
                    theta = 2 * math.pi * j / n_sides
                    px = diacritic_center[0] + diacritic_radius * math.cos(theta)
                    py = diacritic_center[1] + diacritic_radius * math.sin(theta)
                    polygon_points.append((px, py, 0))
                polygon_points.append(polygon_points[0])
                vertices_list.append(np.array(polygon_points))
            elif diacritic_type == "tilde":
                # チルダ：サイン波状の曲線
                tilde_points = []
                num_points = 10
                length = diacritic_radius * 2
                amplitude = diacritic_radius / 2
                start_x = diacritic_center[0] - diacritic_radius
                for k in range(num_points):
                    t = k / (num_points - 1)
                    x = start_x + t * length
                    y = diacritic_center[1] + amplitude * math.sin(math.pi * t)
                    tilde_points.append((x, y, 0))
                vertices_list.append(np.array(tilde_points))
            elif diacritic_type == "grave":
                # グレイブ：左上がりの短い線分
                start = diacritic_center
                end = (diacritic_center[0] - diacritic_radius * 0.8, diacritic_center[1] + diacritic_radius * 0.4, 0)
                vertices_list.append(np.array([start, end]))
            elif diacritic_type == "umlaut":
                # ウムラウト：左右に配置した2つの小さい円（8角形で近似）
                dot_radius = diacritic_radius * 0.3
                n_sides_dot = 8
                offsets = [(-diacritic_radius * 0.5, 0), (diacritic_radius * 0.5, 0)]
                for dx, dy in offsets:
                    center_dot = (diacritic_center[0] + dx, diacritic_center[1] + dy, 0)
                    dot_points = []
                    for j in range(n_sides_dot):
                        theta = 2 * math.pi * j / n_sides_dot
                        px = center_dot[0] + dot_radius * math.cos(theta)
                        py = center_dot[1] + dot_radius * math.sin(theta)
                        dot_points.append((px, py, 0))
                    dot_points.append(dot_points[0])
                    vertices_list.append(np.array(dot_points))
            elif diacritic_type == "acute":
                # アキュート：短い右上がりの斜線
                start = (diacritic_center[0] - diacritic_radius * 0.3, diacritic_center[1] + diacritic_radius * 0.2, 0)
                end = (diacritic_center[0] + diacritic_radius * 0.3, diacritic_center[1] + diacritic_radius * 0.7, 0)
                vertices_list.append(np.array([start, end]))
            elif diacritic_type == "circumflex":
                # サーカムフレックス：左右の端と中央上部の3点で構成するキャレット型
                left = (diacritic_center[0] - diacritic_radius, diacritic_center[1], 0)
                peak = (diacritic_center[0], diacritic_center[1] + diacritic_radius, 0)
                right = (diacritic_center[0] + diacritic_radius, diacritic_center[1], 0)
                vertices_list.append(np.array([left, peak, right]))
            elif diacritic_type == "caron":
                # ハチェク（チェロン）：小さなV字型
                left = (diacritic_center[0] - diacritic_radius, diacritic_center[1] + diacritic_radius * 0.2, 0)
                bottom = (diacritic_center[0], diacritic_center[1] - diacritic_radius * 0.2, 0)
                right = (diacritic_center[0] + diacritic_radius, diacritic_center[1] + diacritic_radius * 0.2, 0)
                vertices_list.append(np.array([left, bottom, right]))
            elif diacritic_type == "cedilla":
                # セディーヤ：文字の下にフック状の曲線
                num_points = 8
                start = (diacritic_center[0] - diacritic_radius * 0.5, diacritic_center[1] - diacritic_radius * 0.2, 0)
                end = (diacritic_center[0] + diacritic_radius * 0.5, diacritic_center[1] - diacritic_radius * 0.2, 0)
                control = (diacritic_center[0], diacritic_center[1] - diacritic_radius * 0.8, 0)
                cedilla_points = []
                for t in np.linspace(0, 1, num_points):
                    # Quadratic Bezier 曲線の計算
                    x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0]
                    y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1]
                    cedilla_points.append((x, y, 0))
                vertices_list.append(np.array(cedilla_points))
            drawn_diacritic = True
            break  # 1回だけ追加するのでループを抜ける


class AsemicGlyph(BaseShape):
    """アセミック文字（抽象的文字）形状生成器。"""
    
    def generate(
        self,
        region: tuple[float, float, float, float] = (-0.5, -0.5, 0.5, 0.5),
        smoothing_radius: float = 0.05,
        diacritic_probability: float = 0.3,
        diacritic_radius: float = 0.04,
        random_seed: float = 42.0,
        **_params: Any
    ) -> list[np.ndarray]:
        """アセミック文字形状を生成する。
        
        Args:
            region: (x0, y0, x1, y1) の領域
            smoothing_radius: 補間用Bézier曲線の半径
            diacritic_probability: 使用ノード付近にディアクリティカルを追加する確率
            diacritic_radius: ディアクリティカル用のサイズ
            random_seed: 乱数シード
            **_params: 追加パラメータ（無視される）
            
        Returns:
            各ストローク・ディアクリティカルの頂点列を格納したリスト
        """
        # 乱数状態の初期化
        random.seed(int(random_seed))
        vertices_list = []
        x0, y0, x1, y1 = region
        cell_width = x1 - x0
        cell_height = y1 - y0
        cell_margin = min(0.025, cell_width / 8, cell_height / 8)

        # ノード生成（配置モードを選択："grid", "hexagon", "poisson" など）
        placement_mode = "poisson"
        nodes = generate_nodes(region, cell_margin, placement_mode)

        # RNG の構築とランダムウォークによるストローク生成
        _, adjacency = relative_neighborhood_graph(nodes)
        strokes_indices = random_walk_strokes(nodes, adjacency)
        used_nodes = {i for stroke in strokes_indices for i in stroke}

        # 各ストロークの生成（スナップ & スムージング）
        for stroke in strokes_indices:
            if len(stroke) < 2:
                continue
            original_stroke = [nodes[i] for i in stroke]
            snapped_stroke = snap_stroke(original_stroke)
            smoothed = smooth_polyline(snapped_stroke, smoothing_radius)
            vertices_list.append(np.array(smoothed))

        # ディアクリティカルの追加
        add_diacritic(vertices_list, nodes, used_nodes, diacritic_probability, diacritic_radius)

        return vertices_list