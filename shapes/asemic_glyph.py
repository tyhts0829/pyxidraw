from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np
from scipy.spatial import cKDTree

from .base import BaseShape

# 型エイリアス
Point3D = tuple[float, float, float]
Region = tuple[float, float, float, float]


@dataclass
class AsemicGlyphConfig:
    """アセミック文字生成の設定パラメータ"""
    min_distance: float = 0.1
    snap_angle_degrees: float = 60.0
    smoothing_points: int = 5
    walk_min_steps: int = 2
    walk_max_steps: int = 4
    poisson_radius_divisor: float = 8.0
    poisson_trials: int = 30


def distance(p: Point3D, q: Point3D) -> float:
    """2次元ユークリッド距離を計算"""
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def relative_neighborhood_graph(nodes: list[Point3D], config: AsemicGlyphConfig) -> tuple[list[tuple[int, int]], dict]:
    """
    RNG (Relative Neighborhood Graph) を構築する。KD-Treeを使用してO(n²log n)で最適化。

    Args:
        nodes: [(x, y, z), ...] の点のリスト
        config: 設定パラメータ

    Returns:
        edges: (i, j) のタプルのリスト
        adjacency: 各ノード番号と隣接ノード番号のリストの辞書
    """
    n = len(nodes)
    if n < 2:
        return [], {i: [] for i in range(n)}
    
    # 2D座標のみを抽出してKD-Treeを構築
    points_2d = np.array([(node[0], node[1]) for node in nodes])
    tree = cKDTree(points_2d)
    
    edges: list[tuple[int, int]] = []
    adjacency = {i: [] for i in range(n)}

    for i in range(n):
        # 近傍ノードを効率的に取得（最大距離で制限）
        max_search_radius = max(abs(points_2d[:, 0].max() - points_2d[:, 0].min()),
                               abs(points_2d[:, 1].max() - points_2d[:, 1].min()))
        neighbors = tree.query_ball_point(points_2d[i], max_search_radius)
        
        for j in neighbors:
            if j <= i:  # 重複を避ける
                continue
                
            dij = distance(nodes[i], nodes[j])
            if dij < config.min_distance:
                continue
                
            # RNG条件チェック：より効率的な近傍探索
            edge_valid = True
            # dijより小さい距離の点のみをチェック
            potential_blockers = tree.query_ball_point(points_2d[i], dij)
            
            for k in potential_blockers:
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


def random_walk_strokes_generator(nodes: list[Point3D], adjacency: dict, config: AsemicGlyphConfig, rng: Optional[random.Random] = None) -> Iterator[list[int]]:
    """
    RNG上でランダムウォークによりストロークを生成する（ジェネレータ版）。
    メモリ効率を向上させ、大きなグラフでも処理可能。

    Args:
        nodes: ノードリスト
        adjacency: 隣接リスト
        config: 設定パラメータ
        rng: 乱数生成器（テスト用）

    Yields:
        各ストロークを構成するノードのインデックスリスト
    """
    if rng is None:
        rng = random
        
    n = len(nodes)
    # 隣接リストのコピー（メモリ効率を考慮してset使用）
    adj = {i: set(neighbors) for i, neighbors in adjacency.items()}
    
    while True:
        # 利用可能なノードを効率的に検索
        candidates = [i for i in range(n) if adj[i]]
        if not candidates:
            break
            
        start = rng.choice(candidates)
        stroke = [start]
        current = start
        steps = rng.randint(config.walk_min_steps, config.walk_max_steps)
        
        for _ in range(steps):
            if not adj[current]:
                break
            next_node = rng.choice(list(adj[current]))
            
            # エッジを削除（双方向）
            adj[current].discard(next_node)
            adj[next_node].discard(current)
            
            stroke.append(next_node)
            current = next_node
            
        if len(stroke) >= 2:  # 最小長さのストロークのみ生成
            yield stroke


def random_walk_strokes(nodes: list[Point3D], adjacency: dict, config: AsemicGlyphConfig, rng: Optional[random.Random] = None) -> list[list[int]]:
    """
    互換性のためのラッパー関数
    """
    return list(random_walk_strokes_generator(nodes, adjacency, config, rng))


def snap_stroke(original: list[Point3D], config: AsemicGlyphConfig) -> list[Point3D]:
    """
    各セグメントの方向を指定角度刻みにスナップする。

    Args:
        original: 元の頂点列
        config: 設定パラメータ

    Returns:
        スナップ後の頂点列
    """
    if len(original) < 2:
        return original
        
    snapped = [original[0]]
    snap_angle = config.snap_angle_degrees
    
    for point in original[1:]:
        last = snapped[-1]
        dx = point[0] - last[0]
        dy = point[1] - last[1]
        L = math.sqrt(dx**2 + dy**2)
        
        if L < 1e-10:  # ほぼ同じ点の場合はスキップ
            continue
            
        theta_deg = math.degrees(math.atan2(dy, dx))
        snapped_theta_deg = round(theta_deg / snap_angle) * snap_angle
        snapped_theta = math.radians(snapped_theta_deg)
        
        new_point = (
            last[0] + L * math.cos(snapped_theta),
            last[1] + L * math.sin(snapped_theta),
            0,
        )
        snapped.append(new_point)
    return snapped


def smooth_polyline(polyline: list[Point3D], smoothing_radius: float, config: AsemicGlyphConfig) -> list[Point3D]:
    """
    quadratic Bézier曲線により各内部コーナーを補間し、なめらかなポリラインを生成する。
    NumPy配列を活用してベクトル演算を最適化。

    Args:
        polyline: 頂点列
        smoothing_radius: 補間用の最大オフセット距離
        config: 設定パラメータ

    Returns:
        補間後の頂点列
    """
    if len(polyline) < 3:
        return polyline
        
    new_points = [polyline[0]]
    num_arc_points = config.smoothing_points
    
    # NumPy配列に変換してベクトル演算を効率化
    points_array = np.array(polyline)
    
    for i in range(1, len(polyline) - 1):
        A, B, C = points_array[i-1], points_array[i], points_array[i+1]
        
        # ベクトル計算を最適化
        vec_BA = A - B
        vec_BC = C - B
        
        dAB = np.linalg.norm(vec_BA)
        dBC = np.linalg.norm(vec_BC)
        
        if dAB < 1e-10 or dBC < 1e-10:
            continue
            
        d = min(smoothing_radius, dAB / 2, dBC / 2)
        
        # 正規化ベクトル
        uBA = vec_BA / dAB
        uBC = vec_BC / dBC
        
        A_prime = B + uBA * d
        C_prime = B - uBC * d
        
        if distance(new_points[-1], tuple(A_prime)) > 0.1:
            new_points.append(tuple(A_prime))
            
        # ベジエ曲線の点を効率的に計算
        t_values = np.linspace(0, 1, num_arc_points + 2)[1:-1]
        t_squared = t_values ** 2
        one_minus_t = 1 - t_values
        one_minus_t_squared = one_minus_t ** 2
        
        # ベクトル化されたベジエ曲線計算
        bezier_points = (one_minus_t_squared[:, np.newaxis] * A_prime[np.newaxis, :] +
                        2 * one_minus_t[:, np.newaxis] * t_values[:, np.newaxis] * B[np.newaxis, :] +
                        t_squared[:, np.newaxis] * C_prime[np.newaxis, :])
        
        for point in bezier_points:
            new_points.append(tuple(point))
            
        new_points.append(tuple(C_prime))
        
    new_points.append(polyline[-1])
    return new_points


def generate_nodes(region: Region, cell_margin: float, placement_mode: str, config: Optional[AsemicGlyphConfig] = None) -> list[Point3D]:
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
        if config is None:
            config = AsemicGlyphConfig()
        r = min(x_max - x_min, y_max - y_min) / config.poisson_radius_divisor  # rを大きくするとノード数が減る
        k = config.poisson_trials  # 試行回数
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


class DiacriticFactory:
    """ディアクリティカル（アクセント記号）生成のファクトリークラス"""
    
    @staticmethod
    def create_circle(center: Point3D, radius: float) -> np.ndarray:
        """円形アクセントを生成する。"""
        n_sides = 20
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        x_coords = center[0] + radius * np.cos(angles)
        y_coords = center[1] + radius * np.sin(angles)
        z_coords = np.zeros(n_sides)
        
        points = np.column_stack((x_coords, y_coords, z_coords))
        # 閉じた形状にする
        return np.vstack([points, points[0:1]])
    
    @staticmethod
    def create_tilde(center: Point3D, radius: float) -> np.ndarray:
        """チルダアクセントを生成する。"""
        num_points = 10
        t_values = np.linspace(0, 1, num_points)
        length = radius * 2
        amplitude = radius / 2
        
        x_coords = center[0] - radius + t_values * length
        y_coords = center[1] + amplitude * np.sin(np.pi * t_values)
        z_coords = np.zeros(num_points)
        
        return np.column_stack((x_coords, y_coords, z_coords))
    
    @staticmethod
    def create_grave(center: Point3D, radius: float) -> np.ndarray:
        """グレイブアクセントを生成する。"""
        start = np.array(center)
        end = np.array([center[0] - radius * 0.8, center[1] + radius * 0.4, 0])
        return np.array([start, end])
    
    @staticmethod
    def create_umlaut(center: Point3D, radius: float) -> list[np.ndarray]:
        """ウムラウトアクセントを生成する。"""
        dot_radius = radius * 0.3
        n_sides = 8
        offsets = [(-radius * 0.5, 0), (radius * 0.5, 0)]
        dots = []
        
        for dx, dy in offsets:
            dot_center = (center[0] + dx, center[1] + dy, 0)
            angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
            x_coords = dot_center[0] + dot_radius * np.cos(angles)
            y_coords = dot_center[1] + dot_radius * np.sin(angles)
            z_coords = np.zeros(n_sides)
            
            points = np.column_stack((x_coords, y_coords, z_coords))
            dots.append(np.vstack([points, points[0:1]]))
        
        return dots
    
    @staticmethod
    def create_acute(center: Point3D, radius: float) -> np.ndarray:
        """アキュートアクセントを生成する。"""
        start = np.array([center[0] - radius * 0.3, center[1] + radius * 0.2, 0])
        end = np.array([center[0] + radius * 0.3, center[1] + radius * 0.7, 0])
        return np.array([start, end])
    
    @staticmethod
    def create_circumflex(center: Point3D, radius: float) -> np.ndarray:
        """サーカムフレックスアクセントを生成する。"""
        left = np.array([center[0] - radius, center[1], 0])
        peak = np.array([center[0], center[1] + radius, 0])
        right = np.array([center[0] + radius, center[1], 0])
        return np.array([left, peak, right])
    
    @staticmethod
    def create_caron(center: Point3D, radius: float) -> np.ndarray:
        """ハチェクアクセントを生成する。"""
        left = np.array([center[0] - radius, center[1] + radius * 0.2, 0])
        bottom = np.array([center[0], center[1] - radius * 0.2, 0])
        right = np.array([center[0] + radius, center[1] + radius * 0.2, 0])
        return np.array([left, bottom, right])
    
    @staticmethod
    def create_cedilla(center: Point3D, radius: float) -> np.ndarray:
        """セディーヤアクセントを生成する。"""
        num_points = 8
        start = np.array([center[0] - radius * 0.5, center[1] - radius * 0.2, 0])
        end = np.array([center[0] + radius * 0.5, center[1] - radius * 0.2, 0])
        control = np.array([center[0], center[1] - radius * 0.8, 0])
        
        t_values = np.linspace(0, 1, num_points)
        # Quadratic Bezier 曲線のベクトル化
        points = ((1 - t_values[:, np.newaxis]) ** 2 * start[np.newaxis, :] +
                 2 * (1 - t_values[:, np.newaxis]) * t_values[:, np.newaxis] * control[np.newaxis, :] +
                 t_values[:, np.newaxis] ** 2 * end[np.newaxis, :])
        
        return points
    
    DIACRITIC_TYPES = {
        "circle": create_circle.__func__,
        "tilde": create_tilde.__func__,
        "grave": create_grave.__func__,
        "umlaut": create_umlaut.__func__,
        "acute": create_acute.__func__,
        "circumflex": create_circumflex.__func__,
        "caron": create_caron.__func__,
        "cedilla": create_cedilla.__func__,
    }
    
    @classmethod
    def create_random_diacritic(cls, center: Point3D, radius: float, rng: Optional[random.Random] = None) -> list[np.ndarray]:
        """ランダムなディアクリティカルを生成する。"""
        if rng is None:
            rng = random
            
        diacritic_type = rng.choice(list(cls.DIACRITIC_TYPES.keys()))
        result = cls.DIACRITIC_TYPES[diacritic_type](center, radius)
        
        # umlautは複数の形状を返す
        if isinstance(result, list):
            return result
        else:
            return [result]


def add_diacritic(
    vertices_list: list[np.ndarray],
    nodes: list[Point3D],
    used_nodes: set,
    diacritic_probability: float,
    diacritic_radius: float,
    rng: Optional[random.Random] = None,
) -> None:
    """
    使用されたノードの中から、一定の確率でディアクリティカル（アクセント記号）を追加する。
    ファクトリーパターンでリファクタリングした版。

    Args:
        vertices_list: 既存の頂点リストに追加する
        nodes: すべてのノードのリスト
        used_nodes: ストローク生成で使用されたノードのインデックス集合
        diacritic_probability: ディアクリティカルを追加する確率
        diacritic_radius: ディアクリティカルのサイズ
        rng: 乱数生成器（テスト用）
    """
    if rng is None:
        rng = random
        
    for i in used_nodes:
        if rng.random() < diacritic_probability:
            offset_x = rng.uniform(-diacritic_radius, diacritic_radius)
            offset_y = rng.uniform(-diacritic_radius, diacritic_radius)
            diacritic_center = (nodes[i][0] + offset_x, nodes[i][1] + offset_y, 0)
            
            diacritic_shapes = DiacriticFactory.create_random_diacritic(
                diacritic_center, diacritic_radius, rng
            )
            vertices_list.extend(diacritic_shapes)
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
        # 乱数状態の初期化（テスト可能性のために分離）
        rng = random.Random(int(random_seed))
        config = AsemicGlyphConfig()
        vertices_list = []
        
        x0, y0, x1, y1 = region
        cell_width = x1 - x0
        cell_height = y1 - y0
        cell_margin = min(0.025, cell_width / 8, cell_height / 8)

        # ノード生成（配置モードを選択："grid", "hexagon", "poisson" など）
        placement_mode = "poisson"
        # グローバルrandom状態を一時的に設定（generate_nodesがglobal randomを使用するため）
        old_state = random.getstate()
        random.setstate(rng.getstate())
        nodes = generate_nodes(region, cell_margin, placement_mode, config)
        rng.setstate(random.getstate())
        random.setstate(old_state)

        # RNG の構築とランダムウォークによるストローク生成
        _, adjacency = relative_neighborhood_graph(nodes, config)
        strokes_indices = random_walk_strokes(nodes, adjacency, config, rng)
        used_nodes = {i for stroke in strokes_indices for i in stroke}

        # 各ストロークの生成（スナップ & スムージング）
        for stroke in strokes_indices:
            if len(stroke) < 2:
                continue
            original_stroke = [nodes[i] for i in stroke]
            snapped_stroke = snap_stroke(original_stroke, config)
            smoothed = smooth_polyline(snapped_stroke, smoothing_radius, config)
            vertices_list.append(np.array(smoothed))

        # ディアクリティカルの追加
        add_diacritic(vertices_list, nodes, used_nodes, diacritic_probability, diacritic_radius, rng)

        return vertices_list