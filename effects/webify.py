from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit

from util.geometry import transform_back, transform_to_xy_plane

from .base import BaseEffect


class Webify(BaseEffect):
    """形状にウェブ状のストリング構造を追加します。"""

    MAX_NUM_CANDIDATE_LINES = 2500
    MAX_RELAXATION_ITERATIONS = 50
    MAX_STEP = 0.5

    def apply(
        self,
        vertices_list: list[np.ndarray],
        num_candidate_lines: float = 0.5,
        relaxation_iterations: float = 0.5,
        step: float = 0.5,
        **_params: Any,
    ) -> list[np.ndarray]:
        """複数の閉曲線にウェブ構造を適用します。

        入力が3次元の場合、まず transform_to_xy_plane でXY平面に変換し、
        create_web で処理後、transform_back で元の座標系に戻します。

        Args:
            vertices_list: 各要素は (N,3) の頂点配列
            num_candidate_lines: 候補線の数（0.0-1.0、デフォルト0.5）
            relaxation_iterations: 弾性平衡シミュレーションの反復回数（0.0-1.0、デフォルト0.5）
            step: シミュレーションのステップサイズ（0.0-1.0、デフォルト0.5）
            **_params: 追加パラメータ（無視される）

        Returns:
            ウェブ構造を追加された頂点配列のリスト
        """
        if not vertices_list:
            return vertices_list

        # パラメータをスケーリング
        num_lines = int(num_candidate_lines * self.MAX_NUM_CANDIDATE_LINES)
        iterations = int(relaxation_iterations * self.MAX_RELAXATION_ITERATIONS)
        step_size = step * self.MAX_STEP

        result = []
        for vertices in vertices_list:
            # 3次元入力の場合、まずXY平面に変換する
            transformed, R, z = transform_to_xy_plane(vertices)
            
            # XY平面上の閉曲線として create_web を実行
            polylines_xy = create_web(
                transformed,
                num_candidate_lines=num_lines,
                relaxation_iterations=iterations,
                step=step_size,
            )
            
            # 生成された各ポリラインを元の座標系に戻す
            for poly in polylines_xy:
                poly_back = transform_back(poly, R, z)
                result.append(poly_back)
                
        return result


@njit(fastmath=True, cache=True)
def line_segment_intersection_nb(Ax, Ay, Bx, By, p0x, p0y, p1x, p1y):
    r_x = Bx - Ax
    r_y = By - Ay
    s_x = p1x - p0x
    s_y = p1y - p0y
    rxs = r_x * s_y - r_y * s_x
    if abs(rxs) < 1e-8:
        return False, 0.0, 0.0, 0.0
    qp_x = p0x - Ax
    qp_y = p0y - Ay
    t = (qp_x * s_y - qp_y * s_x) / rxs
    u = (qp_x * r_y - qp_y * r_x) / rxs
    if t < 0 or t > 1 or u < 0 or u > 1:
        return False, 0.0, 0.0, 0.0
    inter_x = Ax + t * r_x
    inter_y = Ay + t * r_y
    return True, t, inter_x, inter_y


@njit(fastmath=True, cache=True, inline="always")
def fract(x):
    return x - math.floor(x)


@njit(fastmath=True, cache=True)
def generate_candidate_line_from_curve_nb(closed_curve, cl, seed):
    """
    指定された cl と seed に基づいて候補線を生成する。
    """
    N = closed_curve.shape[0]
    # 擬似乱数生成（GLSL的テクニック）
    seed1 = cl * 12.9898 + seed + 78.233
    r1 = fract(math.sin(seed1) * 43758.5453)
    seed2 = cl * 93.9898 + seed + 12.345
    r2 = fract(math.sin(seed2) * 43758.5453)
    seed3 = cl * 45.1234 + seed + 98.765
    r3 = fract(math.sin(seed3) * 43758.5453)
    seed4 = cl * 67.8901 + seed + 23.456
    r4 = fract(math.sin(seed4) * 43758.5453)

    # A点の生成：r1でインデックス、r2で補間
    idx1 = int(r1 * N)
    next_idx1 = (idx1 + 1) % N
    A_x = closed_curve[idx1, 0] * (1 - r2) + closed_curve[next_idx1, 0] * r2
    A_y = closed_curve[idx1, 1] * (1 - r2) + closed_curve[next_idx1, 1] * r2

    # B点の生成：r3でインデックス、r4で補間
    idx2 = int(r3 * N)
    next_idx2 = (idx2 + 1) % N
    B_x = closed_curve[idx2, 0] * (1 - r4) + closed_curve[next_idx2, 0] * r4
    B_y = closed_curve[idx2, 1] * (1 - r4) + closed_curve[next_idx2, 1] * r4

    return A_x, A_y, B_x, B_y


@njit(fastmath=True, cache=True)
def generate_best_candidate_line_from_curve_nb(closed_curve, cl, base_seed, n_attempts):
    """
    同じ cl に対して、base_seed から始め n_attempts 回シードを変化させて候補線を生成し、
    2点間の距離が最大となる候補線を選択する。
    """
    best_dist2 = -1.0
    best_A_x = 0.0
    best_A_y = 0.0
    best_B_x = 0.0
    best_B_y = 0.0
    for i in range(n_attempts):
        current_seed = base_seed + i
        A_x, A_y, B_x, B_y = generate_candidate_line_from_curve_nb(closed_curve, cl, current_seed)
        dx = B_x - A_x
        dy = B_y - A_y
        dist2 = dx * dx + dy * dy
        if dist2 > best_dist2:
            best_dist2 = dist2
            best_A_x = A_x
            best_A_y = A_y
            best_B_x = B_x
            best_B_y = B_y
    return best_A_x, best_A_y, best_B_x, best_B_y


@njit(fastmath=True, cache=True)
def elastic_relaxation_nb(positions, edges, fixed, iterations, step):
    n = positions.shape[0]
    for it in range(iterations):
        forces = np.zeros((n, 2), dtype=positions.dtype)
        m = edges.shape[0]
        for e in range(m):
            i = edges[e, 0]
            j = edges[e, 1]
            diff0 = positions[j, 0] - positions[i, 0]
            diff1 = positions[j, 1] - positions[i, 1]
            forces[i, 0] += diff0
            forces[i, 1] += diff1
            forces[j, 0] -= diff0
            forces[j, 1] -= diff1
        max_force = 10.0
        for i in range(n):
            fx = forces[i, 0]
            fy = forces[i, 1]
            norm = np.sqrt(fx * fx + fy * fy)
            if norm > max_force:
                scale = max_force / norm
                forces[i, 0] *= scale
                forces[i, 1] *= scale
        for i in range(n):
            if not fixed[i]:
                positions[i, 0] += step * forces[i, 0]
                positions[i, 1] += step * forces[i, 1]
    return positions


def merge_edges_into_polylines(nodes, edges):
    """
    ノード集合とエッジリストから隣接エッジを連結して、
    ポリライン（np.array(shape=(N,3))）のリストとして返す。
    """
    graph = {i: set() for i in range(len(nodes))}
    for i, j in edges:
        graph[i].add(j)
        graph[j].add(i)

    visited_edges = set()
    polylines = []

    def edge_key(a, b):
        return tuple(sorted((a, b)))

    # チェーン探索（終端または分岐から）
    for i in range(len(nodes)):
        if len(graph[i]) != 2:
            for j in graph[i]:
                key = edge_key(i, j)
                if key in visited_edges:
                    continue
                chain = [i, j]
                visited_edges.add(key)
                prev, current = i, j
                while len(graph[current]) == 2:
                    next_candidates = list(graph[current] - {prev})
                    if not next_candidates:
                        break
                    next_node = next_candidates[0]
                    key = edge_key(current, next_node)
                    if key in visited_edges:
                        break
                    chain.append(next_node)
                    visited_edges.add(key)
                    prev, current = current, next_node
                if len(chain) >= 2:
                    polyline = np.array([[nodes[k][0], nodes[k][1], 0] for k in chain])
                    polylines.append(polyline)

    # サイクル（全ノードの次数が2の場合）の処理
    remaining_edges = []
    for i in range(len(nodes)):
        for j in graph[i]:
            key = edge_key(i, j)
            if key not in visited_edges:
                remaining_edges.append((i, j))
                visited_edges.add(key)
    used = set()
    for a, b in remaining_edges:
        if (a, b) in used or (b, a) in used:
            continue
        cycle = [a, b]
        used.add((a, b))
        used.add((b, a))
        prev, current = a, b
        while True:
            neighbors = list(graph[current])
            next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
            if next_node == cycle[0]:
                break
            else:
                cycle.append(next_node)
                used.add((current, next_node))
                used.add((next_node, current))
                prev, current = current, next_node
        polyline = np.array([[nodes[k][0], nodes[k][1], 0] for k in cycle])
        polylines.append(polyline)

    return polylines


@njit(fastmath=True, cache=True)
def create_web_nb(closed_curve, num_candidate_lines, relaxation_iterations, step):
    # 初期ノード（閉曲線上の点）を配列として確保
    n = closed_curve.shape[0]
    max_nodes = n + 2 * num_candidate_lines  # 各候補線で最大2点追加
    nodes = np.zeros((max_nodes, 3), dtype=np.float64)
    for i in range(n):
        nodes[i, 0] = closed_curve[i, 0]
        nodes[i, 1] = closed_curve[i, 1]
        nodes[i, 2] = 0.0
    current_n = n

    # エッジ配列の初期化（各エッジは [始点, 終点]）
    max_edges = n + 5 * num_candidate_lines  # 分割ごとにエッジ数が増加
    edges = np.zeros((max_edges, 2), dtype=np.int64)
    valid_edges = np.zeros(max_edges, dtype=np.bool_)
    for i in range(n):
        edges[i, 0] = i
        edges[i, 1] = (i + 1) % n
        valid_edges[i] = True
    current_m = n

    # 候補線によるエッジ分割ループ
    for cl in range(num_candidate_lines):
        A_x, A_y, B_x, B_y = generate_best_candidate_line_from_curve_nb(closed_curve, cl, base_seed=0, n_attempts=2)
        # 各候補線との交点を格納（最大20個まで）
        max_int = 20
        t_vals = np.empty(max_int, dtype=np.float64)
        edge_indices = np.empty(max_int, dtype=np.int64)
        int_x = np.empty(max_int, dtype=np.float64)
        int_y = np.empty(max_int, dtype=np.float64)
        count = 0
        # 現在のすべてのエッジについて交点判定
        for e in range(current_m):
            if not valid_edges[e]:
                continue
            i = edges[e, 0]
            j = edges[e, 1]
            p0x = nodes[i, 0]
            p0y = nodes[i, 1]
            p1x = nodes[j, 0]
            p1y = nodes[j, 1]
            hit, t_val, ix, iy = line_segment_intersection_nb(A_x, A_y, B_x, B_y, p0x, p0y, p1x, p1y)
            if hit:
                if count < max_int:
                    t_vals[count] = t_val
                    edge_indices[count] = e
                    int_x[count] = ix
                    int_y[count] = iy
                    count += 1
        # 交点が2点以上あれば、最もtが小さい2点を採用
        if count >= 2:
            # 単純な最小値探索
            min1 = 1.0e9
            min2 = 1.0e9
            idx1 = -1
            idx2 = -1
            for k in range(count):
                if t_vals[k] < min1:
                    min2 = min1
                    idx2 = idx1
                    min1 = t_vals[k]
                    idx1 = k
                elif t_vals[k] < min2:
                    min2 = t_vals[k]
                    idx2 = k
            if idx1 >= 0 and idx2 >= 0:
                # 1つ目の交点でエッジ分割
                e1 = edge_indices[idx1]
                i1 = edges[e1, 0]
                j1 = edges[e1, 1]
                valid_edges[e1] = False  # 元のエッジを無効化
                new_node1 = current_n
                nodes[new_node1, 0] = int_x[idx1]
                nodes[new_node1, 1] = int_y[idx1]
                nodes[new_node1, 2] = 0.0
                current_n += 1
                edges[current_m, 0] = i1
                edges[current_m, 1] = new_node1
                valid_edges[current_m] = True
                current_m += 1
                edges[current_m, 0] = new_node1
                edges[current_m, 1] = j1
                valid_edges[current_m] = True
                current_m += 1

                # 2つ目の交点でエッジ分割
                e2 = edge_indices[idx2]
                i2 = edges[e2, 0]
                j2 = edges[e2, 1]
                valid_edges[e2] = False
                new_node2 = current_n
                nodes[new_node2, 0] = int_x[idx2]
                nodes[new_node2, 1] = int_y[idx2]
                nodes[new_node2, 2] = 0.0
                current_n += 1
                edges[current_m, 0] = i2
                edges[current_m, 1] = new_node2
                valid_edges[current_m] = True
                current_m += 1
                edges[current_m, 0] = new_node2
                edges[current_m, 1] = j2
                valid_edges[current_m] = True
                current_m += 1

                # 新たに新ノード同士を接続するエッジを追加
                edges[current_m, 0] = new_node1
                edges[current_m, 1] = new_node2
                valid_edges[current_m] = True
                current_m += 1

    # 最終的に有効なエッジのみ抽出
    valid_count = 0
    for e in range(current_m):
        if valid_edges[e]:
            valid_count += 1
    valid_edges_arr = np.empty((valid_count, 2), dtype=np.int64)
    idx = 0
    for e in range(current_m):
        if valid_edges[e]:
            valid_edges_arr[idx, 0] = edges[e, 0]
            valid_edges_arr[idx, 1] = edges[e, 1]
            idx += 1

    # 固定ノード：閉曲線上の初期点は固定（以降の弾性平衡で動かさない）
    fixed = np.zeros(current_n, dtype=np.bool_)
    for i in range(n):
        fixed[i] = True

    # 弾性平衡シミュレーション（positionsは (n,2) 部分のみ更新）
    positions = nodes[:current_n, 0:2].copy()
    positions = elastic_relaxation_nb(positions, valid_edges_arr, fixed, relaxation_iterations, step)
    nodes[:current_n, 0:2] = positions

    return nodes[:current_n], valid_edges_arr


def create_web(closed_curve, num_candidate_lines=10, relaxation_iterations=20, step=0.1):
    """
    closed_curve: np.array(shape=(N,3)) （各点のz=0と仮定）
    """
    nodes, edges = create_web_nb(closed_curve, num_candidate_lines, relaxation_iterations, step)
    # ノードとエッジからポリライン（各頂点が[x,y,0]の形）を生成する
    polylines = merge_edges_into_polylines(nodes, edges)
    return polylines
