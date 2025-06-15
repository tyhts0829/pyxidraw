from __future__ import annotations

from typing import Any, List

import numpy as np
from numba import njit

from .base import BaseEffect

# Constants
MAX_ALPHA = 2.0
MAX_N_POINTS = 50


@njit(fastmath=True, cache=True)
def catmull_rom_spline_alpha(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, 
    n_points: int = 20, alpha: float = 0.5
) -> np.ndarray:
    """α-Catmull-Romスプラインを使用し4点を補間します。"""
    
    def tj(ti, pi, pj):
        # Handle zero distance
        dist = np.linalg.norm(pj - pi)
        if dist < 1e-12:
            dist = 1e-12
        return ti + dist**alpha
    
    tau0 = 0.0
    tau1 = tj(tau0, p0, p1)
    tau2 = tj(tau1, p1, p2)
    tau3 = tj(tau2, p2, p3)
    
    # Pre-allocate result array
    result = np.empty((n_points + 1, 3), dtype=np.float32)
    
    for i in range(n_points + 1):
        t = i / n_points
        t_ = tau1 + t * (tau2 - tau1)
        
        A1 = ((tau1 - t_) / (tau1 - tau0)) * p0 + ((t_ - tau0) / (tau1 - tau0)) * p1
        A2 = ((tau2 - t_) / (tau2 - tau1)) * p1 + ((t_ - tau1) / (tau2 - tau1)) * p2
        A3 = ((tau3 - t_) / (tau3 - tau2)) * p2 + ((t_ - tau2) / (tau3 - tau2)) * p3
        
        B1 = ((tau2 - t_) / (tau2 - tau0)) * A1 + ((t_ - tau0) / (tau2 - tau0)) * A2
        B2 = ((tau3 - t_) / (tau3 - tau1)) * A2 + ((t_ - tau1) / (tau3 - tau1)) * A3
        
        C = ((tau2 - t_) / (tau2 - tau1)) * B1 + ((t_ - tau1) / (tau2 - tau1)) * B2
        
        result[i] = C
    
    return result


@njit(fastmath=True, cache=True)
def _connect_core(
    lines_array: np.ndarray,
    line_starts: np.ndarray,
    line_lengths: np.ndarray,
    n_points: int,
    alpha: float,
    cyclic: bool,
    num_lines: int
):
    """接続処理用のNumba最適化コア関数です。"""
    if cyclic:
        loop_count = num_lines
    else:
        loop_count = num_lines - 1
    
    # Calculate total output size
    total_output_size = 0
    output_starts = np.empty(loop_count, dtype=np.int64)
    output_lengths = np.empty(loop_count, dtype=np.int64)
    
    for i in range(loop_count):
        len_a = line_lengths[i]
        
        if cyclic:
            next_idx = (i + 1) % num_lines
        else:
            next_idx = i + 1
        len_b = line_lengths[next_idx]
        
        if len_a < 2 or len_b < 2:
            output_length = len_a + len_b
        else:
            spline_length = n_points + 1
            output_length = len_a - 1 + spline_length + len_b - 1
        
        output_starts[i] = total_output_size
        output_lengths[i] = output_length
        total_output_size += output_length
    
    # Pre-allocate result array
    result_array = np.empty((total_output_size, 3), dtype=np.float32)
    
    # Process connections
    for i in range(loop_count):
        start_a = line_starts[i]
        len_a = line_lengths[i]
        lineA = lines_array[start_a:start_a + len_a]
        
        if cyclic:
            next_idx = (i + 1) % num_lines
        else:
            next_idx = i + 1
            
        start_b = line_starts[next_idx]
        len_b = line_lengths[next_idx]
        lineB = lines_array[start_b:start_b + len_b]
        
        output_start = output_starts[i]
        
        if len_a < 2 or len_b < 2:
            # Simple concatenation
            result_array[output_start:output_start + len_a] = lineA
            result_array[output_start + len_a:output_start + len_a + len_b] = lineB
        else:
            p0 = lineA[-2]
            p1 = lineA[-1]
            p2 = lineB[0]
            p3 = lineB[1]
            
            spline_points = catmull_rom_spline_alpha(p0, p1, p2, p3, n_points=n_points, alpha=alpha)
            
            # Copy data
            result_array[output_start:output_start + len_a - 1] = lineA[:-1]
            result_array[output_start + len_a - 1:output_start + len_a - 1 + len(spline_points)] = spline_points
            result_array[output_start + len_a - 1 + len(spline_points):output_start + output_lengths[i]] = lineB[1:]
    
    return result_array, output_starts, output_lengths


class Connect(BaseEffect):
    """Catmull-Romスプラインを使用して複数の線を滑らかに接続します。"""
    
    def apply(self, vertices_list: list[np.ndarray], 
             n_points: float = 0.5,
             alpha: float = 0.0,
             cyclic: bool = False,
             **params: Any) -> list[np.ndarray]:
        """接続エフェクトを適用します。
        
        Args:
            vertices_list: 入力頂点配列
            n_points: 補間点数 (0.0-1.0、0-50にマッピング)
            alpha: スプラインテンションパラメータ (0.0-1.0、0-2にマッピング)
            cyclic: 最後の線を最初の線に接続するか
            **params: 追加パラメータ（無視される）
            
        Returns:
            接続された頂点配列
        """
        alpha = alpha * MAX_ALPHA
        n_points = int(n_points * MAX_N_POINTS)
        
        if n_points < 2 or len(vertices_list) <= 1:
            return vertices_list
        
        # Consolidate line data into single array with index info
        total_vertices = sum(len(line) for line in vertices_list)
        lines_array = np.empty((total_vertices, 3), dtype=np.float32)
        line_starts = np.empty(len(vertices_list), dtype=np.int64)
        line_lengths = np.empty(len(vertices_list), dtype=np.int64)
        
        current_pos = 0
        for i, line in enumerate(vertices_list):
            line_starts[i] = current_pos
            line_lengths[i] = len(line)
            lines_array[current_pos:current_pos + len(line)] = line
            current_pos += len(line)
        
        result_array, output_starts, output_lengths = _connect_core(
            lines_array, line_starts, line_lengths, 
            n_points, alpha, cyclic, len(vertices_list)
        )
        
        # Convert back to list of arrays
        new_lines = []
        for i in range(len(output_starts)):
            start = output_starts[i]
            length = output_lengths[i]
            new_lines.append(result_array[start:start + length].copy())
        
        return new_lines