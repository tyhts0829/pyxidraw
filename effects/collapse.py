from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .registry import effect
from engine.core.geometry import Geometry


@njit(fastmath=True, cache=True)
def _subdivide_line(start: np.ndarray, end: np.ndarray, divisions: int) -> np.ndarray:
    """線分を指定された分割数で細分化します。"""
    if divisions <= 1:
        return np.array([start, end])
    
    # 細分化されたポイントを生成
    t_values = np.linspace(0, 1, divisions + 1)
    points = np.empty((divisions + 1, 3), dtype=np.float32)
    
    for i in range(divisions + 1):
        t = t_values[i]
        points[i] = start * (1 - t) + end * t
    
    return points


@njit(fastmath=True, cache=True)
def _apply_collapse_to_coords(
    coords: np.ndarray,
    offsets: np.ndarray,
    intensity: float,
    n_divisions: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """座標とオフセット配列にcollapseエフェクトを適用します。"""
    if intensity == 0.0 or n_divisions == 0:
        return coords.copy(), offsets.copy()
    
    if len(coords) == 0:
        return coords.copy(), offsets.copy()
    
    np.random.seed(seed)
    
    # 結果を格納するリスト
    all_coords = []
    all_offsets = []
    
    # offsetsからポリラインを抽出
    start_idx = 0
    for end_idx in offsets:
        if start_idx >= end_idx:
            start_idx = end_idx
            continue
            
        vertices = coords[start_idx:end_idx]
        
        if vertices.shape[0] < 2:
            # 単一点の場合はそのまま追加
            all_coords.append(vertices)
            all_offsets.append(np.array([vertices.shape[0]], dtype=offsets.dtype))
            start_idx = end_idx
            continue
        
        # ポリライン内の各線分を処理
        polyline_coords = []
        
        for i in range(vertices.shape[0] - 1):
            start_point = vertices[i]
            end_point = vertices[i + 1]
            
            # 線分を細分化
            subdivided = _subdivide_line(start_point, end_point, n_divisions)
            
            # 細分化した各セグメントにノイズを適用
            for j in range(subdivided.shape[0] - 1):
                seg_start = subdivided[j]
                seg_end = subdivided[j + 1]
                
                # メイン方向を求める
                main_dir = seg_end - seg_start
                main_norm = np.linalg.norm(main_dir)
                if main_norm < 1e-12:
                    polyline_coords.append(seg_start)
                    polyline_coords.append(seg_end)
                    continue
                    
                norm_main_dir = main_dir / main_norm
                
                # ノイズベクトルを生成
                noise_vector = np.random.randn(3) / 5.0
                
                # ノイズをメイン方向と直交する方向に変換
                ortho_dir = np.cross(norm_main_dir, noise_vector)
                ortho_norm = np.linalg.norm(ortho_dir)
                if ortho_norm < 1e-12:
                    polyline_coords.append(seg_start)
                    polyline_coords.append(seg_end)
                    continue
                    
                ortho_dir = ortho_dir / ortho_norm
                
                # ノイズを加える
                noise = ortho_dir * intensity
                
                # 変形された線分を追加
                noisy_start = seg_start + noise
                noisy_end = seg_end + noise
                polyline_coords.append(noisy_start)
                polyline_coords.append(noisy_end)
        
        if len(polyline_coords) > 0:
            polyline_array = np.array(polyline_coords, dtype=coords.dtype)
            all_coords.append(polyline_array)
            all_offsets.append(np.array([polyline_array.shape[0]], dtype=offsets.dtype))
        
        start_idx = end_idx
    
    # すべての座標とオフセットを結合
    if len(all_coords) == 0:
        return coords.copy(), offsets.copy()
        
    combined_coords = np.vstack(all_coords)
    combined_offsets = np.cumsum(np.concatenate(all_offsets))
    
    return combined_coords, combined_offsets


@effect()
def collapse(
    g: Geometry,
    *,
    intensity: float = 0.5,
    subdivisions: float = 0.5,
    **_params: Any,
) -> Geometry:
    """線分を細分化してノイズで変形（純関数）。"""
    coords, offsets = g.as_arrays(copy=False)
    if len(coords) == 0 or intensity == 0.0 or subdivisions == 0.0:
        return Geometry(coords.copy(), offsets.copy())
    divisions = max(1, int(subdivisions * 10))
    new_coords, new_offsets = _apply_collapse_to_coords(coords, offsets, float(intensity), divisions)
    return Geometry(new_coords, new_offsets)
