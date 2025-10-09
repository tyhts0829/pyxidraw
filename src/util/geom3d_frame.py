from __future__ import annotations

"""
どこで: `util` の 3D フレーム選択ヘルパ。
何を: ジオメトリ全体に対して、共平面であれば安定な XY 整列フレームを選び、整列済み座標と逆変換情報を返す。
なぜ: `effects.fill`/`effects.partition` 等で、XY 限定ではなく「任意の共平面」入力に対して一貫した処理を行うため。

提供関数:
- `choose_coplanar_frame(coords, offsets, eps_abs=1e-5, eps_rel=1e-4)`
    - 戻り値: `(planar: bool, v2d_all: np.ndarray, R: np.ndarray, z: float, ref_height_global: float)`
    - アルゴリズム: リング優先で姿勢推定（`transform_to_xy_plane`）→ PCA(SVD) フォールバック → 全体 XY 整列 → 絶対/相対閾値で共平面判定。
"""

from typing import Tuple

import numpy as np

from .geom3d_ops import transform_to_xy_plane


def _planarity_threshold(diag: float, eps_abs: float, eps_rel: float) -> float:
    return max(float(eps_abs), float(eps_rel) * float(diag))


def choose_coplanar_frame(
    coords: np.ndarray,
    offsets: np.ndarray,
    *,
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-4,
) -> Tuple[bool, np.ndarray, np.ndarray, float, float]:
    """全体の共平面フレームを選択し、XY に整列する。

    Parameters
    ----------
    coords : np.ndarray
        形状 `(N,3)` の全頂点配列。
    offsets : np.ndarray
        形状 `(M+1,)` の各リング開始 index。
    eps_abs : float, default 1e-5
        平面性判定の絶対許容。
    eps_rel : float, default 1e-4
        平面性判定の相対許容（バウンディング対角に対する比）。

    Returns
    -------
    tuple
        `(planar, v2d_all, R_all, z_all, ref_height_global)` を返す。
        - planar が False の場合、`v2d_all` は単純な同次元コピー相当（z 平行移動のみ）。
    """
    # 1) 非退化リングから姿勢を選ぶ
    R_all = np.eye(3)
    z_all = 0.0
    chosen = False
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e - s < 3:
            continue
        v2d_i, R_i, z_i = transform_to_xy_plane(coords[s:e])
        z_span_i = float(np.max(np.abs(v2d_i[:, 2]))) if v2d_i.size else 0.0
        mins_i = np.min(coords[s:e], axis=0)
        maxs_i = np.max(coords[s:e], axis=0)
        diag_i = float(np.sqrt(np.sum((maxs_i - mins_i) ** 2)))
        if z_span_i <= _planarity_threshold(diag_i, eps_abs, eps_rel):
            R_all = R_i
            z_all = z_i
            chosen = True
            break

    # 2) PCA フォールバック
    if not chosen and coords.shape[0] >= 3:
        P = coords.astype(np.float64, copy=False)
        C = P - np.mean(P, axis=0)
        _u, _s, Vt = np.linalg.svd(C, full_matrices=False)
        normal = Vt[-1, :]
        nz = float(np.linalg.norm(normal))
        if nz > 0.0:
            normal = normal / nz
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            rot_axis = np.cross(normal, z_axis)
            na = float(np.linalg.norm(rot_axis))
            if na > 0.0:
                rot_axis = rot_axis / na
                cos_t = float(np.dot(normal, z_axis))
                cos_t = -1.0 if cos_t < -1.0 else (1.0 if cos_t > 1.0 else cos_t)
                ang = float(np.arccos(cos_t))
                K = np.zeros((3, 3), dtype=np.float64)
                K[0, 1] = -rot_axis[2]
                K[0, 2] = rot_axis[1]
                K[1, 0] = rot_axis[2]
                K[1, 2] = -rot_axis[0]
                K[2, 0] = -rot_axis[1]
                K[2, 1] = rot_axis[0]
                R_all = np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)

    # 3) 全体整列と共平面判定
    v2d_all = coords @ R_all.T
    v2d_all = v2d_all.astype(np.float32, copy=False)
    z_all = float(v2d_all[0, 2]) if v2d_all.size else 0.0
    v2d_all[:, 2] -= z_all
    z_span_all = float(np.max(np.abs(v2d_all[:, 2]))) if v2d_all.size else 0.0
    mins_all = np.min(coords, axis=0)
    maxs_all = np.max(coords, axis=0)
    diag_all = float(np.sqrt(np.sum((maxs_all - mins_all) ** 2)))
    planar = z_span_all <= _planarity_threshold(diag_all, eps_abs, eps_rel)

    # 4) 参照高さ
    if v2d_all.size:
        ref_height_global = float(np.max(v2d_all[:, 1]) - np.min(v2d_all[:, 1]))
    else:
        ref_height_global = 0.0
    return planar, v2d_all, R_all, z_all, ref_height_global
