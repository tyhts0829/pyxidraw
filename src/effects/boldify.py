"""
boldify エフェクト（見かけの太線化）

- 各ポリラインに対し、XY 平面の法線方向に左右へオフセットした平行線を生成し、
  元の線と合わせて 3 本構成にして「太く見える」効果を与えます。
- 端点は隣接セグメントの法線、内部点は前後セグメント法線の平均を用いて滑らかに広げます。

主なパラメータ:
- boldness: 太さ（左右オフセット量の合計）[mm]。0 で変更なし。

特性/注意:
- Z 成分は保持（オフセットは XY のみ）。閉曲線でなくても動作します。
- 頂点数は最大で 3 倍（元＋左＋右）となり、描画コストが増えます。
"""

from __future__ import annotations

import numpy as np

from engine.core.geometry import Geometry

from .registry import effect


def _boldify_coords_with_offsets(
    coords: np.ndarray, offsets: np.ndarray, boldness: float
) -> tuple[np.ndarray, np.ndarray]:
    """太線化（XY平面の法線ベース）。Numba 依存を排し、2 パスで事前確保して高速化。"""
    if boldness <= 0 or coords.size == 0 or offsets.size <= 1:
        return coords.copy(), offsets.copy()

    # 1st pass: 出力サイズ見積もり
    total_vertices = 0
    total_lines = 0
    for i in range(len(offsets) - 1):
        start, end = int(offsets[i]), int(offsets[i + 1])
        L = max(0, end - start)
        if L < 2:
            total_vertices += L
            total_lines += 1
        else:
            total_vertices += 3 * L
            total_lines += 3

    if total_lines == 0:
        return coords.copy(), offsets.copy()

    out_coords = np.empty((total_vertices, 3), dtype=np.float32)
    out_offsets = np.empty(total_lines + 1, dtype=np.int32)
    out_offsets[0] = 0

    half = float(boldness) / 2.0
    ci = 0  # coord cursor
    oi = 0  # offsets cursor (points to last written offset index)

    for i in range(len(offsets) - 1):
        start, end = int(offsets[i]), int(offsets[i + 1])
        v = coords[start:end]
        L = v.shape[0]
        if L == 0:
            continue
        if L < 2:
            out_coords[ci : ci + L] = v
            ci += L
            out_offsets[oi + 1] = ci
            oi += 1
            continue

        # 元のライン
        out_coords[ci : ci + L] = v
        ci += L
        out_offsets[oi + 1] = ci
        oi += 1

        # セグメント法線（XY 平面）
        seg = v[1:] - v[:-1]
        n = np.zeros_like(seg, dtype=np.float32)
        n[:, 0] = -seg[:, 1]
        n[:, 1] = seg[:, 0]
        # 正規化（ゼロ除外）
        ln = np.sqrt(n[:, 0] * n[:, 0] + n[:, 1] * n[:, 1])
        ln = np.where(ln == 0.0, 1.0, ln)
        n[:, 0] /= ln
        n[:, 1] /= ln

        # 頂点法線（端点は隣接の法線、内部は隣接平均→正規化）
        vn = np.zeros_like(v, dtype=np.float32)
        vn[0] = n[0]
        vn[-1] = n[-1]
        if L > 2:
            vn[1:-1] = (n[:-1] + n[1:]) * 0.5
            mag = np.sqrt(vn[1:-1, 0] * vn[1:-1, 0] + vn[1:-1, 1] * vn[1:-1, 1])
            nz = mag > 0
            vn[1:-1][nz] /= mag[nz][:, None]

        left = v + vn * half
        right = v - vn * half

        out_coords[ci : ci + L] = left.astype(np.float32, copy=False)
        ci += L
        out_offsets[oi + 1] = ci
        oi += 1

        out_coords[ci : ci + L] = right.astype(np.float32, copy=False)
        ci += L
        out_offsets[oi + 1] = ci
        oi += 1

    return out_coords, out_offsets


@effect()
def boldify(g: Geometry, *, boldness: float = 0.5) -> Geometry:
    """平行線を追加して線を太く見せる（純関数）。"""
    coords, offsets = g.as_arrays(copy=False)
    new_coords, new_offsets = _boldify_coords_with_offsets(coords, offsets, float(boldness))
    return Geometry(new_coords, new_offsets)
