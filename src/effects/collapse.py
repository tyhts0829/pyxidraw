"""
collapse エフェクト（線の崩し/しわ寄せ）

- 各線分を細分化し、小区間ごとに主方向と直交するランダムベクトルでオフセットして
  「崩れた」「くしゃっとした」見た目を作る。
- ノイズはサブセグメントごとに独立に生成し、激しさは `intensity` で制御する。

主なパラメータ:
- intensity: 変位量（mm 相当, 0–10 推奨）。
- subdivisions: 細分回数（0–10, 0 で未細分化）。

特性/注意:
- 細分化を増やすと微細な揺らぎが増え、頂点数も増加する。
- 細分化後にオフセットを与えた各サブセグメントは互いに接続しない（各 2 頂点の独立ポリライン）。
- 直線長が極端に短い場合やゼロ長は原状維持で独立ポリラインとして出力。
"""

from __future__ import annotations

import math

import numpy as np

from engine.core.geometry import Geometry

from .registry import effect

EPS = 1e-12


def _collapse_numpy_v2(
    coords: np.ndarray,
    offsets: np.ndarray,
    intensity: float,
    divisions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """collapse を分布互換のまま効率化（2 パス + 前方確保）。

    - 非接続仕様（各サブセグメントは 2 頂点の独立ポリライン）を維持。
    - 方向は平面内一様（theta ~ U[0, 2π)）。振幅は `intensity` 一定。
    """
    if coords.shape[0] == 0 or intensity == 0.0 or divisions <= 0:
        return coords.copy(), offsets.copy()

    rng = np.random.default_rng(0)
    n_lines = len(offsets) - 1

    # 第1パス: 出力本数/頂点数をカウント
    total_lines = 0
    total_vertices = 0
    for li in range(n_lines):
        v = coords[offsets[li] : offsets[li + 1]]
        n = v.shape[0]
        if n < 2:
            total_lines += 1
            total_vertices += n
            continue
        seg = v[1:] - v[:-1]
        L = np.sqrt(np.sum(seg.astype(np.float64) ** 2, axis=1))
        nz = L > EPS
        total_lines += int(np.count_nonzero(nz)) * divisions + int(np.count_nonzero(~nz))
        total_vertices += (
            int(np.count_nonzero(nz)) * (2 * divisions) + int(np.count_nonzero(~nz)) * 2
        )

    if total_lines == 0:
        return coords.copy(), offsets.copy()

    # 第2パス: 充填
    out_coords = np.empty((total_vertices, 3), dtype=np.float32)
    out_offsets = np.empty((total_lines + 1,), dtype=np.int32)
    out_offsets[0] = 0
    vc = 0
    oc = 1

    # 共有 t グリッド
    t = np.linspace(0.0, 1.0, divisions + 1, dtype=np.float64)
    t0 = t[:-1]
    t1 = t[1:]

    for li in range(n_lines):
        v = coords[offsets[li] : offsets[li + 1]].astype(np.float64, copy=False)
        n = v.shape[0]
        if n < 2:
            if n > 0:
                out_coords[vc : vc + n] = v.astype(np.float32, copy=False)
                vc += n
            out_offsets[oc] = vc
            oc += 1
            continue

        for j in range(n - 1):
            a = v[j]
            b = v[j + 1]
            d = b - a
            L = float(np.sqrt(np.dot(d, d)))
            if not np.isfinite(L) or L <= EPS:
                out_coords[vc] = a.astype(np.float32)
                vc += 1
                out_coords[vc] = b.astype(np.float32)
                vc += 1
                out_offsets[oc] = vc
                oc += 1
                continue

            n_main = d / L
            # 参照軸（n と非平行）
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(n_main[2]) >= 0.9:
                ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            u = np.cross(n_main, ref)
            ul = float(np.sqrt(np.dot(u, u)))
            if ul <= EPS:
                u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                ul = 1.0
            u /= ul
            v_basis = np.cross(n_main, u)

            # サブセグメント端点（D 本分）
            starts = a * (1.0 - t0[:, None]) + b * t0[:, None]
            ends = a * (1.0 - t1[:, None]) + b * t1[:, None]

            # 平面内一様方向（角度一様）+ 一定振幅 intensity
            theta = rng.random(divisions) * (2.0 * math.pi)
            c = np.cos(theta)
            s = np.sin(theta)
            noise = (c[:, None] * u[None, :] + s[:, None] * v_basis[None, :]) * float(intensity)

            # 書き込み（2 ストライド）
            out_coords[vc : vc + 2 * divisions : 2] = (starts + noise).astype(
                np.float32, copy=False
            )
            out_coords[vc + 1 : vc + 2 * divisions : 2] = (ends + noise).astype(
                np.float32, copy=False
            )
            # offsets をサブセグメントごとに更新
            out_offsets[oc : oc + divisions] = vc + 2 * (np.arange(divisions, dtype=np.int32) + 1)
            vc += 2 * divisions
            oc += divisions

    if oc < out_offsets.shape[0]:
        out_offsets[oc:] = vc
    return out_coords, out_offsets


@effect()
def collapse(
    g: Geometry,
    *,
    intensity: float = 5.0,
    subdivisions: float = 6.0,
) -> Geometry:
    """線分を細分化してノイズで変形。

    Parameters
    ----------
    g : Geometry
        入力ジオメトリ。各行が 1 本のポリラインを表す（`offsets` で区切る）。
    intensity : float, default 5.0
        変位量（mm 相当）。0 で変化なし（no-op）。
    subdivisions : float, default 6.0
        細分回数（実数は丸めて整数に変換）。0 で変化なし（no-op）。
    """
    coords, offsets = g.as_arrays(copy=False)
    if coords.shape[0] == 0 or intensity == 0.0 or subdivisions <= 0.0:
        return Geometry(coords.copy(), offsets.copy())
    divisions = max(1, int(round(subdivisions)))
    new_coords, new_offsets = _collapse_numpy_v2(coords, offsets, float(intensity), divisions)
    return Geometry(new_coords, new_offsets)


collapse.__param_meta__ = {
    "intensity": {"type": "number", "min": 0.0, "max": 10.0},
    "subdivisions": {"type": "integer", "min": 0, "max": 10, "step": 1},
}
