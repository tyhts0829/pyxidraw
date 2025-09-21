"""
collapse エフェクト（線の崩し/しわ寄せ）

- 各線分を細分化し、小区間ごとに主方向と直交するランダムベクトルでオフセットして
  「崩れた」「くしゃっとした」見た目を作ります。
- ノイズはセグメントごとに独立に生成し、激しさは `intensity` で制御します。

主なパラメータ:
- intensity: 変位量（mm 相当, 0–10 推奨）。
- subdivisions: 細分回数（0–10, 0 で未細分化）。

特性/注意:
- 細分化を増やすと鋸歯状の微細な揺らぎが増え、頂点数も増加します。
- 直線長が極端に短い場合やゼロ長はスキップされます。
"""

from __future__ import annotations

import numpy as np
from numba import njit  # type: ignore[attr-defined]

from engine.core.geometry import Geometry

from .registry import effect


@njit(fastmath=True, cache=True)
def _subdivide_line(start: np.ndarray, end: np.ndarray, divisions: int) -> np.ndarray:
    """線分を指定された分割数で細分化します。"""
    if divisions <= 1:
        points = np.empty((2, 3), dtype=np.float32)
        points[0] = start
        points[1] = end
        return points

    # 細分化されたポイントを生成
    t_values = np.linspace(0, 1, divisions + 1)
    points = np.empty((divisions + 1, 3), dtype=np.float32)

    for i in range(divisions + 1):
        t = t_values[i]
        points[i] = start * (1 - t) + end * t

    return points


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
    # 仕様変更: 細分化してオフセットした各線分は互いに接続しない。
    # よって、各セグメントを長さ2の独立ポリラインとして蓄積する。
    all_coords: list[np.ndarray] = []
    all_offsets: list[np.ndarray] = []

    # offsetsからポリラインを抽出
    start_idx = 0
    for end_idx in offsets:
        if start_idx >= end_idx:
            start_idx = end_idx
            continue

        vertices = coords[start_idx:end_idx]

        if vertices.shape[0] < 2:
            # 単一点の場合はそのまま追加（独立ポリライン1本として保持）
            all_coords.append(vertices.astype(coords.dtype, copy=False))
            all_offsets.append(np.array([vertices.shape[0]], dtype=offsets.dtype))
            start_idx = end_idx
            continue

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
                    # 退避: 変形せず、そのまま1本の独立ポリラインとして追加
                    segment = np.stack(
                        [seg_start.astype(coords.dtype), seg_end.astype(coords.dtype)]
                    )
                    all_coords.append(segment)
                    all_offsets.append(np.array([2], dtype=offsets.dtype))
                    continue

                norm_main_dir = main_dir / main_norm

                # ノイズベクトルを生成
                noise_vector = np.random.randn(3).astype(np.float32) / np.float32(5.0)

                # ノイズをメイン方向と直交する方向に変換
                ortho_dir = np.cross(norm_main_dir, noise_vector)
                ortho_norm = np.linalg.norm(ortho_dir)
                if ortho_norm < 1e-12:
                    # 直交方向が得られない場合も非接続で保存
                    segment = np.stack(
                        [seg_start.astype(coords.dtype), seg_end.astype(coords.dtype)]
                    )
                    all_coords.append(segment)
                    all_offsets.append(np.array([2], dtype=offsets.dtype))
                    continue

                ortho_dir = ortho_dir / ortho_norm

                # ノイズを加える
                noise = (ortho_dir * np.float32(intensity)).astype(np.float32)

                # 変形された線分を追加（独立ポリライン2点）
                noisy_start = (seg_start + noise).astype(coords.dtype, copy=False)
                noisy_end = (seg_end + noise).astype(coords.dtype, copy=False)
                segment = np.stack([noisy_start, noisy_end]).astype(coords.dtype, copy=False)
                all_coords.append(segment)
                all_offsets.append(np.array([2], dtype=offsets.dtype))

        start_idx = end_idx

    # すべての座標とオフセットを結合
    if len(all_coords) == 0:
        return coords.copy(), offsets.copy()

    combined_coords = np.vstack(all_coords).astype(coords.dtype, copy=False)
    lengths = np.concatenate(all_offsets)
    # offsets は先頭に 0 を置き、以後は累積和（Geometry の不変条件）
    combined_offsets = np.empty(lengths.size + 1, dtype=offsets.dtype)
    combined_offsets[0] = 0
    np.cumsum(lengths, out=combined_offsets[1:])

    return combined_coords, combined_offsets


@effect()
def collapse(
    g: Geometry,
    *,
    intensity: float = 5.0,
    subdivisions: float = 6.0,
) -> Geometry:
    """線分を細分化してノイズで変形（純関数）。

    仕様: 細分化後にオフセットを与えた各セグメントは互いに接続しない。
    すなわち、元ポリライン内の細分セグメントは長さ2の独立ポリラインとして出力される。
    """
    coords, offsets = g.as_arrays(copy=False)
    if len(coords) == 0 or intensity == 0.0 or subdivisions <= 0.0:
        return Geometry(coords.copy(), offsets.copy())
    divisions = max(1, int(round(subdivisions)))
    new_coords, new_offsets = _apply_collapse_to_coords(
        coords, offsets, float(intensity), divisions
    )
    return Geometry(new_coords, new_offsets)


collapse.__param_meta__ = {
    "intensity": {"type": "number", "min": 0.0, "max": 10.0},
    "subdivisions": {"type": "integer", "min": 0, "max": 10, "step": 1},
}
