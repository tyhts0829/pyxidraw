"""
explode エフェクト（中心からの放射発散・線分分断）

- 全体の重心から各「線分の中点」へ向かう方向ベクトルを正規化し、一定距離だけ
  外側へ平行移動させる（3D ベクトルで XYZ すべて移動）。
- 連続ポリラインは「線分」単位に分断され、各線分は長さ・向きを保ったまま外側へ
  並進する。単一点ポリラインは 1 点のまま移動する。

主なパラメータ:
- factor: 各線分（または単一点）の移動距離 [mm]（0–50）。

注意:
- すべての線分が独立要素となるため、出力ポリライン本数は増加する。
"""

from __future__ import annotations

import numpy as np

from engine.core.geometry import Geometry

from .registry import effect

PARAM_META = {
    "factor": {"type": "number", "min": 0.0, "max": 50.0},
}


@effect()
def explode(g: Geometry, *, factor: float = 25.0) -> Geometry:
    """連続線を線分単位に分断し外側へずらす（全体スケールで短縮）。

    Parameters
    ----------
    g : Geometry
        入力ジオメトリ。
    factor : float, default 25.0
        移動距離の基準 [mm]。0 で no-op。
    """
    coords, offsets = g.as_arrays(copy=False)
    if g.is_empty:
        return Geometry(coords.copy(), offsets.copy())

    center = coords.mean(axis=0)
    amount = float(factor)

    seg_lengths = np.diff(offsets)
    segments_count = int(np.maximum(0, seg_lengths - 1).sum())
    singles_count = int(np.count_nonzero(seg_lengths == 1))

    # --- 1パス目: R_before, R_after を計算（仮想押し出し後の最大半径）
    if coords.shape[0] > 0:
        R_before = float(np.linalg.norm(coords - center, axis=1).max())
    else:
        R_before = 0.0
    R_after = 0.0
    eps = 1e-12

    for i in range(len(offsets) - 1):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        L = e - s
        if L <= 0:
            continue
        if L == 1:
            p = coords[s]
            d = p - center
            len_d = float(np.linalg.norm(d))
            if len_d > eps and amount != 0.0:
                delta = (d / len_d) * amount
            else:
                delta = 0.0
            if isinstance(delta, float):
                p_after = p  # amount==0 or len_d==0 → 変化なし
            else:
                p_after = p + delta
            R_after = max(R_after, float(np.linalg.norm(p_after - center)))
            continue
        for j in range(s, e - 1):
            p0 = coords[j]
            p1 = coords[j + 1]
            mid = (p0 + p1) * 0.5
            d = mid - center
            len_d = float(np.linalg.norm(d))
            if len_d > eps and amount != 0.0:
                delta = (d / len_d) * amount
                p0_after = p0 + delta
                p1_after = p1 + delta
            else:
                p0_after = p0
                p1_after = p1
            R_after = max(
                R_after,
                float(np.linalg.norm(p0_after - center)),
                float(np.linalg.norm(p1_after - center)),
            )

    if R_before > eps and R_after > eps:
        s_coef = R_before / R_after
    else:
        s_coef = 1.0

    # --- 2パス目: 分断しつつ短縮後の座標を書き出し
    out_n_vertices = segments_count * 2 + singles_count
    out_n_lines = segments_count + singles_count
    out_coords = np.empty((out_n_vertices, 3), dtype=np.float32)
    out_offsets = np.empty(out_n_lines + 1, dtype=np.int32)

    write_vtx = 0
    write_line = 0
    out_offsets[0] = 0

    for i in range(len(offsets) - 1):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        L = e - s
        if L <= 0:
            continue
        if L == 1:
            p = coords[s]
            d = p - center
            len_d = float(np.linalg.norm(d))
            if len_d > eps and amount != 0.0:
                delta = (d / len_d) * amount
                p_after = p + delta
            else:
                p_after = p
            p_final = center + s_coef * (p_after - center)
            out_coords[write_vtx] = p_final.astype(np.float32, copy=False)
            write_vtx += 1
            write_line += 1
            out_offsets[write_line] = write_vtx
            continue

        for j in range(s, e - 1):
            p0 = coords[j]
            p1 = coords[j + 1]
            mid = (p0 + p1) * 0.5
            d = mid - center
            len_d = float(np.linalg.norm(d))
            if len_d > eps and amount != 0.0:
                delta = (d / len_d) * amount
                p0_after = p0 + delta
                p1_after = p1 + delta
            else:
                p0_after = p0
                p1_after = p1
            p0_final = center + s_coef * (p0_after - center)
            p1_final = center + s_coef * (p1_after - center)
            out_coords[write_vtx] = p0_final.astype(np.float32, copy=False)
            out_coords[write_vtx + 1] = p1_final.astype(np.float32, copy=False)
            write_vtx += 2
            write_line += 1
            out_offsets[write_line] = write_vtx

    return Geometry(out_coords, out_offsets)


explode.__param_meta__ = PARAM_META
