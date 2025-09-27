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


@effect()
def explode(g: Geometry, *, factor: float = 25.0) -> Geometry:
    """連続線を線分単位に分断し、各線分を中心から外側へ平行移動する。

    各線分の中点と全体重心の差ベクトルを 3D で正規化し、その方向へ `factor` だけ
    並進させる。線分長は不変。単一点ポリラインは 1 点のまま同様に移動させる。

    引数:
        g: 入力ジオメトリ。
        factor: 各線分（単一点含む）の移動距離（mm 単位）。

    返り値:
        線分分断・平行移動後の `Geometry`。
    """
    coords, offsets = g.as_arrays(copy=False)
    if g.is_empty:
        return Geometry(coords.copy(), offsets.copy())

    # 全体の重心（XYZ すべてで平均）
    center = coords.mean(axis=0)
    amount = float(factor)

    # 各ポリラインの頂点数
    seg_lengths = np.diff(offsets)
    # 総線分数（L>=2 のとき L-1 を積算）と単一点数
    segments_count = int(np.maximum(0, seg_lengths - 1).sum())
    singles_count = int(np.count_nonzero(seg_lengths == 1))

    # 出力配列を先に確保
    out_n_vertices = segments_count * 2 + singles_count
    out_n_lines = segments_count + singles_count
    out_coords = np.empty((out_n_vertices, 3), dtype=np.float32)
    out_offsets = np.empty(out_n_lines + 1, dtype=np.int32)

    write_vtx = 0
    write_line = 0
    out_offsets[0] = 0

    # 各ポリラインを走査し、線分単位に独立ポリラインへ展開
    for i in range(len(offsets) - 1):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        L = e - s
        if L <= 0:
            continue

        if L == 1:
            # 単一点ポリラインは 1 点のまま移動
            p = coords[s]
            d = p - center
            len_d = float(np.linalg.norm(d))
            if len_d > 1e-12 and amount != 0.0:
                delta = (d / len_d) * amount
            else:
                delta = np.zeros(3, dtype=np.float32)
            out_coords[write_vtx] = (p + delta).astype(np.float32, copy=False)
            write_vtx += 1
            write_line += 1
            out_offsets[write_line] = write_vtx
            continue

        # L >= 2: 連続ペアごとに線分を作る
        for j in range(s, e - 1):
            p0 = coords[j]
            p1 = coords[j + 1]
            mid = (p0 + p1) * 0.5
            d = mid - center
            len_d = float(np.linalg.norm(d))
            if len_d > 1e-12 and amount != 0.0:
                delta = (d / len_d) * amount
            else:
                delta = np.zeros(3, dtype=np.float32)

            out_coords[write_vtx] = (p0 + delta).astype(np.float32, copy=False)
            out_coords[write_vtx + 1] = (p1 + delta).astype(np.float32, copy=False)
            write_vtx += 2
            write_line += 1
            out_offsets[write_line] = write_vtx

    return Geometry(out_coords, out_offsets)


explode.__param_meta__ = {
    "factor": {"type": "number", "min": 0.0, "max": 50.0},
}
