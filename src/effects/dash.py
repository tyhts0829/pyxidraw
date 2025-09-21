"""
dash エフェクト（破線化）

- 各ポリラインを一定のダッシュ長とギャップ長で切り出し、破線の集合へ変換する。

主なパラメータ:
- dash_length: ダッシュ区間の長さ [mm]。
- gap_length: ギャップ区間の長さ [mm]。

仕様/注意:
- 端部は補間により部分ダッシュになり得る。全長が 0 または頂点数 < 2 の線は原線を保持。
- 長さ単位は座標系の実寸（mm 相当）。
- 実装は 2 パス（count/fill）+ 前方確保で Python ループを最小化（将来 njit しやすい形）。
"""

from __future__ import annotations

import numpy as np

from engine.core.geometry import Geometry

from .registry import effect


@effect()
def dash(
    g: Geometry,
    *,
    dash_length: float = 6.0,
    gap_length: float = 3.0,
) -> Geometry:
    """連続線を破線に変換（純関数）。

    備考:
        - dash_length/gap_length は座標単位（mm 相当）。
        - 線長に応じて端部のダッシュは補間されます（端は部分ダッシュになり得ます）。
        - 既定値（6mm/3mm）は 300mm キャンバス中央の立方体（辺=150mm）で視認性と密度のバランスが良好です。
    """
    coords, offsets = g.as_arrays(copy=False)
    if coords.shape[0] == 0:
        return Geometry(coords.copy(), offsets.copy())

    pattern = float(dash_length + gap_length)
    if not np.isfinite(pattern) or pattern <= 0.0:
        # 不正値や 0 ステップは無限ループを避けるため no-op
        return Geometry(coords.copy(), offsets.copy())

    # ---- 第1パス: 出力行数/頂点数をカウント ---------------------------------
    total_out_vertices = 0
    total_out_lines = 0
    eps = 1e-12

    n_lines = len(offsets) - 1
    for li in range(n_lines):
        v = coords[offsets[li] : offsets[li + 1]]
        n = v.shape[0]
        if n < 2:
            total_out_lines += 1
            total_out_vertices += n
            continue
        seg = v[1:] - v[:-1]
        # float64 弧長で数値安定性を確保
        dist = np.sqrt(np.sum(seg.astype(np.float64) ** 2, axis=1))
        s = np.concatenate(([0.0], np.cumsum(dist)))
        L = float(s[-1])
        if not np.isfinite(L) or L <= 0.0:
            total_out_lines += 1
            total_out_vertices += n
            continue
        starts = np.arange(0.0, L, pattern, dtype=np.float64)
        if starts.size == 0:
            # L が極小で 1 本もダッシュが生成されない場合、原線保持
            total_out_lines += 1
            total_out_vertices += n
            continue
        ends = np.minimum(starts + float(dash_length), L)
        s_idx = np.searchsorted(s, starts, side="left")
        e_idx = np.searchsorted(s, ends, side="left")
        interior = np.maximum(e_idx - s_idx, 0)
        total_out_lines += int(starts.size)
        total_out_vertices += int(np.sum(2 + interior))

    if total_out_lines == 0:
        # すべてが空/単点であった
        return Geometry(coords.copy(), offsets.copy())

    # ---- 第2パス: 事前確保した配列に書き込み --------------------------------
    out_coords = np.empty((total_out_vertices, 3), dtype=np.float32)
    out_offsets = np.empty((total_out_lines + 1,), dtype=np.int32)
    out_offsets[0] = 0
    vc = 0  # 頂点カーソル
    oc = 1  # オフセットカーソル（次書き込み位置）

    for li in range(n_lines):
        v = coords[offsets[li] : offsets[li + 1]]
        n = v.shape[0]
        if n < 2:
            # 原線コピー
            if n > 0:
                out_coords[vc : vc + n] = v
                vc += n
            out_offsets[oc] = vc
            oc += 1
            continue

        seg = v[1:] - v[:-1]
        dist = np.sqrt(np.sum(seg.astype(np.float64) ** 2, axis=1))
        s = np.concatenate(([0.0], np.cumsum(dist)))
        L = float(s[-1])
        if not np.isfinite(L) or L <= 0.0:
            # 原線コピー
            out_coords[vc : vc + n] = v
            vc += n
            out_offsets[oc] = vc
            oc += 1
            continue

        starts = np.arange(0.0, L, pattern, dtype=np.float64)
        if starts.size == 0:
            # 原線コピー（第1パスと一致させる）
            out_coords[vc : vc + n] = v
            vc += n
            out_offsets[oc] = vc
            oc += 1
            continue
        ends = np.minimum(starts + float(dash_length), L)

        s_idx = np.searchsorted(s, starts, side="left")
        e_idx = np.searchsorted(s, ends, side="left")

        # 端点補間（配列）
        s0 = np.clip(s_idx - 1, 0, n - 2)
        e0 = np.clip(e_idx - 1, 0, n - 2)
        den_s = (s[s_idx] - s[s0]).astype(np.float64)
        den_e = (s[e_idx] - s[e0]).astype(np.float64)
        # 0 除算回避
        den_s = np.where(den_s == 0.0, eps, den_s)
        den_e = np.where(den_e == 0.0, eps, den_e)
        ts = ((starts - s[s0]) / den_s).astype(np.float64)
        te = ((ends - s[e0]) / den_e).astype(np.float64)

        start_pts = (
            v[s0].astype(np.float64)
            + (v[s_idx].astype(np.float64) - v[s0].astype(np.float64)) * ts[:, None]
        ).astype(np.float64)
        end_pts = (
            v[e0].astype(np.float64)
            + (v[e_idx].astype(np.float64) - v[e0].astype(np.float64)) * te[:, None]
        ).astype(np.float64)

        # 各ダッシュを書き込み（ラグド結合の最小限ループ）
        for i in range(starts.size):
            # 開始点
            out_coords[vc] = start_pts[i].astype(np.float32)
            vc += 1
            # 中間頂点
            if e_idx[i] > s_idx[i]:
                k = int(e_idx[i] - s_idx[i])
                if k > 0:
                    out_coords[vc : vc + k] = v[s_idx[i] : e_idx[i]]
                    vc += k
            # 終端点
            out_coords[vc] = end_pts[i].astype(np.float32)
            vc += 1
            # 行の終端位置（次行の開始 index）を書き込む
            out_offsets[oc] = vc
            oc += 1

    # 終端 offset（通常は oc==len(out_offsets) だが念のため）
    if oc < out_offsets.shape[0]:
        out_offsets[oc:] = vc
    return Geometry(out_coords, out_offsets)


dash.__param_meta__ = {
    "dash_length": {"type": "number", "min": 0.0, "max": 100.0},
    "gap_length": {"type": "number", "min": 0.0, "max": 100.0},
}
