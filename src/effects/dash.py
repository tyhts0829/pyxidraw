"""
dash エフェクト（破線化）

- 各ポリラインを一定のダッシュ長とギャップ長で切り出し、破線の集合へ変換する。

主なパラメータ:
- dash_length: ダッシュ区間の長さ [mm]。
- gap_length: ギャップ区間の長さ [mm]。

仕様/注意:
- 端部は補間により部分ダッシュになり得る。全長が 0 または頂点数 < 2 の線は原線を保持。
- 長さ単位は座標系の実寸（mm 相当）。
- `dash_length + gap_length <= 0` または非有限値は no-op（入力コピーを返す）。

実装メモ（詳細設計）:
- 2 パス（count/fill）+ 前方確保。端点探索・補間は配列化し、ラグド結合のみ最小限の Python ループ。
- 弧長・補間は float64 で計算し、出力座標は float32 に統一。
- 端点探索は `np.searchsorted(s, ·, side='left')` を使用。0 除算は `eps=1e-12` で回避。
- 任意の Numba 加速に対応（存在時は既定で有効）。`PXD_USE_NUMBA_DASH=0` で無効化可能。
"""

from __future__ import annotations

import numpy as np
from numba import njit  # type: ignore[attr-defined]

from engine.core.geometry import Geometry

from .registry import effect


# ── Kernels（numba があれば JIT、無ければそのまま Python 実行）──────────────
@njit(cache=True, fastmath=True)  # type: ignore[misc]
def _count_line(v: np.ndarray, dash_len: float, gap_len: float) -> tuple:
    n = v.shape[0]
    if n < 2:
        return n, 1
    pattern = dash_len + gap_len
    if pattern <= 0.0 or not np.isfinite(pattern):
        return n, 1
    # 弧長 s（サイズ n）
    s = np.empty(n, dtype=np.float64)
    s[0] = 0.0
    for j in range(n - 1):
        dx = v[j + 1, 0] - v[j, 0]
        dy = v[j + 1, 1] - v[j, 1]
        dz = v[j + 1, 2] - v[j, 2]
        s[j + 1] = s[j] + np.sqrt(dx * dx + dy * dy + dz * dz)
    L = s[n - 1]
    if L <= 0.0 or not np.isfinite(L):
        return n, 1
    # ダッシュ本数（np.arange(0,L,pattern) と等価）
    m = int(np.ceil(L / pattern))
    if m <= 0:
        return n, 1
    # 頂点数合計: 各ダッシュで 2 + interior
    total_vertices = 0
    for i in range(m):
        start = i * pattern
        end = start + dash_len
        if end > L:
            end = L
        # searchsorted（左）
        s_idx = np.searchsorted(s, start)
        e_idx = np.searchsorted(s, end)
        interior = e_idx - s_idx
        if interior < 0:
            interior = 0  # type: ignore[assignment]
        total_vertices += 2 + int(interior)
    return total_vertices, m


@njit(cache=True, fastmath=True)  # type: ignore[misc]
def _fill_line(
    v: np.ndarray,
    dash_len: float,
    gap_len: float,
    out_c: np.ndarray,
    out_o: np.ndarray,
    vc0: int,
    oc0: int,
) -> tuple:
    n = v.shape[0]
    vc = vc0
    oc = oc0
    if n < 2:
        # 原線コピー
        for j in range(n):
            out_c[vc + j, 0] = v[j, 0]
            out_c[vc + j, 1] = v[j, 1]
            out_c[vc + j, 2] = v[j, 2]
        vc += n
        out_o[oc] = vc
        oc += 1
        return vc, oc

    pattern = dash_len + gap_len
    if pattern <= 0.0 or not np.isfinite(pattern):
        # 原線コピー
        for j in range(n):
            out_c[vc + j, 0] = v[j, 0]
            out_c[vc + j, 1] = v[j, 1]
            out_c[vc + j, 2] = v[j, 2]
        vc += n
        out_o[oc] = vc
        oc += 1
        return vc, oc

    # 弧長 s（サイズ n）
    s = np.empty(n, dtype=np.float64)
    s[0] = 0.0
    for j in range(n - 1):
        dx = v[j + 1, 0] - v[j, 0]
        dy = v[j + 1, 1] - v[j, 1]
        dz = v[j + 1, 2] - v[j, 2]
        s[j + 1] = s[j] + np.sqrt(dx * dx + dy * dy + dz * dz)
    L = s[n - 1]
    if L <= 0.0 or not np.isfinite(L):
        for j in range(n):
            out_c[vc + j, 0] = v[j, 0]
            out_c[vc + j, 1] = v[j, 1]
            out_c[vc + j, 2] = v[j, 2]
        vc += n
        out_o[oc] = vc
        oc += 1
        return vc, oc

    m = int(np.ceil(L / pattern))
    if m <= 0:
        for j in range(n):
            out_c[vc + j, 0] = v[j, 0]
            out_c[vc + j, 1] = v[j, 1]
            out_c[vc + j, 2] = v[j, 2]
        vc += n
        out_o[oc] = vc
        oc += 1
        return vc, oc

    for i in range(m):
        start = i * pattern
        end = start + dash_len
        if end > L:
            end = L

        s_idx = np.searchsorted(s, start)
        e_idx = np.searchsorted(s, end)

        # start 補間
        s0 = s_idx - 1
        if s0 < 0:
            s0 = 0  # type: ignore[assignment]
        s1 = s_idx
        den = s[s1] - s[s0]
        if den == 0.0:
            ts = 0.0
        else:
            ts = (start - s[s0]) / den
        x0 = v[s0, 0] + (v[s1, 0] - v[s0, 0]) * ts
        y0 = v[s0, 1] + (v[s1, 1] - v[s0, 1]) * ts
        z0 = v[s0, 2] + (v[s1, 2] - v[s0, 2]) * ts

        # end 補間
        e0 = e_idx - 1
        if e0 < 0:
            e0 = 0  # type: ignore[assignment]
        e1 = e_idx
        dene = s[e1] - s[e0]
        if dene == 0.0:
            te = 0.0
        else:
            te = (end - s[e0]) / dene
        x1 = v[e0, 0] + (v[e1, 0] - v[e0, 0]) * te
        y1 = v[e0, 1] + (v[e1, 1] - v[e0, 1]) * te
        z1 = v[e0, 2] + (v[e1, 2] - v[e0, 2]) * te

        # 書き込み（開始点）
        out_c[vc, 0] = np.float32(x0)
        out_c[vc, 1] = np.float32(y0)
        out_c[vc, 2] = np.float32(z0)
        vc += 1
        # 中間頂点
        if e_idx > s_idx:
            for k in range(s_idx, e_idx):
                out_c[vc, 0] = v[k, 0]
                out_c[vc, 1] = v[k, 1]
                out_c[vc, 2] = v[k, 2]
                vc += 1
        # 終端点
        out_c[vc, 0] = np.float32(x1)
        out_c[vc, 1] = np.float32(y1)
        out_c[vc, 2] = np.float32(z1)
        vc += 1
        # 行の終端 offset
        out_o[oc] = vc
        oc += 1

    return vc, oc


@effect()
def dash(
    g: Geometry,
    *,
    dash_length: float = 6.0,
    gap_length: float = 3.0,
) -> Geometry:
    """連続線を破線に変換。

    Parameters
    ----------
    g : Geometry
        入力ジオメトリ。各行が 1 本のポリラインを表す（`offsets` で区切る）。
    dash_length : float, default 6.0
        ダッシュ（描画区間）の長さ。
    gap_length : float, default 3.0
        ギャップ（非描画区間）の長さ。
    """
    coords, offsets = g.as_arrays(copy=False)
    if coords.shape[0] == 0:
        return Geometry(coords.copy(), offsets.copy())

    pattern = float(dash_length + gap_length)
    if not np.isfinite(pattern) or pattern <= 0.0:
        # 不正値や 0 ステップは無限ループを避けるため no-op
        return Geometry(coords.copy(), offsets.copy())

    # ---- JIT/Python 共通の 2 パス実装（count → fill） -----------------------
    # 第1パス（行数・頂点数をカウント）
    total_out_vertices = 0
    total_out_lines = 0
    n_lines = len(offsets) - 1
    for li in range(n_lines):
        v = coords[offsets[li] : offsets[li + 1]]
        tv, tl = _count_line(v.astype(np.float32, copy=False), float(dash_length), float(gap_length))  # type: ignore[arg-type]
        total_out_vertices += int(tv)
        total_out_lines += int(tl)
    if total_out_lines == 0:
        return Geometry(coords.copy(), offsets.copy())
    out_coords = np.empty((total_out_vertices, 3), dtype=np.float32)
    out_offsets = np.empty((total_out_lines + 1,), dtype=np.int32)
    out_offsets[0] = 0
    vc = 0
    oc = 1
    # 第2パス（書き込み）
    for li in range(n_lines):
        v = coords[offsets[li] : offsets[li + 1]]
        vc, oc = _fill_line(  # type: ignore[arg-type]
            v.astype(np.float32, copy=False),
            float(dash_length),
            float(gap_length),
            out_coords,
            out_offsets,
            vc,
            oc,
        )
    if oc < out_offsets.shape[0]:
        out_offsets[oc:] = vc
    return Geometry(out_coords, out_offsets)


dash.__param_meta__ = {
    "dash_length": {"type": "number", "min": 0.0, "max": 100.0},
    "gap_length": {"type": "number", "min": 0.0, "max": 100.0},
}
