"""
統合 Geometry 型（プロジェクト中核モジュール）

本モジュールは、プロジェクト全体で使用する唯一の幾何表現 `Geometry` を提供する。
`architecture.md` の方針に基づき、生成（Shapes/G）、変換（Geometry）、加工（Effects/E.pipeline）
を明確に分離し、GPU 転送までの境界摩擦を最小化する。

データモデル（不変条件）:
- `coords: float32 ndarray (N, 3)` — 全頂点を 1 本の連続メモリで保持（行は XYZ）。
- `offsets: int32 ndarray (M+1,)` — 各ポリラインの開始 index（末尾は必ず N）。
- i 本目の線分配列は `coords[offsets[i] : offsets[i+1]]` で取り出せる。
- dtype/形状は常に上記に正規化される（入力が 2D の場合は Z を 0 で補う）。

API 方針（ADR 準拠）:
- 変換は `translate/scale/rotate/concat` の最小セットのみを提供。
- すべて純関数（副作用ゼロ）であり、新しい `Geometry` インスタンスを返す。
- 変換チェーンは `Geometry` 側で完結し、エフェクトは `E.pipeline` に委譲する。

構築ユーティリティ:
- `Geometry.from_lines(lines)` は多様な入力（list/ndarray, 2D/3D, 1D ベクトル）から
  上記データモデルへ正規化する。無効な形状は `ValueError`。

性能上の注意:
- 可能な限りコピーを避け、dtype 変換のみで整形する。大規模データでも可読性と速度の両立を狙う。

直感図（複数線の格納）:

    # 例1: 2 本のポリライン（線0は3点、線1は2点）
    #
    # coords (N=5)
    #   idx  xy z
    #   0   [0, 0, 0]
    #   1   [1, 0, 0]
    #   2   [1, 1, 0]
    #   3   [2, 2, 0]
    #   4   [3, 2, 0]
    # offsets (M+1=3): [0, 3, 5]
    #
    # 取り出し:
    #   線0 = coords[0:3]
    #   線1 = coords[3:5]

    # 例2: 3 本のポリライン（2D 入力は Z=0 で補完）
    #   入力線:  L0=[(10,10)] , L1=[(0,0,0),(1,0,0),(1,1,0),(0,1,0)] , L2=[(5,5),(6,6)]
    #   正規化後 coords (N=7):
    #     [[10,10,0], [0,0,0], [1,0,0], [1,1,0], [0,1,0], [5,5,0], [6,6,0]]
    #   offsets (M+1=4): [0, 1, 5, 7]
    #   → L0=coords[0:1], L1=coords[1:5], L2=coords[5:7]

補足:
- 空ジオメトリは `coords.shape==(0,3)`, `offsets==[0]`（線本数 M=0）。
- 単頂点の線も許容（例: `coords=[[0,0,0]]`, `offsets=[0,1]`）。
- `concat` は後続の `offsets[1:]` に先行頂点数を加算して結合する。

使用例:
    from api import G, E
    g = G.grid(divisions=10).scale(50, 50, 1).translate(100, 100, 0)
    pipe = (E.pipeline.rotate(angles_rad=(0.0, 0.0, 0.5)).build())
    out = pipe(g)  # Geometry -> Geometry
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from common.types import Vec3

NumberLike = float | int
LineLike = np.ndarray | Sequence[NumberLike] | Sequence[Sequence[NumberLike]]


def _normalize_geometry_input(
    coords: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """`Geometry` 生成時の内部正規化ヘルパ。"""

    coords_arr = np.asarray(coords, dtype=np.float32)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError("coords は形状 (N, 3) の配列である必要があります。")
    if not coords_arr.flags.c_contiguous:
        coords_arr = np.ascontiguousarray(coords_arr, dtype=np.float32)

    offsets_arr = np.asarray(offsets, dtype=np.int32)
    if offsets_arr.ndim != 1:
        raise ValueError("offsets は 1 次元配列である必要があります。")
    if offsets_arr.size == 0:
        raise ValueError("offsets は少なくとも1要素を含む必要があります。")
    if offsets_arr[0] != 0:
        raise ValueError("offsets[0] は常に 0 である必要があります。")
    if offsets_arr[-1] != coords_arr.shape[0]:
        raise ValueError("offsets[-1] は coords の行数と一致する必要があります。")
    if np.any(np.diff(offsets_arr) < 0):
        raise ValueError("offsets は単調非減少である必要があります。")
    if not offsets_arr.flags.c_contiguous:
        offsets_arr = np.ascontiguousarray(offsets_arr, dtype=np.int32)

    return coords_arr, offsets_arr


class Geometry:
    """統一幾何データ構造。

    フィールド:
    - `coords (N,3) float32`: すべての点列を連結した配列。
    - `offsets (M+1,) int32`: 各ポリラインの開始 index（末尾は N）。
    - digest は廃止（同一性/キャッシュは LazySignature へ移行）。

    設計意図:
    - 表現を 1 種に統一し、Shapes/E.pipeline/Renderer 間の境界を単純化する。
    - 変換はインスタンスを複製する純関数（テスト容易・キャッシュ容易）。
    - 生成時に dtype/形状を検証し、正規化済み状態だけを許容する。
    """

    __slots__ = ("coords", "offsets")

    coords: np.ndarray
    offsets: np.ndarray

    def __init__(self, coords: np.ndarray, offsets: np.ndarray) -> None:
        norm_coords, norm_offsets = _normalize_geometry_input(coords, offsets)
        self.coords = norm_coords
        self.offsets = norm_offsets

    # ── ファクトリ ───────────────────
    @classmethod
    def from_lines(cls, lines: Iterable[LineLike]) -> "Geometry":
        """線分集合を統一表現に正規化して `Geometry` を生成する。

        Parameters
        ----------
        lines : Iterable[LineLike]
            各要素は座標列。`list`/`tuple`/`ndarray` いずれも可。形状は
            - `(K, 2)` の場合は `Z=0` を補完して `(K, 3)` に正規化。
            - `(K, 3)` の場合はそのまま使用。
            - `(3K,)` の 1 次元ベクトルは `(x, y, z)` の並びとして `(-1, 3)` に整形。

        Returns
        -------
        Geometry
            `coords (N, 3) float32` と `offsets (M+1,) int32` を持つジオメトリ。

        Raises
        ------
        ValueError
            形状が `(K,2)/(K,3)/(3K,)` いずれにも適合しない場合、または 1D ベクトル長が
            3 の倍数でない場合。
        """
        np_lines: list[np.ndarray] = []
        for line in lines:
            arr = np.asarray(line, dtype=np.float32)
            if arr.ndim == 1:
                if arr.size % 3 != 0:
                    raise ValueError(
                        "1次元入力の長さは3の倍数である必要があります（(x, y, z) の並び）"
                    )
                arr = arr.reshape(-1, 3)
            elif arr.shape[1] == 2:
                zeros = np.zeros((arr.shape[0], 1), dtype=np.float32)
                arr = np.hstack([arr, zeros])
            elif arr.shape[1] != 3:
                raise ValueError(f"座標配列の形状が不正です: {arr.shape}")
            np_lines.append(arr)

        if not np_lines:
            coords = np.empty((0, 3), dtype=np.float32)
            offsets = np.array([0], dtype=np.int32)
            return cls(coords, offsets)

        offsets = np.empty(len(np_lines) + 1, dtype=np.int32)
        offsets[0] = 0
        for i, arr in enumerate(np_lines, start=1):
            offsets[i] = offsets[i - 1] + arr.shape[0]
        coords = np.concatenate(np_lines, axis=0)
        return cls(coords, offsets)

    # ── 基本操作（すべて純粋） ────────
    def as_arrays(self, *, copy: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """内部配列を返す。

        Parameters
        ----------
        copy : bool, default False
            True の場合はディープコピーを返す。False の場合は読み取り専用ビューを返す。

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            `(coords, offsets)` のタプル。

        Notes
        -----
        `copy=False` は読み取り専用ビュー（`setflags(write=False)`）を返す。外部からの
        就地変更によるキャッシュキー不整合や内容破壊を防ぐ。書き込みが必要な場合は
        `copy=True` を指定する。
        """
        if copy:
            return self.coords.copy(), self.offsets.copy()
        # 読み取り専用ビューを返す（元配列の可変性には影響しない）
        coords_view = self.coords.view()
        offsets_view = self.offsets.view()
        coords_view.setflags(write=False)
        offsets_view.setflags(write=False)
        return coords_view, offsets_view

    @property
    def is_empty(self) -> bool:
        """座標配列が空かの簡易判定（読みやすさのための糖衣）。"""
        return self.coords.size == 0

    def translate(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Geometry":
        """平行移動（純関数）。

        Parameters
        ----------
        dx, dy, dz : float
            各軸の移動量。

        Returns
        -------
        Geometry
            新しい `Geometry`（元は不変）。
        """
        if self.is_empty:
            # 空ジオメトリは移動しても内容は変わらない（no-op）。
            # ただし本APIは常に“新しいインスタンス”を返す純関数で統一しているため、
            # ここでもコピーを作成して返す。これにより呼び出し側での別参照性が保証され、
            # 配列共有による意図せぬエイリアシングを避けられる。
            return Geometry(self.coords.copy(), self.offsets.copy())
        vec = np.array([dx, dy, dz], dtype=np.float32)
        new_coords = self.coords + vec
        return Geometry(new_coords, self.offsets.copy())

    def scale(
        self,
        sx: float = 100,
        sy: float | None = None,
        sz: float | None = None,
        center: Vec3 = (0.0, 0.0, 0.0),
    ) -> "Geometry":
        """拡大縮小（純関数）。

        Parameters
        ----------
        sx : float
            X 軸スケール。`sy/sz` 省略時は等方拡大の係数。
        sy, sz : float, optional
            Y/Z 軸スケール。省略時は `sx` を使用。
        center : Vec3, default (0.0, 0.0, 0.0)
            拡大の基準点（pivot）。

        Returns
        -------
        Geometry
            新しい `Geometry`。
        """
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx
        if self.is_empty:
            return Geometry(self.coords.copy(), self.offsets.copy())
        cx, cy, cz = center
        new = self.coords.copy()
        new[:, 0] -= cx
        new[:, 1] -= cy
        new[:, 2] -= cz
        new[:, 0] *= np.float32(sx)
        new[:, 1] *= np.float32(sy)
        new[:, 2] *= np.float32(sz)
        new[:, 0] += cx
        new[:, 1] += cy
        new[:, 2] += cz
        return Geometry(new, self.offsets.copy())

    def rotate(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        center: Vec3 = (0.0, 0.0, 0.0),
    ) -> "Geometry":
        """回転（純関数）。X→Y→Z の順に右手系で適用。

        Parameters
        ----------
        x, y, z : float, default 0.0
            各軸回転角（ラジアン）。
        center : Vec3, default (0.0, 0.0, 0.0)
            回転中心（pivot）。

        Returns
        -------
        Geometry
            新しい `Geometry`。

        Notes
        -----
        適用順は X→Y→Z。例として `(1, 0, 0)` を Z 軸に `π/2` 回転すると `(0, 1, 0)` に一致。
        """
        if self.is_empty or (x == 0 and y == 0 and z == 0):
            return Geometry(self.coords.copy(), self.offsets.copy())

        cx, cy, cz = center
        c = self.coords.copy()
        # 原点へ移動
        c[:, 0] -= cx
        c[:, 1] -= cy
        c[:, 2] -= cz

        # X 回転
        if x != 0:
            cxr, sxr = np.float32(np.cos(x)), np.float32(np.sin(x))
            y_new = c[:, 1] * cxr - c[:, 2] * sxr
            z_new = c[:, 1] * sxr + c[:, 2] * cxr
            c[:, 1], c[:, 2] = y_new, z_new

        # Y 回転
        if y != 0:
            cyr, syr = np.float32(np.cos(y)), np.float32(np.sin(y))
            x_new = c[:, 0] * cyr + c[:, 2] * syr
            z_new = -c[:, 0] * syr + c[:, 2] * cyr
            c[:, 0], c[:, 2] = x_new, z_new

        # Z 回転
        if z != 0:
            czr, szr = np.float32(np.cos(z)), np.float32(np.sin(z))
            x_new = c[:, 0] * czr - c[:, 1] * szr
            y_new = c[:, 0] * szr + c[:, 1] * czr
            c[:, 0], c[:, 1] = x_new, y_new

        # 中心を戻す
        c[:, 0] += cx
        c[:, 1] += cy
        c[:, 2] += cz
        return Geometry(c, self.offsets.copy())

    def concat(self, other: "Geometry") -> "Geometry":
        """ポリライン集合の連結（純関数）。

        Parameters
        ----------
        other : Geometry
            連結する相手ジオメトリ。

        Returns
        -------
        Geometry
            連結後のジオメトリ。

        Notes
        -----
        `coords` は縦方向に結合し、`offsets` は後段の先頭を `len(self.coords)` だけ
        シフトして統合する。いずれかが空集合の場合は他方のコピーを返す。
        """
        if self.is_empty:
            return Geometry(other.coords.copy(), other.offsets.copy())
        if other.is_empty:
            return Geometry(self.coords.copy(), self.offsets.copy())
        offset_shift = self.coords.shape[0]
        new_coords = np.vstack([self.coords, other.coords]).astype(np.float32, copy=False)
        adjusted_other_offsets = other.offsets[1:] + offset_shift
        new_offsets = np.hstack([self.offsets, adjusted_other_offsets]).astype(np.int32, copy=False)
        return Geometry(new_coords, new_offsets)

    # 演算子糖衣
    def __add__(self, other: "Geometry") -> "Geometry":
        """糖衣: `concat` のエイリアス。"""
        return self.concat(other)

    def __len__(self) -> int:
        """ポリライン本数（`M`）を返す。"""
        return int(self.offsets.shape[0] - 1) if self.offsets.size > 0 else 0

    # ---- DX 向上の小道具 -------------------------------------------------
    @property
    def n_vertices(self) -> int:
        """頂点数 `N` を返す。"""
        return int(self.coords.shape[0])

    @property
    def n_lines(self) -> int:
        """ポリライン本数 `M` を返す。`len(self)` と同義。"""
        return len(self)

    def __repr__(self) -> str:  # pragma: no cover - 表示用
        """形状/件数中心の簡素な表現。

        例: ``Geometry(N=12345, M=120, float32/int32)``
        """
        n = self.n_vertices
        m = self.n_lines
        # dtype 表示は短く（float32/int32）
        c_dt = str(self.coords.dtype)
        o_dt = str(self.offsets.dtype)
        return f"Geometry(N={n}, M={m}, {c_dt}/{o_dt})"
