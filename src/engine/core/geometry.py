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

ダイジェスト（キャッシュ連携）:
- `digest: bytes` は `coords/offsets` 内容から算出する短いハッシュ指紋。
- パイプラインの単層キャッシュ鍵 `(geometry_digest, pipeline_key)` の片翼として用いられる。
- 環境変数 `PXD_DISABLE_GEOMETRY_DIGEST=1` で無効化可能（無効化時の詳細は `digest` docstring 参照）。

構築ユーティリティ:
- `Geometry.from_lines(lines)` は多様な入力（list/ndarray, 2D/3D, 1D ベクトル）から
  上記データモデルへ正規化する。無効な形状は `ValueError`。

性能上の注意:
- 可能な限りコピーを避け、dtype 変換のみで整形する。大規模データでも可読性と速度の両立を狙う。
- ダイジェスト計算では `np.ascontiguousarray(...).tobytes()` により安定したバイト列へ変換して
  ハッシュ化する（初回のみコピー、以後はキャッシュされた `digest` を再利用）。

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

import hashlib
import os
from dataclasses import dataclass, field
from typing import Iterable, Tuple

import numpy as np

from common.types import Vec3


def _digest_enabled() -> bool:
    """環境変数でダイジェスト機能の有効/無効を判定する小関数。"""
    v = os.environ.get("PXD_DISABLE_GEOMETRY_DIGEST", "0")
    return v not in ("1", "true", "TRUE", "True")


def _set_digest_if_enabled(obj: "Geometry") -> None:
    """必要なら `obj._digest` を計算してセットするヘルパ。

    - コンストラクタ直後/変換直後に呼び出す。
    - 無効化時（ベンチ比較用）には何もしない。
    """
    if _digest_enabled():
        obj._digest = obj._compute_digest()


@dataclass(slots=True)
class Geometry:
    """統一幾何データ構造。

    フィールド:
    - `coords (N,3) float32`: すべての点列を連結した配列。
    - `offsets (M+1,) int32`: 各ポリラインの開始 index（末尾は N）。
    - `_digest: bytes | None`: 有効時のみ保持する内容ハッシュ（内部用途）。

    設計意図:
    - 表現を 1 種に統一し、Shapes/E.pipeline/Renderer 間の境界を単純化する。
    - 変換はインスタンスを複製する純関数（テスト容易・キャッシュ容易）。
    """

    coords: np.ndarray  # (N, 3) float32
    offsets: np.ndarray  # (M+1,) int32
    _digest: bytes | None = field(default=None, repr=False, compare=False, init=False)

    # ── ファクトリ ───────────────────
    @classmethod
    def from_lines(cls, lines: Iterable[np.ndarray]) -> "Geometry":
        """多様な線分入力を `Geometry` に正規化するファクトリ。

        受け入れる入力:
        - 各要素は座標列（list/ndarray いずれも可）。
        - 2D は Z=0 を補完。
        - 1D ベクトルは長さが 3 の倍数である必要があり、(x,y,z) の並びとして
          `(-1, 3)` にリシェイプされる。

        戻り値:
        - `coords (N,3) float32` と `offsets (M+1,) int32` を持つ `Geometry`。

        例外:
        - `ValueError`: 列の shape が (K,2)/(K,3)/(3K,) いずれにも適合しない場合、
          または 1D ベクトル長が 3 の倍数でない場合。
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
            obj = cls(coords, offsets)
            _set_digest_if_enabled(obj)
            return obj

        offsets = np.empty(len(np_lines) + 1, dtype=np.int32)
        offsets[0] = 0
        for i, arr in enumerate(np_lines, start=1):
            offsets[i] = offsets[i - 1] + arr.shape[0]
        coords = np.concatenate(np_lines, axis=0)
        obj = cls(coords.astype(np.float32, copy=False), offsets.astype(np.int32, copy=False))
        _set_digest_if_enabled(obj)
        return obj

    # 旧 GeometryData アダプタは撤廃（Geometry 統一）

    # ── 基本操作（すべて純粋） ────────
    def as_arrays(self, *, copy: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """内部配列を返すユーティリティ。

        Args:
            copy: True の場合はディープコピーを返す。

        Returns:
            `(coords, offsets)` のタプル。
        """
        if copy:
            return self.coords.copy(), self.offsets.copy()
        return self.coords, self.offsets

    @property
    def is_empty(self) -> bool:
        """座標配列が空かの簡易判定（読みやすさのための糖衣）。"""
        return self.coords.size == 0

    # ---- ダイジェスト（スナップショットID） ---------------------------------
    # Digest 概要:
    # - Geometry（coords/offsets）の内容から計算する短いハッシュ指紋（スナップショットID）。
    # - 主用途は Pipeline のキャッシュ鍵（入力ジオメトリの digest × パイプライン定義のハッシュ）。
    # - 生成・変換直後に必要があれば遅延計算して保持し、以降の同一性判定を高速化します。
    # - 環境変数 `PXD_DISABLE_GEOMETRY_DIGEST=1` で無効化できます（ベンチ等の比較向け）。
    #   無効化時に `g.digest` を呼ぶと例外を投げますが、Pipeline 側はフォールバックで
    #   配列からハッシュを都度計算するため、キャッシュは引き続き機能します。
    # - 大規模ジオメトリでは初回のハッシュ計算にコストがかかりますが、ヒット率と
    #   同一性判定の安定性のために採用しています。
    def _compute_digest(self) -> bytes:
        """`coords/offsets` から決定的なダイジェストを計算（blake2b-128）。"""
        c = np.ascontiguousarray(self.coords).view(np.uint8)
        o = np.ascontiguousarray(self.offsets).view(np.uint8)
        h = hashlib.blake2b(digest_size=16)
        # mypy: ndarray を bytes に変換して渡す（コピー発生は一度きりのため許容）
        h.update(c.tobytes())
        h.update(o.tobytes())
        return h.digest()

    @property
    def digest(self) -> bytes:
        """ジオメトリのダイジェスト（必要時に遅延計算）。

        注意:
        - `PXD_DISABLE_GEOMETRY_DIGEST=1` 設定時は例外を送出する。
          その場合でも `api.pipeline` 側は配列からの都度計算でフォールバックするため、
          キャッシュ機構は有効に保たれる。
        - 通常は初回アクセスで計算して保持し、以後の同一性判定を高速化する。
        """
        # 環境変数で無効化可能（ベンチ用）
        if not _digest_enabled():
            raise RuntimeError(
                "環境変数によりジオメトリのダイジェスト計算が無効です: PXD_DISABLE_GEOMETRY_DIGEST"
            )
        if self._digest is None:
            self._digest = self._compute_digest()
        return self._digest

    def translate(self, dx: float, dy: float, dz: float = 0.0) -> "Geometry":
        """平行移動（純関数）。

        Args:
            dx, dy, dz: 各軸の移動量。

        Returns:
            新しい `Geometry`（元は不変）。
        """
        if self.is_empty:
            # 空ジオメトリは移動しても内容は変わらない（no-op）。
            # ただし本APIは常に“新しいインスタンス”を返す純関数で統一しているため、
            # ここでもコピーを作成して返す。これにより呼び出し側での別参照性が保証され、
            # digest（有効時）の一貫性や、配列共有による意図せぬエイリアシングを避けられる。
            obj = Geometry(self.coords.copy(), self.offsets.copy())
            _set_digest_if_enabled(obj)
            return obj
        vec = np.array([dx, dy, dz], dtype=np.float32)
        new_coords = self.coords + vec
        obj = Geometry(new_coords, self.offsets.copy())
        _set_digest_if_enabled(obj)
        return obj

    def scale(
        self,
        sx: float,
        sy: float | None = None,
        sz: float | None = None,
        center: Vec3 = (0.0, 0.0, 0.0),
    ) -> "Geometry":
        """拡大縮小（純関数）。

        Args:
            sx, sy, sz: 各軸スケール。`sy/sz` 省略時は等方拡大。
            center: 拡大の基準点（pivot）。

        Returns:
            新しい `Geometry`。
        """
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx
        if self.is_empty:
            obj = Geometry(self.coords.copy(), self.offsets.copy())
            _set_digest_if_enabled(obj)
            return obj
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
        obj = Geometry(new, self.offsets.copy())
        _set_digest_if_enabled(obj)
        return obj

    def rotate(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        center: Vec3 = (0.0, 0.0, 0.0),
    ) -> "Geometry":
        """回転（純関数）。X→Y→Z の順に右手系で適用。

        Args:
            x, y, z: 各軸回転角（ラジアン）。
            center: 回転中心（pivot）。

        Returns:
            新しい `Geometry`。
        """
        if self.is_empty or (x == 0 and y == 0 and z == 0):
            obj = Geometry(self.coords.copy(), self.offsets.copy())
            _set_digest_if_enabled(obj)
            return obj

        cx, cy, cz = center
        c = self.coords.copy()
        # 原点へ移動
        c[:, 0] -= cx
        c[:, 1] -= cy
        c[:, 2] -= cz

        # X 回転
        if x != 0:
            cxr, sxr = np.cos(x), np.sin(x)
            y_new = c[:, 1] * cxr - c[:, 2] * sxr
            z_new = c[:, 1] * sxr + c[:, 2] * cxr
            c[:, 1], c[:, 2] = y_new, z_new

        # Y 回転
        if y != 0:
            cyr, syr = np.cos(y), np.sin(y)
            x_new = c[:, 0] * cyr + c[:, 2] * syr
            z_new = -c[:, 0] * syr + c[:, 2] * cyr
            c[:, 0], c[:, 2] = x_new, z_new

        # Z 回転
        if z != 0:
            czr, szr = np.cos(z), np.sin(z)
            x_new = c[:, 0] * czr - c[:, 1] * szr
            y_new = c[:, 0] * szr + c[:, 1] * czr
            c[:, 0], c[:, 1] = x_new, y_new

        # 中心を戻す
        c[:, 0] += cx
        c[:, 1] += cy
        c[:, 2] += cz
        obj = Geometry(c, self.offsets.copy())
        _set_digest_if_enabled(obj)
        return obj

    def concat(self, other: "Geometry") -> "Geometry":
        """ポリライン集合の連結（純関数）。

        - `coords` を縦方向に結合し、`offsets` は後段の先頭をシフトして統合する。
        - 空集合に対しては相手のコピー/自己のコピーを返す。
        """
        if self.is_empty:
            obj = Geometry(other.coords.copy(), other.offsets.copy())
            _set_digest_if_enabled(obj)
            return obj
        if other.is_empty:
            obj = Geometry(self.coords.copy(), self.offsets.copy())
            _set_digest_if_enabled(obj)
            return obj
        offset_shift = self.coords.shape[0]
        new_coords = np.vstack([self.coords, other.coords]).astype(np.float32, copy=False)
        adjusted_other_offsets = other.offsets[1:] + offset_shift
        new_offsets = np.hstack([self.offsets, adjusted_other_offsets]).astype(np.int32, copy=False)
        obj = Geometry(new_coords, new_offsets)
        _set_digest_if_enabled(obj)
        return obj

    # 演算子糖衣
    def __add__(self, other: "Geometry") -> "Geometry":
        """糖衣: `concat` のエイリアス。"""
        return self.concat(other)

    def __len__(self) -> int:
        """ポリライン本数（`M`）を返す。"""
        return int(self.offsets.shape[0] - 1) if self.offsets.size > 0 else 0
