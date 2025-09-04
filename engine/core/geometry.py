"""
統合 Geometry 型（提案1）

coords   : float32 ndarray  (N, 3)   # 全頂点を 1 本の連続メモリで保持
offsets  : int32   ndarray  (M+1,)   # 各線の開始 index（最後に N を追加）
lines[i] = coords[offsets[i] : offsets[i+1]]

API 方針:
- 変換は translate/scale/rotate/concat の最小セットのみ
- すべて純粋（新インスタンスを返す）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import os
import hashlib

import numpy as np
from common.types import Vec3


@dataclass(slots=True)
class Geometry:
    coords: np.ndarray  # (N, 3) float32
    offsets: np.ndarray  # (M+1,) int32
    _digest: bytes | None = field(default=None, repr=False, compare=False, init=False)

    # ── ファクトリ ───────────────────
    @classmethod
    def from_lines(cls, lines: Iterable[np.ndarray]) -> "Geometry":
        np_lines: list[np.ndarray] = []
        for line in lines:
            arr = np.asarray(line, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 3)
            elif arr.shape[1] == 2:
                zeros = np.zeros((arr.shape[0], 1), dtype=np.float32)
                arr = np.hstack([arr, zeros])
            elif arr.shape[1] != 3:
                raise ValueError(f"Invalid coordinate shape: {arr.shape}")
            np_lines.append(arr)

        if not np_lines:
            coords = np.empty((0, 3), dtype=np.float32)
            offsets = np.array([0], dtype=np.int32)
            obj = cls(coords, offsets)
            if os.environ.get("PXD_DISABLE_GEOMETRY_DIGEST", "0") not in ("1", "true", "TRUE", "True"):
                obj._digest = obj._compute_digest()
            return obj

        offsets = np.empty(len(np_lines) + 1, dtype=np.int32)
        offsets[0] = 0
        for i, arr in enumerate(np_lines, start=1):
            offsets[i] = offsets[i - 1] + arr.shape[0]
        coords = np.concatenate(np_lines, axis=0)
        obj = cls(coords.astype(np.float32, copy=False), offsets.astype(np.int32, copy=False))
        if os.environ.get("PXD_DISABLE_GEOMETRY_DIGEST", "0") not in ("1", "true", "TRUE", "True"):
            obj._digest = obj._compute_digest()
        return obj

    # 旧 GeometryData アダプタは撤廃（Geometry 統一）

    # ── 基本操作（すべて純粋） ────────
    def as_arrays(self, *, copy: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if copy:
            return self.coords.copy(), self.offsets.copy()
        return self.coords, self.offsets

    # ---- ダイジェスト（スナップショット） ---------------------------------
    def _compute_digest(self) -> bytes:
        """coords/offsets から決定的なダイジェストを計算（blake2b-128）。"""
        c = np.ascontiguousarray(self.coords).view(np.uint8)
        o = np.ascontiguousarray(self.offsets).view(np.uint8)
        h = hashlib.blake2b(digest_size=16)
        h.update(c)
        h.update(o)
        return h.digest()

    @property
    def digest(self) -> bytes:
        """ジオメトリのダイジェスト（必要時に遅延計算）。"""
        # 環境変数で無効化可能（ベンチ用）
        if os.environ.get("PXD_DISABLE_GEOMETRY_DIGEST", "0") in ("1", "true", "TRUE", "True"):
            raise RuntimeError("Geometry digest disabled by env: PXD_DISABLE_GEOMETRY_DIGEST")
        if self._digest is None:
            self._digest = self._compute_digest()
        return self._digest

    def translate(self, dx: float, dy: float, dz: float = 0.0) -> "Geometry":
        if self.coords.size == 0:
            obj = Geometry(self.coords.copy(), self.offsets.copy())
            if os.environ.get("PXD_DISABLE_GEOMETRY_DIGEST", "0") not in ("1", "true", "TRUE", "True"):
                obj._digest = obj._compute_digest()
            return obj
        vec = np.array([dx, dy, dz], dtype=np.float32)
        new_coords = self.coords + vec
        obj = Geometry(new_coords, self.offsets.copy())
        if os.environ.get("PXD_DISABLE_GEOMETRY_DIGEST", "0") not in ("1", "true", "TRUE", "True"):
            obj._digest = obj._compute_digest()
        return obj

    def scale(self, sx: float, sy: float | None = None, sz: float | None = None, center: Vec3 = (0.0, 0.0, 0.0)) -> "Geometry":
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx
        if self.coords.size == 0:
            obj = Geometry(self.coords.copy(), self.offsets.copy())
            if os.environ.get("PXD_DISABLE_GEOMETRY_DIGEST", "0") not in ("1", "true", "TRUE", "True"):
                obj._digest = obj._compute_digest()
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
        obj._digest = obj._compute_digest()
        return obj

    def rotate(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        center: Vec3 = (0.0, 0.0, 0.0),
    ) -> "Geometry":
        if self.coords.size == 0 or (x == 0 and y == 0 and z == 0):
            obj = Geometry(self.coords.copy(), self.offsets.copy())
            obj._digest = obj._compute_digest()
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
        if os.environ.get("PXD_DISABLE_GEOMETRY_DIGEST", "0") not in ("1", "true", "TRUE", "True"):
            obj._digest = obj._compute_digest()
        return obj

    def concat(self, other: "Geometry") -> "Geometry":
        if self.coords.size == 0:
            obj = Geometry(other.coords.copy(), other.offsets.copy())
            if os.environ.get("PXD_DISABLE_GEOMETRY_DIGEST", "0") not in ("1", "true", "TRUE", "True"):
                obj._digest = obj._compute_digest()
            return obj
        if other.coords.size == 0:
            obj = Geometry(self.coords.copy(), self.offsets.copy())
            if os.environ.get("PXD_DISABLE_GEOMETRY_DIGEST", "0") not in ("1", "true", "TRUE", "True"):
                obj._digest = obj._compute_digest()
            return obj
        offset_shift = self.coords.shape[0]
        new_coords = np.vstack([self.coords, other.coords]).astype(np.float32, copy=False)
        adjusted_other_offsets = other.offsets[1:] + offset_shift
        new_offsets = np.hstack([self.offsets, adjusted_other_offsets]).astype(np.int32, copy=False)
        obj = Geometry(new_coords, new_offsets)
        if os.environ.get("PXD_DISABLE_GEOMETRY_DIGEST", "0") not in ("1", "true", "TRUE", "True"):
            obj._digest = obj._compute_digest()
        return obj

    # 演算子糖衣
    def __add__(self, other: "Geometry") -> "Geometry":
        return self.concat(other)

    def __len__(self) -> int:
        return int(self.offsets.shape[0] - 1) if self.offsets.size > 0 else 0
