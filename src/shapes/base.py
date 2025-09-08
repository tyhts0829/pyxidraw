"""
シェイプ基底モジュール

概要:
- 図形プリミティブのための抽象基底 `BaseShape` を定義します。
- 形状生成インターフェイス、キャッシュ方針、共通の変換適用順を規定します。

設計意図:
- 各シェイプは `generate(**params)` で「原点基準の素の形状のみ」を返します。
- 変換（`center/scale/rotate`）は `__call__` で一括適用し、
  `engine.core.transform_utils.transform_combined` の規約に従い
  「スケール → 回転 → 平行移動（center）」の順で行います。
- キャッシュは `api/shape_factory.ShapeFactory` を主とし、本クラスの LRU は既定で無効。
  必要に応じて `enable_cache=True` を指定して個別に有効化できます。
- 実装は純粋関数的（副作用なし）を推奨し、テストの再現性とキャッシュの健全性を保ちます。

公開 API:
- `BaseShape.generate(**params) -> Geometry`（抽象）: 変換を行わず、
  `engine.core.geometry.Geometry` を返します。
- `BaseShape.__call__(center, scale, rotate, **params) -> Geometry`:
  キャッシュ＋変換をまとめて実行します。
- `BaseShape._execute(**params) -> Geometry`: キャッシュ用の実行エントリで、
  `__call__` からのみ使用されます。

キャッシュ方針:
- 継承元の `common.cacheable_base.LRUCacheable` により、引数がハッシュ化可能であれば
  メモ化が自動で機能します。
- 既定では `enable_cache=False` のため無効化されます（`ShapeFactory` 側に統一）。
- 環境変数 `PXD_CACHE_DISABLED=1` で全体無効化、`PXD_CACHE_MAXSIZE` でサイズ調整が可能です。

使用例:
    class MyCircle(BaseShape):
        def generate(self, *, radius: float = 1.0, segments: int = 64) -> Geometry:
            # 原点中心・Z=0 の円ポリラインを返す（簡易例）
            import numpy as np  # 使用例内の局所 import
            t = np.linspace(0, 2 * np.pi, segments, endpoint=False, dtype=np.float32)
            xy = np.c_[np.cos(t) * radius, np.sin(t) * radius]
            return Geometry.from_lines([xy])

    # 変換は「スケール → 回転 → 移動」の順
    g = MyCircle(enable_cache=True)(
        center=(10.0, 0.0, 0.0),
        scale=(2.0, 2.0, 1.0),
        rotate=(0.0, 0.0, 0.5),
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from common.cacheable_base import LRUCacheable
from common.types import Vec3
from engine.core.geometry import Geometry


class BaseShape(LRUCacheable, ABC):
    """すべてのシェイプのベースクラス。

    方針: 形状生成のキャッシュは `api/shape_factory.ShapeFactory` 側に一本化します。
    そのため、BaseShape の LRU キャッシュは既定で無効化します（必要なら個別に有効化）。
    """

    def __init__(self, maxsize: int = 128, *, enable_cache: bool = False):
        super().__init__(maxsize=maxsize)
        if not enable_cache:
            self.disable_cache()

    @abstractmethod
    def generate(self, **params: Any) -> Geometry:
        """形状の頂点を生成します。

        返り値:
            形状データを含む `Geometry`。
        """
        pass

    def _execute(self, **params: Any) -> Geometry:
        """実際の処理を実行（キャッシング用）"""
        # transformパラメータを分離
        center = params.pop("center", (0, 0, 0))
        scale = params.pop("scale", (1, 1, 1))
        rotate = params.pop("rotate", (0, 0, 0))

        # 基本形状を生成
        geometry_data = self.generate(**params)

        # 変換が必要な場合は適用
        if center != (0, 0, 0) or scale != (1, 1, 1) or rotate != (0, 0, 0):
            from engine.core import transform_utils as _tf

            geometry_data = _tf.transform_combined(geometry_data, center, scale, rotate)

        return geometry_data

    def __call__(
        self,
        center: Vec3 = (0.0, 0.0, 0.0),
        scale: Vec3 = (1.0, 1.0, 1.0),
        rotate: Vec3 = (0.0, 0.0, 0.0),
        **params: Any,
    ) -> Geometry:
        """キャッシング機能付きで形状を生成"""
        # すべてのパラメータを結合してキャッシング
        all_params = {"center": center, "scale": scale, "rotate": rotate, **params}
        return super().__call__(**all_params)
