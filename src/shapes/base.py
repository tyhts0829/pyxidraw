"""
シェイプ基底モジュール

概要:
- 図形プリミティブのための抽象基底 `BaseShape` を定義する。
- 形状生成インターフェイスと、共通の変換適用順（scale → rotate → translate）を規定する。

設計意図:
- 各シェイプは `generate(**params)` で「原点基準の素の形状のみ」を返す。
- 変換（`center/scale/rotate`）は `__call__` で一括適用し、
  `engine.core.transform_utils.transform_combined` の規約に従い
  「スケール → 回転 → 平行移動（center）」の順で行う。
- キャッシュは上位の `api.shape_factory.ShapeFactory` に集約し、BaseShape 自体はキャッシュを持たない。
- 実装は純粋関数的（副作用なし）を推奨し、テストの再現性を保つ。

公開 API:
- `BaseShape.generate(**params) -> Geometry`（抽象）: 変換を行わず、
  `engine.core.geometry.Geometry` を返す。
- `BaseShape.__call__(center, scale, rotate, **params) -> Geometry`:
  変換をまとめて実行する。

使用例:
    class MyCircle(BaseShape):
        def generate(self, *, radius: float = 1.0, segments: int = 64) -> Geometry:
            # 原点中心・Z=0 の円ポリラインを返す（簡易例）
            import numpy as np  # 使用例内の局所 import
            t = np.linspace(0, 2 * np.pi, segments, endpoint=False, dtype=np.float32)
            xy = np.c_[np.cos(t) * radius, np.sin(t) * radius]
            return Geometry.from_lines([xy])

    # 変換は「スケール → 回転 → 移動」の順
    g = MyCircle()(center=(10.0, 0.0, 0.0), scale=(2.0, 2.0, 1.0), rotate=(0.0, 0.0, 0.5))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from common.types import Vec3
from engine.core.geometry import Geometry


class BaseShape(ABC):
    """すべてのシェイプのベースクラス。

    方針: 形状生成のキャッシュは `api.shape_factory.ShapeFactory` 側に一本化し、
    本クラスはキャッシュを持たない。
    """

    @abstractmethod
    def generate(self, **params: Any) -> Geometry:
        """形状の頂点を生成します。

        返り値:
            形状データを含む `Geometry`。
        """
        pass

    def __call__(
        self,
        center: Vec3 = (0.0, 0.0, 0.0),
        scale: Vec3 = (1.0, 1.0, 1.0),
        rotate: Vec3 = (0.0, 0.0, 0.0),
        **params: Any,
    ) -> Geometry:
        """形状を生成し、必要に応じて変換を適用して返す。"""
        geometry_data = self.generate(**params)
        if center != (0, 0, 0) or scale != (1, 1, 1) or rotate != (0, 0, 0):
            from engine.core import transform_utils as _tf

            return _tf.transform_combined(geometry_data, center, scale, rotate)
        return geometry_data
