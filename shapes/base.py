from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from common.cacheable_base import LRUCacheable
from engine.core.geometry_data import GeometryData


class BaseShape(LRUCacheable, ABC):
    """すべてのシェイプのベースクラス。キャッシング機能付きの形状生成を担当します。"""

    def __init__(self, maxsize: int = 128):
        super().__init__(maxsize=maxsize)

    @abstractmethod
    def generate(self, **params: Any) -> GeometryData:
        """形状の頂点を生成します。

        Returns:
            GeometryData: 形状データを含むオブジェクト
        """
        pass

    def _execute(self, **params: Any) -> GeometryData:
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
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params: Any,
    ) -> GeometryData:
        """キャッシング機能付きで形状を生成"""
        # すべてのパラメータを結合してキャッシング
        all_params = {"center": center, "scale": scale, "rotate": rotate, **params}
        return super().__call__(**all_params)
