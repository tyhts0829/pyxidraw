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

        Returns:
            Geometry: 形状データを含むオブジェクト
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
