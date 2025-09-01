"""
G - 形状ファクトリクラス（次期API仕様）
高性能キャッシュ付きの形状生成API。
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence, Union

import numpy as np

import shapes.asemic_glyph
import shapes.attractor
import shapes.capsule
import shapes.cone
import shapes.cylinder
import shapes.grid
import shapes.lissajous
import shapes.polygon
import shapes.polyhedron

# 形状の登録を実行（全て直接インポート）
import shapes.sphere
import shapes.text
import shapes.torus
from engine.core.geometry_data import GeometryData

from .geometry_api import GeometryAPI
from .shape_registry import get_shape_generator, list_registered_shapes


class ShapeFactoryMeta(type):
    """ShapeFactoryのメタクラス - クラスレベルでの動的属性アクセスを可能にする。"""

    def __getattr__(cls, name: str):
        """クラスレベルでの動的属性アクセス。"""
        try:
            get_shape_generator(name)

            # 動的にクラスメソッドを生成
            def shape_method(**params):
                params_tuple = cls._params_to_tuple(**params)
                data = ShapeFactory._cached_shape(name, params_tuple)
                return GeometryAPI(data)

            return shape_method
        except ValueError:
            raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")


class ShapeFactory(metaclass=ShapeFactoryMeta):
    """高性能キャッシュ付き形状ファクトリ（G）。

    Usage:
        from api import G
        sphere = G.sphere(subdivisions=0.5)
        polygon = G.polygon(n_sides=6)
    """

    @staticmethod
    @lru_cache(maxsize=128)
    def _cached_shape(shape_name: str, params_tuple: tuple) -> GeometryData:
        """内部キャッシュ機能。パラメータのタプルでキャッシュ。"""
        params_dict = dict(params_tuple)

        # レジストリから形状生成器を取得
        generator = get_shape_generator(shape_name)

        # get_shape_generatorはBaseShapeクラス（Type[BaseShape]）を返す
        # クラスをインスタンス化して、generateメソッドを呼び出す
        instance = generator()
        return instance.generate(**params_dict)

    @staticmethod
    def _params_to_tuple(**params) -> tuple:
        """パラメータ辞書をキャッシュ可能なタプルに変換。"""

        # ネストした値も含めて再帰的にソート・変換
        def make_hashable(obj):
            if isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, (list, tuple)):
                return tuple(make_hashable(item) for item in obj)
            elif isinstance(obj, np.ndarray):
                return tuple(obj.flatten())
            else:
                return obj

        return tuple(sorted((k, make_hashable(v)) for k, v in params.items()))

    # === レジストリベース形状生成（メタクラスによる動的アクセス） ===

    # === ユーザー拡張 ===

    @staticmethod
    def from_lines(lines: Sequence[Union[np.ndarray, List]]) -> GeometryAPI:
        """線分リストから形状を作成。

        Args:
            lines: 線分のリスト

        Returns:
            GeometryAPI: 作成された形状
        """
        return GeometryAPI.from_lines(lines)

    @staticmethod
    def empty() -> GeometryAPI:
        """空の形状を作成。

        Returns:
            GeometryAPI: 空の形状
        """
        return GeometryAPI.empty()

    def __getattr__(self, name: str):
        """動的属性アクセスでレジストリの形状を呼び出し可能にする。"""
        # レジストリに登録されているか確認
        try:
            get_shape_generator(name)

            # 動的にメソッドを生成
            def shape_method(**params):
                params_tuple = self._params_to_tuple(**params)
                data = self._cached_shape(name, params_tuple)
                return GeometryAPI(data)

            return shape_method
        except ValueError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @classmethod
    def list_shapes(cls) -> List[str]:
        """利用可能な形状の一覧を返す。"""
        return list_registered_shapes()

    # === キャッシュ管理 ===

    @classmethod
    def clear_cache(cls):
        """形状キャッシュをクリア。"""
        cls._cached_shape.cache_clear()

    @classmethod
    def cache_info(cls):
        """キャッシュ情報を取得。"""
        return cls._cached_shape.cache_info()


# シングルトンインスタンス（G）
G = ShapeFactory()
