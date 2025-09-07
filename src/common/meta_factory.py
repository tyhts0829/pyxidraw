"""
共通メタクラスファクトリ
shapes/ と effects/ の両方で使用する統一されたファクトリパターン
"""

from typing import Any, Callable, Optional

from .base_registry import BaseRegistry


class UnifiedFactoryMeta(type):
    """統一されたファクトリメタクラス"""

    def __init__(cls, name: str, bases: tuple, attrs: dict):
        super().__init__(name, bases, attrs)
        cls._registry: Optional[BaseRegistry] = None

    def set_registry(cls, registry: BaseRegistry) -> None:
        """レジストリを設定"""
        cls._registry = registry

    def __getattr__(cls, name: str) -> Callable:
        """動的属性アクセス"""
        if cls._registry is None:
            raise AttributeError(f"Registry not set for {cls.__name__}")

        if not cls._registry.is_registered(name):
            raise AttributeError(f"'{name}' is not registered in {cls.__name__}")

        def factory_method(*args, **kwargs) -> Any:
            """ファクトリメソッド"""
            assert cls._registry is not None
            item_class = cls._registry.get(name)
            return item_class(*args, **kwargs)

        factory_method.__name__ = name
        factory_method.__doc__ = f"Create {name} instance"

        return factory_method

    def list_available(cls) -> list:
        """利用可能な項目一覧を取得"""
        if cls._registry is None:
            return []
        return cls._registry.list_all()


class ShapeFactoryMeta(UnifiedFactoryMeta):
    """形状専用ファクトリメタクラス"""

    def __getattr__(cls, name: str) -> Callable:
        """形状専用の動的属性アクセス"""
        factory_method = super().__getattr__(name)

        def shape_factory_method(*args, **kwargs) -> Any:
            """形状専用ファクトリメソッド"""
            # 形状生成時の追加処理があればここに記述
            return factory_method(*args, **kwargs)

        shape_factory_method.__name__ = f"create_{name}"
        shape_factory_method.__doc__ = f"Create {name} shape"

        return shape_factory_method


class EffectFactoryMeta(UnifiedFactoryMeta):
    """エフェクト専用ファクトリメタクラス"""

    def __getattr__(cls, name: str) -> Callable:
        """エフェクト専用の動的属性アクセス"""
        factory_method = super().__getattr__(name)

        def effect_factory_method(*args, **kwargs) -> Any:
            """エフェクト専用ファクトリメソッド"""
            # エフェクト生成時の追加処理があればここに記述
            return factory_method(*args, **kwargs)

        effect_factory_method.__name__ = f"apply_{name}"
        effect_factory_method.__doc__ = f"Apply {name} effect"

        return effect_factory_method


class BaseFactory:
    """統一されたファクトリ基底クラス"""

    def __init__(self, registry: BaseRegistry):
        self._registry = registry

    def create(self, name: str, *args, **kwargs) -> Any:
        """項目を作成"""
        if not self._registry.is_registered(name):
            raise ValueError(f"'{name}' is not registered")

        item_class = self._registry.get(name)
        return item_class(*args, **kwargs)

    def list_available(self) -> list:
        """利用可能な項目一覧"""
        return self._registry.list_all()

    def is_available(self, name: str) -> bool:
        """項目が利用可能かチェック"""
        return self._registry.is_registered(name)
