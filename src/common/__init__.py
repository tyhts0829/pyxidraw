"""
共通基盤モジュール
shapes/ と effects/ の両方で使用する統一されたベースクラス群
"""

from .base_registry import BaseRegistry, CacheableRegistry
from .cacheable_base import CacheableBase, LRUCacheable
from .meta_factory import BaseFactory, EffectFactoryMeta, ShapeFactoryMeta, UnifiedFactoryMeta

__all__ = [
    "BaseRegistry",
    "CacheableRegistry",
    "CacheableBase",
    "LRUCacheable",
    "UnifiedFactoryMeta",
    "ShapeFactoryMeta",
    "EffectFactoryMeta",
    "BaseFactory",
]
