"""
ベンチマークプラグインシステム

このパッケージには、異なる種類のベンチマーク対象（エフェクト、形状など）を
プラグインとして実装するためのインターフェースと実装が含まれています。
"""

from .base import BenchmarkPlugin, PluginManager
from .effects import EffectBenchmarkPlugin
from .shapes import ShapeBenchmarkPlugin

__all__ = [
    "BenchmarkPlugin",
    "PluginManager", 
    "EffectBenchmarkPlugin",
    "ShapeBenchmarkPlugin",
]