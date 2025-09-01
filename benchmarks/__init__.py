"""
PyxiDraw ベンチマークシステム

統一ベンチマークシステムのメインパッケージ。
プラグインベースの拡張可能なアーキテクチャを提供します。
"""

from benchmarks.core.runner import UnifiedBenchmarkRunner, run_benchmarks
from benchmarks.core.config import get_config, BenchmarkConfigManager
from benchmarks.core.validator import validate_results, analyze_benchmark_results
from benchmarks.plugins import PluginManager

__version__ = "2.0.0"

__all__ = [
    "UnifiedBenchmarkRunner",
    "run_benchmarks", 
    "get_config",
    "BenchmarkConfigManager",
    "validate_results",
    "analyze_benchmark_results",
    "PluginManager",
]