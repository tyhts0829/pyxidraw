#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマークシステム用型定義モジュール

すべてのベンチマークモジュールで使用される型定義を統一管理します。
型安全性を向上させ、IDEの補完機能を活用できます。
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, TypedDict, Union

import numpy as np
from numpy.typing import NDArray

# ===== 基本型エイリアス =====

# 3D座標データ
Vertices = NDArray[np.float32]  # shape: (N, 3)
VerticesList = List[NDArray[np.float32]]  # 複数の線分

# ===== ベンチマーク結果の新しい型定義 =====

BenchmarkStatus = Literal["success", "failed", "timeout", "error"]


@dataclass
class TimingData:
    """タイミングデータの詳細情報"""
    warm_up_times: List[float]
    measurement_times: List[float]
    total_time: float
    average_time: float
    std_dev: float
    min_time: float
    max_time: float


@dataclass
class BenchmarkMetrics:
    """ベンチマークメトリクス"""
    vertices_count: int
    geometry_complexity: float
    memory_usage: int
    cache_hit_rate: float


@dataclass
class BenchmarkResult:
    """ベンチマーク結果の標準形式（後方互換なしのシンプル版）"""
    target_name: str
    plugin_name: str
    config: dict[str, Any]
    timestamp: float
    success: bool
    error_message: str
    timing_data: TimingData
    metrics: BenchmarkMetrics
    output_data: Optional[Any] = None
    serialization_overhead: float = 0.0
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"


class ModuleFeatures(TypedDict):
    """モジュールの特性情報"""
    has_njit: bool
    has_cache: bool
    function_count: int
    source_lines: int
    import_errors: List[str]


class ValidationResult(TypedDict):
    """検証結果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: BenchmarkMetrics


# ===== 設定関連の型定義 =====

@dataclass
class BenchmarkConfig:
    """ベンチマーク設定クラス"""
    warmup_runs: int = 5
    measurement_runs: int = 20
    timeout_seconds: float = 30.0
    output_dir: Path = Path("benchmark_results")
    
    # エラーハンドリング設定
    continue_on_error: bool = True
    max_errors: int = 10
    
    # 並列実行設定
    parallel: bool = False
    max_workers: Optional[int] = None
    
    # 可視化設定
    generate_charts: bool = True
    chart_format: str = "png"
    chart_dpi: int = 150
    
    # 比較しきい値の柔軟化
    abs_threshold: float = 0.0
    regression_threshold_by_tag: dict[str, float] = field(default_factory=dict)
    regression_threshold_by_target: dict[str, float] = field(default_factory=dict)


class BenchmarkTarget(Protocol):
    """ベンチマーク対象のプロトコル"""
    name: str
    
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """ベンチマーク対象の実行"""
        ...


class BenchmarkRunner(Protocol):
    """ベンチマークランナーのプロトコル"""
    
    def benchmark(self, target: BenchmarkTarget, config: BenchmarkConfig) -> BenchmarkResult:
        """ベンチマークの実行"""
        ...


# ===== エフェクト関連の型定義 =====

EffectFunction = Callable[[Any], Any]  # より汎用的な定義
GeometryEffectFunction = Callable[["Geometry"], "Geometry"]  # type: ignore

# エフェクトパラメータの型定義
EffectParams = Dict[str, Union[int, float, str, tuple, list]]


class EffectVariation(TypedDict):
    """エフェクトのバリエーション定義"""
    name: str
    function: EffectFunction
    params: EffectParams
    expected_performance: Optional[str]  # "fast", "medium", "slow"


# ===== 形状関連の型定義 =====

ShapeFunction = Callable[..., "Geometry"]  # type: ignore


class ShapeVariation(TypedDict):
    """形状のバリエーション定義"""
    name: str
    function: ShapeFunction
    params: EffectParams
    complexity: Literal["simple", "medium", "complex"]


# ===== テスト関連の型定義 =====

class TestCase(TypedDict):
    """テストケースの定義"""
    name: str
    target: BenchmarkTarget
    config: BenchmarkConfig
    expected_status: BenchmarkStatus
    tolerance: float  # パフォーマンス許容値（%）


# ===== 統計・分析関連の型定義 =====

class PerformanceStats(TypedDict):
    """パフォーマンス統計"""
    mean: float
    median: float
    std: float
    min: float
    max: float
    percentile_95: float
    percentile_99: float


class ComparisonResult(TypedDict):
    """比較結果"""
    baseline: BenchmarkResult
    current: BenchmarkResult
    improvement_ratio: float  # 改善率（負の値は悪化）
    is_significant: bool  # 統計的有意性
    p_value: Optional[float]


# ===== 可視化関連の型定義 =====

ChartType = Literal["bar", "line", "scatter", "heatmap", "box"]
ChartFormat = Literal["png", "svg", "pdf", "html"]


class ChartConfig(TypedDict):
    """チャート設定"""
    chart_type: ChartType
    title: str
    xlabel: str
    ylabel: str
    format: ChartFormat
    dpi: int
    figsize: tuple[int, int]


# ===== ファイル形式関連の型定義 =====

ReportFormat = Literal["json", "yaml", "html", "markdown", "csv"]


class ExportConfig(TypedDict):
    """エクスポート設定"""
    format: ReportFormat
    include_charts: bool
    include_raw_data: bool
    compress: bool


# ===== エラー・例外関連の型定義 =====

class BenchmarkError(Exception):
    """ベンチマーク関連の基底例外クラス"""
    
    def __init__(self, message: str, module_name: Optional[str] = None, 
                 error_code: Optional[str] = None):
        super().__init__(message)
        self.module_name = module_name
        self.error_code = error_code
        self.timestamp = datetime.now()


class BenchmarkTimeoutError(BenchmarkError):
    """ベンチマークタイムアウト例外"""
    pass


class BenchmarkConfigError(BenchmarkError):
    """ベンチマーク設定エラー"""
    pass


class ModuleDiscoveryError(BenchmarkError):
    """モジュール探索エラー"""
    pass


class ValidationError(BenchmarkError):
    """検証エラー"""
    pass


# 後方互換コードは削除（シンプルな型のみに統一）
