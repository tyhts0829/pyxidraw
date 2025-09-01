#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマークシステム用型定義モジュール

すべてのベンチマークモジュールで使用される型定義を統一管理します。
型安全性を向上させ、IDEの補完機能を活用できます。
"""

from dataclasses import dataclass
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


@dataclass(init=False)
class BenchmarkResult:
    """ベンチマーク結果の標準形式（柔軟な初期化をサポート）"""
    target_name: str
    plugin_name: str
    config: dict[str, Any]
    timestamp: float
    success: bool
    error_message: str
    timing_data: TimingData
    metrics: BenchmarkMetrics
    output_data: Optional[Any]
    serialization_overhead: float

    def __init__(
        self,
        target_name: Optional[str] = None,
        plugin_name: str = "unknown",
        config: Optional[dict[str, Any]] = None,
        timestamp: float = 0.0,
        success: bool = False,
        error_message: str = "",
        timing_data: Optional[TimingData] = None,
        metrics: Optional[BenchmarkMetrics] = None,
        output_data: Optional[Any] = None,
        serialization_overhead: float = 0.0,
        # 互換引数（レガシー）
        module: Optional[str] = None,
        status: Optional[str] = None,
        timings: Optional[dict[str, list[float]]] = None,
        average_times: Optional[dict[str, float]] = None,
        error: Optional[str] = None,
        **_kwargs: Any,
    ) -> None:
        # レガシー形式が渡された場合のマッピング
        if module is not None and target_name is None:
            target_name = module
        if status is not None:
            success = success if success is not None else (status == "success")
        if error is not None and not error_message:
            error_message = error

        # TimingData の決定
        if timing_data is None:
            warm = []
            meas: list[float] = []
            if timings:
                # 任意のキーの測定値をまとめる（平均値を計算する側は各テスト専用）
                for v in timings.values():
                    meas.extend(v)
            timing_data = TimingData(
                warm_up_times=warm,
                measurement_times=meas,
                total_time=sum(meas) if meas else 0.0,
                average_time=(sum(meas) / len(meas)) if meas else 0.0,
                std_dev=0.0,
                min_time=min(meas) if meas else 0.0,
                max_time=max(meas) if meas else 0.0,
            )

        # Metrics の決定
        if metrics is None or isinstance(metrics, dict):
            metrics = BenchmarkMetrics(vertices_count=0, geometry_complexity=0.0, memory_usage=0, cache_hit_rate=0.0)

        # 実フィールド設定
        self.target_name = target_name or "unknown"
        self.plugin_name = plugin_name
        self.config = config or {}
        self.timestamp = timestamp
        self.success = success
        self.error_message = error_message
        self.timing_data = timing_data
        self.metrics = metrics
        self.output_data = output_data
        self.serialization_overhead = serialization_overhead

        # レガシー互換用に保持
        self._legacy_average_times = average_times or {}
        self._legacy_timings = timings or {}

    # 互換: dict風コピーを提供
    def copy(self) -> dict:
        status = "success" if self.success else "failed"
        return {
            "module": self.target_name,
            "timestamp": self.timestamp,
            "success": self.success,
            "status": status,
            "error": self.error_message,
            "timings": dict(self._legacy_timings) if self._legacy_timings else {},
            "average_times": dict(self._legacy_average_times) if self._legacy_average_times else {},
            "metrics": getattr(self, 'metrics', {}) if isinstance(self.metrics, dict) else {"has_njit": False, "has_cache": True},
        }


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


# ===== 後方互換性のための型エイリアス =====

# 既存コードとの互換性を保つため
LegacyBenchmarkResult = Dict[str, Any]
LegacyTimingData = Dict[str, List[float]]
