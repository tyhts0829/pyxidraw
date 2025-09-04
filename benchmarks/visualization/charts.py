#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク結果チャート生成モジュール

分離されたチャート生成クラス群を統合するファサードクラス。
元の454行のモノリシックな実装を、責務別に分離されたモジュール群で置き換え。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.core.types import BenchmarkResult
from .charts.bar_charts import BarChartGenerator
from .charts.box_charts import BoxPlotGenerator
from .charts.scatter_charts import ScatterPlotGenerator
from .charts.heatmap_charts import HeatmapGenerator
from .charts.base import ChartDataProcessor


class ChartGenerator:
    """統合チャート生成ファサードクラス
    
    分離されたチャート生成クラス群へのシンプルなインターフェースを提供。
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # 分離されたチャート生成クラスを初期化
        self.bar_generator = BarChartGenerator(self.output_dir)
        self.box_generator = BoxPlotGenerator(self.output_dir)
        self.scatter_generator = ScatterPlotGenerator(self.output_dir)
        self.heatmap_generator = HeatmapGenerator(self.output_dir)
        
        # データプロセッサ
        self.data_processor = ChartDataProcessor()
    
    # === 統一API ===
    
    def create_bar_chart(self, data: List[Dict[str, Any]], x_column: str, y_column: str, 
                        title: str, output_path: str, **kwargs) -> str:
        """バーチャート作成"""
        return self.bar_generator.create_bar_chart(data, x_column, y_column, title, output_path, **kwargs)
    
    def create_scatter_plot(self, data: List[Dict[str, Any]], x_column: str, y_column: str, 
                           title: str, output_path: str, **kwargs) -> str:
        """散布図作成"""
        return self.scatter_generator.create_scatter_plot(data, x_column, y_column, title, output_path, **kwargs)
    
    def create_box_plot(self, data: List[Dict[str, Any]], title: str, output_path: str, **kwargs) -> str:
        """ボックスプロット作成"""
        return self.box_generator.create_timing_distribution_plot(data, 
                                                                 title=title, 
                                                                 filename=Path(output_path).name,
                                                                 **kwargs)
    
    def create_heatmap(self, data: List[Dict[str, Any]], title: str, output_path: str, **kwargs) -> str:
        """ヒートマップ作成"""
        return self.heatmap_generator.create_performance_matrix(data, 
                                                               title=title, 
                                                               filename=Path(output_path).name,
                                                               **kwargs)
    
    # === 特化チャート作成メソッド ===
    
    def create_timing_comparison_chart(self, data: List[Dict[str, Any]], output_path: str) -> str:
        """実行時間比較チャート作成"""
        return self.bar_generator.create_timing_chart(data, filename=Path(output_path).name)
    
    def create_success_rate_chart(self, data: List[Dict[str, Any]], output_path: str) -> str:
        """成功率チャート作成"""
        return self.bar_generator.create_success_rate_chart(data, filename=Path(output_path).name)
    
    def create_complexity_analysis_chart(self, data: List[Dict[str, Any]], output_path: str) -> str:
        """複雑度分析チャート作成"""
        return self.scatter_generator.create_complexity_analysis_plot(data, filename=Path(output_path).name)
    
    def create_plugin_comparison_chart(self, data: List[Dict[str, Any]], output_path: str) -> str:
        """プラグイン別比較チャート作成"""
        return self.box_generator.create_plugin_comparison_plot(data, filename=Path(output_path).name)
    
    def create_correlation_matrix_chart(self, data: List[Dict[str, Any]], output_path: str) -> str:
        """相関マトリックスチャート作成"""
        return self.heatmap_generator.create_correlation_matrix(data, filename=Path(output_path).name)
    
    def create_comparison_chart(self, baseline_data: List[Dict[str, Any]], 
                               current_data: List[Dict[str, Any]], output_path: str) -> str:
        """比較チャート作成"""
        comparison_data = self.data_processor.prepare_comparison_data(
            [self._convert_dict_to_result(item) for item in baseline_data],
            [self._convert_dict_to_result(item) for item in current_data]
        )
        return self.bar_generator.create_comparison_chart(comparison_data, filename=Path(output_path).name)
    
    # === ユーティリティメソッド ===
    
    def _convert_results_to_chart_data(self, results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
        """BenchmarkResultをチャートデータ形式に変換"""
        chart_data = []
        for result in results:
            chart_data.append({
                "target": result.target_name,
                "plugin": result.plugin_name,
                "average_time": result.timing_data.average_time * 1000,  # ms変換
                "min_time": result.timing_data.min_time * 1000,
                "max_time": result.timing_data.max_time * 1000,
                "std_dev": result.timing_data.std_dev * 1000,
                "measurements": [t * 1000 for t in result.timing_data.measurement_times],
                "complexity": result.metrics.geometry_complexity,
                "vertices_count": result.metrics.vertices_count
            })
        return chart_data
    
    def _convert_dict_to_result(self, data: Dict[str, Any]) -> BenchmarkResult:
        """辞書データをBenchmarkResult形式に変換"""
        from benchmarks.core.types import BenchmarkResult, TimingData, BenchmarkMetrics
        
        return BenchmarkResult(
            target_name=data.get('target', 'unknown'),
            plugin_name=data.get('plugin', 'unknown'),
            config={},
            timestamp=0.0,
            success=True,
            error_message="",
            timing_data=TimingData(
                warm_up_times=[],
                measurement_times=data.get('measurements', []),
                total_time=0.0,
                average_time=data.get('average_time', 0) / 1000,  # msから秒に変換
                std_dev=data.get('std_dev', 0) / 1000,
                min_time=data.get('min_time', 0) / 1000,
                max_time=data.get('max_time', 0) / 1000
            ),
            metrics=BenchmarkMetrics(
                vertices_count=data.get('vertices_count', 0),
                geometry_complexity=data.get('complexity', 0.0),
                memory_usage=0,
                cache_hit_rate=0.0
            ),
            output_data=None,
            serialization_overhead=0.0
        )


__all__ = ['ChartGenerator']
