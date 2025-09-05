#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク結果チャート生成モジュール（ファサード）

charts パッケージ配下の個別実装を束ねるフロントクラス。
テストでは `benchmarks.visualization.charts.ChartGenerator` を参照するため、
charts/__init__.py でこのクラスを再エクスポートします。
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
    """統合チャート生成ファサードクラス"""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)

        self.bar_generator = BarChartGenerator(self.output_dir)
        self.box_generator = BoxPlotGenerator(self.output_dir)
        self.scatter_generator = ScatterPlotGenerator(self.output_dir)
        self.heatmap_generator = HeatmapGenerator(self.output_dir)

        self.data_processor = ChartDataProcessor()

    # === レガシー互換性API ===
    def create_performance_chart(self, results: Dict[str, BenchmarkResult], chart_type: str = "bar") -> str:
        successful_results = [r for r in results.values() if r.success]
        if not successful_results:
            return ""
        chart_data = self._convert_results_to_chart_data(successful_results)
        if chart_type == "bar":
            return self.bar_generator.create_timing_chart(chart_data, filename="performance_bar.png")
        elif chart_type == "box":
            return self.box_generator.create_timing_distribution_plot(chart_data, filename="performance_box.png")
        elif chart_type == "heatmap":
            return self.heatmap_generator.create_performance_matrix(chart_data, filename="performance_heatmap.png")
        else:
            raise ValueError(f"未対応のチャート種別です: {chart_type}")

    # === 新しい統一API ===
    def create_bar_chart(self, data: List[Dict[str, Any]], x_column: str, y_column: str, title: str, output_path: str, **kwargs) -> str:
        return self.bar_generator.create_bar_chart(data, x_column, y_column, title, output_path, **kwargs)

    def create_scatter_plot(self, data: List[Dict[str, Any]], x_column: str, y_column: str, title: str, output_path: str, **kwargs) -> str:
        return self.scatter_generator.create_scatter_plot(data, x_column, y_column, title, output_path, **kwargs)

    def create_box_plot(self, data: List[Dict[str, Any]], title: str, output_path: str, **kwargs) -> str:
        return self.box_generator.create_timing_distribution_plot(data, title=title, filename=Path(output_path).name, **kwargs)

    def create_heatmap(self, data: List[Dict[str, Any]], title: str, output_path: str, **kwargs) -> str:
        return self.heatmap_generator.create_performance_matrix(data, title=title, filename=Path(output_path).name, **kwargs)

    # === 特化チャート ===
    def create_timing_comparison_chart(self, data: List[Dict[str, Any]], output_path: str) -> str:
        return self.bar_generator.create_timing_chart(data, filename=Path(output_path).name)

    def create_success_rate_chart(self, data: List[Dict[str, Any]], output_path: str) -> str:
        return self.bar_generator.create_success_rate_chart(data, filename=Path(output_path).name)

    def create_complexity_analysis_chart(self, data: List[Dict[str, Any]], output_path: str) -> str:
        return self.scatter_generator.create_complexity_analysis_plot(data, filename=Path(output_path).name)

    def create_plugin_comparison_chart(self, data: List[Dict[str, Any]], output_path: str) -> str:
        return self.box_generator.create_plugin_comparison_plot(data, filename=Path(output_path).name)

    def create_comparison_chart(self, baseline_data: List[Dict[str, Any]], current_data: List[Dict[str, Any]], output_path: str) -> str:
        comparison_data = self.data_processor.prepare_comparison_data(baseline_data, current_data)
        return self.bar_generator.create_comparison_chart(comparison_data, filename=Path(output_path).name)

    # === ユーティリティ ===
    def _convert_results_to_chart_data(self, results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
        chart_data = []
        for result in results:
            chart_data.append({
                "target": result.target_name,
                "plugin": result.plugin_name,
                "average_time": result.timing_data.average_time * 1000,
                "min_time": result.timing_data.min_time * 1000,
                "max_time": result.timing_data.max_time * 1000,
                "std_dev": result.timing_data.std_dev * 1000,
                "measurements": [t * 1000 for t in result.timing_data.measurement_times],
                "complexity": result.metrics.geometry_complexity,
                "vertices_count": result.metrics.vertices_count,
            })
        return chart_data


def create_performance_chart(results: Dict[str, BenchmarkResult], chart_type: str = "bar", output_dir: Optional[Path] = None) -> str:
    generator = ChartGenerator(output_dir)
    return generator.create_performance_chart(results, chart_type)


def create_timing_chart(data: List[Dict[str, Any]], output_path: str) -> str:
    generator = ChartGenerator()
    return generator.create_timing_comparison_chart(data, output_path)


def create_success_chart(data: List[Dict[str, Any]], output_path: str) -> str:
    generator = ChartGenerator()
    return generator.create_success_rate_chart(data, output_path)


__all__ = [
    "ChartGenerator",
    "create_performance_chart",
    "create_timing_chart",
    "create_success_chart",
]
