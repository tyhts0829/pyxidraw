"""
ベンチマーク結果のビジュアライゼーション生成

UnifiedBenchmarkRunnerから分離されたチャート・レポート生成処理
"""
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.core.types import BenchmarkResult
from benchmarks.core.exceptions import benchmark_operation


class BenchmarkVisualizationGenerator:
    """ベンチマーク結果のビジュアライゼーション生成クラス"""
    
    def __init__(self, config, error_collector):
        self.config = config
        self.error_collector = error_collector
    
    def generate_auto_visualization(self, results: Dict[str, BenchmarkResult]) -> None:
        """自動ビジュアライゼーション生成のメインエントリーポイント"""
        if not self.config.generate_charts:
            return
        
        logger = logging.getLogger(__name__)
        try:
            with benchmark_operation("generating visualizations"):
                # チャート生成
                chart_paths = self._generate_charts(results)
                
                # レポート生成
                self._generate_reports(results, chart_paths)
                logger.info("📊 Generated %d charts and reports in %s", len(chart_paths), self.config.output_dir)
                
        except Exception as e:
            error_msg = f"Failed to generate visualizations: {e}"
            logger.warning("⚠️  %s", error_msg)
            self.error_collector.add_error(e)
    
    def _generate_charts(self, results: Dict[str, BenchmarkResult]) -> List[str]:
        """チャート生成の統合処理"""
        chart_paths = []
        
        try:
            from benchmarks.visualization.charts import ChartGenerator
            chart_generator = ChartGenerator()
            
            # 実行時間比較チャート
            timing_chart = self._generate_timing_chart(results, chart_generator)
            if timing_chart:
                chart_paths.append(timing_chart)
            
            # 成功率チャート
            success_chart = self._generate_success_rate_chart(results, chart_generator)
            if success_chart:
                chart_paths.append(success_chart)
            
            # 複雑度分析チャート
            complexity_chart = self._generate_complexity_chart(results, chart_generator)
            if complexity_chart:
                chart_paths.append(complexity_chart)
            
        except ImportError:
            logging.getLogger(__name__).warning("⚠️  Chart generation libraries not available")
        except Exception as e:
            logging.getLogger(__name__).warning("⚠️  Chart generation failed: %s", e)
            self.error_collector.add_error(e)
        
        return chart_paths
    
    def _generate_reports(self, results: Dict[str, BenchmarkResult], chart_paths: List[str]) -> None:
        """レポート生成の統合処理"""
        try:
            # HTMLレポート生成
            self._generate_html_report(results, chart_paths)
            
            # Markdownレポート生成
            self._generate_markdown_report(results, chart_paths)
            
        except Exception as e:
            logging.getLogger(__name__).warning("⚠️  Report generation failed: %s", e)
            self.error_collector.add_error(e)
    
    def _generate_timing_chart(self, results: Dict[str, BenchmarkResult], chart_generator) -> Optional[str]:
        """実行時間比較チャートを生成"""
        try:
            # 成功した結果のみを対象
            successful_results = {k: v for k, v in results.items() if v.success}
            
            if not successful_results:
                return None
            
            # データ準備
            chart_data = []
            for plugin_name, result in successful_results.items():
                chart_data.append({
                    "target": result.target_name,
                    "average_time": result.timing_data.average_time * 1000,  # ms変換
                    "plugin": result.plugin_name
                })
            
            # チャート生成
            output_path = Path(self.config.output_dir) / "timing_comparison.png"
            chart_generator.create_bar_chart(
                data=chart_data,
                x_column="target",
                y_column="average_time",
                title="Benchmark Execution Times",
                output_path=str(output_path)
            )
            
            return str(output_path)
            
        except Exception as e:
            logging.getLogger(__name__).warning("⚠️  Timing chart generation failed: %s", e)
            return None
    
    def _generate_success_rate_chart(self, results: Dict[str, BenchmarkResult], chart_generator) -> Optional[str]:
        """成功率チャートを生成"""
        try:
            # プラグイン別成功率を計算
            plugin_stats = {}
            for result in results.values():
                plugin = result.plugin_name
                if plugin not in plugin_stats:
                    plugin_stats[plugin] = {"total": 0, "successful": 0}
                
                plugin_stats[plugin]["total"] += 1
                if result.success:
                    plugin_stats[plugin]["successful"] += 1
            
            # チャートデータ準備
            chart_data = []
            for plugin, stats in plugin_stats.items():
                success_rate = stats["successful"] / stats["total"] * 100
                chart_data.append({
                    "plugin": plugin,
                    "success_rate": success_rate,
                    "successful": stats["successful"],
                    "total": stats["total"]
                })
            
            # チャート生成
            output_path = Path(self.config.output_dir) / "success_rate.png"
            chart_generator.create_bar_chart(
                data=chart_data,
                x_column="plugin",
                y_column="success_rate",
                title="Benchmark Success Rate by Plugin",
                output_path=str(output_path)
            )
            
            return str(output_path)
            
        except Exception as e:
            logging.getLogger(__name__).warning("⚠️  Success rate chart generation failed: %s", e)
            return None
    
    def _generate_complexity_chart(self, results: Dict[str, BenchmarkResult], chart_generator) -> Optional[str]:
        """複雑度分析チャートを生成"""
        try:
            # 複雑度データがある結果のみを対象
            complexity_data = []
            for result in results.values():
                if result.success and result.metrics.geometry_complexity > 0:
                    complexity_data.append({
                        "target": result.target_name,
                        "complexity": result.metrics.geometry_complexity,
                        "execution_time": result.timing_data.average_time * 1000,
                        "vertices_count": result.metrics.vertices_count
                    })
            
            if not complexity_data:
                return None
            
            # 散布図として生成
            output_path = Path(self.config.output_dir) / "complexity_analysis.png"
            chart_generator.create_scatter_plot(
                data=complexity_data,
                x_column="complexity",
                y_column="execution_time",
                title="Execution Time vs Geometry Complexity",
                output_path=str(output_path)
            )
            
            return str(output_path)
            
        except Exception as e:
            logging.getLogger(__name__).warning("⚠️  Complexity chart generation failed: %s", e)
            return None
    
    def _generate_html_report(self, results: Dict[str, BenchmarkResult], chart_paths: List[str]) -> None:
        """HTMLレポートを生成"""
        try:
            from benchmarks.visualization.reports import HTMLReportGenerator
            
            report_generator = HTMLReportGenerator()
            output_path = Path(self.config.output_dir) / "benchmark_report.html"
            
            report_generator.generate_comprehensive_report(
                results=list(results.values()),
                output_path=str(output_path),
                chart_paths=chart_paths
            )
            
            logging.getLogger(__name__).info("📄 HTML report generated: %s", output_path)
            
        except ImportError:
            logging.getLogger(__name__).warning("⚠️  HTML report generation libraries not available")
        except Exception as e:
            logging.getLogger(__name__).warning("⚠️  HTML report generation failed: %s", e)
    
    def _generate_markdown_report(self, results: Dict[str, BenchmarkResult], chart_paths: List[str]) -> None:
        """Markdownレポートを生成"""
        try:
            from benchmarks.visualization.reports import MarkdownReportGenerator
            
            report_generator = MarkdownReportGenerator()
            output_path = Path(self.config.output_dir) / "benchmark_report.md"
            
            report_generator.generate_summary_report(
                results=list(results.values()),
                output_path=str(output_path),
                include_charts=True
            )
            
            logging.getLogger(__name__).info("📄 Markdown report generated: %s", output_path)
            
        except ImportError:
            logging.getLogger(__name__).warning("⚠️  Markdown report generation libraries not available")
        except Exception as e:
            logging.getLogger(__name__).warning("⚠️  Markdown report generation failed: %s", e)


class ChartDataProcessor:
    """チャートデータの前処理を担当するクラス"""
    
    @staticmethod
    def extract_timing_data(results: Dict[str, BenchmarkResult]) -> List[Dict[str, Any]]:
        """タイミングデータを抽出"""
        timing_data = []
        for result in results.values():
            if result.success and result.timing_data.measurement_times:
                timing_data.append({
                    "target": result.target_name,
                    "plugin": result.plugin_name,
                    "average_time": result.timing_data.average_time,
                    "min_time": result.timing_data.min_time,
                    "max_time": result.timing_data.max_time,
                    "std_dev": result.timing_data.std_dev
                })
        return timing_data
    
    @staticmethod
    def extract_plugin_statistics(results: Dict[str, BenchmarkResult]) -> Dict[str, Dict[str, Any]]:
        """プラグイン別統計を抽出"""
        plugin_stats = {}
        
        for result in results.values():
            plugin = result.plugin_name
            if plugin not in plugin_stats:
                plugin_stats[plugin] = {
                    "total_targets": 0,
                    "successful_targets": 0,
                    "failed_targets": 0,
                    "total_time": 0.0,
                    "errors": []
                }
            
            stats = plugin_stats[plugin]
            stats["total_targets"] += 1
            
            if result.success:
                stats["successful_targets"] += 1
                stats["total_time"] += result.timing_data.total_time
            else:
                stats["failed_targets"] += 1
                if result.error_message:
                    stats["errors"].append(result.error_message)
        
        return plugin_stats
    
    @staticmethod
    def calculate_performance_metrics(results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """パフォーマンスメトリクスを計算"""
        successful_results = [r for r in results.values() if r.success]
        
        if not successful_results:
            return {"message": "No successful results for metrics calculation"}
        
        execution_times = [r.timing_data.average_time for r in successful_results]
        
        return {
            "total_benchmarks": len(results),
            "successful_benchmarks": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "fastest_time": min(execution_times),
            "slowest_time": max(execution_times),
            "average_time": sum(execution_times) / len(execution_times),
            "total_execution_time": sum(r.timing_data.total_time for r in successful_results)
        }
