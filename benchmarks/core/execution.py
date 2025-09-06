"""
ベンチマーク実行エンジン

UnifiedBenchmarkRunnerから分離されたベンチマーク実行の詳細処理
"""
import time
import logging
from typing import Any, Dict, List, Optional
import os
import sys
import platform

import numpy as np

from benchmarks.core.types import (
    BenchmarkResult,
    BenchmarkTarget,
    TimingData,
    BenchmarkMetrics,
)
from benchmarks.core.exceptions import (
    BenchmarkError,
    BenchmarkTimeoutError,
    benchmark_operation,
)
from engine.core.geometry import Geometry


class BenchmarkExecutor:
    """ベンチマーク実行の詳細処理を担当するクラス"""
    
    def __init__(self, config, error_handler, error_collector):
        self.config = config
        self.error_handler = error_handler
        self.error_collector = error_collector
    
    def initialize_benchmark_result(self, target: BenchmarkTarget) -> BenchmarkResult:
        """ベンチマーク結果の初期化"""
        # タグ収集（存在すれば）
        target_tags: list[str] = []
        if hasattr(target, 'tags') and isinstance(getattr(target, 'tags'), (list, set, tuple)):
            target_tags = list(getattr(target, 'tags'))

        return BenchmarkResult(
            target_name=target.name,
            plugin_name=target.plugin_name if hasattr(target, 'plugin_name') else "unknown",
            config=target.config if hasattr(target, 'config') else {},
            timestamp=time.time(),
            success=False,
            error_message="",
            timing_data=TimingData(
                warm_up_times=[],
                measurement_times=[],
                total_time=0.0,
                average_time=0.0,
                std_dev=0.0,
                min_time=0.0,
                max_time=0.0
            ),
            metrics=BenchmarkMetrics(
                vertices_count=0,
                geometry_complexity=0.0,
                memory_usage=0,
                cache_hit_rate=0.0
            ),
            output_data=None,
            serialization_overhead=0.0,
            tags=target_tags,
            meta=self._collect_runtime_meta(),
            schema_version="1.0",
        )
    
    def measure_target_characteristics(self, target: BenchmarkTarget, result: BenchmarkResult) -> None:
        """ターゲットの特性測定"""
        # シリアライゼーションオーバーヘッド測定
        result.serialization_overhead = self._measure_serialization_overhead(target)
        
        # プラグイン特性の分析
        if hasattr(target, 'plugin_name'):
            result.plugin_name = target.plugin_name
    
    def execute_benchmark_measurements(self, target: BenchmarkTarget, result: BenchmarkResult) -> None:
        """実際のベンチマーク測定実行"""
        with benchmark_operation(f"benchmarking {target.name}"):
            # ターゲットの種類に応じて適切なベンチマークを実行
            if self._is_shape_target(target):
                self._benchmark_shape_generation(target, result)
            else:
                self._benchmark_effect_application(target, result)
    
    def calculate_benchmark_statistics(self, result: BenchmarkResult) -> None:
        """ベンチマーク結果の統計計算"""
        times = result.timing_data.measurement_times
        
        if not times:
            return
        
        result.timing_data.total_time = sum(times)
        result.timing_data.average_time = np.mean(times)
        result.timing_data.std_dev = float(np.std(times))
        result.timing_data.min_time = float(np.min(times))
        result.timing_data.max_time = float(np.max(times))
        
        # 成功判定
        result.success = len(times) > 0 and result.error_message == ""
    
    def handle_benchmark_exception(self, result: BenchmarkResult, exception: Exception) -> None:
        """ベンチマーク例外の統一処理"""
        result.success = False
        result.error_message = str(exception)
        
        # エラーの種類に応じた処理
        if isinstance(exception, BenchmarkTimeoutError):
            result.error_message = f"Benchmark timed out: {exception}"
        elif isinstance(exception, BenchmarkError):
            result.error_message = f"Benchmark error: {exception}"
        else:
            result.error_message = f"Unexpected error: {exception}"
        
        # エラーコレクターに登録
        self.error_collector.add_error(exception)
    
    def _is_shape_target(self, target: BenchmarkTarget) -> bool:
        """ターゲットが形状生成かどうかを判定"""
        # プラグイン名から判定
        if hasattr(target, 'plugin_name') and target.plugin_name == 'shape':
            return True
        
        # ターゲット名から判定（プリフィックス.を考慮）
        target_name = target.name.lower()
        
        # エフェクトプリフィックスがある場合は形状ではない
        effect_prefixes = [
            # 新名称
            'transform.', 'scale.', 'translate.', 'rotate.', 'displace.', 'subdivide.', 'extrude.', 'fill.', 'offset.', 'repeat.',
            # 旧名称（互換不要だが誤判定防止のため残置）
            'noise.', 'subdivision.', 'filling.', 'buffer.', 'array.',
        ]
        if any(target_name.startswith(prefix) for prefix in effect_prefixes):
            return False
        
        # 形状キーワードで判定
        shape_keywords = ['polygon', 'sphere', 'grid', 'cylinder', 'cone', 'torus', 'circle', 'square', 'cube', 'icosahedron']
        return any(keyword in target_name for keyword in shape_keywords)
    
    def _benchmark_shape_generation(self, target: BenchmarkTarget, result: BenchmarkResult) -> None:
        """形状生成のベンチマーク"""
        # ウォームアップ実行
        for _ in range(self.config.warmup_runs):
            try:
                start_time = time.perf_counter()
                # 形状生成では引数なしで実行
                if hasattr(target, '_execute_func') and hasattr(target._execute_func, '__call__'):
                    # SerializableShapeTargetなど引数を取らない場合
                    geometry = target._execute_func()
                else:
                    geometry = target.execute()
                end_time = time.perf_counter()
                result.timing_data.warm_up_times.append(end_time - start_time)
                
                # 最初のウォームアップで出力データを保存
                if len(result.timing_data.warm_up_times) == 1:
                    result.output_data = self._extract_geometry_data(geometry)
            except Exception as e:
                raise BenchmarkError(f"Warmup failed: {e}")
        
        # 測定実行
        for _ in range(self.config.measurement_runs):
            try:
                start_time = time.perf_counter()
                # 形状生成では引数なしで実行
                if hasattr(target, '_execute_func') and hasattr(target._execute_func, '__call__'):
                    # SerializableShapeTargetなど引数を取らない場合
                    geometry = target._execute_func()
                else:
                    geometry = target.execute()
                end_time = time.perf_counter()
                
                measurement_time = end_time - start_time
                result.timing_data.measurement_times.append(measurement_time)
                
                # メトリクス更新
                self._update_geometry_metrics(geometry, result)
                
            except Exception as e:
                raise BenchmarkError(f"Measurement failed: {e}")
    
    def _benchmark_effect_application(self, target: BenchmarkTarget, result: BenchmarkResult) -> None:
        """エフェクト適用のベンチマーク"""
        # テスト用ジオメトリを取得
        test_geometry = self._get_test_geometry_for_effect()
        
        # ウォームアップ実行
        for _ in range(self.config.warmup_runs):
            try:
                start_time = time.perf_counter()
                try:
                    output_geometry = target.execute(test_geometry)
                except TypeError:
                    # 引数なし関数にも対応
                    output_geometry = target.execute()
                end_time = time.perf_counter()
                result.timing_data.warm_up_times.append(end_time - start_time)
                
                # 最初のウォームアップで出力データを保存
                if len(result.timing_data.warm_up_times) == 1:
                    result.output_data = self._extract_geometry_data(output_geometry)
            except Exception as e:
                raise BenchmarkError(f"Warmup failed: {e}")
        
        # 測定実行
        for _ in range(self.config.measurement_runs):
            try:
                start_time = time.perf_counter()
                try:
                    output_geometry = target.execute(test_geometry)
                except TypeError:
                    output_geometry = target.execute()
                end_time = time.perf_counter()
                
                measurement_time = end_time - start_time
                result.timing_data.measurement_times.append(measurement_time)
                
                # メトリクス更新
                self._update_geometry_metrics(output_geometry, result)
                
            except Exception as e:
                raise BenchmarkError(f"Measurement failed: {e}")
    
    def _extract_geometry_data(self, geometry: Any) -> Dict[str, Any]:
        """ジオメトリからデータを抽出"""
        if isinstance(geometry, Geometry):
            return {
                "type": "Geometry",
                "vertices_count": len(geometry.coords) if hasattr(geometry, 'coords') and geometry.coords is not None else 0,
                "has_vertices": hasattr(geometry, 'coords') and geometry.coords is not None and len(geometry.coords) > 0
            }
        elif isinstance(geometry, list):
            return {
                "type": "vertices_list",
                "count": len(geometry),
                "total_vertices": sum(len(v) if hasattr(v, '__len__') else 0 for v in geometry)
            }
        else:
            return {
                "type": str(type(geometry).__name__),
                "repr": str(geometry)[:100]
            }

    def _collect_runtime_meta(self) -> Dict[str, Any]:
        """実行環境メタデータを収集（軽量版）"""
        try:
            meta = {
                "python": sys.version.split()[0],
                "os": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cores": os.cpu_count() or 1,
                "env": {k: v for k, v in os.environ.items() if k.startswith("PXD_")},
            }
        except Exception:
            meta = {"python": sys.version.split()[0]}
        return meta
    
    def _update_geometry_metrics(self, geometry: Any, result: BenchmarkResult) -> None:
        """ジオメトリメトリクスを更新"""
        if isinstance(geometry, Geometry) and hasattr(geometry, 'coords') and geometry.coords is not None:
            total_vertices = len(geometry.coords)
            result.metrics.vertices_count = max(result.metrics.vertices_count, total_vertices)
            
            # 複雑度計算（頂点数と線分数から）
            line_count = len(geometry.offsets) - 1 if hasattr(geometry, 'offsets') and geometry.offsets is not None else 1
            complexity = total_vertices * line_count / 1000.0
            result.metrics.geometry_complexity = max(result.metrics.geometry_complexity, complexity)
    
    def _get_test_geometry_for_effect(self) -> Geometry:
        """エフェクト用のテストジオメトリを取得"""
        # シンプルな正方形のテストジオメトリを作成
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)
        
        return Geometry.from_lines([vertices])
    
    def _measure_serialization_overhead(self, target: BenchmarkTarget) -> float:
        """シリアライゼーションオーバーヘッドを測定"""
        try:
            import pickle
            
            start_time = time.perf_counter()
            serialized = pickle.dumps(target)
            pickle.loads(serialized)
            end_time = time.perf_counter()
            
            return end_time - start_time
        except Exception:
            # シリアライゼーションできない場合は0を返す
            return 0.0


class BenchmarkResultProcessor:
    """ベンチマーク結果の後処理を担当するクラス"""
    
    @staticmethod
    def format_execution_summary(results: List[BenchmarkResult]) -> Dict[str, Any]:
        """実行結果のサマリーをフォーマット"""
        if not results:
            return {"message": "No results to summarize"}
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        summary = {
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results)
        }
        
        if successful:
            times = [r.timing_data.average_time for r in successful]
            summary.update({
                "fastest_time": min(times),
                "slowest_time": max(times),
                "average_time": sum(times) / len(times)
            })
        
        return summary
    
    @staticmethod
    def display_execution_status(plugin_name: str, results: List[BenchmarkResult]) -> None:
        """実行ステータスを表示"""
        logger = logging.getLogger(__name__)
        successful = len([r for r in results if r.success])
        total = len(results)
        
        status_icon = "✅" if successful == total else "⚠️" if successful > 0 else "❌"
        level = logging.INFO if successful == total else logging.WARNING if successful > 0 else logging.ERROR
        logger.log(level, "%s %s: %d/%d targets completed", status_icon, plugin_name, successful, total)
        
        # 失敗したターゲットの詳細
        failed_results = [r for r in results if not r.success]
        if failed_results:
            for result in failed_results[:3]:  # 最初の3個まで表示
                logger.warning("   ❌ %s: %s", result.target_name, result.error_message)
            if len(failed_results) > 3:
                logger.warning("   ... and %d more failures", len(failed_results) - 3)
