#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統一ベンチマークランナー

プラグインシステムを使用してベンチマークを実行する統合ランナー。
並列実行、エラーハンドリング、結果収集を管理します。
"""

import pickle
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

from benchmarks.core.types import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkTarget,
    TimingData,
    BenchmarkMetrics,
)
from benchmarks.core.config import get_config
from benchmarks.core.exceptions import (
    BenchmarkError,
    BenchmarkTimeoutError,
    benchmark_operation,
    get_error_collector,
    get_error_handler,
)
from benchmarks.core.execution import BenchmarkExecutor, BenchmarkResultProcessor
from benchmarks.core.visualization import BenchmarkVisualizationGenerator
from benchmarks.plugins.base import PluginManager, create_plugin_manager
from benchmarks.plugins.serializable_targets import init_worker
from engine.core.geometry import Geometry


class UnifiedBenchmarkRunner:
    """統一ベンチマークランナークラス"""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or get_config()
        self.plugin_manager = create_plugin_manager(self.config)
        self.error_handler = get_error_handler()
        self.error_collector = get_error_collector()
        self.logger = logging.getLogger(__name__)
        
        # 新しい分離されたコンポーネント
        self.executor = BenchmarkExecutor(self.config, self.error_handler, self.error_collector)
        self.result_processor = BenchmarkResultProcessor()
        self.visualization_generator = BenchmarkVisualizationGenerator(self.config, self.error_collector)
        
        # エラーハンドリング設定
        self.error_handler.configure(
            max_errors=self.config.max_errors,
            continue_on_error=self.config.continue_on_error
        )
        
        # テストデータの遅延初期化
        self._test_geometries: Optional[Dict[str, Geometry]] = None
    
    @property
    def test_geometries(self) -> Dict[str, Geometry]:
        """テストジオメトリの遅延初期化"""
        if self._test_geometries is None:
            self._test_geometries = self._create_test_geometries()
        return self._test_geometries
    
    def _create_test_geometries(self) -> Dict[str, Geometry]:
        """ベンチマーク用テストジオメトリを作成"""
        geometries = {}
        
        # Simple rectangle
        geometries["small"] = self._create_rectangle(1, 1)
        
        # Medium polygon
        geometries["medium"] = self._create_polygon(20)
        
        # Complex shape (multiple circles)
        geometries["large"] = self._create_complex_shape()
        
        return geometries
    
    @staticmethod
    def _create_rectangle(width: float, height: float) -> Geometry:
        """シンプルな長方形を作成"""
        hw, hh = width / 2, height / 2
        vertices = np.array(
            [[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0], [-hw, -hh, 0.0]], 
            dtype=np.float32
        )
        return Geometry.from_lines([vertices])
    
    @staticmethod
    def _create_polygon(n: int) -> Geometry:
        """n辺の正多角形を作成"""
        angles = np.linspace(0, 2 * np.pi, n + 1)
        vertices = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)]).astype(np.float32)
        return Geometry.from_lines([vertices])
    
    def _create_complex_shape(self) -> Geometry:
        """ベンチマーク用の複雑な形状を作成（複数の円を結合）"""
        circles = []
        for i in range(10):
            radius = 1.0 + i * 0.1
            angles = np.linspace(0, 2 * np.pi, 64 + 1)
            vertices = np.column_stack(
                [radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)]
            ).astype(np.float32)
            circles.append(vertices)
        return Geometry.from_lines(circles)
    
    def run_all_benchmarks(self) -> Dict[str, Dict[str, BenchmarkResult]]:
        """すべてのプラグインのベンチマークを実行"""
        all_results: Dict[str, Dict[str, BenchmarkResult]] = {}
        self.logger.info("Starting unified benchmark run at %s", datetime.now())
        self.logger.info("%s", "=" * 60)
        
        # すべてのプラグインからターゲットを取得
        all_targets = self.plugin_manager.get_all_targets()
        
        total_targets = sum(len(targets) for targets in all_targets.values())
        self.logger.info("Found %d benchmark targets across %d plugins", total_targets, len(all_targets))
        
        # プラグイン別にベンチマーク実行
        for plugin_name, targets in all_targets.items():
            if not targets:
                self.logger.info("Skipping %s: no targets found", plugin_name)
                continue
            
            self.logger.info("--- Benchmarking %s (%d targets) ---", plugin_name, len(targets))
            
            if self.config.parallel:
                plugin_results = self._run_plugin_parallel(plugin_name, targets)
            else:
                plugin_results = self._run_plugin_sequential(plugin_name, targets)

            all_results[plugin_name] = plugin_results
        
        # エラー要約を表示
        if self.error_collector.has_errors() or self.error_collector.has_warnings():
            self.logger.error("%s", "=" * 60)
            self.logger.error("ERROR SUMMARY")
            self.logger.error("%s", "=" * 60)
            self.logger.error("%s", self.error_collector.generate_report())
        
        self.logger.info("Benchmark completed at %s", datetime.now())
        self.logger.info("Total results: %d", sum(len(v) for v in all_results.values()))
        
        # 自動ビジュアライゼーション
        if all_results and self.config.generate_charts:
            # フラットにしてビジュアライゼーションに渡す
            flat: Dict[str, BenchmarkResult] = {}
            for plugin_name, results in all_results.items():
                for target_name, res in results.items():
                    flat_key = f"{plugin_name}.{target_name}"
                    flat[flat_key] = res
            self._generate_auto_visualization(flat)

        return all_results
    
    def _run_plugin_sequential(self, plugin_name: str, targets: List[BenchmarkTarget]) -> Dict[str, BenchmarkResult]:
        """プラグインのベンチマークを順次実行"""
        results = {}
        
        for i, target in enumerate(targets, 1):
            self.logger.info("  [%d/%d] %s...", i, len(targets), target.name)
            
            try:
                result = self.benchmark_target(target)
                results[target.name] = result
                
                if result.success:
                    avg_time = result.timing_data.average_time if result.timing_data.average_time > 0 else 0
                    if avg_time > 0:
                        fps = 1.0 / avg_time
                        self.logger.info("✓ (%.1f fps)", fps)
                    else:
                        self.logger.info("✓ (instant)")
                else:
                    self.logger.warning("✗ (%s)", result.error_message or "unknown error")
                    
            except Exception as e:
                self.logger.exception("✗ (exception: %s)", e)
                # エラーを収集
                error = BenchmarkError(f"Unhandled exception in {target.name}: {e}", module_name=target.name)
                self.error_collector.add_error(error)
        
        return results
    
    def _run_plugin_parallel(self, plugin_name: str, targets: List[BenchmarkTarget]) -> Dict[str, BenchmarkResult]:
        """プラグインのベンチマークを並列実行"""
        # 小規模では並列オーバーヘッドが支配的になるため逐次にフォールバック
        if len(targets) <= 4:
            return self._run_plugin_sequential(plugin_name, targets)

        results = {}
        # CPUバウンドの単純作業のため、ターゲット数に合わせてワーカー数を設定
        max_workers = len(targets)
        
        self.logger.info("  Running %d targets in parallel (max_workers=%d)", len(targets), max_workers)
        
        # 環境制約により、常にスレッドプールを使用（セマフォ未許可環境対策）
        executor_class = ThreadPoolExecutor
        
        # スレッドプール用の引数
        executor_kwargs = {"max_workers": max_workers}
        
        with executor_class(**executor_kwargs) as executor:
            # 全ターゲットを並列実行
            future_to_target = {executor.submit(self.benchmark_target, target): target for target in targets}
            
            completed = 0
            for future in as_completed(future_to_target):
                target = future_to_target[future]
                completed += 1
                
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    results[target.name] = result
                    # 進捗出力は抑制
                    
                except Exception as e:
                    # 進捗出力は抑制
                    error = BenchmarkError(f"Unhandled exception in {target.name}: {e}", module_name=target.name)
                    self.error_collector.add_error(error)
        
        return results
    
    def benchmark_target(self, target: BenchmarkTarget) -> BenchmarkResult:
        """単一ターゲットのベンチマークを実行（BenchmarkResult を返す）"""
        # 新しい分離されたコンポーネントを使用
        result = self.executor.initialize_benchmark_result(target)
        try:
            with benchmark_operation(f"benchmark_{target.name}", target.name):
                self.executor.measure_target_characteristics(target, result)
                self.executor.execute_benchmark_measurements(target, result)
                self.executor.calculate_benchmark_statistics(result)
        except Exception as e:
            self.executor.handle_benchmark_exception(result, e)
        return result
    
    # レガシー辞書互換の簡易分離実行は削除しました
    
    def _benchmark_shape_generation(self, target: BenchmarkTarget) -> Optional[List[float]]:
        """形状生成のベンチマーク"""
        # ウォームアップ
        for _ in range(self.config.warmup_runs):
            try:
                target.execute()
            except Exception:
                return None
        
        # 測定
        times = []
        for _ in range(self.config.measurement_runs):
            try:
                start_time = time.perf_counter()
                result = target.execute()
                end_time = time.perf_counter()
                
                # 結果の妥当性をチェック
                if hasattr(result, 'coords') and len(result.coords) > 0:
                    times.append(end_time - start_time)
                
            except Exception:
                return None
        
        return times if times else None
    
    def _benchmark_effect_application(self, target: BenchmarkTarget, test_geom: Geometry) -> Optional[List[float]]:
        """エフェクト適用のベンチマーク"""
        # ウォームアップ
        for _ in range(self.config.warmup_runs):
            try:
                target.execute(test_geom)
            except Exception:
                return None
        
        # 測定
        times = []
        for _ in range(self.config.measurement_runs):
            try:
                start_time = time.perf_counter()
                result = target.execute(test_geom)
                end_time = time.perf_counter()
                
                # 結果の妥当性をチェック
                if hasattr(result, 'coords') and len(result.coords) > 0:
                    times.append(end_time - start_time)
                
            except Exception:
                return None
        
        return times if times else None
    
    def _is_shape_target(self, target: BenchmarkTarget) -> bool:
        """ターゲットが形状生成かどうかを判定"""
        # メタデータから判定
        if hasattr(target, 'metadata'):
            return target.metadata.get('shape_type') is not None
        
        # 名前から判定
        shape_indicators = [
            'polygon', 'sphere', 'cylinder', 'cone', 'torus', 'capsule',
            'polyhedron', 'lissajous', 'attractor', 'text', 'asemic_glyph', 'grid'
        ]
        return any(indicator in target.name for indicator in shape_indicators)
    
    def _measure_serialization_overhead(self, target: BenchmarkTarget, test_geom: Optional[Any] = None) -> Dict[str, float]:
        """シリアライズ/デシリアライズのオーバーヘッドを測定"""
        overhead = {}
        
        # ターゲットのシリアライズ時間を測定
        start_time = time.perf_counter()
        try:
            serialized_target = pickle.dumps(target)
            overhead['target_serialize_time'] = time.perf_counter() - start_time
            overhead['target_size_bytes'] = len(serialized_target)
            
            # デシリアライズ時間を測定
            start_time = time.perf_counter()
            pickle.loads(serialized_target)
            overhead['target_deserialize_time'] = time.perf_counter() - start_time
        except Exception as e:
            overhead['target_serialize_error'] = str(e)
        
        # Geometryオブジェクトのシリアライズ時間を測定（エフェクトの場合）
        if test_geom is not None:
            start_time = time.perf_counter()
            try:
                serialized_geom = pickle.dumps(test_geom)
                overhead['geometry_serialize_time'] = time.perf_counter() - start_time
                overhead['geometry_size_bytes'] = len(serialized_geom)
                
                # デシリアライズ時間を測定
                start_time = time.perf_counter()
                pickle.loads(serialized_geom)
                overhead['geometry_deserialize_time'] = time.perf_counter() - start_time
            except Exception as e:
                overhead['geometry_serialize_error'] = str(e)
        
        return overhead
    
    def _get_plugin_for_target(self, target: BenchmarkTarget) -> Optional[Any]:
        """ターゲットに対応するプラグインを取得"""
        for plugin in self.plugin_manager.get_all_plugins():
            targets = plugin.get_targets()
            if any(t.name == target.name for t in targets):
                return plugin
        return None
    
    def run_specific_targets(self, target_names: List[str]) -> Dict[str, BenchmarkResult]:
        """特定のターゲットのみベンチマークを実行"""
        all_targets = self.plugin_manager.get_all_targets()
        
        # 名前でターゲットを検索（プラグイン名付きと基本名の両方をサポート）
        selected_targets = []
        for plugin_name, plugin_targets in all_targets.items():
            for target in plugin_targets:
                # プラグイン名付きの完全名と基本名の両方をチェック
                full_name = f"{plugin_name}.{target.name}"
                if target.name in target_names or full_name in target_names:
                    selected_targets.append(target)
        
        if not selected_targets:
            self.logger.warning("No targets found matching: %s", target_names)
            return {}
        
        self.logger.info("Running %d specific targets", len(selected_targets))
        
        # 順次実行
        results = {}
        for target in selected_targets:
            self.logger.info("Benchmarking %s...", target.name)
            result = self.benchmark_target(target)
            results[target.name] = result
            
            status = "✓" if result.success else "✗"
            self.logger.info("%s", status)
        
        # 自動ビジュアライゼーション
        if results and self.config.generate_charts:
            self._generate_auto_visualization(results)
        
        return results
    
    def get_available_targets(self) -> Dict[str, List[str]]:
        """利用可能なターゲット一覧を取得"""
        all_targets = self.plugin_manager.get_all_targets()
        return {
            plugin_name: [target.name for target in targets]
            for plugin_name, targets in all_targets.items()
        }
    
    def list_available_targets(self) -> List[str]:
        """利用可能なターゲット一覧をフラットなリストで取得（CLI用）"""
        all_targets = self.plugin_manager.get_all_targets()
        target_list = []
        for plugin_name, targets in all_targets.items():
            for target in targets:
                # プラグイン名を含めた完全な名前を使用
                target_list.append(f"{plugin_name}.{target.name}")
        return sorted(target_list)
    
    def _generate_auto_visualization(self, results: Dict[str, BenchmarkResult]) -> None:
        """ベンチマーク結果の自動ビジュアライゼーション"""
        # 新しい分離されたビジュアライゼーション生成コンポーネントに委譲
        self.visualization_generator.generate_auto_visualization(results)


# 便利関数
def create_runner(config: Optional[BenchmarkConfig] = None) -> UnifiedBenchmarkRunner:
    """ベンチマークランナーを作成する便利関数"""
    return UnifiedBenchmarkRunner(config)


def run_benchmarks(config: Optional[BenchmarkConfig] = None) -> Dict[str, BenchmarkResult]:
    """ベンチマークを実行する便利関数"""
    runner = create_runner(config)
    return runner.run_all_benchmarks()
