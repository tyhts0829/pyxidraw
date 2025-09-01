#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマークシステム パフォーマンステスト

システム自体のパフォーマンスと回帰検出のためのテスト
"""

import time
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from statistics import mean, stdev

import numpy as np

from benchmarks.core.config import BenchmarkConfig
from benchmarks.core.runner import UnifiedBenchmarkRunner
from benchmarks.core.execution import BenchmarkExecutor
from benchmarks.benchmark_result_manager import BenchmarkResultManager
from benchmarks.plugins.base import PluginManager, BaseBenchmarkTarget
from benchmarks.core.types import BenchmarkResult, TimingData, BenchmarkMetrics
from benchmarks.core.exceptions import get_error_handler, get_error_collector
from engine.core.geometry import Geometry


class PerformanceBenchmarkTarget(BaseBenchmarkTarget):
    """パフォーマンステスト用のベンチマークターゲット"""
    
    def __init__(self, name: str, work_complexity: int = 1000):
        self.work_complexity = work_complexity
        
        def performance_function():
            # CPU集約的な作業をシミュレート
            result = 0
            for i in range(self.work_complexity):
                result += i * i
            
            # Geometryオブジェクトを作成
            vertices = np.random.rand(min(100, self.work_complexity // 10), 3).astype(np.float32)
            return Geometry.from_lines([vertices])
        
        super().__init__(name, performance_function)


class TestBenchmarkSystemPerformance(unittest.TestCase):
    """ベンチマークシステム自体のパフォーマンステスト"""
    
    @classmethod
    def setUpClass(cls):
        """クラス全体のセットアップ"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.config = BenchmarkConfig(
            warmup_runs=2,
            measurement_runs=5,
            parallel=False,
            output_dir=Path(cls.temp_dir)
        )
        
        # パフォーマンス基準値（秒）
        cls.PERFORMANCE_THRESHOLDS = {
            'single_target_execution': 0.1,    # 単一ターゲット実行時間
            'result_save_time': 0.05,          # 結果保存時間
            'result_load_time': 0.02,          # 結果読み込み時間
            'plugin_discovery_time': 0.1,      # プラグイン発見時間
            'benchmark_initialization': 0.05,  # ベンチマーク初期化時間
        }
    
    @classmethod
    def tearDownClass(cls):
        """クラス全体のクリーンアップ"""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """各テストのセットアップ"""
        self.performance_data = {}
    
    def measure_time(self, operation_name):
        """時間測定用のコンテキストマネージャー"""
        class TimeContext:
            def __init__(self, test_instance, name):
                self.test_instance = test_instance
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.perf_counter()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.perf_counter()
                execution_time = end_time - self.start_time
                self.test_instance.performance_data[self.name] = execution_time
                
                # 閾値チェック
                threshold = self.test_instance.PERFORMANCE_THRESHOLDS.get(self.name)
                if threshold and execution_time > threshold:
                    print(f"⚠️  Performance warning: {self.name} took {execution_time:.3f}s (threshold: {threshold:.3f}s)")
        
        return TimeContext(self, operation_name)
    
    def test_single_target_execution_performance(self):
        """単一ターゲット実行のパフォーマンステスト"""
        executor = BenchmarkExecutor(
            self.config,
            get_error_handler(),
            get_error_collector()
        )
        
        target = PerformanceBenchmarkTarget("perf_test", work_complexity=1000)
        
        with self.measure_time('single_target_execution'):
            result = executor.initialize_benchmark_result(target)
            executor.measure_target_characteristics(target, result)
            executor.execute_benchmark_measurements(target, result)
            executor.calculate_benchmark_statistics(result)
        
        # 結果検証
        self.assertTrue(result.success)
        self.assertGreater(len(result.timing_data.measurement_times), 0)
        
        # パフォーマンス検証
        execution_time = self.performance_data['single_target_execution']
        self.assertLess(execution_time, self.PERFORMANCE_THRESHOLDS['single_target_execution'],
                       f"Single target execution too slow: {execution_time:.3f}s")
    
    def test_result_persistence_performance(self):
        """結果永続化のパフォーマンステスト"""
        # 大きな結果データセットを作成
        results = {}
        for i in range(20):  # 20個のターゲット結果
            results[f"target_{i}"] = BenchmarkResult(
                target_name=f"target_{i}",
                plugin_name="perf_test_plugin",
                config={},
                timestamp=time.time(),
                success=True,
                error_message="",
                timing_data=TimingData(
                    warm_up_times=[0.001, 0.002],
                    measurement_times=[0.003, 0.004, 0.005, 0.003, 0.004],
                    total_time=0.019,
                    average_time=0.0038,
                    std_dev=0.0008,
                    min_time=0.003,
                    max_time=0.005
                ),
                metrics=BenchmarkMetrics(
                    vertices_count=100 * (i + 1),
                    geometry_complexity=1.5 * (i + 1),
                    memory_usage=1024 * (i + 1),
                    cache_hit_rate=0.8
                ),
                output_data={"data": list(range(i * 10))},  # 可変サイズデータ
                serialization_overhead=0.0001
            )
        
        manager = BenchmarkResultManager(str(self.config.output_dir))
        
        # 保存パフォーマンステスト
        with self.measure_time('result_save_time'):
            saved_file = manager.save_results(results)
        
        # 読み込みパフォーマンステスト
        with self.measure_time('result_load_time'):
            loaded_results = manager.load_results(saved_file)
        
        # 結果検証
        self.assertIsNotNone(loaded_results)
        self.assertEqual(len(loaded_results), 20)
        
        # パフォーマンス検証
        save_time = self.performance_data['result_save_time']
        load_time = self.performance_data['result_load_time']
        
        self.assertLess(save_time, self.PERFORMANCE_THRESHOLDS['result_save_time'],
                       f"Result save too slow: {save_time:.3f}s")
        self.assertLess(load_time, self.PERFORMANCE_THRESHOLDS['result_load_time'],
                       f"Result load too slow: {load_time:.3f}s")
    
    def test_plugin_discovery_performance(self):
        """プラグイン発見のパフォーマンステスト"""
        with patch.object(PluginManager, '_auto_discover_plugins') as mock_discovery:
            # 実際の発見処理をモック
            mock_discovery.side_effect = lambda: time.sleep(0.01)  # 軽い処理をシミュレート
            
            with self.measure_time('plugin_discovery_time'):
                plugin_manager = PluginManager(self.config)
            
            # パフォーマンス検証
            discovery_time = self.performance_data['plugin_discovery_time']
            self.assertLess(discovery_time, self.PERFORMANCE_THRESHOLDS['plugin_discovery_time'],
                           f"Plugin discovery too slow: {discovery_time:.3f}s")
    
    def test_benchmark_initialization_performance(self):
        """ベンチマーク初期化のパフォーマンステスト"""
        with patch.object(PluginManager, '_auto_discover_plugins'):
            with self.measure_time('benchmark_initialization'):
                runner = UnifiedBenchmarkRunner(self.config)
                
                # 複数のモックプラグインを登録
                for i in range(5):
                    mock_plugin = MagicMock()
                    mock_plugin.name = f"perf_plugin_{i}"
                    mock_plugin.get_targets.return_value = [
                        PerformanceBenchmarkTarget(f"target_{j}", 100) for j in range(3)
                    ]
                    runner.plugin_manager.register_plugin(mock_plugin)
            
            # パフォーマンス検証
            init_time = self.performance_data['benchmark_initialization']
            self.assertLess(init_time, self.PERFORMANCE_THRESHOLDS['benchmark_initialization'],
                           f"Benchmark initialization too slow: {init_time:.3f}s")
    
    def test_parallel_vs_sequential_performance(self):
        """並列実行 vs 逐次実行のパフォーマンス比較"""
        targets = [PerformanceBenchmarkTarget(f"parallel_target_{i}", 500) for i in range(4)]
        
        # 逐次実行の測定
        sequential_config = BenchmarkConfig(
            warmup_runs=1,
            measurement_runs=3,
            parallel=False,
            output_dir=self.config.output_dir
        )
        
        with patch.object(PluginManager, '_auto_discover_plugins'):
            sequential_runner = UnifiedBenchmarkRunner(sequential_config)
            mock_plugin = MagicMock()
            mock_plugin.name = "sequential_plugin"
            mock_plugin.get_targets.return_value = targets
            sequential_runner.plugin_manager.register_plugin(mock_plugin)
            
            start_time = time.perf_counter()
            sequential_results = sequential_runner.run_all_benchmarks()
            sequential_time = time.perf_counter() - start_time
        
        # 並列実行の測定
        parallel_config = BenchmarkConfig(
            warmup_runs=1,
            measurement_runs=3,
            parallel=True,
            max_workers=2,
            output_dir=self.config.output_dir
        )
        
        with patch.object(PluginManager, '_auto_discover_plugins'):
            parallel_runner = UnifiedBenchmarkRunner(parallel_config)
            mock_plugin = MagicMock()
            mock_plugin.name = "parallel_plugin"
            mock_plugin.get_targets.return_value = targets
            parallel_runner.plugin_manager.register_plugin(mock_plugin)
            
            start_time = time.perf_counter()
            parallel_results = parallel_runner.run_all_benchmarks()
            parallel_time = time.perf_counter() - start_time
        
        # 結果検証
        self.assertEqual(len(sequential_results["sequential_plugin"]), 4)
        self.assertEqual(len(parallel_results["parallel_plugin"]), 4)
        
        # パフォーマンス比較（並列実行の方が速いはず）
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1
        print(f"並列実行スピードアップ: {speedup:.2f}x (逐次: {sequential_time:.3f}s, 並列: {parallel_time:.3f}s)")
        
        # 並列実行は少なくとも同程度以上の性能を示すはず
        self.assertLessEqual(parallel_time, sequential_time * 1.2,  # 20%のオーバーヘッドまで許容
                           f"Parallel execution not efficient: sequential={sequential_time:.3f}s, parallel={parallel_time:.3f}s")
    
    def test_memory_usage_performance(self):
        """メモリ使用量のパフォーマンステスト"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 大量のベンチマーク結果を処理
        large_results = {}
        for i in range(100):
            large_results[f"memory_target_{i}"] = BenchmarkResult(
                target_name=f"memory_target_{i}",
                plugin_name="memory_test_plugin",
                config={},
                timestamp=time.time(),
                success=True,
                error_message="",
                timing_data=TimingData(
                    warm_up_times=[0.001] * 10,
                    measurement_times=[0.002] * 50,  # 大きなデータ
                    total_time=0.1,
                    average_time=0.002,
                    std_dev=0.0001,
                    min_time=0.001,
                    max_time=0.003
                ),
                metrics=BenchmarkMetrics(
                    vertices_count=1000,
                    geometry_complexity=5.0,
                    memory_usage=10240,
                    cache_hit_rate=0.9
                ),
                output_data={"large_data": list(range(1000))},  # 大きなデータ
                serialization_overhead=0.001
            )
        
        # 結果管理での処理
        manager = BenchmarkResultManager(str(self.config.output_dir))
        saved_file = manager.save_results(large_results)
        loaded_results = manager.load_results(saved_file)
        
        # メモリ使用量確認
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"メモリ使用量増加: {memory_increase:.1f}MB (初期: {initial_memory:.1f}MB, 最終: {final_memory:.1f}MB)")
        
        # メモリ使用量が過度に増加していないことを確認（100MB以下）
        self.assertLess(memory_increase, 100,
                       f"Memory usage increased too much: {memory_increase:.1f}MB")
        
        # 結果の正確性も確認
        self.assertEqual(len(loaded_results), 100)
    
    def test_error_handling_performance(self):
        """エラーハンドリングのパフォーマンステスト"""
        # エラーが発生する多数のターゲット
        error_targets = []
        for i in range(10):
            target = MagicMock()
            target.name = f"error_target_{i}"
            target.execute.side_effect = RuntimeError(f"Error {i}")
            error_targets.append(target)
        
        executor = BenchmarkExecutor(
            self.config,
            get_error_handler(),
            get_error_collector()
        )
        
        start_time = time.perf_counter()
        
        # 複数のエラーを処理
        for target in error_targets:
            result = executor.initialize_benchmark_result(target)
            try:
                executor.execute_benchmark_measurements(target, result)
            except Exception as e:
                executor.handle_benchmark_exception(result, e)
        
        error_handling_time = time.perf_counter() - start_time
        
        # エラーハンドリングが高速であることを確認
        self.assertLess(error_handling_time, 0.1,
                       f"Error handling too slow: {error_handling_time:.3f}s for 10 errors")


class TestBenchmarkPerformanceRegression(unittest.TestCase):
    """パフォーマンス回帰検出テスト"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_file = Path(self.temp_dir) / "baseline_performance.json"
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_performance_regression_detection(self):
        """パフォーマンス回帰検出のテスト"""
        # ベースライン結果の作成
        baseline_results = {
            "fast_target": {
                "average_time": 0.001,
                "min_time": 0.0008,
                "max_time": 0.0012
            },
            "slow_target": {
                "average_time": 0.005,
                "min_time": 0.004,
                "max_time": 0.006
            }
        }
        
        # 現在の結果（一部で性能劣化）
        current_results = {
            "fast_target": {
                "average_time": 0.0015,  # 50%劣化
                "min_time": 0.0012,
                "max_time": 0.0018
            },
            "slow_target": {
                "average_time": 0.0045,  # 10%改善
                "min_time": 0.0035,
                "max_time": 0.0055
            }
        }
        
        # 回帰検出ロジック
        regressions = []
        improvements = []
        
        for target_name in baseline_results:
            if target_name in current_results:
                baseline_time = baseline_results[target_name]["average_time"]
                current_time = current_results[target_name]["average_time"]
                
                change_ratio = (current_time - baseline_time) / baseline_time
                
                if change_ratio > 0.2:  # 20%以上の劣化
                    regressions.append({
                        "target": target_name,
                        "change": change_ratio,
                        "baseline": baseline_time,
                        "current": current_time
                    })
                elif change_ratio < -0.1:  # 10%以上の改善
                    improvements.append({
                        "target": target_name,
                        "change": change_ratio,
                        "baseline": baseline_time,
                        "current": current_time
                    })
        
        # 検出結果の検証
        self.assertEqual(len(regressions), 1)
        self.assertEqual(regressions[0]["target"], "fast_target")
        self.assertGreater(regressions[0]["change"], 0.4)  # 40%以上の劣化
        
        self.assertEqual(len(improvements), 1)
        self.assertEqual(improvements[0]["target"], "slow_target")
        self.assertLess(improvements[0]["change"], -0.05)  # 5%以上の改善
        
        # 回帰が検出された場合の警告
        if regressions:
            print(f"⚠️  Performance regressions detected:")
            for regression in regressions:
                print(f"  {regression['target']}: {regression['change']:+.1%} "
                     f"({regression['baseline']:.3f}s → {regression['current']:.3f}s)")
        
        if improvements:
            print(f"✅ Performance improvements detected:")
            for improvement in improvements:
                print(f"  {improvement['target']}: {improvement['change']:+.1%} "
                     f"({improvement['baseline']:.3f}s → {improvement['current']:.3f}s)")


if __name__ == '__main__':
    # パフォーマンステストの実行
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 基本パフォーマンステスト
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarkSystemPerformance))
    
    # 回帰検出テスト
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarkPerformanceRegression))
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # パフォーマンステスト結果の要約
    if result.wasSuccessful():
        print("\n🚀 All performance tests passed!")
    else:
        print(f"\n⚠️  {len(result.failures)} test(s) failed, {len(result.errors)} error(s) occurred")
        print("Performance regression may have been detected.")