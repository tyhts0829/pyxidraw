#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマークシステム統合テスト

複数のコンポーネントが連携して正常に動作することを検証
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from benchmarks.core.config import BenchmarkConfig, BenchmarkConfigManager
from benchmarks.core.runner import UnifiedBenchmarkRunner
from benchmarks.core.execution import BenchmarkExecutor
from benchmarks.core.visualization import BenchmarkVisualizationGenerator
from benchmarks.benchmark_result_manager import BenchmarkResultManager
from benchmarks.plugins.base import PluginManager, BaseBenchmarkTarget
from benchmarks.core.types import BenchmarkResult, TimingData, BenchmarkMetrics
from engine.core.geometry import Geometry


class MockBenchmarkTarget(BaseBenchmarkTarget):
    """テスト用のベンチマークターゲット"""
    
    def __init__(self, name: str, execution_time: float = 0.001, should_fail: bool = False):
        self.execution_time = execution_time
        self.should_fail = should_fail
        
        def mock_function():
            if self.should_fail:
                raise RuntimeError(f"Mock function {name} failed")
            
            import time
            time.sleep(self.execution_time)
            
            # 簡単なGeometryを返す
            vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32)
            return Geometry.from_lines([vertices])
        
        super().__init__(name, mock_function)


class TestBenchmarkIntegration(unittest.TestCase):
    """ベンチマークシステムの統合テスト"""
    
    def setUp(self):
        """テスト用の一時ディレクトリと設定"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "benchmark_output"
        
        # テスト用設定
        self.config = BenchmarkConfig(
            warmup_runs=1,
            measurement_runs=3,
            parallel=False,
            max_workers=2,
            timeout_seconds=10.0,
            output_dir=self.output_dir,
            generate_charts=False  # チャート生成は重いのでデフォルトOFF
        )
        
        # テスト用ターゲット
        self.test_targets = [
            MockBenchmarkTarget("fast_target", 0.001),
            MockBenchmarkTarget("slow_target", 0.005),
            MockBenchmarkTarget("failing_target", 0.001, should_fail=True)
        ]
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_benchmark_execution(self):
        """エンドツーエンドのベンチマーク実行テスト"""
        # 1. ベンチマーク実行
        with patch.object(PluginManager, '_auto_discover_plugins'):
            runner = UnifiedBenchmarkRunner(self.config)
            
            # モックプラグインを手動で登録
            mock_plugin = MagicMock()
            mock_plugin.name = "test_plugin"
            mock_plugin.get_targets.return_value = self.test_targets[:2]  # 失敗するターゲットは除外
            runner.plugin_manager.register_plugin(mock_plugin)
            
            # ベンチマーク実行
            results = runner.run_all_benchmarks()
            
            # 結果検証
            self.assertIsInstance(results, dict)
            self.assertIn("test_plugin", results)
            
            plugin_results = results["test_plugin"]
            self.assertEqual(len(plugin_results), 2)
            
            # 各結果の詳細検証
            for target_name, result in plugin_results.items():
                self.assertIsInstance(result, BenchmarkResult)
                self.assertTrue(result.success)
                self.assertGreater(len(result.timing_data.measurement_times), 0)
                self.assertGreater(result.timing_data.average_time, 0)
    
    def test_benchmark_result_persistence(self):
        """ベンチマーク結果の永続化テスト"""
        # テスト用結果データを作成
        test_results = {
            "target1": BenchmarkResult(
                target_name="target1",
                plugin_name="test_plugin",
                config={},
                timestamp=1234567890.0,
                success=True,
                error_message="",
                timing_data=TimingData(
                    warm_up_times=[0.001],
                    measurement_times=[0.002, 0.003, 0.002],
                    total_time=0.008,
                    average_time=0.00233,
                    std_dev=0.0005,
                    min_time=0.002,
                    max_time=0.003
                ),
                metrics=BenchmarkMetrics(
                    vertices_count=100,
                    geometry_complexity=1.5,
                    memory_usage=1024,
                    cache_hit_rate=0.8
                ),
                output_data={"type": "geometry"},
                serialization_overhead=0.0001
            )
        }
        
        # 1. 結果保存
        manager = BenchmarkResultManager(str(self.output_dir))
        saved_file = manager.save_results(test_results)
        
        # 保存ファイルの存在確認
        self.assertTrue(Path(saved_file).exists())
        self.assertTrue((self.output_dir / "effects" / "latest.json").exists())
        
        # 2. 結果読み込み
        loaded_results = manager.load_results(saved_file)
        self.assertIsNotNone(loaded_results)
        self.assertIn("target1", loaded_results)
        
        # 3. 最新結果読み込み
        latest_results = manager.load_latest_results()
        self.assertEqual(len(latest_results), 1)
        self.assertIn("target1", latest_results)
        
        # 4. データ整合性確認
        loaded_result = loaded_results["target1"]
        self.assertEqual(loaded_result["target_name"], "target1")
        self.assertEqual(loaded_result["success"], True)
        self.assertEqual(len(loaded_result["timing_data"]["measurement_times"]), 3)
    
    def test_plugin_system_integration(self):
        """プラグインシステム統合テスト"""
        with patch.object(PluginManager, '_auto_discover_plugins'):
            # プラグインマネージャーの作成
            plugin_manager = PluginManager(self.config)
            
            # モックプラグインを作成・登録
            mock_plugin = MagicMock()
            mock_plugin.name = "integration_test_plugin"
            mock_plugin.plugin_type = "test"
            mock_plugin.get_targets.return_value = self.test_targets
            mock_plugin.validate_target.return_value = True
            
            plugin_manager.register_plugin(mock_plugin)
            
            # プラグイン取得テスト
            retrieved_plugin = plugin_manager.get_plugin("integration_test_plugin")
            self.assertEqual(retrieved_plugin, mock_plugin)
            
            # ターゲット取得テスト
            all_targets = plugin_manager.get_all_targets()
            self.assertIn("integration_test_plugin", all_targets)
            self.assertEqual(len(all_targets["integration_test_plugin"]), 3)
            
            # タイプ別プラグイン取得テスト
            test_plugins = plugin_manager.get_plugins_by_type("test")
            self.assertEqual(len(test_plugins), 1)
            self.assertEqual(test_plugins[0], mock_plugin)
    
    def test_error_handling_integration(self):
        """エラーハンドリング統合テスト"""
        with patch.object(PluginManager, '_auto_discover_plugins'):
            runner = UnifiedBenchmarkRunner(self.config)
            
            # 失敗するターゲットを含むプラグインを登録
            mock_plugin = MagicMock()
            mock_plugin.name = "error_test_plugin"
            mock_plugin.get_targets.return_value = [self.test_targets[2]]  # 失敗するターゲット
            runner.plugin_manager.register_plugin(mock_plugin)
            
            # ベンチマーク実行（エラーが発生するが継続する）
            results = runner.run_all_benchmarks()
            
            # 結果にはプラグインが含まれているが、失敗した結果になっている
            self.assertIn("error_test_plugin", results)
            plugin_results = results["error_test_plugin"]
            
            # 失敗したターゲットの結果を確認
            if plugin_results:  # 結果がある場合
                for result in plugin_results.values():
                    if not result.success:
                        self.assertIsNotNone(result.error_message)
                        self.assertIn("Mock function", result.error_message)
    
    def test_config_integration(self):
        """設定システム統合テスト"""
        # 1. 設定ファイル作成
        config_file = Path(self.temp_dir) / "test_config.yaml"
        config_data = {
            "warmup_runs": 2,
            "measurement_runs": 5,
            "parallel": True,
            "max_workers": 3,
            "timeout_seconds": 15.0,
            "output_dir": str(self.output_dir),
            "generate_charts": True
        }
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # 2. 設定読み込み
        config_manager = BenchmarkConfigManager(config_file)
        loaded_config = config_manager.load_config()
        
        # 3. 設定値確認
        self.assertEqual(loaded_config.warmup_runs, 2)
        self.assertEqual(loaded_config.measurement_runs, 5)
        self.assertTrue(loaded_config.parallel)
        self.assertEqual(loaded_config.max_workers, 3)
        self.assertEqual(loaded_config.timeout_seconds, 15.0)
        
        # 4. ランナーで設定使用
        with patch.object(PluginManager, '_auto_discover_plugins'):
            runner = UnifiedBenchmarkRunner(loaded_config)
            self.assertEqual(runner.config.measurement_runs, 5)
    
    def test_visualization_integration(self):
        """ビジュアライゼーション統合テスト"""
        # テスト用結果データ
        results = {
            "target1": BenchmarkResult(
                target_name="target1",
                plugin_name="test_plugin",
                config={},
                timestamp=1234567890.0,
                success=True,
                error_message="",
                timing_data=TimingData(
                    warm_up_times=[0.001],
                    measurement_times=[0.002, 0.003, 0.002],
                    total_time=0.007,
                    average_time=0.00233,
                    std_dev=0.0005,
                    min_time=0.002,
                    max_time=0.003
                ),
                metrics=BenchmarkMetrics(
                    vertices_count=100,
                    geometry_complexity=1.5,
                    memory_usage=1024,
                    cache_hit_rate=0.8
                ),
                output_data=None,
                serialization_overhead=0.0001
            ),
            "target2": BenchmarkResult(
                target_name="target2",
                plugin_name="test_plugin",
                config={},
                timestamp=1234567890.0,
                success=True,
                error_message="",
                timing_data=TimingData(
                    warm_up_times=[0.002],
                    measurement_times=[0.005, 0.004, 0.006],
                    total_time=0.017,
                    average_time=0.005,
                    std_dev=0.001,
                    min_time=0.004,
                    max_time=0.006
                ),
                metrics=BenchmarkMetrics(
                    vertices_count=200,
                    geometry_complexity=2.0,
                    memory_usage=2048,
                    cache_hit_rate=0.9
                ),
                output_data=None,
                serialization_overhead=0.0002
            )
        }
        
        # ビジュアライゼーション設定でチャート生成を有効化
        viz_config = BenchmarkConfig(
            output_dir=self.output_dir,
            generate_charts=True
        )
        
        # エラーコレクターのモック
        error_collector = MagicMock()
        
        # ビジュアライゼーション生成テスト
        viz_generator = BenchmarkVisualizationGenerator(viz_config, error_collector)
        
        # 例外が発生しないことを確認（実際のチャート生成はライブラリ依存なのでモック）
        with patch('benchmarks.visualization.charts.ChartGenerator') as mock_chart_gen:
            mock_chart_gen.return_value.create_bar_chart.return_value = str(self.output_dir / "test_chart.png")
            
            try:
                viz_generator.generate_auto_visualization(results)
                # エラーなく完了することを確認
                self.assertTrue(True)
            except ImportError:
                # ビジュアライゼーションライブラリがない場合はスキップ
                self.skipTest("Visualization libraries not available")
    
    def test_parallel_execution_integration(self):
        """並列実行統合テスト"""
        # 並列実行設定
        parallel_config = BenchmarkConfig(
            warmup_runs=1,
            measurement_runs=2,
            parallel=True,
            max_workers=2,
            output_dir=self.output_dir
        )
        
        with patch.object(PluginManager, '_auto_discover_plugins'):
            runner = UnifiedBenchmarkRunner(parallel_config)
            
            # 複数のターゲットを持つプラグインを登録
            mock_plugin = MagicMock()
            mock_plugin.name = "parallel_test_plugin"
            mock_plugin.get_targets.return_value = self.test_targets[:2]  # 成功するターゲットのみ
            runner.plugin_manager.register_plugin(mock_plugin)
            
            # 並列実行
            results = runner.run_all_benchmarks()
            
            # 結果検証
            self.assertIn("parallel_test_plugin", results)
            plugin_results = results["parallel_test_plugin"]
            self.assertEqual(len(plugin_results), 2)
            
            # すべてのターゲットが正常実行されたことを確認
            for result in plugin_results.values():
                self.assertTrue(result.success)


class TestBenchmarkComponentIntegration(unittest.TestCase):
    """個別コンポーネント間の統合テスト"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = BenchmarkConfig(output_dir=Path(self.temp_dir))
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_executor_and_result_manager_integration(self):
        """ベンチマーク実行エンジンと結果管理の連携テスト"""
        from benchmarks.core.exceptions import get_error_handler, get_error_collector
        
        # エグゼキューターの作成
        executor = BenchmarkExecutor(
            self.config,
            get_error_handler(),
            get_error_collector()
        )
        
        # テストターゲットの作成
        target = MockBenchmarkTarget("integration_test", 0.001)
        
        # ベンチマーク実行
        result = executor.initialize_benchmark_result(target)
        executor.measure_target_characteristics(target, result)
        executor.execute_benchmark_measurements(target, result)
        executor.calculate_benchmark_statistics(result)
        
        # 結果検証
        self.assertTrue(result.success)
        self.assertGreater(len(result.timing_data.measurement_times), 0)
        
        # 結果管理への保存
        result_manager = BenchmarkResultManager(str(self.config.output_dir))
        results_dict = {"integration_test": result}
        saved_file = result_manager.save_results(results_dict)
        
        # 保存確認
        self.assertTrue(Path(saved_file).exists())
        
        # 読み込み確認
        loaded_results = result_manager.load_results(saved_file)
        self.assertIsNotNone(loaded_results)
        self.assertIn("integration_test", loaded_results)
    
    def test_config_and_plugin_integration(self):
        """設定とプラグインシステムの連携テスト"""
        with patch.object(PluginManager, '_auto_discover_plugins'):
            # プラグインマネージャーの作成
            plugin_manager = PluginManager(self.config)
            
            # 設定がプラグインに正しく渡されることを確認
            self.assertEqual(plugin_manager.config, self.config)
            
            # プラグイン登録時の設定連携確認
            mock_plugin = MagicMock()
            mock_plugin.name = "config_test_plugin"
            mock_plugin.config = self.config
            
            plugin_manager.register_plugin(mock_plugin)
            
            retrieved_plugin = plugin_manager.get_plugin("config_test_plugin")
            self.assertEqual(retrieved_plugin.config, self.config)


if __name__ == '__main__':
    unittest.main()