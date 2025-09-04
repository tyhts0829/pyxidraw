#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク実行モジュール(runner.py)のテスト
"""

import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np

from benchmarks.core.config import BenchmarkConfig
from benchmarks.core.runner import UnifiedBenchmarkRunner
from benchmarks.core.types import BenchmarkResult, TimingData, BenchmarkMetrics
from benchmarks.plugins.base import BaseBenchmarkTarget

# --- ヘルパー関数とモッククラス ---

class MockPlugin:
    def __init__(self, name, targets):
        self.name = name
        self._targets = targets

    def get_targets(self, refresh=False):
        return self._targets

    def analyze_target_features(self, target):
        return {"has_njit": False, "has_cache": False, "function_count": 1}

def create_typed_result(name: str, avg_time: float) -> BenchmarkResult:
    return BenchmarkResult(
        target_name=name,
        plugin_name="test_plugin",
        config={},
        timestamp=0.0,
        success=True,
        error_message="",
        timing_data=TimingData(
            warm_up_times=[avg_time],
            measurement_times=[avg_time, avg_time],
            total_time=avg_time * 2,
            average_time=avg_time,
            std_dev=0.0,
            min_time=avg_time,
            max_time=avg_time,
        ),
        metrics=BenchmarkMetrics(
            vertices_count=0,
            geometry_complexity=0.0,
            memory_usage=0,
            cache_hit_rate=0.0,
        ),
    )

# --- テストクラス本体 ---

class TestUnifiedBenchmarkRunner(unittest.TestCase):
    """UnifiedBenchmarkRunnerのテスト"""

    def setUp(self):
        self.config = BenchmarkConfig(warmup_runs=1, measurement_runs=2)
        self.runner = UnifiedBenchmarkRunner(config=self.config)

        # シリアライズ可能なGeometryオブジェクトを作成
        from engine.core.geometry import Geometry
        self.test_geometry = Geometry.from_lines([np.zeros((10, 3), dtype=np.float32)])
        
        # シリアライズ可能な関数を使用
        def simple_effect(geom):
            return Geometry.from_lines([geom.coords * 2])
        
        def simple_shape():
            return Geometry.from_lines([np.random.rand(10, 3).astype(np.float32)])
        
        self.effect_target = BaseBenchmarkTarget(name="effects.target1", execute_func=simple_effect)
        self.shape_target = BaseBenchmarkTarget(name="shapes.polygon", execute_func=simple_shape)

        self.runner.plugin_manager = MagicMock()
        self.runner.plugin_manager.get_all_targets.return_value = {
            "effects": [self.effect_target],
            "shapes": [self.shape_target],
        }
        self.runner.plugin_manager.get_all_plugins.return_value = [
            MockPlugin("effects", [self.effect_target]),
            MockPlugin("shapes", [self.shape_target]),
        ]

    def test_benchmark_target_effect(self):
        """Typed BenchmarkResult が返り、平均値が計算される"""
        # executor をモックして測定時間を差し込む
        def fake_execute_measurements(target, result):
            result.timing_data.measurement_times.extend([0.1, 0.2])
        with patch.object(self.runner.executor, 'execute_benchmark_measurements', side_effect=fake_execute_measurements):
            res = self.runner.benchmark_target(self.effect_target)
            self.assertIsInstance(res, BenchmarkResult)
            self.assertTrue(res.success)
            self.assertAlmostEqual(res.timing_data.average_time, np.mean([0.1, 0.2]))

    def test_benchmark_target_shape(self):
        """形状ターゲットでも平均が計算される"""
        def fake_execute_measurements(target, result):
            result.timing_data.measurement_times.extend([0.3, 0.4])
        with patch.object(self.runner, '_is_shape_target', return_value=True), \
             patch.object(self.runner.executor, 'execute_benchmark_measurements', side_effect=fake_execute_measurements):
            res = self.runner.benchmark_target(self.shape_target)
            self.assertIsInstance(res, BenchmarkResult)
            self.assertTrue(res.success)
            self.assertAlmostEqual(res.timing_data.average_time, np.mean([0.3, 0.4]))

    def test_is_shape_target(self):
        """ターゲットが形状生成かどうかの判定テスト"""
        # metadataアトリビュートを直接設定
        shape_with_meta = BaseBenchmarkTarget("s", lambda: None)
        shape_with_meta.metadata = {"shape_type": "polygon"}
        self.assertTrue(self.runner._is_shape_target(shape_with_meta))

    @patch("benchmarks.core.runner.UnifiedBenchmarkRunner.benchmark_target")
    def test_run_all_benchmarks_sequential(self, mock_benchmark_target):
        """全ベンチマークの順次実行テスト"""
        self.runner.config.parallel = False
        mock_benchmark_target.return_value = create_typed_result("mock", 0.1)
        results = self.runner.run_all_benchmarks()
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_benchmark_target.call_count, 2)

    def test_run_all_benchmarks_parallel(self):
        """全ベンチマークの並列実行テスト（実際の並列実行）"""
        self.runner.config.parallel = True
        self.runner.config.max_workers = 1  # テスト環境では1ワーカーに制限
        
        # 実際に並列実行を試みる
        results = self.runner.run_all_benchmarks()
        
        # 結果の確認（実行成功/失敗に関わらず結果が返されることを確認）
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)

if __name__ == "__main__":
    unittest.main()
