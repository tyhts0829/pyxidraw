#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマークプラグインモジュールのテスト
"""

import unittest
from unittest.mock import patch

from benchmarks.core.config import BenchmarkConfig
from benchmarks.plugins.effects import EffectBenchmarkPlugin
from benchmarks.plugins.shapes import ShapeBenchmarkPlugin
from benchmarks.plugins.base import ParametrizedBenchmarkTarget


class TestEffectBenchmarkPlugin(unittest.TestCase):
    """EffectBenchmarkPluginのテスト"""

    def setUp(self):
        self.config = BenchmarkConfig()
        self.plugin = EffectBenchmarkPlugin(name="effects", config=self.config)

    def test_plugin_type(self):
        """プラグインタイプのテスト"""
        self.assertEqual(self.plugin.plugin_type, "effects")

    def test_discover_targets(self):
        """ターゲット発見のテスト"""
        targets = self.plugin.discover_targets()
        self.assertGreater(len(targets), 0)
        # noiseターゲットが存在することを確認
        self.assertTrue(any(t.name.startswith("noise.") for t in targets))
        # transformターゲットが存在することを確認
        self.assertTrue(any(t.name.startswith("transform.") for t in targets))

    def test_create_benchmark_target(self):
        """ベンチマークターゲット作成のテスト"""
        target = self.plugin.create_benchmark_target("noise.low_intensity")
        self.assertIsInstance(target, ParametrizedBenchmarkTarget)
        self.assertEqual(target.name, "noise.low_intensity")
        self.assertEqual(target.parameters["intensity"], 0.1)

    def test_create_invalid_target(self):
        """無効なターゲット作成時のエラーテスト"""
        with self.assertRaises(ValueError):
            self.plugin.create_benchmark_target("invalid.target")

    def test_analyze_target_features(self):
        """ターゲット特性分析のテスト"""
        target = self.plugin.create_benchmark_target("noise.high_frequency")
        features = self.plugin.analyze_target_features(target)
        self.assertTrue(features["has_njit"]) # noiseはnjit利用と判定されるはず
        self.assertTrue(features["has_cache"])


class TestShapeBenchmarkPlugin(unittest.TestCase):
    """ShapeBenchmarkPluginのテスト"""

    def setUp(self):
        self.config = BenchmarkConfig()
        self.plugin = ShapeBenchmarkPlugin(name="shapes", config=self.config)

    def test_plugin_type(self):
        """プラグインタイプのテスト"""
        self.assertEqual(self.plugin.plugin_type, "shapes")

    def test_discover_targets(self):
        """ターゲット発見のテスト"""
        targets = self.plugin.discover_targets()
        self.assertGreater(len(targets), 0)
        # polygonターゲットが存在することを確認
        self.assertTrue(any(t.name.startswith("polygon.") for t in targets))
        # sphereターゲットが存在することを確認
        self.assertTrue(any(t.name.startswith("sphere.") for t in targets))

    def test_create_benchmark_target(self):
        """ベンチマークターゲット作成のテスト"""
        target = self.plugin.create_benchmark_target("polygon.triangle")
        self.assertIsInstance(target, ParametrizedBenchmarkTarget)
        self.assertEqual(target.name, "polygon.triangle")
        self.assertEqual(target.parameters["n_sides"], 3)

    def test_analyze_target_features(self):
        """ターゲット特性分析のテスト"""
        target = self.plugin.create_benchmark_target("sphere.high_res")
        features = self.plugin.analyze_target_features(target)
        self.assertTrue(features["has_njit"]) # 複雑な形状はnjit利用と判定されるはず


if __name__ == "__main__":
    unittest.main()
