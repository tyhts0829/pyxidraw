#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク実行モジュール

エフェクトモジュールのパフォーマンス測定を実行するクラス。
"""

import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from effect_module_discovery import EffectModuleDiscovery
from test_data_generator import TestDataGenerator

# Type aliases
VerticesList = List[NDArray[np.float32]]
EffectFunction = Callable[[VerticesList], VerticesList]
BenchmarkResult = Dict[str, Any]


class BenchmarkRunner:
    """ベンチマーク実行を管理するクラス"""

    def __init__(self, warmup_runs: int = 5, benchmark_runs: int = 20):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.data_generator = TestDataGenerator()
        self.module_discovery = EffectModuleDiscovery()

    def benchmark_single_size(self, effect_func: EffectFunction, 
                            test_shapes: VerticesList, 
                            size_name: str) -> Optional[List[float]]:
        """単一サイズのデータでベンチマークを実行"""
        # Warmup
        for _ in range(self.warmup_runs):
            try:
                effect_func(test_shapes)
            except Exception:
                return None
        
        # Benchmark
        times: List[float] = []
        for _ in range(self.benchmark_runs):
            try:
                start_time = time.perf_counter()
                effect_func(test_shapes)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception:
                return None
        
        return times

    def benchmark_effect(self, module_name: str) -> BenchmarkResult:
        """単一のエフェクトモジュールをベンチマーク"""
        results = {
            "module": module_name,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": None,
            "njit_functions": {},
            "timings": {},
            "average_times": {},
        }

        try:
            # Check njit usage
            results["njit_functions"] = self.module_discovery.check_njit_usage(module_name)

            # Find and instantiate effect class
            effect_func = self.module_discovery.get_effect_function(module_name)
            if effect_func is None:
                results["error"] = f"No effect class with apply method found in {module_name}"
                return results

            # Get test shapes
            test_shapes = self.data_generator.generate_test_shapes()

            # Benchmark for different data sizes
            for size_name, shapes in test_shapes.items():
                times = self.benchmark_single_size(effect_func, shapes, size_name)
                
                if times is None:
                    results["error"] = f"Error during benchmark for {size_name} size"
                    break
                    
                if times:
                    results["timings"][size_name] = times
                    results["average_times"][size_name] = np.mean(times)

            results["success"] = bool(results["timings"])

        except Exception as e:
            results["error"] = f"Failed to benchmark {module_name}: {str(e)}"
            results["traceback"] = traceback.format_exc()

        return results

    def run_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """すべてのエフェクトモジュールのベンチマークを実行"""
        modules = self.module_discovery.get_effect_modules()
        all_results = {}

        print(f"Found {len(modules)} effect modules to benchmark")
        print("-" * 50)

        for module in modules:
            print(f"Benchmarking {module}...", end=" ")
            result = self.benchmark_effect(module)

            if result["success"]:
                avg_time = np.mean(list(result["average_times"].values()))
                avg_fps = 1.0 / avg_time if avg_time > 0 else float('inf')
                print(f"◯ (avg: {avg_fps:.1f} fps)")
            else:
                print(f"× ({result['error']})")

            all_results[module] = result

        return all_results