#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シェイプモジュールベンチマークスイート

このスクリプトは、shapes/ディレクトリ内のすべてのシェイプモジュールをベンチマークし、
実行時間の測定、キャッシュ使用状況の確認、失敗の追跡を行います。
結果はタイムスタンプ付きで保存され、履歴比較が可能です。
"""

import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Type aliases
Vertices = NDArray[np.float32]  # Single array of shape (N, 3)
VerticesList = List[NDArray[np.float32]]  # List of vertex arrays
BenchmarkResult = Dict[str, Any]
TimingData = Dict[str, List[float]]
ShapeFunction = Callable[..., VerticesList]


class ShapeBenchmark:
    """シェイプモジュール用ベンチマークシステム"""

    def __init__(self, output_dir: str = "benchmark_results", warmup_runs: int = 5, benchmark_runs: int = 20) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.shapes_dir = self.output_dir / "shapes"
        self.shapes_dir.mkdir(exist_ok=True)

        # Benchmark parameters
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

        # Initialize test parameters lazily
        self._test_params: Optional[Dict[str, List[Dict[str, Any]]]] = None

    @property
    def test_params(self) -> Dict[str, List[Dict[str, Any]]]:
        """テストパラメータの遅延初期化"""
        if self._test_params is None:
            self._test_params = {
                # Common shape parameters with different complexity levels
                "polygon": [
                    {"n_sides": 3},  # Triangle
                    {"n_sides": 6},  # Hexagon
                    {"n_sides": 20},  # High-poly
                ],
                "sphere": [
                    {"rows": 10, "cols": 10},  # Low res
                    {"rows": 20, "cols": 20},  # Medium res
                    {"rows": 50, "cols": 50},  # High res
                ],
                "torus": [
                    {"major_segments": 10, "minor_segments": 10},  # Low res
                    {"major_segments": 20, "minor_segments": 20},  # Medium res
                    {"major_segments": 40, "minor_segments": 40},  # High res
                ],
                "grid": [
                    {"rows": 5, "cols": 5},  # Small grid
                    {"rows": 10, "cols": 10},  # Medium grid
                    {"rows": 20, "cols": 20},  # Large grid
                ],
                "lissajous": [
                    {"a": 3, "b": 2, "samples": 100},  # Simple pattern
                    {"a": 5, "b": 4, "samples": 200},  # Medium pattern
                    {"a": 7, "b": 6, "samples": 500},  # Complex pattern
                ],
                "attractor": [
                    {"attractor_type": "lorenz", "points": 1000},  # Low complexity
                    {"attractor_type": "rossler", "points": 5000},  # Medium complexity
                    {"attractor_type": "aizawa", "points": 10000},  # High complexity
                ],
                "capsule": [
                    {"radius": 0.5, "height": 1.0, "segments": 16},  # Low res
                    {"radius": 0.5, "height": 1.0, "segments": 32},  # Medium res
                    {"radius": 0.5, "height": 1.0, "segments": 64},  # High res
                ],
                "asemic_glyph": [
                    {"region": (-0.5, -0.5, 0.5, 0.5), "random_seed": 42.0},  # Default size
                    {"region": (-1.0, -1.0, 1.0, 1.0), "random_seed": 123.0},  # Larger size
                    {"region": (-0.25, -0.25, 0.25, 0.25), "random_seed": 456.0},  # Smaller size
                ],
                "cone": [
                    {"segments": 8},   # Low res
                    {"segments": 32},  # Medium res (default)
                    {"segments": 128}, # High res
                ],
                "cylinder": [
                    {"segments": 8},   # Low res
                    {"segments": 32},  # Medium res (default)
                    {"segments": 128}, # High res
                ],
                "polyhedron": [
                    {"polygon_type": "tetrahedron"},  # Low complexity (6 edges)
                    {"polygon_type": "hexahedron"},   # Medium complexity (12 edges)
                    {"polygon_type": "icosahedron"},  # High complexity (30 edges)
                ],
                "text": [
                    {"text": "A"},           # Low complexity
                    {"text": "HELLO"},       # Medium complexity
                    {"text": "HELLO WORLD"}, # High complexity
                ],
                # Default parameters for shapes without specific test params
                "default": [
                    {},  # Default parameters
                    {"scale": (2, 2, 2)},  # Scaled version
                    {"rotate": (0.5, 0.5, 0.5)},  # Rotated version
                ],
            }
        return self._test_params

    @staticmethod
    def get_shape_modules() -> List[str]:
        """ベンチマーク対象のシェイプモジュールリストを取得"""
        shapes_path = Path("shapes")
        excluded_files = {"__init__.py", "base.py", "factory.py"}

        return sorted(
            [
                file.stem
                for file in shapes_path.glob("*.py")
                if not file.name.startswith("__") and file.name not in excluded_files
            ]
        )

    @staticmethod
    def check_cache_usage(module_name: str) -> Dict[str, bool]:
        """モジュール内の関数がlru_cacheデコレータを使用しているかチェック"""
        cache_info: Dict[str, bool] = {}
        excluded_names = {"annotations", "np", "lru_cache", "Any", "BaseShape"}

        try:
            # Import module directly to avoid circular imports
            spec = importlib.util.spec_from_file_location(
                f"shapes.{module_name}", 
                Path("shapes") / f"{module_name}.py"
            )
            if spec is None or spec.loader is None:
                return cache_info
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, obj in inspect.getmembers(module):
                if name.startswith("__") or name in excluded_names:
                    continue

                # Check if it's a function with lru_cache
                has_cache = hasattr(obj, "__wrapped__") and hasattr(obj, "cache_info")

                if has_cache or inspect.isfunction(obj):
                    cache_info[name] = has_cache

        except Exception:
            pass

        return cache_info

    def benchmark_shape(self, module_name: str) -> Dict[str, Any]:
        """単一のシェイプモジュールをベンチマーク"""
        results = {
            "module": module_name,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": None,
            "cached_functions": {},
            "timings": {},
            "average_times": {},
            "vertex_counts": {},
        }

        try:
            # Import module directly to avoid circular imports
            spec = importlib.util.spec_from_file_location(
                f"shapes.{module_name}", 
                Path("shapes") / f"{module_name}.py"
            )
            if spec is None or spec.loader is None:
                results["error"] = f"Could not load module spec for {module_name}"
                return results
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check cache usage
            results["cached_functions"] = self.check_cache_usage(module_name)

            # Find and instantiate shape class
            shape_func = self._get_shape_function(module, module_name)
            if shape_func is None:
                results["error"] = f"No shape class with generate method found in {module_name}"
                return results

            # Get test parameters for this shape (shape-specific or default)
            test_params_list = self.test_params.get(module_name, self.test_params["default"])

            # Benchmark for different parameter sets to measure complexity scaling
            for i, params in enumerate(test_params_list):
                param_name = f"params_{i}"
                times, vertex_count = self._benchmark_single_params(shape_func, params, param_name)

                if times is None:
                    results["error"] = f"Error during benchmark for {param_name}"
                    break

                if times:
                    results["timings"][param_name] = times
                    results["average_times"][param_name] = np.mean(times)
                    results["vertex_counts"][param_name] = vertex_count

            results["success"] = bool(results["timings"])

        except Exception as e:
            results["error"] = f"Failed to benchmark {module_name}: {str(e)}"
            results["traceback"] = traceback.format_exc()

        return results

    def run_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """すべてのシェイプモジュールのベンチマークを実行"""
        modules = self.get_shape_modules()
        all_results = {}

        print(f"Found {len(modules)} shape modules to benchmark")
        print("-" * 50)

        for module in modules:
            print(f"Benchmarking {module}...", end=" ")
            result = self.benchmark_shape(module)

            if result["success"]:
                avg_time = np.mean(list(result["average_times"].values()))
                print(f"◯ (avg: {avg_time*1000:.2f}ms)")
            else:
                print(f"× ({result['error']})")

            all_results[module] = result

        return all_results

    def save_results(self, results: Dict[str, Dict[str, Any]]) -> str:
        """タイムスタンプ付きでベンチマーク結果を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.shapes_dir / f"benchmark_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        # Also save as latest
        latest_file = self.shapes_dir / "latest.json"
        with open(latest_file, "w") as f:
            json.dump(results, f, indent=2)

        return str(filename)

    def visualize_results(self, results: BenchmarkResult, save_path: Optional[str] = None) -> None:
        """ベンチマーク結果の横棒グラフを作成"""
        # Extract visualization data
        viz_data = self._extract_visualization_data(results)

        # Create charts
        fig, _ = self._create_benchmark_charts(viz_data)

        # Save chart
        self._save_chart(fig, save_path)

    def compare_historical(self, num_recent: int = 5) -> None:
        """最近のベンチマーク結果を比較して、時間経過による改善を表示"""
        # Get all benchmark files
        benchmark_files = sorted(self.shapes_dir.glob("benchmark_*.json"))

        if len(benchmark_files) < 2:
            print("Not enough historical data for comparison")
            return

        # Load recent results
        recent_files = benchmark_files[-num_recent:]
        historical_data: Dict[str, Dict[str, Any]] = {}

        for file in recent_files:
            with open(file, "r") as f:
                data = json.load(f)
                timestamp = file.stem.replace("benchmark_", "")
                historical_data[timestamp] = data

        # Prepare comparison data
        modules_set: Set[str] = set()
        for data in historical_data.values():
            modules_set.update(data.keys())

        modules: List[str] = sorted(modules_set)

        # Create comparison chart
        _, ax = plt.subplots(figsize=(14, max(8, len(modules) * 0.5)))

        timestamps = sorted(historical_data.keys())
        x = np.arange(len(timestamps))
        width = 0.8 / len(modules)

        for i, module in enumerate(modules):
            times: List[Union[float, None]] = []
            for ts in timestamps:
                if module in historical_data[ts] and historical_data[ts][module]["success"]:
                    # Use average of all parameter sets
                    avg_times = historical_data[ts][module]["average_times"]
                    if avg_times:
                        times.append(np.mean(list(avg_times.values())) * 1000)  # type: ignore
                    else:
                        times.append(None)
                else:
                    times.append(None)

            # Plot only if we have data
            if any(t is not None for t in times):
                positions = x + i * width - 0.4 + width / 2
                valid_times = [t if t is not None else 0 for t in times]
                ax.bar(positions, valid_times, width, label=module, alpha=0.8)

        ax.set_xlabel("Benchmark Run")
        ax.set_ylabel("Average Time (ms)")
        ax.set_title(f"Historical Performance Comparison (Last {num_recent} Runs)")
        ax.set_xticks(x)
        ax.set_xticklabels([ts[:8] + "\\n" + ts[9:].replace("_", ":") for ts in timestamps], rotation=45, ha="right")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        comparison_file = self.shapes_dir / "historical_comparison.png"
        plt.savefig(comparison_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Historical comparison saved to {comparison_file}")

    def _get_shape_function(self, module: Any, module_name: str) -> Optional[ShapeFunction]:
        """シェイプクラスを検索してインスタンス化"""
        # Try to find shape class by name matching
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and name.lower() == module_name.lower():
                try:
                    instance = obj()
                    if hasattr(instance, "generate") or hasattr(instance, "__call__"):
                        # Return instance directly for cache access
                        return instance
                except Exception:
                    pass

        # Try any class with generate method or __call__
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and (hasattr(obj, "generate") or hasattr(obj, "__call__")):
                try:
                    instance = obj()
                    # Return instance directly for cache access
                    return instance
                except Exception:
                    pass

        return None

    def _benchmark_single_params(
        self, shape_func: ShapeFunction, params: Dict[str, Any], param_name: str
    ) -> Tuple[Optional[List[float]], Optional[int]]:
        """単一パラメータセットでベンチマークを実行"""
        # shape_func is now the instance directly
        shape_instance = shape_func
        
        # Warmup
        vertex_count = None
        for _ in range(self.warmup_runs):
            try:
                # Call the instance directly or use generate method
                if hasattr(shape_instance, "__call__"):
                    vertices_list = shape_instance(**params)
                else:
                    vertices_list = shape_instance.generate(**params)
                if vertex_count is None:
                    vertex_count = sum(len(v) for v in vertices_list)
            except Exception:
                return None, None

        # Clear cache after warmup to ensure benchmark measures actual computation
        if hasattr(shape_instance, 'clear_cache'):
            shape_instance.clear_cache()

        # Benchmark
        times: List[float] = []
        for _ in range(self.benchmark_runs):
            try:
                start_time = time.perf_counter()
                # Call the instance directly or use generate method
                if hasattr(shape_instance, "__call__"):
                    shape_instance(**params)
                else:
                    shape_instance.generate(**params)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                
                # Clear cache after each run to measure actual computation time
                if hasattr(shape_instance, 'clear_cache'):
                    shape_instance.clear_cache()
            except Exception:
                return None, None

        return times, vertex_count

    def _extract_visualization_data(self, results: BenchmarkResult) -> Dict[str, List[Any]]:
        """可視化用のデータを抽出"""
        modules: List[str] = []
        param_0_times: List[float] = []
        param_1_times: List[float] = []
        param_2_times: List[float] = []
        has_cache: List[str] = []
        success_status: List[str] = []

        for module, data in sorted(results.items()):
            modules.append(module)

            if data["success"]:
                # Get times for each parameter set
                times = []
                for i in range(3):
                    param_key = f"params_{i}"
                    if param_key in data["average_times"]:
                        times.append(data["average_times"][param_key] * 1000)
                    else:
                        times.append(0)
                
                param_0_times.append(times[0])
                param_1_times.append(times[1])
                param_2_times.append(times[2])
                success_status.append("◯")
            else:
                param_0_times.append(0)
                param_1_times.append(0)
                param_2_times.append(0)
                success_status.append("×")

            cached_funcs = data.get("cached_functions", {})
            has_cache.append("◆" if any(cached_funcs.values()) else "◇")

        return {
            "modules": modules,
            "param_0_times": param_0_times,
            "param_1_times": param_1_times,
            "param_2_times": param_2_times,
            "has_cache": has_cache,
            "success_status": success_status,
        }

    def _create_benchmark_charts(self, viz_data: Dict[str, List[Any]]) -> Tuple[Figure, List[Axes]]:
        """ベンチマークチャートを作成"""
        modules = viz_data["modules"]
        fig, axes = plt.subplots(1, 3, figsize=(18, max(8, len(modules) * 0.4)))
        y_pos = np.arange(len(modules))

        # Chart configurations
        chart_configs = [
            ("param_0_times", "lightblue", "Parameter Set 1 (Simple)"),
            ("param_1_times", "lightgreen", "Parameter Set 2 (Medium)"),
            ("param_2_times", "lightcoral", "Parameter Set 3 (Complex)"),
        ]

        # Create charts
        for ax, (data_key, color, title) in zip(axes, chart_configs):
            bars = ax.barh(y_pos, viz_data[data_key], color=color)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(
                [
                    f"{m} {s} {c}"
                    for m, s, c in zip(viz_data["modules"], viz_data["success_status"], viz_data["has_cache"])
                ]
            )
            ax.set_xlabel("Time (ms)")
            ax.set_title(title)
            ax.grid(axis="x", alpha=0.3)

            # Add value labels
            for bar in bars:
                width = bar.get_width()
                if width > 0:
                    ax.text(
                        width, bar.get_y() + bar.get_height() / 2, f"{width:.1f}", ha="left", va="center", fontsize=8
                    )

        plt.suptitle(f'Shape Module Benchmarks - {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        plt.tight_layout()
        # Adjust subplot positions to make room for the legend at the bottom
        plt.subplots_adjust(bottom=0.125)
        fig.text(0.5, 0.05, "◯ = Success, × = Failed, ◆ = Uses cache, ◇ = No cache", ha="center", fontsize=10)

        return fig, axes

    def _save_chart(self, fig: Figure, save_path: Optional[str] = None) -> None:
        """チャートを保存"""
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = self.shapes_dir / f"benchmark_chart_{timestamp}.png"
            fig.savefig(save_file, dpi=150, bbox_inches="tight")

            # Also save as latest
            latest_chart = self.shapes_dir / "latest_chart.png"
            fig.savefig(latest_chart, dpi=150, bbox_inches="tight")

        plt.close(fig)


def main() -> None:
    benchmark = ShapeBenchmark()

    print("Starting Shape Module Benchmark Suite")
    print("=" * 50)

    # Run benchmarks
    results = benchmark.run_benchmarks()

    # Save results
    saved_file = benchmark.save_results(results)
    print(f"\\nResults saved to: {saved_file}")

    # Visualize results
    benchmark.visualize_results(results)
    print(f"Visualization saved to: {benchmark.shapes_dir / 'latest_chart.png'}")

    # Compare historical data
    print("\\nGenerating historical comparison...")
    benchmark.compare_historical()

    # Summary
    print("\\nSummary:")
    print("-" * 30)
    successful = sum(1 for r in results.values() if r["success"])
    failed = len(results) - successful
    print(f"Total modules: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # Show vertex counts
    print("\\nVertex counts by shape:")
    for module, data in sorted(results.items()):
        if data["success"] and data["vertex_counts"]:
            counts = ", ".join(f"{k}: {v}" for k, v in data["vertex_counts"].items())
            print(f"  - {module}: {counts}")

    if failed > 0:
        print("\\nFailed modules:")
        for module, data in results.items():
            if not data["success"]:
                print(f"  - {module}: {data['error']}")


if __name__ == "__main__":
    main()