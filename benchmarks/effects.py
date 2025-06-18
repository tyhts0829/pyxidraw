#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
エフェクトモジュールベンチマークスイート

このスクリプトは、effects/ディレクトリ内のすべてのエフェクトモジュールをベンチマークし、
実行時間の測定、njitデコレータの使用状況の確認、失敗の追跡を行います。
結果はタイムスタンプ付きで保存され、履歴比較が可能です。
"""

import importlib
import inspect
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define types for 3D coordinate data
Vertices = np.ndarray  # Single array of shape (N, 3)
VerticesList = List[np.ndarray]  # List of vertex arrays


class EffectBenchmark:
    """エフェクトモジュール用ベンチマークシステム"""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.effects_dir = self.output_dir / "effects"
        self.effects_dir.mkdir(exist_ok=True)

        # Test data sizes - all as list[np.ndarray(N, 3)]
        self.small_shape = [self._create_rectangle(1, 1)]  # Small: simple rectangle
        self.medium_shape = [self._create_polygon(20)]  # Medium: 20-sided polygon
        self.large_shape = self._create_large_shape()  # Large: complex shape (multiple arrays)

        self.test_shapes = {"small": self.small_shape, "medium": self.medium_shape, "large": self.large_shape}

        # Benchmark parameters
        self.warmup_runs = 5
        self.benchmark_runs = 20

    def _create_rectangle(self, width: float, height: float) -> Vertices:
        """シンプルな長方形を作成（3D座標）"""
        hw, hh = width / 2, height / 2
        # Add z=0 coordinate for 3D
        return np.array(
            [[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0], [-hw, -hh, 0.0]], dtype=np.float32
        )

    def _create_polygon(self, n: int) -> Vertices:
        """n辺の正多角形を作成（3D座標）"""
        angles = np.linspace(0, 2 * np.pi, n + 1)
        x = np.cos(angles)
        y = np.sin(angles)
        z = np.zeros_like(x)
        return np.column_stack([x, y, z]).astype(np.float32)

    def _create_circle(self, radius: float, segments: int = 64) -> Vertices:
        """円を作成（3D座標）"""
        angles = np.linspace(0, 2 * np.pi, segments + 1)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z = np.zeros_like(x)
        return np.column_stack([x, y, z]).astype(np.float32)

    def _create_large_shape(self) -> VerticesList:
        """ベンチマーク用の大きく複雑な形状を作成（複数の3D配列のリスト）"""
        shapes = []
        for i in range(10):
            shapes.append(self._create_circle(1.0 + i * 0.1))

        return shapes

    def get_effect_modules(self) -> List[str]:
        """ベンチマーク対象のエフェクトモジュールリストを取得"""
        effects_path = Path("effects")
        modules = []

        for file in effects_path.glob("*.py"):
            if file.name.startswith("__") or file.name in ["base.py", "pipeline.py"]:
                continue
            modules.append(file.stem)

        return sorted(modules)

    def check_njit_usage(self, module_name: str) -> Dict[str, bool]:
        """モジュール内の関数がnjitデコレータを使用しているかチェック"""
        njit_info = {}

        try:
            module = importlib.import_module(f"effects.{module_name}")

            for name, obj in inspect.getmembers(module):
                # Check all members (including private functions starting with _)
                # Skip imports and special attributes
                if name.startswith("__") or name in ["annotations", "np", "njit", "Any", "BaseEffect"]:
                    continue
                    
                # Check if it's a numba compiled function (CPUDispatcher)
                is_njit = "numba.core.registry.CPUDispatcher" in str(type(obj))
                
                if is_njit or inspect.isfunction(obj):
                    njit_info[name] = is_njit

        except Exception:
            pass

        return njit_info

    def benchmark_effect(self, module_name: str) -> Dict[str, Any]:
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
            # Import module
            module = importlib.import_module(f"effects.{module_name}")

            # Check njit usage
            results["njit_functions"] = self.check_njit_usage(module_name)

            # Find the effect class (usually capitalized version of module name)
            effect_class_name = module_name.capitalize()
            effect_class = None

            # Try to find the effect class
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name.lower() == module_name.lower():
                    effect_class = obj
                    break

            if not effect_class:
                # Try alternative naming patterns
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and hasattr(obj, "apply"):
                        effect_class = obj
                        break

            if not effect_class:
                results["error"] = f"No effect class found in {module_name}"
                return results

            # Create instance and get apply method
            try:
                effect_instance = effect_class()
                effect_func = lambda shapes: effect_instance.apply(shapes)
            except:
                # If instantiation fails, try to use the class directly
                results["error"] = f"Failed to instantiate effect class in {module_name}"
                return results

            # Benchmark for different data sizes
            for size_name, test_shapes in self.test_shapes.items():
                times = []

                # Warmup
                for _ in range(self.warmup_runs):
                    try:
                        _ = effect_func(test_shapes)
                    except:
                        pass

                # Actual benchmark
                for _ in range(self.benchmark_runs):
                    try:
                        start_time = time.perf_counter()
                        _ = effect_func(test_shapes)
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
                    except Exception as e:
                        results["error"] = f"Error during benchmark: {str(e)}"
                        break

                if times:
                    results["timings"][size_name] = times
                    results["average_times"][size_name] = np.mean(times)

            results["success"] = bool(results["timings"])

        except Exception as e:
            results["error"] = f"Failed to benchmark {module_name}: {str(e)}"
            results["traceback"] = traceback.format_exc()

        return results

    def run_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """すべてのエフェクトモジュールのベンチマークを実行"""
        modules = self.get_effect_modules()
        all_results = {}

        print(f"Found {len(modules)} effect modules to benchmark")
        print("-" * 50)

        for module in modules:
            print(f"Benchmarking {module}...", end=" ")
            result = self.benchmark_effect(module)

            if result["success"]:
                avg_time = np.mean(list(result["average_times"].values()))
                print(f"◯ (avg: {avg_time*1000:.2f}ms)")
            else:
                print(f"× ({result['error']})")

            all_results[module] = result

        return all_results

    def save_results(self, results: Dict[str, Dict]) -> str:
        """タイムスタンプ付きでベンチマーク結果を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.effects_dir / f"benchmark_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        # Also save as latest
        latest_file = self.effects_dir / "latest.json"
        with open(latest_file, "w") as f:
            json.dump(results, f, indent=2)

        return str(filename)

    def visualize_results(self, results: Dict[str, Dict], save_path: Optional[str] = None) -> None:
        """ベンチマーク結果の横棒グラフを作成"""
        # Prepare data for visualization
        modules: List[str] = []
        small_times: List[float] = []
        medium_times: List[float] = []
        large_times: List[float] = []
        has_njit: List[str] = []
        success_status: List[str] = []

        for module, data in sorted(results.items()):
            modules.append(module)

            if data["success"]:
                small_times.append(data["average_times"].get("small", 0) * 1000)
                medium_times.append(data["average_times"].get("medium", 0) * 1000)
                large_times.append(data["average_times"].get("large", 0) * 1000)
                success_status.append("◯")
            else:
                small_times.append(0)
                medium_times.append(0)
                large_times.append(0)
                success_status.append("×")

            # Check if any function uses njit
            njit_funcs = data.get("njit_functions", {})
            has_njit.append("◆" if any(njit_funcs.values()) else "◇")

        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, max(8, len(modules) * 0.4)))

        y_pos = np.arange(len(modules))

        # Small data size chart
        bars1 = ax1.barh(y_pos, small_times, color="lightblue")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{m} {s} {n}" for m, s, n in zip(modules, success_status, has_njit)])
        ax1.set_xlabel("Time (ms)")
        ax1.set_title("Small Data Size")
        ax1.grid(axis="x", alpha=0.3)

        # Medium data size chart
        bars2 = ax2.barh(y_pos, medium_times, color="lightgreen")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"{m} {s} {n}" for m, s, n in zip(modules, success_status, has_njit)])
        ax2.set_xlabel("Time (ms)")
        ax2.set_title("Medium Data Size")
        ax2.grid(axis="x", alpha=0.3)

        # Large data size chart
        bars3 = ax3.barh(y_pos, large_times, color="lightcoral")
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f"{m} {s} {n}" for m, s, n in zip(modules, success_status, has_njit)])
        ax3.set_xlabel("Time (ms)")
        ax3.set_title("Large Data Size")
        ax3.grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                width = bar.get_width()
                if width > 0:
                    bar.axes.text(
                        width, bar.get_y() + bar.get_height() / 2, f"{width:.1f}", ha="left", va="center", fontsize=8
                    )

        plt.suptitle(f'Effect Module Benchmarks - {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        plt.tight_layout()

        # Add legend
        fig.text(0.5, 0.02, "◯ = Success, × = Failed, ◆ = Uses njit, ◇ = No njit", ha="center", fontsize=10)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = self.effects_dir / f"benchmark_chart_{timestamp}.png"
            plt.savefig(save_file, dpi=150, bbox_inches="tight")

            # Also save as latest
            latest_chart = self.effects_dir / "latest_chart.png"
            plt.savefig(latest_chart, dpi=150, bbox_inches="tight")

        plt.close()

    def compare_historical(self, num_recent: int = 5) -> None:
        """最近のベンチマーク結果を比較して、時間経過による改善を表示"""
        # Get all benchmark files
        benchmark_files = sorted(self.effects_dir.glob("benchmark_*.json"))

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
        fig, ax = plt.subplots(figsize=(14, max(8, len(modules) * 0.5)))

        timestamps = sorted(historical_data.keys())
        x = np.arange(len(timestamps))
        width = 0.8 / len(modules)

        for i, module in enumerate(modules):
            times: List[Optional[float]] = []
            for ts in timestamps:
                if module in historical_data[ts] and historical_data[ts][module]["success"]:
                    # Use average of all data sizes
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
        ax.set_xticklabels([ts[:8] + "\n" + ts[9:].replace("_", ":") for ts in timestamps], rotation=45, ha="right")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        comparison_file = self.effects_dir / "historical_comparison.png"
        plt.savefig(comparison_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Historical comparison saved to {comparison_file}")


def main():
    """メインのベンチマーク実行"""
    benchmark = EffectBenchmark()

    print("Starting Effects Module Benchmark Suite")
    print("=" * 50)

    # Run benchmarks
    results = benchmark.run_benchmarks()

    # Save results
    saved_file = benchmark.save_results(results)
    print(f"\nResults saved to: {saved_file}")

    # Visualize results
    benchmark.visualize_results(results)
    print(f"Visualization saved to: {benchmark.effects_dir / 'latest_chart.png'}")

    # Compare historical data
    print("\nGenerating historical comparison...")
    benchmark.compare_historical()

    # Summary
    print("\nSummary:")
    print("-" * 30)
    successful = sum(1 for r in results.values() if r["success"])
    failed = len(results) - successful
    print(f"Total modules: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed modules:")
        for module, data in results.items():
            if not data["success"]:
                print(f"  - {module}: {data['error']}")


if __name__ == "__main__":
    main()
