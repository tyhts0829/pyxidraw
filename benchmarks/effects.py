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
EffectFunction = Callable[[VerticesList], VerticesList]


class EffectBenchmark:
    """エフェクトモジュール用ベンチマークシステム"""

    def __init__(self, output_dir: str = "benchmark_results", warmup_runs: int = 5, benchmark_runs: int = 20) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.effects_dir = self.output_dir / "effects"
        self.effects_dir.mkdir(exist_ok=True)

        # Benchmark parameters
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

        # Initialize test shapes lazily
        self._test_shapes: Optional[Dict[str, VerticesList]] = None

    @property
    def test_shapes(self) -> Dict[str, VerticesList]:
        """テストデータの遅延初期化"""
        if self._test_shapes is None:
            self._test_shapes = {
                "small": [self._create_rectangle(1, 1)],
                "medium": [self._create_polygon(20)],
                "large": self._create_large_shape(),
            }
        return self._test_shapes

    @staticmethod
    def _create_rectangle(width: float, height: float) -> Vertices:
        """シンプルな長方形を作成（3D座標）"""
        hw, hh = width / 2, height / 2
        return np.array(
            [[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0], [-hw, -hh, 0.0]], dtype=np.float32
        )

    @staticmethod
    def _create_polygon(n: int) -> Vertices:
        """n辺の正多角形を作成（3D座標）"""
        angles = np.linspace(0, 2 * np.pi, n + 1)
        return np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)]).astype(np.float32)

    @staticmethod
    def _create_circle(radius: float, segments: int = 64) -> Vertices:
        """円を作成（3D座標）"""
        angles = np.linspace(0, 2 * np.pi, segments + 1)
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)]).astype(
            np.float32
        )

    def _create_large_shape(self) -> VerticesList:
        """ベンチマーク用の大きく複雑な形状を作成（複数の3D配列のリスト）"""
        return [self._create_circle(1.0 + i * 0.1) for i in range(10)]

    @staticmethod
    def get_effect_modules() -> List[str]:
        """ベンチマーク対象のエフェクトモジュールリストを取得"""
        effects_path = Path("effects")
        excluded_files = {"__init__.py", "base.py", "pipeline.py"}

        return sorted(
            [
                file.stem
                for file in effects_path.glob("*.py")
                if not file.name.startswith("__") and file.name not in excluded_files
            ]
        )

    @staticmethod
    def check_njit_usage(module_name: str) -> Dict[str, bool]:
        """モジュール内の関数がnjitデコレータを使用しているかチェック"""
        njit_info: Dict[str, bool] = {}
        excluded_names = {"annotations", "np", "njit", "Any", "BaseEffect"}

        try:
            module = importlib.import_module(f"effects.{module_name}")

            for name, obj in inspect.getmembers(module):
                if name.startswith("__") or name in excluded_names:
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

            # Find and instantiate effect class
            effect_func = self._get_effect_function(module, module_name)
            if effect_func is None:
                results["error"] = f"No effect class with apply method found in {module_name}"
                return results

            # Benchmark for different data sizes
            for size_name, test_shapes in self.test_shapes.items():
                times = self._benchmark_single_size(effect_func, test_shapes, size_name)

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

    def visualize_results(self, results: BenchmarkResult, save_path: Optional[str] = None) -> None:
        """ベンチマーク結果の横棒グラフを作成"""
        # Extract visualization data
        viz_data = self._extract_visualization_data(results)

        # Create charts
        fig, axes = self._create_benchmark_charts(viz_data)

        # Save chart
        self._save_chart(fig, save_path)

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

    def _get_effect_function(self, module: Any, module_name: str) -> Optional[EffectFunction]:
        """エフェクトクラスを検索してインスタンス化"""
        # Try to find effect class by name matching
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and name.lower() == module_name.lower():
                try:
                    instance = obj()
                    if hasattr(instance, "apply"):
                        return lambda shapes: instance.apply(shapes)
                except Exception:
                    pass

        # Try any class with apply method
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and hasattr(obj, "apply"):
                try:
                    instance = obj()
                    return lambda shapes: instance.apply(shapes)
                except Exception:
                    pass

        return None

    def _benchmark_single_size(
        self, effect_func: EffectFunction, test_shapes: VerticesList, size_name: str
    ) -> Optional[List[float]]:
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

    def _extract_visualization_data(self, results: BenchmarkResult) -> Dict[str, List[Any]]:
        """可視化用のデータを抽出"""
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

            njit_funcs = data.get("njit_functions", {})
            has_njit.append("◆" if any(njit_funcs.values()) else "◇")

        return {
            "modules": modules,
            "small_times": small_times,
            "medium_times": medium_times,
            "large_times": large_times,
            "has_njit": has_njit,
            "success_status": success_status,
        }

    def _create_benchmark_charts(self, viz_data: Dict[str, List[Any]]) -> Tuple[Figure, List[Axes]]:
        """ベンチマークチャートを作成"""
        modules = viz_data["modules"]
        fig, axes = plt.subplots(2, 3, figsize=(18, max(12, len(modules) * 0.6)))
        y_pos = np.arange(len(modules))

        # Chart configurations
        chart_configs = [
            ("small_times", "lightblue", "Small Data Size"),
            ("medium_times", "lightgreen", "Medium Data Size"),
            ("large_times", "lightcoral", "Large Data Size"),
        ]

        # Create charts with linear and log scale
        for col, (data_key, color, title) in enumerate(chart_configs):
            times = viz_data[data_key]

            # Linear scale chart (top row)
            ax_linear = axes[0, col]
            bars = ax_linear.barh(y_pos, times, color=color)
            ax_linear.set_yticks(y_pos)
            ax_linear.set_yticklabels(
                [
                    f"{m} {s} {n}"
                    for m, s, n in zip(viz_data["modules"], viz_data["success_status"], viz_data["has_njit"])
                ]
            )
            ax_linear.set_xlabel("Time (ms)")
            ax_linear.set_title(f"{title} - Linear Scale")
            ax_linear.grid(axis="x", alpha=0.3)

            # Add value labels for linear scale
            for bar in bars:
                width = bar.get_width()
                if width > 0:
                    label = f"{width:.3f}" if width < 1000 else f"{width/1000:.3f}s"
                    ax_linear.text(
                        min(width, ax_linear.get_xlim()[1] * 0.95),
                        bar.get_y() + bar.get_height() / 2,
                        label,
                        ha="left" if width < ax_linear.get_xlim()[1] * 0.9 else "right",
                        va="center",
                        fontsize=8,
                    )

            # Log scale chart (bottom row)
            ax_log = axes[1, col]
            # Replace zeros with small value for log scale
            log_times = [max(t, 0.01) if t > 0 else 0.01 for t in times]
            bars_log = ax_log.barh(y_pos, log_times, color=color)
            ax_log.set_xscale("log")
            ax_log.set_yticks(y_pos)
            ax_log.set_yticklabels(
                [
                    f"{m} {s} {n}"
                    for m, s, n in zip(viz_data["modules"], viz_data["success_status"], viz_data["has_njit"])
                ]
            )
            ax_log.set_xlabel("Time (ms) - Log Scale")
            ax_log.set_title(f"{title} - Log Scale")
            ax_log.grid(axis="x", alpha=0.3, which="both")

            # Add value labels for log scale
            for bar, orig_time in zip(bars_log, times):
                if orig_time > 0:
                    label = f"{orig_time:.3f}" if orig_time < 1000 else f"{orig_time/1000:.3f}s"
                    ax_log.text(
                        bar.get_width() * 1.1,
                        bar.get_y() + bar.get_height() / 2,
                        label,
                        ha="left",
                        va="center",
                        fontsize=8,
                    )

        plt.suptitle(f'Effect Module Benchmarks - {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])  # type: ignore
        fig.text(0.5, 0.01, "◯ = Success, × = Failed, ◆ = Uses njit, ◇ = No njit", ha="center", fontsize=10)

        return fig, axes

    def _save_chart(self, fig: Figure, save_path: Optional[str] = None) -> None:
        """チャートを保存"""
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


def main() -> None:
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
