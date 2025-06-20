#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク結果可視化モジュール

ベンチマーク結果のグラフ生成と可視化を行うクラス。
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

BenchmarkResult = Dict[str, Any]


class BenchmarkVisualizer:
    """ベンチマーク結果の可視化を行うクラス"""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.effects_dir = self.output_dir / "effects"

    def visualize_results(self, results: Dict[str, BenchmarkResult], save_path: Optional[str] = None) -> None:
        """ベンチマーク結果の横棒グラフを作成"""
        # Extract visualization data
        viz_data = self._extract_visualization_data(results)

        # Create charts
        fig, axes = self._create_benchmark_charts(viz_data)

        # Save chart
        self._save_chart(fig, save_path)

    def compare_historical(self, historical_data: Dict[str, Dict[str, BenchmarkResult]], num_recent: int = 5) -> None:
        """最近のベンチマーク結果を比較して、時間経過による改善を表示"""
        if len(historical_data) < 2:
            print("Not enough historical data for comparison")
            return

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
                        avg_time = np.mean(list(avg_times.values()))
                        fps = 1.0 / avg_time if avg_time > 0 else float("inf")
                        times.append(float(fps))
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
        ax.set_ylabel("Average FPS")
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

    def _extract_visualization_data(self, results: Dict[str, BenchmarkResult]) -> Dict[str, List[Any]]:
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
                small_time = data["average_times"].get("small", 0)
                medium_time = data["average_times"].get("medium", 0)
                large_time = data["average_times"].get("large", 0)
                small_times.append(1.0 / small_time if small_time > 0 else 0)
                medium_times.append(1.0 / medium_time if medium_time > 0 else 0)
                large_times.append(1.0 / large_time if large_time > 0 else 0)
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
        fig, axes = plt.subplots(1, 3, figsize=(18, max(8, len(modules) * 0.4)))
        y_pos = np.arange(len(modules))

        # Chart configurations
        chart_configs = [
            ("small_times", "lightblue", "Small Data Size"),
            ("medium_times", "lightgreen", "Medium Data Size"),
            ("large_times", "lightcoral", "Large Data Size"),
        ]

        # Create charts
        for ax, (data_key, color, title) in zip(axes, chart_configs):
            bars = ax.barh(y_pos, viz_data[data_key], color=color)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(
                [
                    f"{m} {s} {n}"
                    for m, s, n in zip(viz_data["modules"], viz_data["success_status"], viz_data["has_njit"])
                ]
            )
            ax.set_xlabel("FPS")
            ax.set_title(title)
            ax.grid(axis="x", alpha=0.3)

            # Add value labels
            for bar in bars:
                width = bar.get_width()
                if width > 0:
                    ax.text(
                        width, bar.get_y() + bar.get_height() / 2, f"{width:.1f}", ha="left", va="center", fontsize=8
                    )

        plt.suptitle(f'Effect Module Benchmarks - {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        plt.subplots_adjust(bottom=0.1)
        plt.tight_layout()
        fig.text(0.5, 0.02, "◯ = Success, × = Failed, ◆ = Uses njit, ◇ = No njit", ha="center", fontsize=10)

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
