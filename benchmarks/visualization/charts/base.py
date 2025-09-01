"""
ベーシックチャート生成基底クラス

全チャート生成クラスの共通機能と基底実装
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

try:
    import japanize_matplotlib
    japanize_matplotlib.japanize()
    HAS_JAPANESE = True
except ImportError:
    HAS_JAPANESE = False

from benchmarks.core.types import BenchmarkResult


class BaseChartGenerator(ABC):
    """チャート生成基底クラス"""
    
    def __init__(self, output_dir: Optional[Path] = None, style: str = "seaborn-v0_8"):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # スタイル設定
        self._setup_style(style)
        
        # 共通設定
        self.figure_size = (12, 8)
        self.dpi = 300
        self.colors = sns.color_palette("husl", 12)
    
    def _setup_style(self, style: str):
        """スタイル設定"""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Seabornテーマ設定
        sns.set_theme()
        
        # 日本語フォント設定
        if HAS_JAPANESE:
            japanize_matplotlib.japanize()
    
    def _create_figure(self, figsize: Optional[Tuple[float, float]] = None) -> Tuple[Figure, Axes]:
        """Figure とAxes を作成"""
        figsize = figsize or self.figure_size
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        return fig, ax
    
    def _save_chart(self, fig: Figure, filename: str, **kwargs) -> str:
        """チャートを保存"""
        output_path = self.output_dir / filename
        
        # デフォルトの保存オプション
        save_options = {
            'dpi': self.dpi,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        save_options.update(kwargs)
        
        fig.savefig(output_path, **save_options)
        plt.close(fig)
        
        return str(output_path)
    
    def _extract_timing_data(self, results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
        """タイミングデータを抽出"""
        timing_data = []
        for result in results:
            if result.success and result.timing_data.measurement_times:
                timing_data.append({
                    "target": result.target_name,
                    "plugin": result.plugin_name,
                    "average_time": result.timing_data.average_time * 1000,  # ms変換
                    "min_time": result.timing_data.min_time * 1000,
                    "max_time": result.timing_data.max_time * 1000,
                    "std_dev": result.timing_data.std_dev * 1000,
                    "measurements": [t * 1000 for t in result.timing_data.measurement_times]
                })
        return timing_data
    
    def _format_chart_title(self, title: str, subtitle: Optional[str] = None) -> str:
        """チャートタイトルをフォーマット"""
        if subtitle:
            return f"{title}\n{subtitle}"
        return title
    
    def _add_chart_metadata(self, ax: Axes, title: str, subtitle: Optional[str] = None):
        """チャートにメタデータを追加"""
        # タイトル設定
        formatted_title = self._format_chart_title(title, subtitle)
        ax.set_title(formatted_title, fontsize=16, fontweight='bold', pad=20)
        
        # グリッド設定
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 軸ラベル設定
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    @abstractmethod
    def create_chart(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """チャートを作成（サブクラスで実装）"""
        pass


class ChartDataProcessor:
    """チャートデータ前処理ユーティリティ"""
    
    @staticmethod
    def filter_successful_results(results: List[BenchmarkResult]) -> List[BenchmarkResult]:
        """成功した結果のみをフィルタ"""
        return [r for r in results if r.success]
    
    @staticmethod
    def group_by_plugin(results: List[BenchmarkResult]) -> Dict[str, List[BenchmarkResult]]:
        """プラグイン別にグループ化"""
        groups = {}
        for result in results:
            plugin = result.plugin_name
            if plugin not in groups:
                groups[plugin] = []
            groups[plugin].append(result)
        return groups
    
    @staticmethod
    def calculate_statistics(values: List[float]) -> Dict[str, float]:
        """統計値を計算"""
        if not values:
            return {}
        
        return {
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75)
        }
    
    @staticmethod
    def prepare_comparison_data(baseline: List[BenchmarkResult], 
                              current: List[BenchmarkResult]) -> List[Dict[str, Any]]:
        """比較用データを準備"""
        baseline_dict = {r.target_name: r for r in baseline if r.success}
        current_dict = {r.target_name: r for r in current if r.success}
        
        comparison_data = []
        for target_name in set(baseline_dict.keys()) & set(current_dict.keys()):
            baseline_result = baseline_dict[target_name]
            current_result = current_dict[target_name]
            
            baseline_time = baseline_result.timing_data.average_time
            current_time = current_result.timing_data.average_time
            
            change_ratio = (current_time - baseline_time) / baseline_time
            
            comparison_data.append({
                "target": target_name,
                "baseline_time": baseline_time * 1000,
                "current_time": current_time * 1000,
                "change_ratio": change_ratio,
                "improvement": change_ratio < 0
            })
        
        return comparison_data


class ChartColorManager:
    """チャート色管理ユーティリティ"""
    
    def __init__(self):
        self.plugin_colors = {}
        self.color_palette = sns.color_palette("husl", 12)
        self.color_index = 0
    
    def get_plugin_color(self, plugin_name: str) -> Tuple[float, float, float]:
        """プラグイン固有の色を取得"""
        if plugin_name not in self.plugin_colors:
            self.plugin_colors[plugin_name] = self.color_palette[self.color_index % len(self.color_palette)]
            self.color_index += 1
        return self.plugin_colors[plugin_name]
    
    def get_performance_color(self, value: float, threshold: float = 0.0) -> str:
        """パフォーマンス値に基づく色を取得"""
        if value < threshold:
            return '#2ecc71'  # 緑（改善）
        elif value > abs(threshold):
            return '#e74c3c'  # 赤（劣化）
        else:
            return '#f39c12'  # オレンジ（変化なし）