"""
バーチャート生成

実行時間比較、成功率、パフォーマンス比較などのバーチャート
"""
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from benchmarks.core.types import BenchmarkResult
from .base import BaseChartGenerator, ChartDataProcessor, ChartColorManager


class BarChartGenerator(BaseChartGenerator):
    """バーチャート生成クラス"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_manager = ChartColorManager()
    
    def create_chart(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """汎用バーチャート作成"""
        chart_type = kwargs.get('chart_type', 'timing')
        
        if chart_type == 'timing':
            return self.create_timing_chart(data, **kwargs)
        elif chart_type == 'success_rate':
            return self.create_success_rate_chart(data, **kwargs)
        elif chart_type == 'comparison':
            return self.create_comparison_chart(data, **kwargs)
        else:
            raise ValueError(f"未対応のチャート種別です: {chart_type}")
    
    def create_timing_chart(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """実行時間バーチャート作成"""
        if not data:
            return ""
        
        fig, ax = self._create_figure(kwargs.get('figsize'))
        
        # データ準備
        targets = [item['target'] for item in data]
        times = [item['average_time'] for item in data]
        plugins = [item.get('plugin', 'unknown') for item in data]
        
        # プラグイン別に色分け
        colors = [self.color_manager.get_plugin_color(plugin) for plugin in plugins]
        
        # バーチャート作成
        bars = ax.bar(range(len(targets)), times, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # エラーバー追加（標準偏差がある場合）
        if all('std_dev' in item for item in data):
            error_bars = [item['std_dev'] for item in data]
            ax.errorbar(range(len(targets)), times, yerr=error_bars, 
                       fmt='none', ecolor='black', capsize=3, alpha=0.7)
        
        # チャートの装飾
        ax.set_xlabel('Benchmark Targets', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
        
        title = kwargs.get('title', 'Benchmark Execution Times')
        subtitle = f'Total: {len(data)} targets'
        self._add_chart_metadata(ax, title, subtitle)
        
        # X軸ラベル設定
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=45, ha='right')
        
        # 値をバーの上に表示
        for i, (bar, time) in enumerate(zip(bars, times)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(times),
                   f'{time:.2f}ms', ha='center', va='bottom', fontsize=8)
        
        # 凡例追加（プラグインが複数ある場合）
        unique_plugins = list(set(plugins))
        if len(unique_plugins) > 1:
            legend_elements = [plt.Rectangle((0,0),1,1, 
                                           color=self.color_manager.get_plugin_color(plugin),
                                           label=plugin) for plugin in unique_plugins]
            ax.legend(handles=legend_elements, loc='upper right')
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        filename = kwargs.get('filename', 'timing_chart.png')
        return self._save_chart(fig, filename)
    
    def create_success_rate_chart(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """成功率バーチャート作成"""
        if not data:
            return ""
        
        fig, ax = self._create_figure(kwargs.get('figsize'))
        
        # データ準備
        plugins = [item['plugin'] for item in data]
        success_rates = [item['success_rate'] for item in data]
        successful_counts = [item['successful'] for item in data]
        total_counts = [item['total'] for item in data]
        
        # 成功率に基づく色分け
        colors = ['#2ecc71' if rate >= 90 else '#f39c12' if rate >= 70 else '#e74c3c' 
                  for rate in success_rates]
        
        # バーチャート作成
        bars = ax.bar(range(len(plugins)), success_rates, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=0.5)
        
        # チャートの装飾
        ax.set_xlabel('Plugins', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        
        title = kwargs.get('title', 'Benchmark Success Rate by Plugin')
        subtitle = f'Total plugins: {len(data)}'
        self._add_chart_metadata(ax, title, subtitle)
        
        # X軸ラベル設定
        ax.set_xticks(range(len(plugins)))
        ax.set_xticklabels(plugins, rotation=45, ha='right')
        
        # 値をバーの上に表示
        for i, (bar, rate, successful, total) in enumerate(zip(bars, success_rates, successful_counts, total_counts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{rate:.1f}%\n({successful}/{total})', ha='center', va='bottom', fontsize=9)
        
        # 基準線追加
        ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Good (90%)')
        ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Fair (70%)')
        ax.legend()
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        filename = kwargs.get('filename', 'success_rate_chart.png')
        return self._save_chart(fig, filename)
    
    def create_comparison_chart(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """比較バーチャート作成（ベースライン vs 現在）"""
        if not data:
            return ""
        
        fig, ax = self._create_figure(kwargs.get('figsize'))
        
        # データ準備
        targets = [item['target'] for item in data]
        baseline_times = [item['baseline_time'] for item in data]
        current_times = [item['current_time'] for item in data]
        
        x = np.arange(len(targets))
        width = 0.35
        
        # バーチャート作成
        bars1 = ax.bar(x - width/2, baseline_times, width, label='Baseline', 
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, current_times, width, label='Current', 
                      color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # チャートの装飾
        ax.set_xlabel('Benchmark Targets', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
        
        title = kwargs.get('title', 'Performance Comparison: Baseline vs Current')
        subtitle = f'Targets: {len(data)}'
        self._add_chart_metadata(ax, title, subtitle)
        
        # X軸ラベル設定
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=45, ha='right')
        
        # 値をバーの上に表示
        for bars, times in [(bars1, baseline_times), (bars2, current_times)]:
            for bar, time in zip(bars, times):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(max(baseline_times), max(current_times)),
                       f'{time:.2f}ms', ha='center', va='bottom', fontsize=8)
        
        # 改善・劣化の矢印表示
        for i, item in enumerate(data):
            change_ratio = item['change_ratio']
            if abs(change_ratio) > 0.05:  # 5%以上の変化のみ表示
                y_pos = max(baseline_times[i], current_times[i]) * 1.15
                color = '#2ecc71' if change_ratio < 0 else '#e74c3c'
                symbol = '↓' if change_ratio < 0 else '↑'
                ax.text(i, y_pos, f'{symbol}{abs(change_ratio)*100:.1f}%', 
                       ha='center', va='bottom', color=color, fontweight='bold', fontsize=10)
        
        ax.legend()
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        filename = kwargs.get('filename', 'comparison_chart.png')
        return self._save_chart(fig, filename)
    
    def create_bar_chart(self, data: List[Dict[str, Any]], x_column: str, y_column: str, 
                        title: str, output_path: str, **kwargs) -> str:
        """汎用バーチャート作成（新しいvisualizationモジュールとの互換性）"""
        if not data:
            return ""
        
        fig, ax = self._create_figure()
        
        # データ抽出
        x_values = [item[x_column] for item in data]
        y_values = [item[y_column] for item in data]
        
        # 色設定
        color = kwargs.get('color', '#3498db')
        if isinstance(color, str):
            colors = [color] * len(x_values)
        else:
            colors = color
        
        # バーチャート作成
        bars = ax.bar(range(len(x_values)), y_values, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=0.5)
        
        # チャート装飾
        ax.set_xlabel(kwargs.get('xlabel', x_column), fontsize=12, fontweight='bold')
        ax.set_ylabel(kwargs.get('ylabel', y_column), fontsize=12, fontweight='bold')
        self._add_chart_metadata(ax, title)
        
        # X軸ラベル
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels(x_values, rotation=45, ha='right')
        
        # 値表示
        for bar, value in zip(bars, y_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(y_values),
                   f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # 保存
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
