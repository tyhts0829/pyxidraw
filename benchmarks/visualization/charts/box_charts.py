"""
ボックスプロット生成

実行時間分布、プラグイン別パフォーマンス分析のボックスプロット
"""
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from benchmarks.core.types import BenchmarkResult
from .base import BaseChartGenerator, ChartDataProcessor, ChartColorManager


class BoxPlotGenerator(BaseChartGenerator):
    """ボックスプロット生成クラス"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_manager = ChartColorManager()
    
    def create_chart(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """汎用ボックスプロット作成"""
        chart_type = kwargs.get('chart_type', 'timing_distribution')
        
        if chart_type == 'timing_distribution':
            return self.create_timing_distribution_plot(data, **kwargs)
        elif chart_type == 'plugin_comparison':
            return self.create_plugin_comparison_plot(data, **kwargs)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def create_timing_distribution_plot(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """実行時間分布ボックスプロット作成"""
        if not data:
            return ""
        
        fig, ax = self._create_figure(kwargs.get('figsize'))
        
        # データ準備
        plot_data = []
        labels = []
        
        for item in data:
            if 'measurements' in item and item['measurements']:
                plot_data.append(item['measurements'])
                labels.append(item['target'])
        
        if not plot_data:
            return ""
        
        # ボックスプロット作成
        box_plot = ax.boxplot(plot_data, labels=labels, patch_artist=True, 
                             showmeans=True, meanline=True)
        
        # 色設定
        colors = [self.color_manager.get_plugin_color(data[i].get('plugin', 'unknown')) 
                  for i in range(len(plot_data))]
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 平均線の色設定
        for line in box_plot['means']:
            line.set_color('red')
            line.set_linewidth(2)
        
        # チャートの装飾
        ax.set_xlabel('Benchmark Targets', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
        
        title = kwargs.get('title', 'Execution Time Distribution')
        subtitle = f'Targets: {len(plot_data)}'
        self._add_chart_metadata(ax, title, subtitle)
        
        # X軸ラベルの回転
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 統計情報を表示
        for i, measurements in enumerate(plot_data):
            stats = ChartDataProcessor.calculate_statistics(measurements)
            if stats:
                # 外れ値を上部に表示
                outliers = [x for x in measurements if x > stats['q75'] + 1.5 * (stats['q75'] - stats['q25'])]
                if outliers:
                    ax.text(i+1, max(measurements) * 1.05, f'{len(outliers)} outliers',
                           ha='center', va='bottom', fontsize=8, alpha=0.7)
        
        # 凡例追加
        ax.legend([plt.Line2D([0], [0], color='red', linewidth=2)], ['Mean'], loc='upper right')
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        filename = kwargs.get('filename', 'timing_distribution.png')
        return self._save_chart(fig, filename)
    
    def create_plugin_comparison_plot(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """プラグイン別比較ボックスプロット作成"""
        if not data:
            return ""
        
        # プラグイン別にデータをグループ化
        plugin_data = {}
        for item in data:
            plugin = item.get('plugin', 'unknown')
            if plugin not in plugin_data:
                plugin_data[plugin] = []
            
            if 'measurements' in item and item['measurements']:
                plugin_data[plugin].extend(item['measurements'])
        
        if not plugin_data:
            return ""
        
        fig, ax = self._create_figure(kwargs.get('figsize'))
        
        # データ準備
        plot_data = list(plugin_data.values())
        labels = list(plugin_data.keys())
        
        # ボックスプロット作成
        box_plot = ax.boxplot(plot_data, labels=labels, patch_artist=True, 
                             showmeans=True, meanline=True)
        
        # 色設定
        colors = [self.color_manager.get_plugin_color(plugin) for plugin in labels]
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 平均線の色設定
        for line in box_plot['means']:
            line.set_color('red')
            line.set_linewidth(2)
        
        # チャートの装飾
        ax.set_xlabel('Plugins', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
        
        title = kwargs.get('title', 'Plugin Performance Comparison')
        subtitle = f'Plugins: {len(labels)}'
        self._add_chart_metadata(ax, title, subtitle)
        
        # プラグイン別統計を表示
        for i, (plugin, measurements) in enumerate(plugin_data.items()):
            stats = ChartDataProcessor.calculate_statistics(measurements)
            if stats:
                info_text = f"n={len(measurements)}\nμ={stats['mean']:.2f}ms\nσ={stats['std']:.2f}ms"
                ax.text(i+1, max(measurements) * 0.9, info_text,
                       ha='center', va='top', fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 凡例追加
        ax.legend([plt.Line2D([0], [0], color='red', linewidth=2)], ['Mean'], loc='upper right')
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        filename = kwargs.get('filename', 'plugin_comparison.png')
        return self._save_chart(fig, filename)
    
    def create_violin_plot(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """バイオリンプロット作成（分布形状も表示）"""
        if not data:
            return ""
        
        fig, ax = self._create_figure(kwargs.get('figsize'))
        
        # データ準備
        plot_data = []
        labels = []
        
        for item in data:
            if 'measurements' in item and item['measurements']:
                plot_data.append(item['measurements'])
                labels.append(item['target'])
        
        if not plot_data:
            return ""
        
        # バイオリンプロット作成
        parts = ax.violinplot(plot_data, positions=range(1, len(plot_data) + 1), 
                             showmeans=True, showmedians=True)
        
        # 色設定
        colors = [self.color_manager.get_plugin_color(data[i].get('plugin', 'unknown')) 
                  for i in range(len(plot_data))]
        
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # チャートの装飾
        ax.set_xlabel('Benchmark Targets', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
        
        title = kwargs.get('title', 'Execution Time Distribution (Violin Plot)')
        subtitle = f'Targets: {len(plot_data)}'
        self._add_chart_metadata(ax, title, subtitle)
        
        # X軸ラベル設定
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        filename = kwargs.get('filename', 'timing_violin.png')
        return self._save_chart(fig, filename)