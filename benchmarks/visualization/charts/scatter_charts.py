"""
散布図生成

複雑度 vs 実行時間、相関分析などの散布図
"""
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from benchmarks.core.types import BenchmarkResult
from .base import BaseChartGenerator, ChartDataProcessor, ChartColorManager


class ScatterPlotGenerator(BaseChartGenerator):
    """散布図生成クラス"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_manager = ChartColorManager()
    
    def create_chart(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """汎用散布図作成"""
        chart_type = kwargs.get('chart_type', 'complexity_vs_time')
        
        if chart_type == 'complexity_vs_time':
            return self.create_complexity_analysis_plot(data, **kwargs)
        elif chart_type == 'correlation':
            return self.create_correlation_plot(data, **kwargs)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def create_complexity_analysis_plot(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """複雑度 vs 実行時間の散布図作成"""
        if not data:
            return ""
        
        fig, ax = self._create_figure(kwargs.get('figsize'))
        
        # データ準備
        complexities = [item.get('complexity', 0) for item in data]
        times = [item.get('execution_time', 0) for item in data]
        targets = [item.get('target', f'target_{i}') for i, item in enumerate(data)]
        plugins = [item.get('plugin', 'unknown') for item in data]
        
        # 有効なデータのみをフィルタ
        valid_data = [(c, t, target, plugin) for c, t, target, plugin in zip(complexities, times, targets, plugins) 
                      if c > 0 and t > 0]
        
        if not valid_data:
            return ""
        
        complexities, times, targets, plugins = zip(*valid_data)
        
        # プラグイン別に色分け
        unique_plugins = list(set(plugins))
        plugin_colors = {plugin: self.color_manager.get_plugin_color(plugin) 
                        for plugin in unique_plugins}
        
        # 散布図作成
        for plugin in unique_plugins:
            plugin_indices = [i for i, p in enumerate(plugins) if p == plugin]
            plugin_complexities = [complexities[i] for i in plugin_indices]
            plugin_times = [times[i] for i in plugin_indices]
            
            ax.scatter(plugin_complexities, plugin_times, 
                      c=[plugin_colors[plugin]], label=plugin, 
                      alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        # 回帰直線追加
        if len(complexities) > 2:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(complexities, times)
                x_line = np.linspace(min(complexities), max(complexities), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'r--', alpha=0.8, linewidth=2, 
                       label=f'Regression (R²={r_value**2:.3f})')
            except Exception:
                pass  # 回帰分析に失敗した場合はスキップ
        
        # チャートの装飾
        ax.set_xlabel('Geometry Complexity', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
        
        title = kwargs.get('title', 'Execution Time vs Geometry Complexity')
        subtitle = f'Data points: {len(valid_data)}'
        self._add_chart_metadata(ax, title, subtitle)
        
        # 各点にラベル追加（オプション）
        if kwargs.get('show_labels', False) and len(valid_data) <= 20:
            for i, (c, t, target) in enumerate(zip(complexities, times, targets)):
                ax.annotate(target, (c, t), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        # 凡例追加
        ax.legend(loc='upper left')
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        filename = kwargs.get('filename', 'complexity_analysis.png')
        return self._save_chart(fig, filename)
    
    def create_correlation_plot(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """相関分析散布図作成"""
        if not data:
            return ""
        
        x_column = kwargs.get('x_column', 'complexity')
        y_column = kwargs.get('y_column', 'execution_time')
        
        fig, ax = self._create_figure(kwargs.get('figsize'))
        
        # データ準備
        x_values = [item.get(x_column, 0) for item in data]
        y_values = [item.get(y_column, 0) for item in data]
        
        # 有効なデータのみをフィルタ
        valid_pairs = [(x, y) for x, y in zip(x_values, y_values) if x > 0 and y > 0]
        
        if not valid_pairs:
            return ""
        
        x_values, y_values = zip(*valid_pairs)
        
        # 散布図作成
        ax.scatter(x_values, y_values, alpha=0.7, s=60, 
                  c='#3498db', edgecolors='black', linewidth=0.5)
        
        # 相関係数計算と回帰直線
        if len(x_values) > 2:
            try:
                correlation = np.corrcoef(x_values, y_values)[0, 1]
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
                
                x_line = np.linspace(min(x_values), max(x_values), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'r--', alpha=0.8, linewidth=2)
                
                # 相関情報を表示
                info_text = f'Correlation: {correlation:.3f}\nR²: {r_value**2:.3f}\np-value: {p_value:.3e}'
                ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
            except Exception:
                pass  # 統計分析に失敗した場合はスキップ
        
        # チャートの装飾
        ax.set_xlabel(kwargs.get('xlabel', x_column), fontsize=12, fontweight='bold')
        ax.set_ylabel(kwargs.get('ylabel', y_column), fontsize=12, fontweight='bold')
        
        title = kwargs.get('title', f'{y_column} vs {x_column}')
        subtitle = f'Data points: {len(valid_pairs)}'
        self._add_chart_metadata(ax, title, subtitle)
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        filename = kwargs.get('filename', 'correlation_plot.png')
        return self._save_chart(fig, filename)
    
    def create_scatter_plot(self, data: List[Dict[str, Any]], x_column: str, y_column: str, 
                           title: str, output_path: str, **kwargs) -> str:
        """汎用散布図作成（新しいvisualizationモジュールとの互換性）"""
        if not data:
            return ""
        
        fig, ax = self._create_figure()
        
        # データ抽出
        x_values = [item[x_column] for item in data]
        y_values = [item[y_column] for item in data]
        
        # 有効なデータのみをフィルタ
        valid_pairs = [(x, y) for x, y in zip(x_values, y_values) 
                       if x is not None and y is not None]
        
        if not valid_pairs:
            return ""
        
        x_values, y_values = zip(*valid_pairs)
        
        # 色設定
        color = kwargs.get('color', '#3498db')
        size = kwargs.get('size', 60)
        alpha = kwargs.get('alpha', 0.7)
        
        # 散布図作成
        ax.scatter(x_values, y_values, c=color, s=size, alpha=alpha,
                  edgecolors='black', linewidth=0.5)
        
        # 回帰直線追加（オプション）
        if kwargs.get('show_regression', True) and len(x_values) > 2:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
                x_line = np.linspace(min(x_values), max(x_values), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'r--', alpha=0.8, linewidth=2)
                
                # R²値を表示
                ax.text(0.05, 0.95, f'R² = {r_value**2:.3f}', transform=ax.transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            except Exception:
                pass
        
        # チャート装飾
        ax.set_xlabel(kwargs.get('xlabel', x_column), fontsize=12, fontweight='bold')
        ax.set_ylabel(kwargs.get('ylabel', y_column), fontsize=12, fontweight='bold')
        self._add_chart_metadata(ax, title)
        
        plt.tight_layout()
        
        # 保存
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def create_bubble_chart(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """バブルチャート作成（3次元データの可視化）"""
        if not data:
            return ""
        
        x_column = kwargs.get('x_column', 'complexity')
        y_column = kwargs.get('y_column', 'execution_time')
        size_column = kwargs.get('size_column', 'vertices_count')
        
        fig, ax = self._create_figure(kwargs.get('figsize'))
        
        # データ準備
        x_values = [item.get(x_column, 0) for item in data]
        y_values = [item.get(y_column, 0) for item in data]
        size_values = [item.get(size_column, 10) for item in data]
        plugins = [item.get('plugin', 'unknown') for item in data]
        
        # 有効なデータのみをフィルタ
        valid_data = [(x, y, s, p) for x, y, s, p in zip(x_values, y_values, size_values, plugins) 
                      if x > 0 and y > 0 and s > 0]
        
        if not valid_data:
            return ""
        
        x_values, y_values, size_values, plugins = zip(*valid_data)
        
        # サイズを正規化
        min_size, max_size = min(size_values), max(size_values)
        if max_size > min_size:
            normalized_sizes = [(s - min_size) / (max_size - min_size) * 400 + 50 
                               for s in size_values]
        else:
            normalized_sizes = [100] * len(size_values)
        
        # プラグイン別に色分け
        unique_plugins = list(set(plugins))
        plugin_colors = {plugin: self.color_manager.get_plugin_color(plugin) 
                        for plugin in unique_plugins}
        
        # バブルチャート作成
        for plugin in unique_plugins:
            plugin_indices = [i for i, p in enumerate(plugins) if p == plugin]
            plugin_x = [x_values[i] for i in plugin_indices]
            plugin_y = [y_values[i] for i in plugin_indices]
            plugin_sizes = [normalized_sizes[i] for i in plugin_indices]
            
            ax.scatter(plugin_x, plugin_y, s=plugin_sizes, 
                      c=[plugin_colors[plugin]], label=plugin, 
                      alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # チャートの装飾
        ax.set_xlabel(kwargs.get('xlabel', x_column), fontsize=12, fontweight='bold')
        ax.set_ylabel(kwargs.get('ylabel', y_column), fontsize=12, fontweight='bold')
        
        title = kwargs.get('title', f'Bubble Chart: {y_column} vs {x_column}')
        subtitle = f'Bubble size: {size_column}'
        self._add_chart_metadata(ax, title, subtitle)
        
        # 凡例追加
        ax.legend(loc='upper left')
        
        # サイズ凡例追加
        size_legend_values = [min_size, (min_size + max_size) / 2, max_size]
        size_legend_labels = [f'{int(v)}' for v in size_legend_values]
        size_legend_sizes = [50, 150, 300]
        
        for i, (label, size) in enumerate(zip(size_legend_labels, size_legend_sizes)):
            ax.scatter([], [], s=size, c='gray', alpha=0.6, edgecolors='black', 
                      label=f'{size_column}: {label}')
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        filename = kwargs.get('filename', 'bubble_chart.png')
        return self._save_chart(fig, filename)