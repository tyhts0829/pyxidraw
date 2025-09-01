"""
ヒートマップ生成

プラグイン vs ターゲット、時間変化、相関マトリックスなどのヒートマップ
"""
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from benchmarks.core.types import BenchmarkResult
from .base import BaseChartGenerator, ChartDataProcessor, ChartColorManager


class HeatmapGenerator(BaseChartGenerator):
    """ヒートマップ生成クラス"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_manager = ChartColorManager()
    
    def create_chart(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """汎用ヒートマップ作成"""
        chart_type = kwargs.get('chart_type', 'performance_matrix')
        
        if chart_type == 'performance_matrix':
            return self.create_performance_matrix(data, **kwargs)
        elif chart_type == 'correlation_matrix':
            return self.create_correlation_matrix(data, **kwargs)
        elif chart_type == 'time_series':
            return self.create_time_series_heatmap(data, **kwargs)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def create_performance_matrix(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """パフォーマンスマトリックスヒートマップ作成"""
        if not data:
            return ""
        
        # データをプラグイン x ターゲット マトリックスに変換
        matrix_data = {}
        plugins = set()
        targets = set()
        
        for item in data:
            plugin = item.get('plugin', 'unknown')
            target = item.get('target', 'unknown')
            time = item.get('average_time', 0)
            
            plugins.add(plugin)
            targets.add(target)
            matrix_data[(plugin, target)] = time
        
        if not matrix_data:
            return ""
        
        # DataFrameを作成
        plugins = sorted(list(plugins))
        targets = sorted(list(targets))
        
        matrix = np.zeros((len(plugins), len(targets)))
        for i, plugin in enumerate(plugins):
            for j, target in enumerate(targets):
                matrix[i, j] = matrix_data.get((plugin, target), np.nan)
        
        # ヒートマップ作成
        fig, ax = self._create_figure(kwargs.get('figsize', (12, 8)))
        
        # カラーマップ設定（速い=緑、遅い=赤）
        cmap = kwargs.get('cmap', 'RdYlGn_r')
        
        # ヒートマップ生成
        sns.heatmap(matrix, 
                   xticklabels=targets, 
                   yticklabels=plugins,
                   annot=True, 
                   fmt='.2f',
                   cmap=cmap,
                   cbar_kws={'label': 'Execution Time (ms)'},
                   ax=ax)
        
        # チャートの装飾
        ax.set_xlabel('Benchmark Targets', fontsize=12, fontweight='bold')
        ax.set_ylabel('Plugins', fontsize=12, fontweight='bold')
        
        title = kwargs.get('title', 'Performance Matrix Heatmap')
        subtitle = f'Plugins: {len(plugins)}, Targets: {len(targets)}'
        self._add_chart_metadata(ax, title, subtitle)
        
        # ラベルの回転
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        filename = kwargs.get('filename', 'performance_matrix.png')
        return self._save_chart(fig, filename)
    
    def create_correlation_matrix(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """相関マトリックスヒートマップ作成"""
        if not data:
            return ""
        
        # 数値列を抽出
        numeric_columns = ['average_time', 'complexity', 'vertices_count', 'min_time', 'max_time', 'std_dev']
        
        # DataFrameを作成
        df_data = []
        for item in data:
            row = {}
            for col in numeric_columns:
                value = item.get(col, None)
                if value is not None and not np.isnan(value):
                    row[col] = value
            if len(row) >= 2:  # 最低2つの数値列が必要
                df_data.append(row)
        
        if not df_data:
            return ""
        
        df = pd.DataFrame(df_data)
        
        # 相関行列を計算
        correlation_matrix = df.corr()
        
        # ヒートマップ作成
        fig, ax = self._create_figure(kwargs.get('figsize', (10, 8)))
        
        # マスクを作成（上三角を隠す）
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # ヒートマップ生成
        sns.heatmap(correlation_matrix,
                   mask=mask,
                   annot=True,
                   fmt='.3f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Correlation Coefficient'},
                   ax=ax)
        
        # チャートの装飾
        title = kwargs.get('title', 'Correlation Matrix')
        subtitle = f'Features: {len(correlation_matrix.columns)}'
        self._add_chart_metadata(ax, title, subtitle)
        
        # ラベルの回転
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        filename = kwargs.get('filename', 'correlation_matrix.png')
        return self._save_chart(fig, filename)
    
    def create_time_series_heatmap(self, data: List[Dict[str, Any]], **kwargs) -> str:
        """時系列ヒートマップ作成（複数回実行の結果の場合）"""
        if not data:
            return ""
        
        # 時系列データの準備
        time_series_data = {}
        
        for item in data:
            target = item.get('target', 'unknown')
            measurements = item.get('measurements', [])
            
            if measurements:
                time_series_data[target] = measurements
        
        if not time_series_data:
            return ""
        
        # 最大測定回数を取得
        max_measurements = max(len(measurements) for measurements in time_series_data.values())
        
        # マトリックスを作成
        targets = sorted(list(time_series_data.keys()))
        matrix = np.full((len(targets), max_measurements), np.nan)
        
        for i, target in enumerate(targets):
            measurements = time_series_data[target]
            for j, measurement in enumerate(measurements):
                if j < max_measurements:
                    matrix[i, j] = measurement
        
        # ヒートマップ作成
        fig, ax = self._create_figure(kwargs.get('figsize', (max_measurements * 0.5 + 4, len(targets) * 0.4 + 4)))
        
        # カラーマップ設定
        cmap = kwargs.get('cmap', 'viridis')
        
        # ヒートマップ生成
        sns.heatmap(matrix,
                   xticklabels=range(1, max_measurements + 1),
                   yticklabels=targets,
                   cmap=cmap,
                   cbar_kws={'label': 'Execution Time (ms)'},
                   ax=ax)
        
        # チャートの装飾
        ax.set_xlabel('Measurement Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Benchmark Targets', fontsize=12, fontweight='bold')
        
        title = kwargs.get('title', 'Time Series Performance Heatmap')
        subtitle = f'Targets: {len(targets)}, Max measurements: {max_measurements}'
        self._add_chart_metadata(ax, title, subtitle)
        
        # ラベルの回転
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        filename = kwargs.get('filename', 'time_series_heatmap.png')
        return self._save_chart(fig, filename)
    
    def create_comparison_heatmap(self, baseline_data: List[Dict[str, Any]], 
                                 current_data: List[Dict[str, Any]], **kwargs) -> str:
        """比較ヒートマップ作成（改善/劣化の可視化）"""
        if not baseline_data or not current_data:
            return ""
        
        # ベースラインと現在のデータを辞書に変換
        baseline_dict = {item['target']: item.get('average_time', 0) for item in baseline_data}
        current_dict = {item['target']: item.get('average_time', 0) for item in current_data}
        
        # 共通のターゲットを取得
        common_targets = sorted(list(set(baseline_dict.keys()) & set(current_dict.keys())))
        
        if not common_targets:
            return ""
        
        # 変化率を計算
        change_matrix = np.zeros((1, len(common_targets)))
        for j, target in enumerate(common_targets):
            baseline_time = baseline_dict[target]
            current_time = current_dict[target]
            
            if baseline_time > 0:
                change_ratio = (current_time - baseline_time) / baseline_time
                change_matrix[0, j] = change_ratio
            else:
                change_matrix[0, j] = np.nan
        
        # ヒートマップ作成
        fig, ax = self._create_figure(kwargs.get('figsize', (len(common_targets) * 0.6 + 2, 3)))
        
        # カラーマップ設定（改善=緑、劣化=赤）
        cmap = 'RdYlGn_r'
        
        # ヒートマップ生成
        sns.heatmap(change_matrix,
                   xticklabels=common_targets,
                   yticklabels=['Change Ratio'],
                   annot=True,
                   fmt='.2%',
                   cmap=cmap,
                   center=0,
                   cbar_kws={'label': 'Performance Change'},
                   ax=ax)
        
        # チャートの装飾
        ax.set_xlabel('Benchmark Targets', fontsize=12, fontweight='bold')
        
        title = kwargs.get('title', 'Performance Comparison Heatmap')
        subtitle = f'Targets: {len(common_targets)}'
        self._add_chart_metadata(ax, title, subtitle)
        
        # ラベルの回転
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        filename = kwargs.get('filename', 'comparison_heatmap.png')
        return self._save_chart(fig, filename)