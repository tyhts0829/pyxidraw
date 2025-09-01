"""
ベンチマークチャート生成モジュール

チャート種類別に分離された実装群
"""
from .base import BaseChartGenerator, ChartDataProcessor, ChartColorManager
from .bar_charts import BarChartGenerator
from .box_charts import BoxPlotGenerator
from .scatter_charts import ScatterPlotGenerator
from .heatmap_charts import HeatmapGenerator

# 互換API: テストでは ChartGenerator をモック差し替えするため、
# ここでは最小のスタブを提供（実体は charts_facade.ChartGenerator 参照推奨）
class ChartGenerator:  # pragma: no cover - only a stub for patch target
    def __init__(self, *args, **kwargs):
        pass
    def create_bar_chart(self, *args, **kwargs):
        return ""
    def create_scatter_plot(self, *args, **kwargs):
        return ""

__all__ = [
    'BaseChartGenerator',
    'ChartDataProcessor', 
    'ChartColorManager',
    'BarChartGenerator',
    'BoxPlotGenerator',
    'ScatterPlotGenerator',
    'HeatmapGenerator',
    'ChartGenerator'
]
