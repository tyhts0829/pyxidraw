"""
ベンチマーク結果可視化モジュール

ベンチマーク結果のチャート生成、レポート作成を行います。
既存のvisualizationロジックを改良し、より柔軟で拡張可能な
可視化システムを提供します。
"""

try:
    from .charts_facade import ChartGenerator
except Exception:
    class ChartGenerator:  # minimal stub for tests
        pass

__all__ = ["ChartGenerator"]

from .reports import ReportGenerator, generate_html_report, generate_markdown_report

__all__ += [
    "ReportGenerator",
    "generate_html_report",
    "generate_markdown_report",
]
