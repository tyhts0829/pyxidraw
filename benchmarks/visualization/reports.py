#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク結果レポート生成モジュール

HTML、Markdown形式でベンチマーク結果レポートを生成します。
チャートの埋め込み、統計情報の表示、詳細分析を含みます。
"""

import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.core.types import BenchmarkResult, ValidationResult
from benchmarks.core.validator import BenchmarkResultAnalyzer


class ReportGenerator:
    """ベンチマークレポート生成クラス"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        self.analyzer = BenchmarkResultAnalyzer()
    
    def generate_html_report(self, 
                            results: Dict[str, BenchmarkResult],
                            chart_paths: Optional[List[Path]] = None,
                            save_path: Optional[Path] = None) -> Path:
        """HTMLレポートを生成"""
        
        # 結果を分析
        analysis = self.analyzer.analyze_results(results)
        
        # HTMLテンプレート
        html_content = self._generate_html_content(results, analysis, chart_paths)
        
        # 保存
        save_path = save_path or (self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return save_path
    
    def generate_markdown_report(self,
                                results: Dict[str, BenchmarkResult],
                                chart_paths: Optional[List[Path]] = None,
                                save_path: Optional[Path] = None) -> Path:
        """Markdownレポートを生成"""
        
        # 結果を分析
        analysis = self.analyzer.analyze_results(results)
        
        # Markdownコンテンツ
        md_content = self._generate_markdown_content(results, analysis, chart_paths)
        
        # 保存
        save_path = save_path or (self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return save_path
    
    def _generate_html_content(self, 
                              results: Dict[str, BenchmarkResult],
                              analysis: Dict[str, Any],
                              chart_paths: Optional[List[Path]]) -> str:
        """HTMLコンテンツを生成"""
        
        summary = analysis["summary"]
        validation = analysis["validation"]
        ranking = analysis["performance_ranking"]
        stats = analysis["statistics"]
        
        # チャートをBase64エンコード
        chart_embeds = []
        if chart_paths:
            for chart_path in chart_paths:
                if chart_path.exists():
                    with open(chart_path, 'rb') as f:
                        chart_data = base64.b64encode(f.read()).decode()
                        chart_embeds.append(f"data:image/png;base64,{chart_data}")
        
        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyxiDraw ベンチマークレポート</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #34495e;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .success {{ color: #27ae60; }}
        .failure {{ color: #e74c3c; }}
        .warning {{ color: #f39c12; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        
        .chart {{
            text-align: center;
            margin: 30px 0;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        
        .status-success {{ background-color: #d4edda; color: #155724; }}
        .status-failure {{ background-color: #f8d7da; color: #721c24; }}
        
        .validation-section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 6px;
        }}
        
        .error-list, .warning-list {{
            list-style-type: none;
            padding: 0;
        }}
        .error-list li, .warning-list li {{
            margin: 5px 0;
            padding: 8px;
            border-radius: 4px;
        }}
        .error-list li {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .warning-list li {{
            background-color: #fff3cd;
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PyxiDraw ベンチマークレポート</h1>
        <p><strong>生成日時:</strong> {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        
        <h2>📊 実行結果サマリー</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>総モジュール数</h3>
                <div class="value">{summary['total_modules']}</div>
            </div>
            <div class="summary-card">
                <h3>成功</h3>
                <div class="value success">{summary['successful']}</div>
            </div>
            <div class="summary-card">
                <h3>失敗</h3>
                <div class="value failure">{summary['failed']}</div>
            </div>
            <div class="summary-card">
                <h3>成功率</h3>
                <div class="value">{summary['success_rate']:.1%}</div>
            </div>
        </div>
        
        {self._generate_performance_summary_html(summary)}
        
        {self._generate_charts_html(chart_embeds)}
        
        {self._generate_ranking_html(ranking)}
        
        {self._generate_detailed_results_html(results)}
        
        {self._generate_validation_html(validation)}
        
        {self._generate_statistics_html(stats)}
        
        <hr style="margin: 40px 0;">
        <footer style="text-align: center; color: #7f8c8d;">
            <p>Generated by PyxiDraw Benchmark System v2.0</p>
        </footer>
    </div>
</body>
</html>
        """
        
        return html
    
    def _generate_performance_summary_html(self, summary: Dict[str, Any]) -> str:
        """パフォーマンスサマリーHTML"""
        if summary['successful'] == 0:
            return ""
        
        return f"""
        <h3>⚡ パフォーマンス統計</h3>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>最速時間</h3>
                <div class="value">{summary['fastest_time']*1000:.3f}ms</div>
            </div>
            <div class="summary-card">
                <h3>最遅時間</h3>
                <div class="value">{summary['slowest_time']*1000:.3f}ms</div>
            </div>
            <div class="summary-card">
                <h3>平均時間</h3>
                <div class="value">{summary['average_time']*1000:.3f}ms</div>
            </div>
        </div>
        """
    
    def _generate_charts_html(self, chart_embeds: List[str]) -> str:
        """チャートHTML"""
        if not chart_embeds:
            return ""
        
        charts_html = "<h2>📈 ベンチマーク結果チャート</h2>"
        
        for i, chart_embed in enumerate(chart_embeds):
            charts_html += f"""
            <div class="chart">
                <img src="{chart_embed}" alt="Benchmark Chart {i+1}">
            </div>
            """
        
        return charts_html
    
    def _generate_ranking_html(self, ranking: List[tuple]) -> str:
        """ランキングHTML"""
        if not ranking:
            return ""
        
        html = """
        <h2>🏆 パフォーマンスランキング</h2>
        <table>
            <thead>
                <tr>
                    <th>順位</th>
                    <th>モジュール</th>
                    <th>平均実行時間</th>
                    <th>FPS</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for rank, (module_name, avg_time) in enumerate(ranking[:20], 1):
            fps = 1.0 / avg_time if avg_time > 0 else float('inf')
            html += f"""
                <tr>
                    <td>{rank}</td>
                    <td>{module_name}</td>
                    <td>{avg_time*1000:.3f}ms</td>
                    <td>{fps:.1f}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def _generate_detailed_results_html(self, results: Dict[str, BenchmarkResult]) -> str:
        """詳細結果HTML"""
        html = """
        <h2>📋 詳細ベンチマーク結果</h2>
        <table>
            <thead>
                <tr>
                    <th>モジュール</th>
                    <th>ステータス</th>
                    <th>平均時間</th>
                    <th>最適化</th>
                    <th>シリアライズ</th>
                    <th>エラー</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for module_name, result in sorted(results.items()):
            status_class = "status-success" if result.success else "status-failure"
            status_text = "✓ 成功" if result.success else "✗ 失敗"
            avg_time_str = f"{result.timing_data.average_time*1000:.3f}ms" if result.success and result.timing_data.average_time > 0 else "-"
            
            # 表示用: タグを最適化欄に出す（簡潔）
            optimization_str = ", ".join(result.tags) if getattr(result, 'tags', None) else "-"
            
            # シリアライズ情報（ms）
            serialization_str = f"{result.serialization_overhead*1000:.2f}ms" if result.serialization_overhead > 0 else "-"
            
            error_str = result.error_message or "-" if not result.success else "-"
            
            html += f"""
                <tr class="{status_class}">
                    <td>{module_name}</td>
                    <td>{status_text}</td>
                    <td>{avg_time_str}</td>
                    <td>{optimization_str}</td>
                    <td>{serialization_str}</td>
                    <td>{error_str}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def _generate_validation_html(self, validation: ValidationResult) -> str:
        """検証結果HTML"""
        html = """
        <div class="validation-section">
            <h2>🔍 検証結果</h2>
        """
        
        if validation["is_valid"]:
            html += '<p class="success">✓ すべての検証項目が合格しました</p>'
        else:
            html += '<p class="failure">✗ 検証エラーが検出されました</p>'
        
        if validation["errors"]:
            html += f"""
            <h3 class="failure">エラー ({len(validation["errors"])}件)</h3>
            <ul class="error-list">
            """
            for error in validation["errors"]:
                html += f"<li>{error}</li>"
            html += "</ul>"
        
        if validation["warnings"]:
            html += f"""
            <h3 class="warning">警告 ({len(validation["warnings"])}件)</h3>
            <ul class="warning-list">
            """
            for warning in validation["warnings"]:
                html += f"<li>{warning}</li>"
            html += "</ul>"
        
        html += "</div>"
        return html
    
    def _generate_statistics_html(self, stats: Dict[str, Any]) -> str:
        """統計情報HTML"""
        if not stats:
            return ""
        
        html = """
        <h2>📊 統計情報</h2>
        <div class="summary-grid">
        """
        
        if "njit_usage_rate" in stats:
            html += f"""
            <div class="summary-card">
                <h3>NJIT使用率</h3>
                <div class="value">{stats['njit_usage_rate']:.1%}</div>
            </div>
            """
        
        if "cache_usage_rate" in stats:
            html += f"""
            <div class="summary-card">
                <h3>キャッシュ使用率</h3>
                <div class="value">{stats['cache_usage_rate']:.1%}</div>
            </div>
            """
        
        if "avg_measurements" in stats:
            html += f"""
            <div class="summary-card">
                <h3>平均測定回数</h3>
                <div class="value">{stats['avg_measurements']:.1f}</div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_markdown_content(self,
                                  results: Dict[str, BenchmarkResult],
                                  analysis: Dict[str, Any],
                                  chart_paths: Optional[List[Path]]) -> str:
        """Markdownコンテンツを生成"""
        
        summary = analysis["summary"]
        validation = analysis["validation"]
        ranking = analysis["performance_ranking"]
        stats = analysis["statistics"]
        
        md = f"""# PyxiDraw ベンチマークレポート

**生成日時:** {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

## 📊 実行結果サマリー

| 項目 | 値 |
|------|-----|
| 総モジュール数 | {summary['total_modules']} |
| 成功 | {summary['successful']} |
| 失敗 | {summary['failed']} |
| 成功率 | {summary['success_rate']:.1%} |

"""
        
        # パフォーマンス統計
        if summary['successful'] > 0:
            md += f"""
### ⚡ パフォーマンス統計

| 項目 | 時間 |
|------|------|
| 最速時間 | {summary['fastest_time']*1000:.3f}ms |
| 最遅時間 | {summary['slowest_time']*1000:.3f}ms |
| 平均時間 | {summary['average_time']*1000:.3f}ms |

"""
        
        # チャート
        if chart_paths:
            md += "## 📈 ベンチマーク結果チャート\n\n"
            for i, chart_path in enumerate(chart_paths):
                if chart_path.exists():
                    md += f"![Benchmark Chart {i+1}]({chart_path.name})\n\n"
        
        # ランキング
        if ranking:
            md += """## 🏆 パフォーマンスランキング

| 順位 | モジュール | 平均実行時間 | FPS |
|------|------------|-------------|-----|
"""
            for rank, (module_name, avg_time) in enumerate(ranking[:20], 1):
                fps = 1.0 / avg_time if avg_time > 0 else float('inf')
                md += f"| {rank} | {module_name} | {avg_time*1000:.3f}ms | {fps:.1f} |\n"
            md += "\n"
        
        # 詳細結果
        md += """## 📋 詳細ベンチマーク結果

| モジュール | ステータス | 平均時間 | 最適化 | エラー |
|------------|------------|----------|---------|-------|
"""
        
        for module_name, result in sorted(results.items()):
            status_text = "✓ 成功" if result.success else "✗ 失敗"
            avg_time_str = f"{result.timing_data.average_time*1000:.3f}ms" if result.success and result.timing_data.average_time > 0 else "-"
            optimization_str = ", ".join(result.tags) if getattr(result, 'tags', None) else "-"
            error_str = result.error_message or "-" if not result.success else "-"
            md += f"| {module_name} | {status_text} | {avg_time_str} | {optimization_str} | {error_str} |\n"
        
        md += "\n"
        
        # 検証結果
        md += "## 🔍 検証結果\n\n"
        
        if validation["is_valid"]:
            md += "✓ すべての検証項目が合格しました\n\n"
        else:
            md += "✗ 検証エラーが検出されました\n\n"
        
        if validation["errors"]:
            md += f"### エラー ({len(validation['errors'])}件)\n\n"
            for error in validation["errors"]:
                md += f"- {error}\n"
            md += "\n"
        
        if validation["warnings"]:
            md += f"### 警告 ({len(validation['warnings'])}件)\n\n"
            for warning in validation["warnings"]:
                md += f"- {warning}\n"
            md += "\n"
        
        # 統計情報
        if stats:
            md += "## 📊 統計情報\n\n"
            md += "| 項目 | 値 |\n|------|-----|\n"
            
            if "njit_usage_rate" in stats:
                md += f"| NJIT使用率 | {stats['njit_usage_rate']:.1%} |\n"
            if "cache_usage_rate" in stats:
                md += f"| キャッシュ使用率 | {stats['cache_usage_rate']:.1%} |\n"
            if "avg_measurements" in stats:
                md += f"| 平均測定回数 | {stats['avg_measurements']:.1f} |\n"
            
            md += "\n"
        
        md += """---

*Generated by PyxiDraw Benchmark System v2.0*
"""
        
        return md


# 便利関数
def generate_html_report(results: Dict[str, BenchmarkResult],
                        chart_paths: Optional[List[Path]] = None,
                        output_dir: Optional[Path] = None) -> Path:
    """HTMLレポートを生成する便利関数"""
    generator = ReportGenerator(output_dir)
    return generator.generate_html_report(results, chart_paths)


def generate_markdown_report(results: Dict[str, BenchmarkResult],
                           chart_paths: Optional[List[Path]] = None,
                           output_dir: Optional[Path] = None) -> Path:
    """Markdownレポートを生成する便利関数"""
    generator = ReportGenerator(output_dir)
    return generator.generate_markdown_report(results, chart_paths)
