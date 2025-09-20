"""
ベンチ結果（summary.json もしくは perf.json）から静的 HTML レポートを生成する。

特徴:
- 追加の Python 依存なし（標準ライブラリのみ）。
- Chart.js を CDN から読み込み（オフライン環境ではグラフのみ非表示になる可能性）。
- ケース名で簡易グルーピング（pipeline の miss/hit, digest on/off, cache on/off, geometry micro）。

使い方:
    # 1) JSON 生成（いずれか）
    pytest -q tests/perf -m perf --benchmark-only --benchmark-json=perf.json
    python -m tools.bench.summarize perf.json --out-json benchmark_results/summary.json

    # 2) HTML 生成
    python -m tools.bench.html_report benchmark_results/summary.json --out benchmark_results/index.html

備考:
- 入力に perf.json を渡すことも可能（自動判別）。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

try:
    # summarize から Row/load_rows を再利用（perf.json 時に便利）
    from tools.bench.summarize import Row, load_rows  # type: ignore
except Exception:  # pragma: no cover - summarize が無い場合
    Row = object  # type: ignore
    load_rows = None  # type: ignore


def _is_perf_json(data: Any) -> bool:
    return isinstance(data, Mapping) and "benchmarks" in data


def _rows_from_input(data: Any) -> List[dict]:
    # perf.json -> rows（summarize.load_rows を利用）
    if _is_perf_json(data):
        if load_rows is None:
            raise RuntimeError("perf.json を直接扱うには tools.bench.summarize が必要です")
        rows = load_rows(data)
        return [asdict(r) for r in rows]  # type: ignore[arg-type]
    # summary.json（Row 辞書の列）
    if isinstance(data, list):
        return [dict(x) for x in data]
    raise TypeError("入力 JSON の形式を認識できません（perf.json または summary.json を想定）")


def _to_js_array(values: Iterable[Any]) -> str:
    # シンプルな JSON.stringify 同等（改行抑制）
    return json.dumps(list(values), ensure_ascii=False)


def render_html(rows: List[dict]) -> str:
    # 並びを安定化
    rows = sorted(rows, key=lambda r: (str(r.get("case", "")), str(r.get("name", ""))))

    # 全ケースの棒グラフ（mean [ms]）
    labels_all = [str(r.get("case") or r.get("name")) for r in rows]
    means_all_ms = [float(r.get("mean", 0.0)) * 1e3 for r in rows]

    # グループ抽出ユーティリティ
    def by_prefix(prefix: str) -> tuple[list[str], list[float]]:
        lbls, vals = [], []
        for r in rows:
            case = str(r.get("case") or "")
            if case.startswith(prefix):
                lbls.append(case)
                vals.append(float(r.get("mean", 0.0)) * 1e3)
        return lbls, vals

    lbl_miss_hit, val_miss_hit = by_prefix("pipeline/miss_vs_hit/")
    lbl_digest, val_digest = by_prefix("pipeline/digest/")
    lbl_cache, val_cache = by_prefix("pipeline/cache/")
    lbl_micro, val_micro = by_prefix("geometry/micro_ops/")
    lbl_shapes, val_shapes = by_prefix("shape/")
    lbl_effects, val_effects = by_prefix("effect/")

    # 表用データ
    table_rows = rows

    # HTML（Chart.js は CDN 読み込み）
    html = f"""<!DOCTYPE html>
<html lang=\"ja\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Benchmark Report</title>
  <style>
    body {{ font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
    h1, h2 {{ margin: 0.6em 0 0.3em; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; min-height: 0; }}
    .chart-container {{ position: relative; width: 100%; }}
    .chart-container canvas {{ width: 100% !important; height: 100% !important; display: block; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #eee; padding: 4px 6px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .note {{ color: #666; font-size: 12px; }}
  </style>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
</head>
<body>
  <h1>Benchmark Report</h1>
  <p class=\"note\">単位は ms（mean）。Chart.js は CDN から取得します。オフライン環境では表のみ表示されます。</p>

  <div class=\"grid\">
    <div class=\"card\">
      <h2>All Cases</h2>
      <div class=\"chart-container\"><canvas id=\"chart_all\"></canvas></div>
      <p class=\"note\">全ベンチの平均実行時間（mean, ms）。ケース（case）はテスト側で付与したラベルで、処理の種類や条件（S/M、miss/hit など）を示す。値が小さいほど高速。個々の差分は相対比較（同一環境・直近 autosave との比較）で読む。</p>
    </div>
    <div class=\"card\">
      <h2>Pipeline miss vs hit</h2>
      <div class=\"chart-container\"><canvas id=\"chart_miss_hit\"></canvas></div>
      <p class=\"note\">パイプラインのキャッシュ <b>miss</b> と <b>hit</b> の比較。miss はキャッシュをクリアして初回実行（再計算あり）、hit は同一入力での再実行（キャッシュ取得）。hit が miss より十分小さいほどキャッシュが有効。</p>
    </div>
    <div class=\"card\">
      <h2>Digest on/off</h2>
      <div class=\"chart-container\"><canvas id=\"chart_digest\"></canvas></div>
      <p class=\"note\">Geometry.digest の有効/無効比較。<b>on</b> は `Geometry.digest` を鍵に使用、<b>off</b> は配列からのハッシュ計算にフォールバック（`PXD_DISABLE_GEOMETRY_DIGEST=1`）。off が相対的に大きいほど digest の前計算効果が大きい。</p>
    </div>
    <div class=\"card\">
      <h2>Cache on/off</h2>
      <div class=\"chart-container\"><canvas id=\"chart_cache\"></canvas></div>
      <p class=\"note\">パイプライン内部キャッシュの <b>on</b>/<b>off</b>（on は事前にウォームアップしてヒットさせた値）。off は各回で再計算するため大きく、on はキャッシュ取得のオーバーヘッドのみ。差分が小さい場合、計算自体が軽いか、キャッシュの効果が限定的。</p>
    </div>
    <div class=\"card\">
      <h2>Geometry micro ops</h2>
      <div class=\"chart-container\"><canvas id=\"chart_micro\"></canvas></div>
      <p class=\"note\">Geometry の基本操作（rotate/scale/concat）の平均時間。入力 N/M は一定（テスト内で固定）。相対差でボトルネック候補を把握し、アルゴリズム変更の影響を観察する。</p>
    </div>
  </div>

  <div class=\"card\">\n    <h2>Shapes (all)</h2>\n    <div class=\"chart-container\"><canvas id=\"chart_shapes\"></canvas></div>\n    <p class=\"note\">登録済み shape を既定引数で実行した生成時間（mean, ms）。case は shape 名。負荷が大きいものはパラメータ既定値の性質（分割数など）が影響する可能性がある。</p>\n  </div>\n  <div class=\"card\">\n    <h2>Effects (all)</h2>\n    <div class=\"chart-container\"><canvas id=\"chart_effects\"></canvas></div>\n    <p class=\"note\">登録済み effect を代表ジオメトリへ適用した実行時間（mean, ms）。case は effect 名。オプショナル依存が不足するものはスキップされる。</p>\n  </div>

  <div class=\"card\" style=\"margin-top:16px;\">
    <h2>Raw Table</h2>
    <table>
      <thead>
        <tr>
          <th>case</th><th>name</th><th>N</th><th>M</th>
          <th>mean [s]</th><th>median [s]</th><th>stddev</th><th>ops</th><th>rounds</th><th>rss_delta</th><th>alloc_peak</th>
        </tr>
      </thead>
      <tbody>
        {''.join(f"<tr><td>{r.get('case','')}</td><td>{r.get('name','')}</td><td>{r.get('N','')}</td><td>{r.get('M','')}</td><td>{r.get('mean',0):.6e}</td><td>{r.get('median',0):.6e}</td><td>{r.get('stddev',0):.3e}</td><td>{r.get('ops',0):.2f}</td><td>{r.get('rounds','')}</td><td>{r.get('rss_delta','')}</td><td>{r.get('alloc_peak','')}</td></tr>" for r in table_rows)}
      </tbody>
    </table>
  </div>

  <script>
    function mkBar(id, labels, values) {{
      const canvas = document.getElementById(id);
      const container = canvas.parentElement; // .chart-container
      // ラベル数に応じて親コンテナ高さを固定（Chart.js は親の高さに追従）
      const rows = Array.isArray(labels) ? labels.length : 0;
      const h = Math.max(220, Math.min(1000, 24 * rows + 80));
      container.style.height = h + 'px';
      const ctx = canvas.getContext('2d');
      new Chart(ctx, {{ type: 'bar', data: {{ labels: labels, datasets: [{{
          label: 'mean [ms]', data: values, backgroundColor: 'rgba(54, 162, 235, 0.6)'
      }}] }}, options: {{
          indexAxis: 'y',                 // 水平バー
          responsive: true,
          maintainAspectRatio: false,
          resizeDelay: 200,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ beginAtZero: true, title: {{ display: true, text: 'ms (lower is faster)' }} }},
            y: {{ ticks: {{ autoSkip: false }} }}
          }}
      }} }});
    }}

    mkBar('chart_all', {_to_js_array(labels_all)}, {_to_js_array(means_all_ms)});
    mkBar('chart_miss_hit', {_to_js_array(lbl_miss_hit)}, {_to_js_array(val_miss_hit)});
    mkBar('chart_digest', {_to_js_array(lbl_digest)}, {_to_js_array(val_digest)});
    mkBar('chart_cache', {_to_js_array(lbl_cache)}, {_to_js_array(val_cache)});
    mkBar('chart_micro', {_to_js_array(lbl_micro)}, {_to_js_array(val_micro)});
    mkBar('chart_shapes', {_to_js_array(lbl_shapes)}, {_to_js_array(val_shapes)});
    mkBar('chart_effects', {_to_js_array(lbl_effects)}, {_to_js_array(val_effects)});
  </script>
</body>
</html>
"""
    return html


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate static HTML report from summary.json or perf.json"
    )
    p.add_argument("input", help="summary.json (preferred) or perf.json")
    p.add_argument("--out", required=True, help="output HTML path")
    args = p.parse_args(argv)

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = _rows_from_input(data)
    html = render_html(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI エントリ
    raise SystemExit(main())
