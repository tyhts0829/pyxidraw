# ベンチマーク運用仕様（pytest-benchmark＋メモリ計測）

本ドキュメントは、性能回帰の検知と改善効果の定量化を、`pytest-benchmark` による統計・レポートと、`tracemalloc`/`psutil` によるメモリ/アロケーション観測を前提に標準化する。編集ファイル優先の高速ループと CI 比較に両対応する。

## 目的（どこで・何を・なぜ）
- どこで: `src/` の CPU 側ロジック（幾何/エフェクト/パイプライン/ランタイム/レンダ前処理）。
- 何を: 代表的パスの経過時間（wall time）を、入力規模とキャッシュ条件を固定して測定。
- なぜ: コード変更やアルゴリズム改修が「どの程度速く/遅くなったか」を定量把握し、回帰を早期検出するため。

## スコープ（レイヤ別の代表処理）
- Geometry（`engine.core.geometry`）
  - `Geometry.from_lines`（大規模結合/正規化）
  - `translate/scale/rotate/concat`（ベクトル化演算）
  - `digest`（有効/無効の差分計測: `PXD_DISABLE_GEOMETRY_DIGEST`）
- Effects/Pipeline（`api.effects` + `effects/*`）
  - Pipeline 実行（キャッシュ無効/有効、初回/再実行）
  - 代表エフェクト（例: `offset`, `displace`, `subdivide`）
- Shapes（`api.shapes` + `shapes/*`）
  - 代表形状生成（キャッシュ無効/有効、入力規模 S/M/L）
- Render 前処理（`engine.render.renderer._geometry_to_vertices_indices`）
  - VBO/IBO 生成（GPU 依存なしで計測可能）
- Runtime（`engine.runtime.WorkerPool`）
  - 1 tick あたりのタスク投入/取り出しのオーバーヘッド（inline 実行）

GPU 描画（ModernGL）は環境依存が大きいため、初期スコープ外（任意/別ジョブ）。

## 指標（Metrics）
- 時間（`pytest-benchmark`）: min/median/mean/std/rounds を取得（デフォルト列は `--benchmark-columns` で調整）。
- 規模: 入力頂点数 `N` / ポリライン数 `M`（`Geometry.n_vertices` / `n_lines`）。
- キャッシュ条件: 有効/無効、初回/再実行（Pipeline/Shapes LRU・Geometry digest）。
- メモリ: 
  - RSS 差分 [bytes]（`psutil.Process(...).memory_info().rss`）。
  - Python ヒープのピーク割当 [bytes]（`tracemalloc.get_traced_memory()[1]`）。
  - 必要に応じてアロケーション統計（`tracemalloc.take_snapshot()`）。

## 依存関係（Ask-first 方針）
- 追加（dev）: `pytest-benchmark`, `psutil`
  - `pyproject.toml` の `[project.optional-dependencies].dev` に追記（別PR）。
  - インストール: `python3.10 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e .[dev]`

依存追加は Ask-first（AGENTS.md）に従い、別PRで行う。

## 実行方法（pytest-benchmark ベース）
- 全体（perf マーカーのみ）: `pytest -q -m perf --benchmark-only`
- 対象限定: `pytest -q tests/perf/test_geometry_perf.py::test_from_lines_s --benchmark-only`
- ベースライン保存: `pytest -q -m perf --benchmark-only --benchmark-autosave`
- JSON 出力: `pytest -q -m perf --benchmark-only --benchmark-json=perf.json`
- 比較: `pytest -q -m perf --benchmark-only --benchmark-compare`（直近 autosave と比較）

推奨オプション（任意）:
- `--benchmark-min-rounds=10`（短時間ケースの反復数確保）
- `--benchmark-warmup=on --benchmark-warmup-iterations=2`（ウォームアップ）
- `--benchmark-columns=min,median,mean,stddev,ops,rounds`（表示列）
- `--benchmark-sort=name`（安定表示）

## ベンチテストの形（テンプレート）
`tests/perf/` に配置し、すべて `@pytest.mark.perf` を付与。`pytest-benchmark` の `benchmark` フィクスチャを用いる。

```python
# 例: tests/perf/test_geometry_perf.py
import os
import gc
import numpy as np
import psutil
import tracemalloc
import pytest

from api import G, E
from engine.core.geometry import Geometry
from engine.render.renderer import _geometry_to_vertices_indices


def measure_memory(fn, *args, **kwargs):
    proc = psutil.Process()
    gc.collect()
    rss_before = proc.memory_info().rss
    tracemalloc.start()
    try:
        result = fn(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    rss_after = proc.memory_info().rss
    return result, {"rss_delta": int(rss_after - rss_before), "alloc_peak": int(peak)}


@pytest.mark.perf
def test_from_lines_s(benchmark):
    # 入力 S 規模（例: 10_000 頂点程度）
    xs = np.linspace(0, 1, 5_000, dtype=np.float32)
    ys = np.sin(xs * np.pi).astype(np.float32)
    line = np.stack([xs, ys, np.zeros_like(xs)], axis=1)
    lines = [line, line[::-1]]

    def target():
        g = Geometry.from_lines(lines)
        # 軽い利用で最適化を回避
        _ = g.n_vertices, g.n_lines
        return g

    g, mem = measure_memory(target)
    res = benchmark.pedantic(lambda: Geometry.from_lines(lines), rounds=10, warmup_rounds=2)
    benchmark.extra_info.update({
        "case": "from_lines/S",
        "N": g.n_vertices,
        "M": g.n_lines,
        **mem,
    })


@pytest.mark.perf
def test_pipeline_cache_hit_vs_miss(benchmark):
    g = G.grid(subdivisions=(100, 100)).scale(200, 200, 1)
    pipe = (E.pipeline.offset(distance=2.0).rotate(z=0.3).cache(maxsize=128).build())

    def miss():
        pipe.clear_cache(); return pipe(g)

    def hit():
        _ = pipe(g); return pipe(g)

    miss_res = benchmark(miss)
    hit_res = benchmark(hit)
    benchmark.extra_info.update({
        "case": "pipeline/miss_vs_hit",
        "N": g.n_vertices,
        "M": g.n_lines,
    })
```

注意:
- 乱数は固定（`np.random.default_rng(0)`）。
- `benchmark.extra_info` に `N/M/cache/digest` などのメタ情報を付与（レポート集計に利用）。
- GC の影響を見る/隠すは `--benchmark-disable-gc` で切替可能。

## 入力規模（S/M/L の目安）
- S: ~1e4 頂点（< 100ms）— プルリク確認向け最小
- M: ~1e5 頂点（< 1s）— 代表的実用ケース
- L: ~1e6 頂点（数秒）— 回帰調査/ローカル限定（CI 非推奨）

規模はテスト内で生成（`numpy` ベクトル化）し、外部入出力は行わない。

## キャッシュ条件の切替
- Geometry digest: `PXD_DISABLE_GEOMETRY_DIGEST=1` で無効（`api.effects` は配列ハッシュにフォールバック）。
- Pipeline キャッシュ: `E.pipeline.cache(maxsize=...)` で制御。`0` で無効。
- Shapes LRU: `api.shapes.ShapesAPI.clear_cache()` を適宜呼び分ける。

## 回帰判定のルール（推奨）
- ローカル: `--benchmark-autosave` でベースライン保存 → `--benchmark-compare` で差分確認。中央値/平均が +10〜20% 超で調査。
- CI（将来）: autosave 生成物（`.benchmarks/`）または `--benchmark-json` の成果物をアーティファクト化し、前回比 +20% 超で警告/失敗（閾値はジョブで制御）。

補助方針:
- 初期は失敗させない（開発速度優先）。安定後に CI で失敗閾値を導入。

## レポート/アーティファクト
- `pytest-benchmark` の保存先: `.benchmarks/`（デフォルト）。
- JSON: `--benchmark-json=perf.json` を CI で保存（比較スクリプトは後続対応）。
- 任意: `data/perf/` へ独自集計を書き出す場合は `.gitignore` に追加。スナップショットのコミット更新は Ask-first。

## テスト配置と命名
- ルート: `tests/perf/`
- 命名: `test_*.py`（対象モジュールに対応付ける）
- マーカー: すべて `@pytest.mark.perf`
- 粒度: 
  - Micro: 個々の関数/小ブロック（例: `Geometry.rotate`、`_geometry_to_vertices_indices`）
  - Macro: 代表パイプライン（shape → effects 2–3 段 → 配列変換）

## 今後の拡張（Ask-first）
- 詳細なアロケーション統計（`tracemalloc` snapshot の比較）
- GPU パスのベンチ（実機ラボ/CI マトリクス）

---

運用メモ
- perf テストは「非失敗」を原則とし、回帰の兆候を発見したら PR 説明に中央値/入力規模/条件（キャッシュ有無/シード）を記載する。
- 編集ファイルに関係するテストから先に回す（`pytest -q -k <expr>`）。
- 仕様はコード（`src/`）と同期し、代表ケースの更新があれば本ファイルも併せて更新する。
