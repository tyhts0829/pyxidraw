# HUD/メトリクスのオプション化・リファクタリング計画（チェックリスト）

目的
- HUD に表示する項目（FPS/頂点数/CPU/MEM/CACHE など）をオプション化し、将来の項目追加に強い構造へ整理する。
- CACHE 表示（shape/effect ヒット判定）によるパフォーマンス影響を計測できるよう、収集処理を明示的にオン/オフ可能にする。

基本方針（シンプル・変更容易）
- HUD の「何を表示するか」を宣言的に渡す。Overlay は宣言に従い描画するだけにする。
- 収集（サンプリング）と表示（レンダリング）を明確に分離し、収集は個別に有効/無効を切替可能にする。
- 既定は“現状と同等の表示”、ただし CACHE 計測のみデフォルト無効（計測目的で切替しやすい）。

オプションOFF時の挙動（取得処理も止める）
- enabled=False（HUD全体OFF）
  - `MetricSampler`/`OverlayHUD` を生成しない。タイマ/スケジューラ登録も行わない。
  - `metrics_snapshot=None` を Worker に渡し、CACHE 計測も完全に無効化。
  - `StreamReceiver` の `on_metrics` 配線も行わない。
- show_cache_status=False（CACHE表示OFF）
  - `metrics_snapshot=None` を Worker に渡し、各ワーカー（マルチ/インライン）での描画前後スナップショット取得をしない。
  - 受信側の `on_metrics` 配線も行わず、`MetricSampler.data` に `CACHE/*` キーを追加しない。
- show_cpu_mem=False（CPU/MEM表示OFF）
  - `psutil` の遅延 import を採用し、このフラグが False の場合は import も `Process(...)` 生成も行わない。
  - `cpu_percent()`/`memory_info()` の呼び出しは発生しない。
- show_vertex_count=False（頂点数OFF）
  - `SwapBuffer.get_front()` の呼び出しや `len(geometry.coords)` の計算を行わない。
- show_fps=False（FPS OFF）
  - `SwapBuffer.version()` の差分計算や時間計測を行わない。
- Overlay 側の最適化
  - `MetricSampler.data` に存在するキーのみラベルを生成・更新する（OFFの項目はキー自体が入らない）。
  - すべての項目がOFFで `enabled=True` の場合、ラベルは1つも作られず、描画オーバーヘッドは極小。
  - 追加最適化（任意）: 表示対象が空なら Overlay の生成自体をスキップしてもよい。

API 仕様案（最小差分で導入）
- `api.sketch.run_sketch(...)` に `hud_config: HUDConfig | None = None` を追加。
  - `HUDConfig`（新規）
    - `enabled: bool = True` HUD 自体の有効/無効
    - `show_fps: bool = True`
    - `show_vertex_count: bool = True`
    - `show_cpu_mem: bool = True`
    - `show_cache_status: bool = False`  ← CACHE は既定で無効
    - `order: list[str] | None = None`  表示順（例: ["FPS", "VERTEX", "CPU", "MEM", "CACHE/SHAPE", "CACHE/EFFECT"]）。`None` は既定順
    - `sample_interval: float = 0.2`  MetricSampler のサンプリング周期
- 互換: 既存呼び出しは `hud_config=None` で同等表示（CACHE を除く）。

構成変更（責務分離）
- `engine/ui/hud/`（パッケージ）
  - `config.py`: `HUDConfig` 定義とユーティリティ（注文からフィールド名列生成）。
  - `fields.py`: 標準項目名の定義（文字列定数）と表示キーのマップ（例: `FPS`, `VERTEX`, `CPU`, `MEM`, `CACHE_SHAPE`, `CACHE_EFFECT`）。
  - `sampler.py`: `MetricSampler`（`HUDConfig` を受け、不要な計測をスキップ）。
  - `overlay.py`: 描画は `HUDConfig.order` の並びのみ。`MetricSampler.data` にないキーは無視。
- `api.sketch`
  - `hud_config` を受け取り、HUD 無効時は Overlay/MetricSampler を生成しない。
  - `show_cache_status=False` の場合、`metrics_snapshot` を `None` にして Worker へ渡す。
  - `StreamReceiver` への `on_metrics` フックも `show_cache_status=True` の時だけ配線。

実装手順（チェックリスト）
- [ ] `engine/ui/hud/config.py` に `HUDConfig` を追加（dataclass, 既定値は上記）。
- [ ] `engine/ui/hud/fields.py` を追加（標準キー名を定義: `FPS`, `VERTEX`, `CPU`, `MEM`, `CACHE_SHAPE`, `CACHE_EFFECT`）。
- [ ] `engine/ui/monitor.py` の `MetricSampler` に `HUDConfig` を受けさせ、`show_cpu_mem`/`sample_interval` を反映。`data` は既存キーを維持。
- [ ] `engine/ui/overlay.py` に `HUDConfig` を渡し、`order` に基づき `MetricSampler.data` を選択描画するように変更。
- [ ] `api/sketch.py` に `hud_config: HUDConfig | None` を追加。`None` の場合は互換既定を作成。
- [ ] `api/sketch.py` で `show_cache_status` に応じて `metrics_snapshot` と `on_metrics` の配線を切替。
- [ ] HUDConfig の簡易説明を README に追記（docs/overlay_cache_status_plan.md は廃止）。
- [ ] `architecture.md` の「UI/HUD」周辺に `engine/ui/hud` の役割を追記。
- [ ] 変更ファイルに対する `ruff/black/isort/mypy` を実行。
- [ ] smoke 確認: `python main.py` で HUD 正常（CACHE OFF）。`show_cache_status=True` で CACHE 行が表示され、ワーカ数 0/複数で動作。

表示例（既定）
- `order` 省略時は左下に次を表示: `FPS`, `VERTEX`, `CPU`, `MEM`（CACHE は OFF）。
- `show_cache_status=True` のとき `CACHE/SHAPE`, `CACHE/EFFECT` を末尾に追加。

パフォーマンス計測計画
- 計測観点: `show_cache_status=False/True` 時のフレーム時間（`MetricSampler` の FPS 実効値）、CPU% の差分。
- 手順: 同一スケッチで `hud_config.show_cache_status` を切替し 60 秒平均を比較（HUD 表示はオンのまま）。
- 期待: CACHE ON の追加コストは「1フレームあたり2回の軽量辞書走査 + `lru_cache.cache_info()` 呼び出し」程度。ヒット数が多いほど増加寄与は一定。

リスクとフォールバック
- psutil が未導入環境: 既定依存に含まれており問題は小さい。`show_cpu_mem=False` であれば import 自体を遅延/回避可能にする。
- 表示項目の追加: `fields.py` に定数を追加し、`MetricSampler` でキーを埋めれば Overlay は変更不要。

確認事項（要返信）
1) 既定で `show_cache_status=False`（OFF）にする方針でよいか。
2) 表示順の既定は現行どおり `FPS, VERTEX, CPU, MEM` でよいか。
3) 導入位置は `engine/ui/hud` というパッケージ名で問題ないか（`ui/overlay.py` や `ui/monitor.py` は既存のまま、依存方向のみ調整）。

メモ
- 実装後は `main.py` で `hud_config` を組み立てて `api.run_sketch` に渡すフックを追加してもよい（CLI オプションは別途）。
