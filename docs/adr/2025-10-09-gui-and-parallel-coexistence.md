# ADR: パラメータGUIと並列（WorkerPool）を両立させる設計

状況: 現在は `use_parameter_gui=True` 時に `workers=0`（Inline 実行）へ強制され、並列が無効化される。GUI と並列を両立し、GUI を保ったまま `workers>=1` の多プロセスで `draw(t)` を実行できるようにする。

## 背景 / 課題
- 現仕様（根拠）
  - `main.py:38-47` で `use_parameter_gui=True, workers=3` を指定しても、
    `src/api/sketch.py:252-259` により `worker_count=0` に上書きされる。
  - `src/engine/runtime/worker.py:153-160` で `num_workers<1` の場合、Inline モード（単一プロセス）に切替わる。
  - 設計意図は `architecture.md:66-74` に記載（GUI と spawn 方式の相性配慮）。
- 技術的制約
  - macOS 等の `spawn` 方式で、GUI ランタイム（`ParameterManager.draw` のバウンドメソッド）はピクル不可。
  - Parameter GUI は「未指定引数のみを GUI で上書き」し、優先順位は「明示引数 > GUI > 既定値」。この解決は `ParameterRuntime` が担っている。
  - Engine 層は API/UI 層に依存させない方針のため、ランタイム注入は関数渡し（インジェクション）で行う必要がある（`api.cc.set_snapshot` と同様）。

## 目標（スコープ）
- `use_parameter_gui=True` でも `workers>=1` の並列実行を有効化する。
- GUI 側のパラメータ上書きを、ワーカープロセス側の `draw(t)` に正しく反映する。
- 既存のパラメータ規約/量子化/署名生成の整合を崩さない。
- Engine の依存境界（engine→api/ui 直接依存なし）を維持する。

非目標:
- Dear PyGui のイベント統合方法やテーマなど、GUI 表示の大規模改修。
- 既存 API（`G`/`E`/`Geometry`）のシグネチャ変更。

## 決定（概要）
1) Parameter GUI の「現在値スナップショット」をフレーム毎に取得し、`RenderTask` に同梱する。
2) ワーカープロセス側で、そのスナップショットを解釈する軽量ランタイム（SnapshotRuntime）を一時的に有効化してから `user_draw(t)` を実行する。
3) Engine 層には UI 依存を持ち込まず、スナップショット適用はトップレベル関数（picklable）をインジェクトして実行する（`apply_param_snapshot(...)`）。

これにより、GUI はメインスレッドで継続動作しつつ、`draw(t)` は N プロセスで並列化できる。

## 詳細設計

### 1. Parameter スナップショット
- 形式: `dict[str, Any]`（キーは `"{scope}.{name}#{index}.{param}"`。例: `"effect.fill#0.density"`）。
- 値: GUI 適用後の「実値」。float の量子化は保存/転送時は行わない（量子化は署名生成にて実施）。
- 送付タイミング: `WorkerPool.tick()` のタスク投入時に同梱。
- サイズ最適化: 原則「override のみ」を送る（original と一致するものは省略）。

備考（規約との整合）:
- RangeHint は UI 表示にのみ使用。スナップショットは実値を保持。
- 署名生成（キャッシュ鍵）は `common.param_utils.params_signature` で行うため、Effect 実行への量子化適用は従来通り（Shape は鍵のみ量子化）。

### 2. SnapshotRuntime（ワーカ側）
- 目的: `ParameterRuntime` 相当の最小 API を、静的スナップショットから再現。
- 実装: `before_shape_call(...)` と `before_effect_call(...)` を提供。
  - `scope/name/index` と呼び出し時の `params` からパラメータIDを合成し、スナップショットに該当値があれば「未指定の引数」に限って上書きする（「明示引数 > GUI > 既定値」を維持）。
  - `t` 注入: `inspect.signature(fn)` で `t` を受け取る関数に対し、`params` 未指定なら `t` を注入する（現行 Runtime と同様）。
  - インデックス: shape 呼び出しは「名称ごとにフレーム内カウントアップ」、effect はステップ index を受け取る（現行と同じ）。
- フレーム境界: `begin_frame()` 相当で shape 呼び出しカウンタをリセット。`set_inputs(t)` で時刻を保持。

### 3. RenderTask の拡張
- 変更: `src/engine/runtime/task.py` の `RenderTask` に `param_overrides: Mapping[str, Any] | None` を追加（pickle 可能な型のみ）。
- 互換性: 省略時 `None` を許容し、GUI 無効時はペイロードを送らない。

### 4. ワーカーへの適用（関数インジェクション）
- 追加インジェクション: `apply_param_snapshot(overrides: Mapping[str, Any] | None, t: float) -> None` をトップレベルに実装（例: `engine.ui.parameters.snapshot`）。
  - 中で `SnapshotRuntime(overrides).set_inputs(t)` を生成→`engine.ui.parameters.runtime.activate_runtime(...)` を呼ぶ。
  - `None` / 空の場合は `deactivate_runtime()` を呼ぶ。
- `_WorkerProcess.run()` の先頭で、`apply_cc_snapshot(task.cc_state)` に続けて `apply_param_snapshot(task.param_overrides, task.t)` を呼ぶ。
- Engine 層（`engine.runtime.worker`）は関数を受け取って呼ぶだけで、UI 層に依存しない。

### 5. api.sketch の更新
- GUI 有効時でも `worker_count = workers` をそのまま渡す。
- `draw_callback` は `user_draw` を直接使用（`ParameterManager.draw` のバウンド関数は渡さない）。
- `ParameterManager` はメインスレッドで初期トレースと GUI 管理のみを行い、`store` から「差分スナップショット（override のみ）」を生成する `param_snapshot_fn()` を提供。
- `WorkerPool` 生成時に `apply_param_snapshot` を注入。`tick()` で `task = RenderTask(..., param_overrides=param_snapshot_fn())` を作る。

### 6. エラーハンドリング
- スナップショット適用は best-effort。未知キーや型不一致は黙ってスキップする（現行 persistence と同様の方針）。
- 署名/量子化は従来のパスで実行されるため、キャッシュや一致性は変わらない。

## 代替案（検討済み）
- GUI 側 Runtime/Store を共有メモリ/Manager 経由で共有: 設計が複雑化し、ロック/IPC コストが高い。spawn 下でのデッドロック/終了順序もリスク。
- `resolve_without_runtime()` を拡張してグローバル状態から適用: Engine/API の依存境界が曖昧になる。SnapshotRuntime + 注入の方が明快。

## 影響範囲
- 追加/更新ファイル（予定）
  - `src/engine/runtime/task.py`（`RenderTask` 拡張）
  - `src/engine/runtime/worker.py`（`apply_param_snapshot` 呼び出し追加）
  - `src/api/sketch.py`（GUI 時も `workers` をそのまま使用、`param_snapshot_fn` の導入、関数インジェクション）
  - 新規: `src/engine/ui/parameters/snapshot.py`（SnapshotRuntime と `apply_param_snapshot`）

## リスク / 留意点
- 形状呼び出し回数（index）がフレーム間で変動するスケッチでは、GUI とワーカーで index がずれる可能性（初期トレースと実行経路が乖離）。既知の制約として明記。
- スナップショットのサイズ増大: override のみ送付で抑制。必要に応じ圧縮や差分転送も検討可能。
- 関数の picklability: インジェクトする関数は必ずトップレベル定義にする（`spawn` 互換）。

## 検証計画（抜粋）
- 単体/限定チェック（変更ファイル優先）
  - `ruff check --fix {changed}`、`black {changed} && isort {changed}`、`mypy {changed}`
  - `pytest -q -m smoke` と Worker 経路の最小動作確認
- 手動確認
  - `use_parameter_gui=True, workers>0` で起動し、プロセスが複数起動することを確認。
  - GUI スライダ操作でワーカー側の描画に反映されることを確認。
  - キャッシュ HIT/MISS/HUD の表示が従来通りであること。

## 実装チェックリスト（要確認）
- [x] `RenderTask` に `param_overrides` を追加（None 許容、型: Mapping[str, Any]）。
- [x] 新規 `SnapshotRuntime` と `apply_param_snapshot()` を追加（トップレベル関数、picklable）。
- [x] `_WorkerProcess.run()` に `apply_param_snapshot(task.param_overrides, task.t)` を追加。
- [x] `ParameterManager` に「override だけのスナップショット」を返す関数を追加（実装は `snapshot.extract_overrides(store)` を利用）。
- [x] `run_sketch()` を更新（GUI 有効時でも並列を維持、`user_draw` を直接渡し、`param_snapshot_fn` を組み込み）。
- [x] `architecture.md` の該当節を更新（GUI と並列の両立、SnapshotRuntime の注記）。
- [ ] 変更ファイルに対して ruff/mypy/pytest を実行し緑化（mypy は未導入のため未実施）。

## DoD（完了の定義）
- GUI 有効時に `workers>=1` で並列実行が動作し、パラメータの GUI 操作がワーカー側に反映される。
- 署名生成/キャッシュ挙動が従来と一致。
- 各チェック（ruff/black/isort/mypy/pytest）が成功し、影響ドキュメント（architecture.md）を更新済み。

---

承認が得られ次第、このチェックリストに沿って最小差分で実装します。懸念点や要望（例: index ずれの扱い、override 全量送付か差分か等）があればコメントください。
