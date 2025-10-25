パラメータGUI: パイプライン識別子（pipeline_uid）導入 実装チェックリスト

目的
- 複数の `Pipeline` インスタンスで同名エフェクト・同一ステップが並存した場合でも、Parameter GUI の Descriptor ID 衝突を回避し、GUI 操作をパイプライン単位で分離する。

スコープ
- API 層（`src/api/effects.py`）の Pipeline/Builder
- パラメータランタイム層（`src/engine/ui/parameters/`）の runtime, value_resolver（コンテキスト）
- 永続化（`persistence.py`）のキー移行（必要に応じて）
- スタブ自動生成（必要に応じて）とテスト

実装タスク
- [x] PipelineBuilder に `label(uid: str) -> PipelineBuilder` を追加（任意設定）
  - [x] 受け取った `uid` を内部保持（例: `_uid: str | None`）。
  - [x] 既存のビルドフローに影響しないことを確認。
- [x] 自動 UID 付与の仕組みを追加
  - [x] `PipelineBuilder.build()`（または `Pipeline.__init__`）で、`_uid` 未設定時に自動 UID（連番 `p0`, `p1`, ...）を採番。
  - [x] 実行スレッド/プロセス単位で衝突しないシンプルな連番カウンタ（モジュールスコープの整数 + Lock）を用意。
- [x] Pipeline に `pipeline_uid: str` を保持
  - [x] `PipelineBuilder._ensure_pipeline()` で `Pipeline` 作成時に UID を注入。
  - [x] `Pipeline.__repr__` に `pipeline_uid` を含める（開発支援、任意）。
- [x] ランタイムへの UID 伝播
  - [x] `Pipeline.__call__` → `runtime.before_effect_call(...)` 呼び出しに `pipeline_uid` を新規引数として渡す。
  - [x] 呼び出し元がランタイム非介在の経路（`resolve_without_runtime`）でも動作に影響しないことを確認。
- [x] ParameterRuntime/ParameterContext の拡張
  - [x] `ParameterRuntime.before_effect_call(..., pipeline_uid: str, ...)` にシグネチャ拡張。
  - [x] `ParameterContext` に `pipeline: str` フィールドを追加。
  - [x] `ParameterContext.descriptor_prefix` を `f"effect@{self.pipeline}.{self.name}#{self.index}"` に変更。
- [x] ValueResolver 側の ID 生成が新 prefix に追随することを確認
  - [x] `descriptor_id = f"{context.descriptor_prefix}.{key}"` はそのまま利用。
- [x] 既存の shape 側（`before_shape_call`）は変更しない（影響範囲最小化）。

永続化と互換
- [x] 既存保存データ（`effect.<name>#<idx>.*`）はそのままでは一致しないため、移行方針を決定:
  - [x] 最小案: 自動移行無し（既存 override は無視される）。
  - [ ] 任意案: `load_overrides()` で旧キーを検出したら、最初に出現した該当 Descriptor へ 1 度だけ移譲（ドキュメント化）。

スタブ/ドキュメント/テスト
- [x] 説明ドキュメント更新（本件の調査メモへ方針Bを確定記載済み）。
  - `docs/investigations/parameter_gui_pipeline_effect_id_collision.md`
- [ ] 生成スタブ（`tools/gen_g_stubs.py`）の検討:
  - [ ] `PipelineBuilder.label()` を Protocol に追加するかを判断（IDE補完目的。テストは未要求だが追加可）。
- [ ] テスト追加
  - [ ] 2 パイプライン（同一構成）で `affine`/`fill` の Descriptor ID が衝突しない（ID 生成を比較）。
  - [ ] 片方の GUI override が他方へ影響しない（`ParameterStore` 経由の適用確認）。
  - [ ] 任意: `.label("base")` と `.label("text")` を与えた場合、ID に反映される。

作業順序（推奨）
1) effects 層（Builder/Pipeline）の UID 付与/伝播を実装
2) runtime/context 層の拡張 → 動作確認（スケッチを軽く実行）
3) persistence の移行方針を選定・必要なら実装
4) スタブ生成の更新（必要なら）
5) テスト追加

高速チェック（変更ファイル限定）
- 例コマンド:
  - ruff: `ruff check --fix src/api/effects.py src/engine/ui/parameters/runtime.py src/engine/ui/parameters/value_resolver.py`
  - format: `black {changed} && isort {changed}`
  - mypy: `mypy {changed}`
  - テスト最小: `pytest -q -k parameters` または `pytest -q tests/ui/parameters/test_runtime.py`

備考
- 破壊的変更（Descriptor ID 体系）であるため、`docs` に互換性メモを残し、初回のみ override が空になる可能性を周知する。
