# Parameter GUI: チェーン順（パイプライン呼び出し順）に揃える計画（汎用）

目的（Why）
- 任意のスケッチで、ユーザーが `E.pipeline...` をチェーンした順番（左→右の宣言順）どおりに、Parameter GUI のエフェクト項目を表示する。
- 複数パイプラインが同時に存在する場合でも、パイプライン単位でグルーピングし、各グループ内はチェーン順で並べる。
- 既存仕様（「未指定=既定値採用」の引数のみ GUI に表示、RangeHint の扱い、cc は `api.cc` に閉じる）を維持。

非目標（Out of scope）
- パラメータ ID 体系（例: `effect.displace#0.amplitude_mm`）を変更しない。
- キャッシュ鍵生成やパイプライン実行ロジックには手を入れない。
- 量子化や `__param_meta__` の既存仕様は不変（署名生成のみ量子化）。

現状把握（関連ファイル）
- `src/api/effects.py`: `Pipeline.__call__` 内で `runtime.before_effect_call(...)` を呼ぶ。`pipeline_uid` は `ParameterRuntime.next_pipeline_uid()` から供給。
- `src/engine/ui/parameters/runtime.py`: `ParameterRuntime/ParameterContext`。`before_effect_call` で `step_index` と `pipeline_uid` を `ParameterValueResolver` に渡す。
- `src/engine/ui/parameters/value_resolver.py`: GUI 登録（`ParameterDescriptor` 生成と `ParameterStore.register`）。Descriptor の `id` は `effect@{pipeline}.{name}#{index}.{key}`（pipeline 無しは `effect.{name}#{index}.{key}`）。カテゴリは effect の場合 `pipeline_uid`（無ければ `effect`）。
- `src/engine/ui/parameters/dpg_window.py`: GUI 構築。現状 `_build_grouped_table()` の並び替えで呼び出し順が失われうる（`sorted(..., key=(category,id))`）。
  

アプローチ（安定ソートのための明示メタ導入）
1) Descriptor に並び順メタを追加（後方互換の optional）
   - `src/engine/ui/parameters/state.py` の `ParameterDescriptor` にフィールドを追加:
     - `pipeline_uid: str | None`（effect のみ設定）
     - `step_index: int | None`（パイプライン内でのエフェクト呼び出し順。0 始まり）
     - `param_order: int | None`（関数シグネチャ内のパラメータ出現順。必要なら）
   - 既存の `id`/`category`/保存形式は維持して互換確保。

2) Resolver で並び順メタを埋める
   - `src/engine/ui/parameters/value_resolver.py` で `ParameterDescriptor` 生成時に、`ParameterContext` から `pipeline_uid` と `step_index` を設定。
   - `param_order` は signature 列挙順 0..N を割り当て（必要に応じて）。

3) GUI のソート/グルーピング更新
   - `src/engine/ui/parameters/dpg_window.py`:
     - 初期の一律ソート（`id`/`category` 基準）をやめ、グループ順は「最初に出現した順」を保持。
     - グループは従来どおり `category`（= `pipeline_uid`）でまとめる。複数パイプラインがある場合、登録順（ランタイムで最初に観測された順）に `p0`, `p1`, ... の順で表示。
     - effect グループ内の並びは `step_index` → `param_order` → `id` の優先で安定ソート。
     - shape/その他グループは従来どおり（アルファ or 既存順）。

4) 代替（最小変更フォールバック）
   - もし Descriptor 拡張が難しい場合、`desc.id` から `#{index}` をパースして整数化、`category` ごとに安定ソート。ただし文字列依存で脆いので非推奨。まずは 1)～3) を採用。

検証方針（編集ファイル優先の高速ループ）
- 手動確認（推奨）
  - 任意のスケッチ（例: `sketch/251101.py`、`sketch/251107.py` など）を起動。
  - Parameter GUI の Effect セクションで、各パイプラインのグループ内の並びが、ユーザーコードのチェーン宣言順と一致することを確認。
- 単体テスト（任意）
  - `tests/ui/parameters/` に簡易テストを追加し、2 本の `Pipeline` を構成して `store.descriptors()` をフィルタ、`step_index` 順に並んでいることを検証。
- 最小チェック
  - 変更ファイルに対し `ruff check --fix {path}` / `black {path} && isort {path}` / `mypy {path}` を実行。

影響と互換性
- `ParameterDescriptor` の optional フィールド追加は後方互換。既存保存形式・復元・ID・スタブに影響なし。
- `persistence.py` のキーや `id` は変更しないため、保存済み設定との互換性を保つ。

作業チェックリスト（承認後に実施）
- [ ] 現状の GUI 並び順を任意のスケッチで事前スナップショット（目視/スクショ）
- [x] `ParameterDescriptor` に `pipeline_uid`/`step_index`/`param_order` を追加（`src/engine/ui/parameters/state.py`）
- [x] `value_resolver.py` で生成時に各メタを設定（scalar/vector/passthrough すべて）
- [x] `dpg_window.py` でグルーピング順=出現順、effect 内= `step_index` 優先のソートに変更（初期 `mount` の id ソート撤廃含む）
- [x] 変更ファイルの `ruff/black/isort/mypy`（ファイル単位）を通す
- [ ] 手動確認：複数スケッチで `p0`/`p1` ... の並びがチェーン順に一致
- [ ] （任意）テスト追加：effect 並び順の単体テスト（2 パイプラインを含む）
- [ ] 必要に応じて `architecture.md` に Parameter GUI の並び順仕様を追記

参考/関連
- 類似計画: `docs/plans/parameter_gui_effect_order.md`（一般化計画）。本計画は全スケッチ対象の実装手順と検証観点を具体化。

承認後の実装対象ファイル（予定）
- `src/engine/ui/parameters/state.py`
- `src/engine/ui/parameters/value_resolver.py`
- `src/engine/ui/parameters/dpg_window.py`

以上。問題なければ本チェックリストに沿って実装に着手します。
