Parameter GUI: 形状/パイプライン単位のグルーピング計画（提案）

目的
- 現状の GUI は `shape` グループ配下に全ジオメトリ、`effect` グループ配下に全エフェクトのパラメータが並ぶ。
- これを、スケッチ内の「実体」ごとにグルーピングする。
  - 例（sketch/251022.py）: `sphere` グループ、`text` グループ、`e_pipe1` グループ、`e_pipe2` グループ。

現状の整理（コード確認）
- グループ見出しは `ParameterDescriptor.category` を Dear PyGui 側でヘッダ表示。
  - src/engine/ui/parameters/dpg_window.py:534（`collapsing_header(label=category)`）
- `category` は Resolver が Descriptor 登録時に付与しており、現状は `context.scope`（= `shape` または `effect`）。
  - src/engine/ui/parameters/value_resolver.py:135, 173, 210（いずれも `category=context.scope`）
- パイプライン UID は導入済みで、ランタイムから `ParameterContext.pipeline` に渡る。
  - runtime → resolver まで `pipeline_uid` が伝播（`effect@{uid}.name#index.param` のID生成）。

要件定義
- 形状（shape）: `category` を形状関数名（例: `sphere`, `text`）にする。
  - 単一スケッチ内で同名 shape を複数回呼ぶケースでは、同一グループ（`sphere`）にまとまる。
  - 将来的に「インスタンス別に分けたい」場合に備え、切替可能な設計（config）を検討（今回は不要）。
- エフェクト（effect）: `category` をパイプライン UID（例: `e_pipe1`, `e_pipe2`）にする。
  - 明示ラベルが無い場合はランタイムの自動 UID（`p0`, `p1`, ...）を用いる。
  - 同一パイプライン内の複数ステップ（affine/fill など）は同一グループにまとまる。
- 表示順序: 既存の `Display/HUD` → shape 群 → pipeline 群。現行ソート（`sorted(filtered, key=(category, id))`）で概ね自然な並びになる見込み。
- 永続化（persistence）: 影響なし（保存キーは `id` のみで `category` は関与しない）。

設計方針（最小変更）
1) `ParameterValueResolver` の Descriptor 生成時に `category` を計算する関数を導入。
   - `scope == "shape"` のとき: `category = context.name`（例: `sphere`, `text`）。
   - `scope == "effect"` のとき: `category = context.pipeline or "effect"`（pipeline UID があればそれ、無ければ従来フォールバック）。
   - `label` は従来通り `f"{context.label_prefix}: {param_name}"`（例: `affine#0: scale`）。
2) Dear PyGui 側はそのまま（`category` によるグループ化）。
3) ランタイム/スナップショットは既に `pipeline_uid` を提供しているため変更不要。
4) コンフィグ（任意）:
   - `parameter_gui.layout.grouping_mode` を将来オプションとして受ける余地（`by_kind` / `by_entity`）。本実装では固定 `by_entity` を採用し、設定導入は見送る。

影響/互換性
- GUI の表示グループが変更される（視覚上の改善）。
- Descriptor ID は不変（前回変更済みの `effect@{uid}.name#index.param` 仕様を維持）。
- 保存データ（overrides）の互換性に影響なし。

実装ステップ（チェックリスト）
- [x] Resolver: `category` 算出の導入
  - [x] shape: `context.name`
  - [x] effect: `context.pipeline or "effect"`
- [x] Resolver: `_resolve_scalar/_resolve_vector/_resolve_passthrough` で `category` を新ルールに差し替え
- [x] 動作確認（静的）: 既存ユニットテストが通る（カテゴリ変更はテスト非依存）
  - [x] `tests/ui/parameters/test_runtime.py`（Descriptor ID 依存のみ）
  - [x] `tests/ui/parameters/test_value_resolver.py`（ID/Range のみ）
- [ ] 任意: スケッチ実行で GUI 上のグループ表示を目視確認（`sphere/text/e_pipe1/e_pipe2`）
  - [ ] `.label("e_pipe1")`, `.label("e_pipe2")` 指定で見出しに反映
  - [ ] 未指定時は `p0`, `p1` 等の自動 UID が見出しになる

実装詳細（差分見込み）
- ファイル: src/engine/ui/parameters/value_resolver.py
  - `ParameterContext` は前回拡張済み（`pipeline` 保持）。
  - `ParameterDescriptor(...)` の `category` 引数を `context.scope` から `_category_for(context)` に置換。

将来拡張（メモ）
- shape のインスタンス単位グルーピング（`sphere#0`, `sphere#1`）へ切替可能なオプション。
- パイプラインのグループ表示順を `PipelineBuilder.label_order()` のようなAPIで制御。
- UI でのグループ折り畳み状態の永続化。

以上。
