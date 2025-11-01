# Parameter GUI: エフェクト並び順をパイプライン呼び出し順に揃える（計画）

目的
- `sketch/251101.py` のようなスケッチで、Parameter GUI 上の「エフェクト」項目を、実行パイプライン内の呼び出し順（左から右の宣言順）に並べる。
- 既存仕様（「未指定＝既定値採用」の引数のみ GUI に表示、RangeHint の解釈、cc は `api.cc` に閉じる）を保持する。

非目標
- エフェクト/シェイプのパラメータ定義や既存の ID 体系（`effect.displace#0.amplitude_mm` 等）を変更しない。
- キャッシュ設計やパイプラインの実行ロジックには手を入れない。

関連ファイル（現状把握）
- `src/api/effects.py`: `Pipeline.__call__` で `runtime.before_effect_call(...)` を呼び出し。`pipeline_uid` を `ParameterRuntime.next_pipeline_uid()` から取得。
- `src/engine/ui/parameters/runtime.py`: `ParameterRuntime` と `ParameterContext`。`before_effect_call` で `step_index` と `pipeline_uid` を `ParameterValueResolver` に渡す。
- `src/engine/ui/parameters/value_resolver.py`: GUI 登録（`ParameterDescriptor` の生成と `ParameterStore.register`）。Descriptor の `id` は `effect@{pipeline}.{name}#{index}.{key}`（pipeline 無し時は `effect.{name}#{index}.{key}`）。カテゴリは effect の場合 `pipeline_uid`（無ければ `effect`）。
- `src/engine/ui/parameters/dpg_window.py`: GUI 構築。現在は `_build_grouped_table()` で `sorted(filtered, key=lambda d: (d.category, d.id))` としており、呼び出し順が失われる。
- `src/engine/ui/parameters/state.py`: `ParameterDescriptor` の定義・`ParameterStore`。

アプローチ（推奨: 明示メタによる安定ソート）
1) Descriptor に「並び順ヒント」を追加（後方互換、任意フィールド）
   - `ParameterDescriptor` に以下の optional フィールドを追加：
     - `pipeline_uid: str | None`（effect のみ設定）
     - `step_index: int | None`（パイプライン内でのエフェクト呼び出し順。0 始まり）
     - `param_order: int | None`（関数シグネチャ内のパラメータ出現順。必要なら）
   - 既存の `id`/`category` はそのまま維持（保存/復元や互換のため）。

2) Resolver で並び順ヒントを埋める
   - `value_resolver.py` 内で `ParameterDescriptor` 生成時に、`ParameterContext` の `pipeline` と `index` を各フィールドへ設定。
   - `param_order` は signature 列挙順で 0..N を割り振る（必要に応じて）。

3) GUI の並べ替えロジックを更新
   - `dpg_window.py`:
     - `mount()` の初期並べ替え（現状 `sorted(..., key=lambda d: d.id)`）を撤廃し、ストア返却順 or 新キーを使用。
     - `_build_grouped_table()` を、グルーピングは `category` で行いつつ、グループ順は「最初の出現順」を保持。
     - effect グループ（`desc.source == "effect"`）内は `step_index` → `param_order` → `id` の順で安定ソート。
     - 既存の `Display`/`HUD` や shape グループは従来どおり（アルファ or 既存順）で良い。

4) 代替・フォールバック（最小変更案）
   - Descriptor 拡張が難しい場合は、`desc.id` から `#{index}` をパースして `int` を抽出、`category`（= pipeline_uid）ごとに安定ソートする。
   - ただし文字列依存で脆いので非推奨。まずは 1) の実装を目指す。

検証方針（編集ファイル優先の高速ループ）
- 手動確認（推奨）
  - `sketch/251101.py` を起動（`python main.py`）。Parameter GUI の Effect セクションで、`affine → scale → translate → fill → subdivide → displace → mirror` の順に並ぶことを目視確認。
- 単体テスト（任意）
  - `tests/ui/parameters/` に簡易テストを追加し、`Pipeline` に 2-3 個の effect を既定値で並べ、`store.descriptors()` をフィルタして step 順で並んでいることを検証。
- 最小チェック
  - 変更ファイルに対して `ruff/black/isort/mypy` を通す。

影響と互換性
- Descriptor へのフィールド追加は後方互換（デフォルト値 `None`）。既存テストで `ParameterDescriptor(...)` を直接生成している箇所は、引数追加不要（デフォルトで通る）ことを確認。
- パラメータ ID/保存形式（`persistence.py`）には影響しない。

作業チェックリスト
- [ ] 現行の GUI 並び順の実測（`sketch/251101.py` で事前確認）
- [ ] `ParameterDescriptor` に `pipeline_uid`/`step_index`/`param_order` を追加（`src/engine/ui/parameters/state.py`）
- [ ] `value_resolver.py` で各生成箇所に並び順ヒントを設定（scalar/vector/passthrough）
- [ ] `dpg_window.py` の `mount()` 初期ソートを撤廃 or 新しいキーに変更
- [ ] `_build_grouped_table()` をグループ順=出現順、effect グループ内= `step_index` 優先に変更
- [ ] 変更ファイルの `ruff/black/isort/mypy`（ファイル単位）を通す
- [ ] 手動確認：`sketch/251101.py` でエフェクト順がパイプライン順に一致
- [ ] （任意）テスト追加：effect 並び順の単体テスト
- [ ] 必要に応じて `architecture.md` に GUI 並び順の扱いを追記

メモ / オープンクエスチョン
- グループラベル（現状 `category = pipeline_uid` → `p0/p1/...`）は人間に優しくない可能性。`Pipeline p0` のような表示名変換を行うかは別改善とする。
- 明示 `pipeline_uid` を `Pipeline(..., pipeline_uid="...")` で付与している場合は、その値がグループキー/表示ラベルに使われる。順序は `ParameterRuntime.next_pipeline_uid()` に依存せず、登録順を尊重する。

承認後の実装対象ファイル（予定）
- `src/engine/ui/parameters/state.py`
- `src/engine/ui/parameters/value_resolver.py`
- `src/engine/ui/parameters/dpg_window.py`

以上。ご確認ください。問題なければ実装に着手します。

