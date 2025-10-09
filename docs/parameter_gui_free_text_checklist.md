# Parameter GUI: 自由文字列入力対応（チェックリスト）

目的

- 未指定（既定値採用）の文字列引数（例: `text`）を GUI から自由入力できるようにする。
- 既存の数値/ベクトル/列挙（choices あり）への影響は最小に保つ。

現状の問題（要約）

- 文字列は `ValueType` に明示的な `string` 型がなく、`enum` と同一視されている。
- `choices` がない文字列は GUI 非対応（supported=False）となり、ウィジェットが生成されない。
  - 判定: `src/engine/ui/parameters/value_resolver.py:222-238`
  - 表示フィルタ: `src/engine/ui/parameters/dpg_window.py:172-206`
- そのため、`G.text()` で `text` を未指定にしても、GUI に入力欄が出ない。

提案仕様（最終像）

- `ValueType` に `"string"` を追加する。
- 値種別の判定:
  - `param_meta.type == "string"` → `string`
  - `choices` がある → `enum`
  - 上記以外で値が `str` → `string`
- `string` は GUI 対応（supported=True）。Dear PyGui は `input_text` を用いる。
- `enum`（choices あり）は従来通り、ラジオ/コンボを使用する。
- 複数行入力は要件に応じて選択（下記「確認事項」参照）。
- 永続化/スナップショット適用は `enum` と同様に文字列を保存・復元する。

変更対象ファイル（想定）

- `src/engine/ui/parameters/state.py`
  - `ValueType` に `"string"` を追加。
- `src/engine/ui/parameters/value_resolver.py`
  - `_determine_value_type()` で `string` を正しく判定。
  - `_resolve_passthrough()` で `string` を supported=True として `ParameterDescriptor` 登録。
- `src/engine/ui/parameters/dpg_window.py`
  - `_create_widget()` に `vt == "string"` 分岐を追加し、`dpg.add_input_text(...)` を利用。
  - 既存の `_on_widget_change()` は `app_data` をそのまま `set_override()` へ渡すため流用可能。
- `src/engine/ui/parameters/persistence.py`
  - `save_overrides()/load_overrides()` に `string` 型の保存/復元分岐を追加。
- （任意）`src/shapes/text.py`
  - `__param_meta__` に `"text": {"type": "string"}` を追加（明示化）。

実装手順（チェックリスト）

- [x] `state.py` の `ValueType` に `"string"` を追加する。
- [x] `value_resolver.py` の `_determine_value_type()` を以下に変更: - `meta.type == "string"` → `"string"` - `"choices" in meta` → `"enum"` - それ以外で `default_value` もしくは `raw_value` が `str` → `"string"`
- [x] `value_resolver.py` の `_resolve_passthrough()` を更新: - `supported` 判定に `"string"` を含める。 - `choices_list` が無い `string` でも `ParameterDescriptor` を登録する。
- [x] `dpg_window.py` の `_create_widget()` に `string` 用ウィジェットを追加: - `dpg.add_input_text(tag=desc.id, default_value=str(value) or "", callback=_on_widget_change, user_data=desc.id, multiline=<設定>)` - `multiline` の扱いは「確認事項」に従う。
- [x] `persistence.py` を更新: - `save_overrides()` は `string` をそのまま保存（JSON）。 - `load_overrides()` は `string` を `str(value)` として復元。
- [x] （任意）`src/shapes/text.py` の `__param_meta__` に `text` の型を明示（併せて font は GUI 非表示に設定）。
- [x] 変更ファイルに対するローカル検証（ruff/mypy/pytest 一部）を実行。
- [x] `architecture.md` に小さな差分（ValueType に `string` 追加）を反映。

検証計画（編集ファイル優先）

- Lint/Format/Type（対象ファイルのみ）
  - `ruff check --fix src/engine/ui/parameters/state.py src/engine/ui/parameters/value_resolver.py src/engine/ui/parameters/dpg_window.py src/engine/ui/parameters/persistence.py`
  - `black src/engine/ui/parameters/state.py src/engine/ui/parameters/value_resolver.py src/engine/ui/parameters/dpg_window.py src/engine/ui/parameters/persistence.py`
  - `isort src/engine/ui/parameters/state.py src/engine/ui/parameters/value_resolver.py src/engine/ui/parameters/dpg_window.py src/engine/ui/parameters/persistence.py`
  - `mypy src/engine/ui/parameters/state.py src/engine/ui/parameters/value_resolver.py src/engine/ui/parameters/dpg_window.py src/engine/ui/parameters/persistence.py`
- 手動動作確認
  - `python main.py`（`use_parameter_gui=True` 既定）を実行。
  - `G.text(...)` の `text` を未指定のまま GUI に自由入力欄が出ること。
  - 入力変更が即時描画に反映されること。
  - 終了後に `data/gui/<script>.json` が保存され、再起動で復元されること。

確認事項（要回答）

- 自由入力欄は「複数行（multiline）」にしますか？（改行を含む文を入力可能）
  - 例: `dpg.add_input_text(..., multiline=True, height=80)` を採用。
  - もしくは単一行で最小 UI とし、将来要望時に拡張しますか？：こちらを採用
- `font` 引数（`text.py`）も自由入力欄として表示してよいですか？（現状 meta は `{"type": "string"}` 相当）：No。font は自由入力にしない。
- 表示カテゴリは `shape` のままで問題ありませんか？（今の分類ルール踏襲）：はい。

リスク/影響

- `ValueType` の追加に伴う型分岐の取りこぼし（persistence/snapshot など）
- DPG 未導入/ヘッドレス環境ではウィンドウはダミー（既存と同じフォールバック）。

備考

- 実装はシンプルさを優先し、`string` は最小限の自由入力のみを提供（バリデーションや補完は行わない）。
- 既存の `enum`（choices あり）は従来通りの UI（ラジオ/コンボ）を維持する。
