# Parameter GUI: 文字列入力の複数行対応（拡張）チェックリスト（明示制御）

目的
- 既存の文字列パラメータ（ValueType: `string`）について、GUI 入力を単一行から複数行（テキストエリア）に拡張する。
- 既存の数値/ベクトル/列挙への影響は最小化する。

設計（明示制御の方針）
- 既定は単一行（multiline=False）。
- `__param_meta__` に以下のキーがある場合のみ複数行にする。
  - `type: "string"`（既存）
  - `multiline: true`（新規）
  - `height: <int>`（任意。DPG テキストエリアの高さ。省略時は 80）
- 例: `{"type": "string", "multiline": true, "height": 80}`
- フォント `font` は引き続き GUI 非表示（choices 空）を維持。

実装対象（仕様）
- `src/engine/ui/parameters/state.py`
  - `ParameterDescriptor` に UI ヒントを追加: `string_multiline: bool = False`, `string_height: int | None = None`。
- `src/engine/ui/parameters/value_resolver.py`
  - passthrough 解決時に `meta_entry` を参照し、上記ヒントを `ParameterDescriptor` に設定。
    - `multiline: true/false`、`height: int` を解釈。
- `src/engine/ui/parameters/dpg_window.py`
  - `vt == "string"` で `desc.string_multiline` が True の場合、`dpg.add_input_text(..., multiline=True, height=desc.string_height or 80)` を使用。
  - False の場合は単一行（現状の input_text）を維持。
- 永続化/スナップショット
  - 現状のままで変更不要（文字列はそのまま保存/復元）。
- ドキュメント
  - `architecture.md` の文字列入力の説明に「`__param_meta__.multiline/height` で明示制御」を追記。

作業ステップ
- [x] `state.py` の `ParameterDescriptor` に `string_multiline`/`string_height` を追加。
- [x] `value_resolver.py` で `meta_entry` からフラグを抽出し、`ParameterDescriptor` にセット。
- [x] `dpg_window.py` で `desc.string_multiline` に応じて input_text の `multiline/height` を切替。
- [x] `architecture.md` を更新（明示制御の仕様を追記）。
- [x] Lint/Format/Type（編集ファイル限定）を実行。
- [x] 対象テストの追加/更新（必要最小限）。

検証計画（編集ファイル優先）
- `ruff check --fix {changed}` / `black {changed}` / `isort {changed}` / `mypy {changed}` を実行。
- `pytest -q tests/ui/parameters/test_persistence.py`（文字列の保存/復元に回帰がないこと）
- 任意の手動確認:
  - `__param_meta__` 未指定 → 単一行表示。
  - `{"type": "string", "multiline": true}` → 複数行表示、高さ既定 80。
  - `{"type": "string", "multiline": true, "height": 120}` → 複数行表示、高さ 120。

リスク/備考
- 全 `string` を複数行化するため、単一行で十分なケースでもテキストエリア表示になる（UI の見た目が大きくなる）。
- 必要に応じて将来 `__param_meta__` で `multiline` の明示制御に拡張可。
