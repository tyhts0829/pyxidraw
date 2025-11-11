# Parameter GUI: Shape/Pipeline カテゴリ背景色の独立制御 計画

目的: Parameter GUI における「shape のカテゴリ」と「pipeline（effects）のカテゴリ」の背景色を独立して制御できるようにする。既存テーマの互換を維持しつつ、構成可能性を最小限の追加で拡張する。

## 現状と課題

- 現状は `parameter_gui.theme.colors.header*`（グローバル）で全カテゴリの見出し背景色が一律に適用される。
- カテゴリは `src/engine/ui/parameters/dpg_window.py` 内 `_build_grouped_table()` で `dpg.collapsing_header` として構築され、shape/effect の混在にかかわらず同一テーマが適用される。
- 要件: shape 由来のカテゴリと pipeline/effect 由来のカテゴリで別々の見た目（少なくとも Header 背景）を設定できるようにする。

## 仕様（提案）

- 設定は既存 `parameter_gui.theme` の下に新設キー `categories` を追加（後方互換: 未指定時は従来の `colors.header*` を適用）。
- キー構造（例）:
  - `parameter_gui.theme.categories.shape.header`
  - `parameter_gui.theme.categories.shape.header_hovered`
  - `parameter_gui.theme.categories.shape.header_active`
  - `parameter_gui.theme.categories.pipeline.header`
  - `parameter_gui.theme.categories.pipeline.header_hovered`
  - `parameter_gui.theme.categories.pipeline.header_active`
  - 追加（任意/将来拡張）: `table_row_bg`, `table_row_bg_alt`（可能ならカテゴリー配下テーブルの行背景にも適用）
- 値の形式: 既存テーマと同じ（RGBA 0..255 / 0..1 / #hex を許容、内部で u8 RGBA に正規化）。
- 適用対象:
  - `dpg.collapsing_header` に対して `mvThemeCol_Header`, `mvThemeCol_HeaderHovered`, `mvThemeCol_HeaderActive` をバインド。
  - 行背景は Dear PyGui が提供する場合のみ `mvThemeCol_TableRowBg`, `mvThemeCol_TableRowBgAlt` をテーブル側にバインド（存在しない場合は無視）。
- 選択ロジック:
  - カテゴリ内の `ParameterDescriptor` に `source=="effect"` が 1 つでもあれば pipeline カテゴリとして扱う。それ以外は shape カテゴリ。
  - 両者混在時は pipeline 優先（より目立たせる想定）。

## 実装方針

- データモデル最小拡張:
  - `src/engine/ui/parameters/state.py:ParameterThemeConfig` に `categories: dict[str, dict[str, any]] = {}` を追加（任意キー辞書として保持）。
  - 型の厳密化は避け、キー存在チェックでフェイルソフトに適用。
- テーマ構築:
  - `src/engine/ui/parameters/dpg_window.py` にカテゴリ専用テーマを生成するヘルパを追加し、インスタンス内にキャッシュ（例: `_cat_theme_shape`, `_cat_theme_pipeline`, `_cat_table_theme_shape`, `_cat_table_theme_pipeline`）。
  - `_build_grouped_table().flush(...)` 内で `collapsing_header` 作成直後に `dpg.bind_item_theme(...)` を実行。
  - 既存グローバルテーマはそのまま維持。カテゴリテーマは「ヘッダ/テーブル」に限定し、その他の色/スタイルはグローバルに委譲。
- コンフィグ読込:
  - `src/engine/ui/parameters/manager.py` で `parameter_gui.theme.categories` を取り出し、`ParameterThemeConfig(categories=...)` に受け渡し。
  - 既存の `style/colors` と同等の扱いで、未指定は空辞書。
- フォールバックと安全性:
  - カテゴリごとのキーが無ければ何も作らず（バインドしない）。
  - DPG 側に該当 `mvThemeCol_*` が無い場合はスキップ（ログは debug か無音）。
  - 例外は握りつぶしつつ継続（GUI 全体の動作を阻害しない）。

## 変更ファイル（予定）

- src/engine/ui/parameters/state.py
- src/engine/ui/parameters/manager.py
- src/engine/ui/parameters/dpg_window.py
- configs/default.yaml（サンプル設定の追記）
- architecture.md（Parameter GUI の外観設定の説明を更新）

## 受け入れ条件（DoD）

- 未設定時は従来どおり（グローバル `header*` の色）で表示される。
- `theme.categories.shape.header*` のみ指定で shape カテゴリのヘッダ色が変わる（pipeline は従来色）。
- `theme.categories.pipeline.header*` のみ指定で pipeline カテゴリのヘッダ色が変わる（shape は従来色）。
- DPG 未導入環境でも例外なく import/生成が行える（スキップやフェイルソフトを維持）。

## 実装タスクリスト（チェックリスト）

- [x] state: `ParameterThemeConfig` に `categories` フィールド追加（デフォルト `{}`）。
- [x] manager: YAML から `theme.categories` を読み込み `ParameterThemeConfig(categories=...)` に渡す。
- [x] dpg_window: カテゴリテーマ生成ヘルパを追加し、`collapsing_header` とテーブルへテーマを条件バインド。
- [x] dpg_window: Display/HUD ヘッダにもカテゴリテーマを適用（`build_display_controls`）。
- [x] configs/default.yaml: `parameter_gui.theme.categories` のサンプル値を追記（shape/pipeline で色差が分かるもの）。
- [x] architecture.md: Parameter GUI のテーマ設定に「カテゴリ別ヘッダ色」を追記。
- [x] 変更ファイルに対して `ruff/black/isort/mypy` を通す（編集ファイル限定）。
- [ ] DPG 環境ありでの手動確認（最小）: サンプルスケッチ起動で shape/pipeline の見出し色が独立して変わること。

## 動作確認（手動）

- main 実行: `python main.py`（適当な sketch を既定動作で読み込み）。
- `config.yaml` などで以下のように試す:
  - shape だけ指定
    - `parameter_gui.theme.categories.shape.header: [90, 80, 60, 255]`
  - pipeline だけ指定
    - `parameter_gui.theme.categories.pipeline.header: [40, 70, 110, 255]`
  - hover/active も指定して反応を確認
- shape/effect が混在するカテゴリがある場合は pipeline 色が採用されることを確認（優先順位仕様）。

## 確認事項（要回答）

- 背景色の対象範囲:
  - a) 見出し（collapsing header）のみで十分か。；はい
  - b) カテゴリ配下のテーブル行背景（TableRowBg/Alt）にも色を適用するか（DPG に色定義がある場合）。いいえ。
- フォールバックの希望:
  - 未指定時は完全に「グローバル header\* にフォールバック」で良いか（現行案）。
- 将来的拡張の要否:
  - `general` 用のカテゴリ色（shape/effect 以外）や、カテゴリ名でのパターンマッチ適用（例: `Display`, `HUD`）も必要か。

---

承認いただければ、上記チェックリストに沿って実装に着手します。加筆/修正の要望があればコメントください。

## 補足: Display/HUD カテゴリ対応（追記）

- `parameter_gui.theme.categories` は `Display` と `HUD` のキーも受け付ける。
  - 例: `parameter_gui.theme.categories.Display.header`, `...Display.header_hovered`, `...Display.header_active`
  - 例: `parameter_gui.theme.categories.HUD.header`, `...HUD.header_hovered`, `...HUD.header_active`
- `Display`/`HUD` は `build_display_controls()` で作成する独立ヘッダに直に適用される（テーブル行背景も任意で適用）。
- グループ化テーブル側は従来どおり shape/pipeline の 2 種を自動判定し、`categories.shape`/`categories.pipeline` を適用する（該当キーが無ければフォールバック）。
