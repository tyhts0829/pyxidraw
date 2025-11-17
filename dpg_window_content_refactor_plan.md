# ParameterWindowContentBuilder 実装改善計画

目的: `src/engine/ui/parameters/dpg_window_content.py` の責務を整理し、「何をしているか」が読み取りやすい構造にしつつ、挙動と外部 API（`ParameterWindow` から見た使い方）を変えない。

## 現状の課題整理（要約）

- 1 クラスに Display/HUD・カテゴリグルーピング・テーブル構築・型別ウィジェット生成・CC バインディング・Store 同期がすべて詰め込まれており、メソッド数と分岐が多い。
- Dear PyGui の低レベル API を型/レイアウトごとに手書きしているため、似たようなテーブル構築コードが繰り返されている。
- レイアウト計算（比率・padding）とウィジェット生成が同じメソッド内で混在しており、「何を描いているか」が一見で追いにくい。
- エラー処理・後方互換ロジックもこの層で行っており、純粋な UI ロジックと guard ロジックが絡み合っている。

## 改善方針（設計レベル）

1. **責務ごとのサブルーチン/小クラスへの分割（ただしモジュールは据え置き）**
   - モジュールは現状のまま `dpg_window_content.py` に保ちつつ、クラス/メソッドの役割を分割する。
   - 目標は「Display/HUD」「カテゴリテーブル」「型別ウィジェット」「CC ビュー」「Store 同期」がそれぞれ見通せる粒度に整理されている状態。

2. **レイアウトとウィジェット生成の分離**
   - テーブル/行/列の構造（レイアウト）と、個々の DPG ウィジェット追加（ウィジェット生成）を分ける。
   - 例えば、`_create_row_3cols` は「行の枠（Label/Bars/CC の3セル）を用意して、各セルに対してコールバックを呼ぶ」程度に薄くし、その中で `_build_label_cell` / `_build_bars_cell` / `_build_cc_cell` に分ける。

3. **共通パターンのヘルパー化**
   - 「padding 付きの 1 列テーブル」「CellPadding 付きのサブテーブル」など、繰り返し発生する DPG テーブルパターンを小さなヘルパに切り出す。
   - 型別ウィジェット生成でも、`RangeHint`/`VectorRangeHint` から min/max を計算する部分や `value_precision` の組み立てをユーティリティ関数に寄せることで、各 `_create_*` メソッドを短くする。

4. **Store 同期ロジックの局所化**
   - `_on_widget_change` / `on_store_change` / `store_rgb01` / `_safe_norm` は、概念的には「値同期」層なので、レイアウト構築とは切り離して読めるように並び順と構造を整理する（モジュール内でセクションを分ける）。

5. **挙動は変えない（互換性重視）**
   - 外部 API（`ParameterWindow` から見た `build_root_window` / `mount_descriptors` / `on_store_change` など）の署名と意味はそのまま維持する。
   - Descriptor の解釈（`category_kind` や vector スライダ数、CC の扱いなど）は変えない。

## 具体的なリファクタリングタスク

### 1. Display/HUD ビューの整理

- [ ] `build_display_controls` を「Display セクション」「HUD セクション」の2ブロックに明確に分割し、それぞれを private ヘルパに切り出す。
  - 候補: `_build_display_section(parent: int | str)` / `_build_hud_section(parent: int | str)`
- [ ] HUD セクション内でも、`Show HUD` トグル行とカラー系行（Text/Meter/Meter BG）をヘルパに分離する。
  - 例: `_build_hud_toggle_row(table_parent)` / `_build_hud_color_rows(table_parent)`

### 2. カテゴリテーブル構築の整理

- [ ] `_build_grouped_table` の中で行っている
  - Excluded ID フィルタ
  - `(category_kind, category)` ごとのグルーピング
  - effect グループ用のソート
  をそれぞれ小さなヘルパ関数に分解する。
  - 例: `_filter_descriptors_for_table`, `_group_descriptors_by_category`, `_sort_group_items`
- [ ] `_flush_group` に「テーマ適用」「テーブル構築」「行追加」の3段階が混ざっているので、1 段階ずつ別ヘルパに切り出す。
  - 例: `_apply_category_themes(header, kind)`, `_create_category_table(parent, kind)`, `_populate_category_rows(table, items)`

### 3. 型別ウィジェット生成の整理

- [ ] `_create_bars` 内の vector/scalar 分岐を、vector 専用ヘルパと scalar 専用ヘルパに分ける。
  - `_create_vector_bars(parent, desc, value)` / `_create_scalar_bars(parent, desc, value, hint)`
- [ ] `_create_widget` も「型ごとのディスパッチ」だけを行い、実際の DPG 呼び出しは `_create_bool_widget` / `_create_enum_widget` / `_create_string_widget` / `_create_vector_widget` などに分ける。
- [ ] `_create_int` / `_create_float` と CC 列の構造がよく似ているので、「左にスライダ、右に CC 入力を持つ 2 列テーブル」構築ヘルパを切り出す。

### 4. CC ビューと Store 同期の整理

- [ ] `_create_cc_inputs` 内の vector/scalar 分岐を、それぞれ `_create_vector_cc_inputs` / `_create_scalar_cc_inputs` に分ける。
- [ ] `_add_cc_binding_input` / `_add_cc_binding_input_component` に共通するロジック（`default_text` の決定、`kwargs` 組み立て）を内部ヘルパにまとめ、メソッド本体を短くする。
- [ ] `on_store_change` 内の分岐（Display/HUD カラー / style.color / vector）を、それぞれ専用の更新ヘルパに分けて、ループ内の if/elif を浅くする。

### 5. セクション別の並び替えとコメント整理

- [ ] クラス内のメソッド定義順を「公開インタフェース → Display/HUD → テーブル構築 → 型別ウィジェット → CC → 同期/ヘルパ」の順に並べ替え、docstring もこの構造に合わせる。
- [ ] フローが直感的でない箇所（例えば vector ブラウザ用の dim 補正、`category_kind` のフォールバック）には短い説明コメントを追加する（既存コメントのトーンに合わせる）。

## 検証方針

- [ ] リファクタリング後も、以下の観点で動作確認を行う:
  - 各型（bool/enum/string/vector/int/float/style.color）のウィジェットが以前と同じ見た目・挙動であること。
  - `Show HUD` トグルが `runner.show_hud` と正しく連携していること。
  - カテゴリヘッダのグルーピングと順序（shape/effect/HUD/Display）が変わっていないこと。
- [ ] 型チェックと Lint:
  - `pyright src/engine/ui/parameters/dpg_window_content.py`
  - `mypy src/engine/ui/parameters/dpg_window_content.py`
  - `ruff check src/engine/ui/parameters/dpg_window_content.py`

## 今後の拡張候補（今回のスコープ外）

- ParameterWindowContentBuilder を「形状パラメータビュー」と「Effects パラメータビュー」に分割し、カテゴリごとのビュークラスを持つ構成にする。
- DPG 依存を薄めて、将来別 GUI バックエンド（例: Qt-based parameter view）とインタフェースを共有できるようにする（backends パッケージ化）。

---

この計画に沿って進めれば、挙動を変えずに `dpg_window_content.py` の読みやすさと保守性を上げられるはずです。実装着手前に、分割の粒度（例えば CC 部分を別クラスにするかどうか）について希望があればここに追記します。 

