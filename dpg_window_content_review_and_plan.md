# ParameterWindowContentBuilder コードレビューと改善計画

## 対象

- ファイル: `src/engine/ui/parameters/dpg_window_content.py`
- 主な責務: ParameterStore/Descriptor から Style セクションとパラメータテーブルを構築し、Dear PyGui ウィジェットと Store を双方向同期する。

## 概要レビュー

- 良い点
  - クラス/メソッドごとに日本語 docstring と型ヒントが整備されており、役割が明確。
  - `ParameterWindowThemeManager` や `ParameterLayoutConfig` との分離ができていて、レイアウト/テーマ/状態のレイヤ分けは概ね良好。
  - `ParameterDescriptor.value_type`/`range_hint`/`vector_hint` などのメタ情報を用いて、型ごとに汎用的なウィジェット生成を行っている。
  - `store_rgb01` や `_safe_norm` によるカラー正規化、`_style_param_ids` による Style セクションの同期対象管理など、状態同期の考え方が整理されている。
- 改善余地（サマリ）
  - クラス/ファイルが肥大化しており、Style セクション構築・パラメータテーブル構築・各種ウィジェット生成・CC バインディング・同期処理が 1 クラスに詰め込まれていて読み解きにくい。
  - Dear PyGui テーブル/テーマ設定/スライダ生成/カラー編集まわりでパターンが重複しており、DRY さと一貫性の面で改善余地がある。
  - Style/HUD カラー処理やベクトル/スカラー向けウィジェット生成ロジックに重複やデッドパスがあり、将来の修正時にバグを生みやすい構造になっている。

## 詳細レビュー

### 良い点

1. **ドキュメントと型付けが整っている**
   - ファイル先頭に「どこで/何を/なぜ」のヘッダがあり、`ParameterWindowContentBuilder` の役割が一目で分かる。
   - 公開メソッドに日本語 docstring と型ヒントが付いており、外部からの利用方法が読み取りやすい。
2. **責務分離の大枠は良い**
   - テーマ/フォント管理は `ParameterWindowThemeManager` に委譲し、`ParameterWindowContentBuilder` はレイアウトと値同期に集中している。
   - Descriptor/Store/Theme/Layout の境界が比較的明確で、UI 層にビジネスロジックを持ち込んでいない。
3. **値種別に応じた汎用的なウィジェット生成**
   - `value_type`（bool/enum/string/vector/int/float）と RangeHint を組み合わせてウィジェットを切り替える構造は拡張しやすい。
   - CC バインディングやベクトル成分用のタグ命名（`param::x` など）が一貫しており、テストしやすい。

### 気になる点 / 改善余地

1. **High – クラス/ファイルの肥大化と責務混在**
   - `ParameterWindowContentBuilder` が Style セクション構築、カテゴリ別パラメータテーブル構築、各種ウィジェット生成、CC バインディング、Store との同期ユーティリティをすべて抱えており、ファイル全体で 1000 行超のボリュームになっている。
   - `build_style_controls` や `_build_grouped_table`、`_create_bars` など 1 メソッドが長大で、ローカルな文脈を追い続けないと意図が分かりにくい。
   - 「Style セクション全体の設計」や「テーブルレイアウト全体像」がコードから直感的に読み取りにくく、変更時の認知負荷が高い。
2. **Medium – DPG テーブル構築/テーマ設定コードの重複**
   - `mvStyleVar_CellPadding` を使ったセルパディング付きテーブル構築（`with dpg.table(...) as tbl: ... dpg.bind_item_theme(tbl, theme)`）が、Style テーブル、カテゴリテーブル、Bars/CC/ベクトル用のサブテーブルなど複数箇所でほぼ同じ形で繰り返されている。
   - `width_stretch=True` と `init_width_or_weight` の追加に対する `try/except TypeError` パターンも、複数メソッドに分散して書かれている。
   - `dpg.set_item_width(..., -1)` を `try/except Exception` で囲むコードが多数あり、テーブル構築ロジックの視認性を下げている。
3. **Medium – Style/HUD カラー処理ロジックの重複と散在**
   - 背景色/ライン色/HUD テキスト・メータ色などの初期値計算が `build_style_controls` 内で個別に書かれており、`_resolve_canvas_colors`・Store の override/元値取得・`_safe_norm` が混在している。
   - 同じ ID セット（`runner.background`, `runner.line_color`, `runner.hud_text_color`, `runner.hud_meter_color`, `runner.hud_meter_bg_color` など）が `build_style_controls` と `on_store_change` 両方にハードコードされ、追加・変更時に漏れやすい構造になっている。
   - `store_rgb01` と `_safe_norm` でも Style/Laye r 色の特別扱い条件式が重なっており、意図がコードから少し読み取りにくい。
4. **Medium – ベクトル/スカラー向けウィジェット生成ロジックの二重化とデッドパス**
   - `_create_bars` と `_create_widget` の両方がベクトル・スカラー向けスライダを生成する役割を持っており、似たようなコード（ベクトル dim の算出、`("x","y","z","w")` ループ、セルパディング付きサブテーブル）が重複している。
   - `_create_bars` 内で `value_type in {"int","float"}` の場合はそこでスライダを完結させて `return` しているため、`_create_widget` 内の int/float 分岐および `_create_int` / `_create_float` への経路は現状到達しない（事実上のデッドコード）。
   - 将来 `_create_widget` の使い方を変えたときにレイアウトが二系統に分かれる可能性があり、今のうちに責務を整理しておきたい。
5. **Low – 小さな try/except の乱立とマジックナンバー**
   - `dpg.set_item_width(..., -1)` や `dpg.add_table_column(..., init_width_or_weight=...)` 周りで汎用的な `except Exception: pass` が多用されており、実際に握りつぶしたい例外とそうでないものの区別がコードから分かりにくい。
   - 行太さ既定値 `0.0006` や Style/HUD 色の RGBA 既定値（例: `(0.196, 0.196, 0.196, 1.0)`）が生値のまま複数箇所に現れており、意味付けや調整ポイントが把握しづらい。
6. **Low – Style ID 群のハードコードと on_store_change の分岐**
   - `sync_style_from_store` と `on_store_change` がスタイル用 ID セットに依存しているが、その定義がクラス内部に散らばっており、Style パラメータ追加時に更新漏れが起こりやすい。
   - `on_store_change` の分岐が「Style/HUD などの特殊 ID」「レイヤー style.color」「その他 vector 成分」の順に重なっていて、処理フローがやや追いづらい。

## 実装改善計画（チェックリスト）

### 1. 構造の整理・責務分割

- [ ] `ParameterWindowContentBuilder` 内のメソッド順序を「公開 API → Style セクション → パラメータテーブル → ウィジェット生成 → CC/同期ユーティリティ → 共通ヘルパー」に再構成する（外部公開 API シグネチャは変更しない）。
- [ ] Style セクション構築を `build_style_controls()` から、グローバル色/HUD 色/レイヤー行ごとの小さなプライベートメソッド（例: `_build_global_style_rows`, `_build_hud_rows`, `_build_layer_style_rows`）に分割し、1 メソッドあたりの行数を減らす。
- [ ] パラメータテーブル構築ロジック（`_build_grouped_table`, `_flush_group`, `_create_row_3cols`）の責務を明文化し、必要に応じて内部ヘルパ（例: `_ParameterTableBuilder` 的な小さなクラス or 関数群）に抽出するかを検討し、採用案を決める。

### 2. テーブル/テーマ周りの共通化

- [ ] `mvStyleVar_CellPadding` を使ったセルパディング付きテーブル構築パターンを 1 箇所に集約するヘルパ（例: `_build_stretch_table(parent, columns, *, theme_kind=None)`）を導入し、`_build_grouped_table` / `_flush_group` / `_create_bars` / `_create_cc_inputs` / `_create_widget` から重複コードを置き換える。
- [x] `width_stretch=True` と `init_width_or_weight` の追加に対する `try/except TypeError` パターンを、小さなユーティリティ（例: `_add_stretch_column(label, weight)`）に寄せて行数を削減しつつ、旧 Dear PyGui との互換を維持する。
- [x] `dpg.set_item_width(item_id, -1)` をラップするヘルパ（例: `_set_full_width(item_id)`）を導入し、`try/except Exception` のノイズを減らす。

### 3. Style/HUD カラー処理の整理

- [x] Style/HUD に関わるパラメータ ID をモジュールレベルの定数（例: `STYLE_COLOR_IDS`, `HUD_COLOR_IDS`）として定義し、`build_style_controls` / `on_store_change` / `sync_style_from_store` で共有する。
- [x] 背景/ライン/HUD 色の初期値解決ロジックを `_resolve_canvas_colors`（config 読み込み）と Store の override/元値適用を行う小さなヘルパ（例: `_resolve_style_color_from_store(pid, fallback_rgba)`）に分離し、`build_style_controls` 内の if/代入ブロックを簡潔にする。
- [x] `store_rgb01` と `_safe_norm` での Style/Layer 色の特別扱い条件式を 1 箇所に集約し、意味のある名前（例: `_is_layer_style_color_id(pid)`）で表現する。

### 4. ウィジェット生成ロジックの整理

- [ ] `_create_bars` / `_create_widget` / `_create_int` / `_create_float` の責務を整理し、「Bars 列は値のバー群」「CC 列は常に `_create_cc_inputs` 経由で生成」といった最終レイアウト仕様を明確化する（必要なら仕様メモを残す）。
- [ ] 上記仕様に基づき、現状到達しない `_create_widget` 内の int/float 分岐と `_create_int` / `_create_float` が本当に不要かを確認する。不要と判断できれば削除または `_create_bars` 側に統合してコードパスを 1 系統にする。
- [x] ベクトル用スライダ生成パターン（`_create_bars` と `_create_widget` に重複する部分）を 1 箇所にまとめ、dim（2–4）と range を受け取るヘルパ（例: `_create_vector_sliders(parent, desc, value_vec, vmin, vmax)`）に切り出す。

### 5. CC バインディングと同期処理の整理

- [ ] CC 入力テーブル生成ロジック（`_create_cc_inputs`）内のセルパディング・列追加パターンを、2. のテーブルヘルパを使う形にリファクタリングする。
- [x] `_on_cc_binding_change` の分岐を簡素化し、入力文字列処理（trim → パース → clamp）の部分を別ヘルパに分離してテストしやすくする（既存の `tests/ui/parameters/test_cc_binding.py` を活かす）。
- [ ] `on_store_change` のカラー系特別扱い（Style/HUD/layer color）とベクトル成分（`pid::x` 等）の更新ロジックを整理し、早期 `continue` の多用を避けて処理フローを読みやすくする。

### 6. マジックナンバー・定数整理

- [x] 行太さの既定値や HUD メータ色など、UI 既定値として意味を持つ数値をモジュールレベルの定数にまとめ、`build_style_controls` から生値を排除する。
- [x] 列幅クランプに使っている閾値（`0.1`/`0.9`/`0.05`/`0.95` など）に簡単なコメントまたは定数名を付け、レイアウト意図をコードから推測しやすくする。

### 7. テスト・ドキュメント更新

- [ ] 既存の `tests/ui/parameters/test_cc_binding.py` を再確認し、内部構造を変えても公開挙動（CC バインディングの解釈と clamp）が変わっていないことを確認する。
- [ ] 必要に応じて、Style セクションや vector スライダ周辺の挙動をカバーする追加テスト（UI 初期値と Store の同期など）を検討し、追加するかどうかを決める。
- [ ] `architecture.md:80-89` 付近の Parameter GUI 説明と、実際の `ParameterWindowContentBuilder` の責務分割に差異が生じた場合は、最終的な構造に合わせて記述を更新する。

## メモ / オープン質問

- Style セクションと一般パラメータテーブルをクラスレベルで分割する（`StyleSectionBuilder` と `ParameterTableBuilder` に分ける）か、1 クラスのままメソッドレベルで整理するかは、どこまで構造を変えてよいか相談したい。
- CC バインディング UI の仕様（ベクトル成分ごとの入力を常に表示するか、簡略表示モードを認めるか）について、将来的な拡張の方向性があればそれに合わせて設計したい。
- 現状の API を壊してもよい範囲（例: private メソッドのシグネチャや内部タグ命名規則）に何か制約があれば教えてほしい。
