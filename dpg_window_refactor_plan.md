# dpg_window クラス分割・モジュール分割計画

目的: `src/engine/ui/parameters/dpg_window.py` に集中している Dear PyGui 実装を、責務ごとにクラス/モジュール分割しつつ、外部 API（`ParameterWindow` の生成/表示切替/終了）を変えずに見通しを良くする。

## ターゲット構成（クラス / モジュール案）

### モジュール構成案

- `src/engine/ui/parameters/dpg_window.py`
  - 公開エントリポイント。
  - `ParameterWindowBase` と、薄いオーケストレータとしての `ParameterWindow` のみを持つ。
  - DPG 依存の詳細実装は下記ヘルパクラスに委譲する。

- `src/engine/ui/parameters/dpg_window_theme.py`
  - `ParameterWindowThemeManager`（仮称）を定義。
  - フォント解決・テーマ生成・カテゴリ別テーマ（Shape/Pipeline/HUD/Display）の構築を担当。

- `src/engine/ui/parameters/dpg_window_content.py`
  - `ParameterWindowContentBuilder`（仮称）を定義。
  - ルートウィンドウ/Display/HUD の構築、Descriptor グルーピングとテーブル生成、型別ウィジェット生成、Store との同期ロジックを担当。

### クラス単位の責務案

- `ParameterWindowBase`（現状維持）
  - UI 実装の最小インタフェース。
  - `set_visible() / close() / mount()` の抽象契約のみを持つ。

- `ParameterWindow`（オーケストレータ）
  - DPG コンテキストとビューポートの生成/破棄を管理。
  - `ParameterStore`/`ParameterLayoutConfig`/`ParameterThemeConfig` を受け取り、内部で
    `ParameterWindowThemeManager` と `ParameterWindowContentBuilder` を生成して利用する。
  - 外部からの操作は現状どおり:
    - 生成時にウィンドウを構築（`auto_show` オプション付き）。
    - `set_visible()` で `show/hide` とドライバ起動/停止を切り替える。
    - `close()` で購読解除→ドライバ停止→ビューポート非表示→コンテキスト破棄を行う。

- `ParameterWindowThemeManager`（新規 / `dpg_window_theme.py`）
  - 依存: `ParameterLayoutConfig`, `ParameterThemeConfig | None`, `dearpygui.dearpygui`, 任意で `util.color`。
  - 責務:
    - フォント登録（`_setup_fonts` 相当）
      - `config.yaml` からフォント名/検索ディレクトリを解決し、Dear PyGui の font registry に登録。
    - グローバルテーマの構築・適用（`_setup_theme` 相当）
      - `ParameterThemeConfig.style/colors` と `layout.*` を元に、スタイルとカラーを Dear PyGui テーマとして構築。
    - カテゴリ別テーマのキャッシュ管理
      - `category_kind`（`"shape"`, `"pipeline"`, `"hud"`, `"display"` など）ごとに、
        collapsing header / table 行背景のテーマを生成し再利用。
    - カラー値の正規化（`_to_dpg_color` 相当）
      - `util.color.to_u8_rgba` を優先しつつ、フォールバックとして 0..1 / 0..255 のリストも受け付ける。

- `ParameterWindowContentBuilder`（新規 / `dpg_window_content.py`）
  - 依存: `ParameterStore`, `ParameterLayoutConfig`, `ParameterWindowThemeManager`, `ParameterDescriptor`, `dearpygui.dearpygui`.
  - 責務:
    - ルートウィンドウ/Display/HUD の構築
      - ルートウィンドウの作成（`_build_root_window`）。
      - Display/HUD セクションの構築と、カテゴリ別テーマの適用（`build_display_controls`, `sync_display_from_store`）。
    - Descriptor グルーピングとテーブル構築
      - Display/HUD 関連 ID を除外し、`(category_kind, category)` 単位でグルーピング（`_build_grouped_table`）。
      - カテゴリごとに collapsible header + 3 列テーブル（Label/Bars/CC）を生成。
    - 型別ウィジェット生成
      - `value_type` に応じて各種ウィジェットを生成（`_create_widget`, `_create_bool`, `_create_enum`,
        `_create_string`, `_create_vector`, `_create_int`, `_create_float`, `_create_style_color_picker` など）。
      - RangeHint/VectorRangeHint と layout 設定を用いたスライダのレンジ決定。
    - CC 入力・ Store 連携
      - CC 番号入力の生成と変更ハンドリング（`_add_cc_binding_input`, `_add_cc_binding_input_component`,
        `_on_cc_binding_change`）。
      - ウィジェット変更 → Store override 反映（`_on_widget_change`, `_current_or_default`）。
      - Store 変更 → ウィジェット値反映（`_on_store_change`）。
    - カラー/正規化ヘルパ
      - `force_set_rgb_u8`, `store_rgb01`, `_safe_norm` のようなヘルパを内包し、Display/HUD と style.color 両方に利用。
    - Dear PyGui ヘルパ
      - DPG バージョン差分を吸収する `_dpg_policy` / テーブルカラム比率計算（`_label_value_ratio`, `_add_two_columns` など）。

- ドライバ制御（`_tick`, `_start_driver`, `_stop_driver`）
  - ひとまず `ParameterWindow` 内に残す想定。
  - 必要なら後続で `ParameterWindowDriver` のような小さなクラスに切り出すオプションを残す。

## 実装タスクチェックリスト

### 1. テーマ管理クラスの導入

- [ ] `src/engine/ui/parameters/dpg_window_theme.py` を新規追加し、`ParameterWindowThemeManager` を定義する。
  - [ ] コンストラクタで `layout: ParameterLayoutConfig` と `theme_cfg: ParameterThemeConfig | None` を受け取る。
  - [ ] フォント登録メソッド `setup_fonts()` を実装し、既存 `_setup_fonts` のロジックを移植する。
  - [ ] グローバルテーマ構築メソッド `setup_theme()` を実装し、`_apply_default_styles` / `_apply_styles_from_config` / `_apply_colors_from_config` を内部メソッドとして組み込む。
  - [ ] カテゴリ別テーマ取得メソッド `get_category_header_theme(kind: str)` / `get_category_table_theme(kind: str)` を実装し、現状の `_get_category_header_theme` / `_get_category_table_theme` を移植する。
  - [ ] カラー正規化メソッド `to_dpg_color(value: Any)` を実装し、既存 `_to_dpg_color` を移植する。

- [ ] `dpg_window.ParameterWindow` からテーマ関連メソッドを削除し、`ParameterWindowThemeManager` への委譲に差し替える。
  - [ ] `__init__` 内で `self._theme_mgr = ParameterWindowThemeManager(layout=self._layout, theme_cfg=self._theme)` を生成する。
  - [ ] フォント設定は `self._theme_mgr.setup_fonts()` を呼び出す形に書き換える。
  - [ ] ルートウィンドウ構築後に `self._theme_mgr.setup_theme()` を呼び出してグローバルテーマを適用する。
  - [ ] Display/HUD やカテゴリヘッダで使用している `_get_category_header_theme` / `_get_category_table_theme` 呼び出しを、`self._theme_mgr.get_category_header_theme()` / `get_category_table_theme()` に置き換える。

### 2. コンテンツ構築クラスの導入

- [ ] `src/engine/ui/parameters/dpg_window_content.py` を新規追加し、`ParameterWindowContentBuilder` を定義する。
  - [ ] コンストラクタで `store: ParameterStore`, `layout: ParameterLayoutConfig`, `theme_mgr: ParameterWindowThemeManager` を受け取る。
  - [ ] ルートウィンドウ構築メソッド `build_root_window(root_tag: str)` を実装し、既存 `_build_root_window` のロジック（Display/HUD セクションの構築含む）を移植する。
  - [ ] Display/HUD 用メソッド `build_display_controls(parent: int | str)` / `sync_display_from_store()` を移植し、`ParameterStore` はコンストラクタから参照する形にまとめる。
  - [ ] パラメータテーブル構築メソッド `mount_descriptors(root_tag: str, descriptors: list[ParameterDescriptor])` を用意し、`mount()` 内の stage + `_build_grouped_table` 呼び出しをここに集約する。
  - [ ] グルーピングとテーブル構築ロジック（`_build_grouped_table`, `_label_value_ratio`, `_add_two_columns`, `_create_row_3cols`）を builder 内のプライベートメソッドとして移植する。
  - [ ] 型別ウィジェット生成ロジック（`_create_bars`, `_create_cc_inputs`, `_create_widget`, `_create_bool`, `_create_enum`, `_create_string`, `_create_vector`, `_create_int`, `_create_float`, `_create_style_color_picker`）を builder に移し、必要に応じて小さなヘルパに整理する。
  - [ ] Store との値連携メソッド（`_current_or_default`, `_on_widget_change`, `_on_cc_binding_change`, `_add_cc_binding_input`, `_add_cc_binding_input_component`, `_on_store_change`）を builder に移す。
  - [ ] カラー/正規化ヘルパ（`force_set_rgb_u8`, `store_rgb01`, `_safe_norm`）や DPG ヘルパ（`_dpg_policy`）も builder に集約する。

- [ ] `dpg_window.ParameterWindow` から上記メソッドを削除し、`ParameterWindowContentBuilder` への委譲に差し替える。
  - [ ] `__init__` 内で `self._content = ParameterWindowContentBuilder(store=self._store, layout=self._layout, theme_mgr=self._theme_mgr)` を生成する。
  - [ ] ルート構築は `self._content.build_root_window(ROOT_TAG)` を呼ぶ形に変更する。
  - [ ] `mount()` は stage の管理と root_tag を渡すだけにし、実際のテーブル構築は `self._content.mount_descriptors(ROOT_TAG, descriptors)` に任せる。
  - [ ] Store の購読コールバックは `self._store_listener = lambda ids: self._content.handle_store_change(ids)` のようにし、内部で `_on_store_change` と Display/HUD 同期をまとめて呼ぶメソッドを用意する。

### 3. ParameterWindow 本体の整理

- [ ] `ParameterWindow` の責務を「ライフサイクル管理 + ドライバ起動/停止 + ヘルパクラスの組み立て」に限定するよう、フィールドとメソッドを最小化する。
  - [ ] DPG ドライバ関連（`_tick`, `_start_driver`, `_stop_driver`）はそのまま維持しつつ、他のロジックと混ざらないようにセクション分けを明確化する。
  - [ ] `set_visible` / `close` / `mount` は、新クラスの API を利用する薄いラッパとして再実装する。
  - [ ] `ParameterWindowBase` のインタフェースや `ParameterWindow` のコンストラクタシグネチャは変更しない（既存の controller/test からの利用を温存する）。

## テスト / 検証タスク

- [ ] 既存テストの再確認
  - [ ] `tests/ui/parameters/test_dpg_mount_smoke.py` が引き続き `ParameterWindow` を直接 import し、生成/表示切替/終了が例外なく動作することを確認する。
  - [ ] `engine.ui.parameters.controller.ParameterWindowController` からの遅延 import (`from .dpg_window import ParameterWindow`) がそのまま動作することを確認する。

- [ ] 新構成に対する追加テスト（必要なら）
  - [ ] `ParameterWindowContentBuilder` 単体を Dear PyGui import 可能な環境で生成し、最小 Descriptor セットに対して `build_root_window` / `mount_descriptors` が例外なく動作する smoke テストを追加するか検討する。
  - [ ] `ParameterWindowThemeManager` の `setup_theme()` が、テーマ設定なしの場合でもフェイルソフトに動作することを確認するテストを検討する。

- [ ] 静的チェック / Lint
  - [ ] 変更ファイル（`dpg_window.py`, `dpg_window_theme.py`, `dpg_window_content.py`）に対して `ruff check --fix` を実行する。
  - [ ] 同じく `black` / `isort` / `mypy` を対象ファイルに対して実行する。

## ドキュメント / AGENTS 更新タスク

- [ ] `architecture.md` に parameter_gui の UI 層構成を追記し、
  - ParameterWindow が `ParameterWindowThemeManager` / `ParameterWindowContentBuilder` と協調していること、
  - shape/effect/HUD/Display のカテゴリ種別とテーマ適用の流れ
  を簡潔に整理する。

- [ ] `src/engine/ui/parameters/AGENTS.md` の Overview に
  - `dpg_window.py`: 公開エントリ（ParameterWindow）
  - `dpg_window_theme.py`: テーマ/フォント管理
  - `dpg_window_content.py`: Dear PyGui レイアウト（Display + Parameter テーブル）
  を追記する。

## 要確認事項（ユーザーに確認したい点）

- [ ] モジュール名/クラス名について
  - `dpg_window_theme.py` / `dpg_window_content.py` と
    `ParameterWindowThemeManager` / `ParameterWindowContentBuilder` という名前で問題ないか。
  - もっと短い名前（例: `DpgTheme`, `DpgContent`）のほうがよければ教えてほしい。

- [ ] ドライバ制御の切り出し範囲
  - Dear PyGui の `render_dearpygui_frame` / `pyglet` 連携を、
    追加のヘルパクラス（例: `ParameterWindowDriver`）に分離する案もあるが、
    現時点では ParameterWindow 内に残す想定でよいか。

- [ ] Display/HUD セクションの扱い
  - Display/HUD の構築ロジックを `ParameterWindowContentBuilder` にまとめる方針でよいか。
  - 将来的に HUD 専用のクラス（例: `ParameterHudView`）に分ける余地を残すが、
    今回のリファクタではそこまで分割しない前提で問題ないか。

## 今後の拡張アイデア（任意）

- [ ] `ParameterWindowContentBuilder` を Dear PyGui に強く依存しないインタフェースでラップし、
  UI 抽象層（別実装への差し替え）を検討する。
- [ ] Display/HUD の値同期を、将来的に他の UI（例: CLI/別 GUI）と共有できるようにするため、
  `ParameterStore` 側に汎用的なビュー/バインディングヘルパを追加する案を検討する。

---

この計画内容と命名/分割方針で問題なければ、このチェックリストに従って `dpg_window.py` のリファクタリングを進めていきます。変更前に調整したい点や、分割の粒度についての希望があれば、このファイルへの追記内容として指示してください。

