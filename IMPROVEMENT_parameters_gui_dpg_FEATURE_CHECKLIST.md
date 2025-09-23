# Parameter GUI（Dear PyGui）未活用機能・導入チェックリスト

目的

- Dear PyGui の「便利だが未採用」機能を parameter_gui に段階導入するためのチェックリスト。
- 採用したい項目に [x] を入れてください。合意後、最小差分で実装します。

関連

- 既存実装: `src/engine/ui/parameters/dpg_window.py`
- 設計メモ: `IMPROVEMENT_parameters_gui_dpg.md`

優先候補（提案順）

- 1. 折りたたみグループ + 2 カラム
- 2. 検索/フィルタバー
- 3. ドラッグ数値入力 + カラーエディタ
- 4. 変更ハイライト

---

チェックリスト（採用にチェックを入れてください）

- [x] 折りたたみグループ（カテゴリ別） — 実装済

  - 効果: 多パラメータ時の見通し向上（カテゴリ単位で開閉）。
  - DPG: `add_collapsing_header` / `add_tree_node`。
  - 差し込み: `mount()` 内で `ParameterDescriptor.category` ごとにヘッダ生成し、その配下に 2 カラムテーブルで行を構築。
  - 実装箇所: `src/engine/ui/parameters/dpg_window.py:160` 付近（カテゴリ分割〜ヘッダ/テーブル生成）

- [x] 2 カラム/テーブルレイアウト（左ラベル/右入力） — 実装済

  - 効果: 視線移動と縦スクロール量の削減。
  - DPG: `add_table` + `table_column`/`table_row`（または `same_line`）。
  - 差し込み: 行構築時にラベルとウィジェットを同一行へ配置。
  - 参考: `src/engine/ui/parameters/dpg_window.py:168`（ラベル生成）、`src/engine/ui/parameters/dpg_window.py:210-251`（各入力）。
  - 実装箇所: `src/engine/ui/parameters/dpg_window.py:196`（mount 内テーブル生成）、`src/engine/ui/parameters/dpg_window.py:206`（\_create_row で table_row に配置）

- [ ] 検索/フィルタバー

  - 効果: パラメータ到達時間を大幅短縮。
  - DPG: `add_input_text(hint="Search")` + 行の `configure_item(show=False)` で絞り込み。
  - 差し込み: ルート作成直後に検索入力を追加し、コールバックで行の表示/非表示を切替。
  - 参考: `src/engine/ui/parameters/dpg_window.py:95-103`（ルート/スクロール領域）。

- [ ] ドラッグ入力（drag_float/int/float3/4）

  - 効果: 微調整と粗い移動を両立（マウスドラッグで連続変更）。
  - DPG: `add_drag_float`/`add_drag_int`/`add_drag_float3/4`（`speed`/`min`/`max`）。
  - 差し込み: 数値系でスライダーの代替/併設（`RangeHint.step` を `speed` へ反映）。
  - 参考: `src/engine/ui/parameters/dpg_window.py:230-251`（int/float 入力）。

- [ ] カラーエディタ/ピッカー（RGB/RGBA）

  - 効果: 色系パラメータの操作性を大幅改善。
  - DPG: `add_color_edit`/`add_color_picker`。
  - 差し込み: `value_type == "vector"` かつ色系（名称/グループ判定）で自動切替。値スケール（0..1/0..255）は既存レンジに合わせて変換。
  - 参考: `src/engine/ui/parameters/dpg_window.py:209-223`（vector 入力）。

- [ ] ツールチップ（help_text の活用）

  - 効果: UI を騒がせず説明を提供。
  - DPG: `with dpg.tooltip(item): add_text(...)`。
  - 差し込み: 各ウィジェット作成直後、`desc.help_text` があれば付与。

- [ ] 右クリック・コンテキストメニュー（Reset/コピー/貼付け 等）

  - 効果: 頻用操作を手元で完結。プリセット運用とも相性良。
  - DPG: `add_popup(parent=item, mousebutton=...)`。
  - 差し込み: 行またはウィジェット単位でメニューを定義。

- [ ] 変更ハイライト（差分時テーマ付与）

  - 効果: 上書き中の値が一目で分かる。
  - DPG: `theme`/`bind_item_theme` で背景/枠色を切替。
  - 差し込み: Store→UI 反映時に既定値と比較し、差分ならテーマ付与/解除。
  - 参考: `src/engine/ui/parameters/dpg_window.py:312-331`（テーマ実装の流儀）。

- [ ] キーボードショートカット（検索フォーカス/enum 移動 など）

  - 効果: キーボード中心の高速操作。
  - DPG: `handler_registry` + `add_key_press_handler` 等。
  - 差し込み: ルート生成時にハンドラ登録。`Ctrl+F` で検索へフォーカス、`←/→` で enum 遷移、`1..9` でクイック選択。

- [ ] フォント/スケール（可読性/HiDPI）

  - 効果: 可読性と一貫した見た目。
  - DPG: `add_font_registry`/`add_font`/`bind_font`、`set_global_font_scale`。
  - 差し込み: `create_context` 後、`setup_dearpygui` 前。
  - 参考: `src/engine/ui/parameters/dpg_window.py:89-93`。

- [ ] ドッキング/複数パネル

  - 効果: プレビュー/ログ/パラメータの自由配置で作業効率向上。
  - DPG: `configure_app(docking=True, docking_space=True)`。
  - 差し込み: `setup_dearpygui` 前にアプリ設定を有効化。
  - 参考: `src/engine/ui/parameters/dpg_window.py:89-93`。

- [ ] レイアウト保存/復元（ウィンドウ配置のみ）

  - 効果: ワークスペースの再現が容易。
  - DPG: `save_init_file(...)` / `load_init_file(...)`。
  - 差し込み: メニュー/ショートカットから保存/読込 API を呼出し。

- [ ] フォーカス/スクロール制御

  - 効果: 大量項目でも目的行へ即移動。
  - DPG: `focus_item(id)`、`set_y_scroll(...)`。
  - 差し込み: 値更新直後や検索ジャンプ時に対象行へフォーカス。

- [ ] 対数スライダー（log スケール）

  - 効果: 広いダイナミックレンジの操作が容易。
  - DPG: スライダーの対数スケールフラグ（Dear PyGui のバージョンにより引数名/可否は要確認）。
  - 差し込み: `RangeHint.scale == "log"` のときにスライダーへ対数フラグを付与。
  - 参考: `src/engine/ui/parameters/state.py` の `RangeHint.scale`。

- [ ] 無効/Read-only 表示（supported=False でも表示）
  - 効果: 文脈把握（現在は非対応項目を非表示）。
  - DPG: `configure_item(item, enabled=False)`。
  - 差し込み: `supported=False` の項目も行を作成し、無効化して表示。

---

補足

- 実装は「最小差分・段階導入」。各項目は独立導入/検証が可能です。
- 変更ファイルに限定して `ruff/black/isort/mypy/pytest -q -m smoke` を回します（CI/ヘッドレス配慮）。
- 採用チェック後、着手順（優先度）と範囲を確認します。
