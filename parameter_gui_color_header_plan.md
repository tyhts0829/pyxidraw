# parameter_gui カラー UI 集約計画（案）

目的: Parameter GUI に散在している Display/HUD/pipeline の色編集 UI を 1 つのヘッダに集約し、将来のカラーパレット機能を組み込みやすくする。

現状メモ:

- Display: 背景色/ライン色のカラーピッカーが `Display` ヘッダ内にある（`canvas.background_color`/`line_color` を初期値に使用）。
- HUD: `HUD` ヘッダに Text/Meter/Meter BG の色と Show HUD トグルが同居している。
- pipeline/shape: `.style#...color` の Descriptor をパラメータテーブル内で検出し、その場でカラーピッカーを出している。
- テーマ設定は `parameter_gui.theme.colors.*` と `categories`（Display/HUD/shape/pipeline）の header 色に分かれている。色の正規化は `util.color.normalize_color` / `to_u8_rgba` を使用。

スコープ:

- 対象: Display 背景/ライン、HUD テキスト/メーター色、style.color（pipeline/shape）の GUI 表示位置とレイアウト。
- 非対象: レンダラー内部の色処理や色値の保存形式（RGBA/vec3）は変更しない。

## 実装タスク案（チェックリスト）

- [ ] UI 仕様の確定
  - [ ] 新規ヘッダ名（例: `Colors`）と配置（起動時デフォルト open、Display/HUD の上部）を決める。
  - [ ] 小見出し/行構成: Display 色 2 件、HUD 色 3 件（Show HUD は従来ヘッダ側に残す想定）、style.color は pipeline/shape をラベル単位でまとめる案を Fix。
  - [ ] カテゴリテーマ: 新ヘッダ用に `parameter_gui.theme.categories.Colors` を追加するか、既定 `colors.header*` を使うか決める（後方互換優先なら既定色を使用）。
- [ ] UI 構築ロジックの整理（`src/engine/ui/parameters/dpg_window_content.py`）
  - [ ] Display/HUD のカラーピッカー生成を新ヘッダ構築関数へ移し、既存 Display/HUD ヘッダには非カラー項目だけ残す。
  - [ ] style.color Descriptor を事前に抽出するヘルパ（例: `_collect_style_color_descriptors`）を追加し、集約ヘッダで表示する。
  - [ ] 集約済み style.color は通常パラメータテーブルから除外し、重複表示を避ける。
  - [ ] 既存カラー入力挙動（正規化・vec3 保存・`force_set_rgb_u8`）を共通化し、将来パレット差し替えポイントを 1 箇所に寄せる。
- [ ] 同期/保存経路の調整
  - [ ] `sync_display_from_store` 相当の同期ルーチンを新ヘッダに対応させ、style.color も含めた色項目をまとめて更新する。
  - [ ] `on_store_change` のカラー分岐を重複なく扱えるよう整理し、既存の `store_rgb01` を流用する。
- [ ] テーマ/設定の反映
  - [ ] 新ヘッダがテーマにぶら下がるよう ThemeManager にフックを追加し、新カテゴリキーを導入する場合は config の例も更新する。
  - [ ] `architecture.md` の Parameter GUI 記述に「カラー UI は 1 箇所に集約される」旨と設定キーの変更点を同期する。
- [ ] 将来のパレット拡張の余地
  - [ ] カラーピッカー生成を `ColorControlSpec` のようなシンプルなデータ構造にまとめ、後から「プリセット適用/パレット選択」に差し替えられるようにする。
- [ ] 検証
  - [ ] DPG を直接叩かずに済む部分（style.color 抽出やフィルタリング）のユニットテストを `tests/ui/parameters` に追加。
  - [ ] 変更ファイルに対して `ruff`/`black`/`isort`/`mypy` を実行。GUI は簡易手動確認（`python main.py` で `use_parameter_gui=True`）を想定。

## オープンな確認事項

- [ ] Colors ヘッダの名称・並び順は上記案でよいか（Display/HUD より上を想定）。；はい
- [ ] style.color のグルーピングは「パイプライン表示ラベル単位」で良いか（shape の style.color も同じヘッダに入れる前提で問題ないか）。；はい。
- [ ] HUD の Show HUD トグルは従来どおり HUD ヘッダ側に残す形でよいか（色だけを Colors に寄せる想定）。；はい
