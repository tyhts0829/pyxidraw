# フォント関連の後方互換（status_manager.*）削除 計画チェックリスト

目的: `status_manager.font` / `status_manager.font_size` への後方互換参照を削除し、設定の単純化（`hud.*` と `parameter_gui.layout.*` のみを正とする）。

背景/現状:
- 現行コードは以下の互換参照を持つ。
  - HUD: `src/engine/ui/hud/overlay.py` で `hud.font_name` / `hud.font_size` を優先し、無ければ `status_manager.font` / `status_manager.font_size` をフォールバック。
  - Parameter GUI: `src/engine/ui/parameters/dpg_window.py` で `parameter_gui.layout.font_name` → `hud.font_name` → `status_manager.font` の順で決定。
- ドキュメント（architecture.md / docs/plan_font_search_from_config.md）にもフォールバックの記述が残存。
- `status_manager` の他項目（color/pos/pad/info）は現行未使用。

スコープ:
- コードから `status_manager.*` を読むフォールバック実装を削除。
- 関連ドキュメントから当該記述を除去/更新。
- `configs/default.yaml` の `status_manager:` セクションの扱いは要判断（削除 or レガシー注記の残置）。

作業手順（チェックリスト）:
1. [x] overlay.py: `status_manager.font` / `status_manager.font_size` フォールバック削除（該当ブロックの除去）。
2. [x] overlay.py: コメントから互換参照の言及を削除（hud.* のみを明記）。
3. [x] dpg_window.py: `status_manager.font` フォールバック削除（決定順は `parameter_gui.layout.font_name` → `hud.font_name` のみ）。
4. [x] dpg_window.py: コメント更新（互換参照の文言を削除）。
5. [x] architecture.md: フォント探索/適用の説明から `status_manager.*` フォールバックの記述を削除。
6. [x] docs/plan_font_search_from_config.md: 同様に記述更新。
7. [x] configs/default.yaml: `status_manager:` セクションの扱いを決定し反映（A: 丸ごと削除（推奨・破壊的）/ B: レガシー注記を付けて残置）。
8. [x] 変更ファイルに限定した `ruff/black/isort/mypy`（編集ファイルのみ）。
9. [x] 最小スモーク（必要に応じて GUI/pyglet 環境が無い場合はスキップ可）。

互換性への影響:
- 既存の `config.yaml` で `status_manager.font` / `font_size` を指定していた場合、以降は反映されません。
- 移行先: HUD のフォントは `hud.font_name` / `hud.font_size`（HUD）、Parameter GUI は `parameter_gui.layout.font_name` を設定してください。

確認事項（要回答）:
1) `configs/default.yaml` の `status_manager:` セクションは削除してよいですか？
   - A: 削除（推奨。未使用項目の撤去で混乱を回避）
   - B: 残置（レガシー注記を付ける）

ロールバック指針:
- もし必要になった場合は、削除したブロックを元の位置に戻し、architecture.md の記述も復元します。
