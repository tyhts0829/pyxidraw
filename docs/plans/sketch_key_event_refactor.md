# キーイベント処理のヘルパ化計画（チェックリスト）

目的（前提: 互換維持・シンプル優先）
- [ ] `on_key_press` 内の分岐（P/Shift+P/G/Shift+G/V/Shift+V/ESC）を小さなヘルパへ分離
- [ ] `sketch.py` をさらに簡素化し、テストしやすい形にする（依存注入で関数単体テスト可能）
- [ ] 既存のキー割り当てと挙動は完全踏襲（機能追加・変更は行わない）

設計要点
- [ ] 新規ファイル `src/api/sketch_runner/keymap.py`（内部専用）を追加
- [ ] 薄いデータ駆動：`KeyHandlers`（単純な名前付きタプル/辞書）に動作を集約
- [ ] 依存（overlay/recorder/export/pyglet/renderer 等）はコンストラクタに注入し、ロジックを関数に閉じ込める
- [ ] `handle_key_event(sym, mods, handlers)` を公開（`sketch.py` から呼ぶ）

対象イベント（踏襲）
- [ ] `ESC` → window.close()
- [ ] `P` → PNG 保存（HUD 含む）
- [ ] `Shift+P` → 高解像度 PNG（HUD なし/FBO）
- [ ] `G` → G-code エクスポート開始
- [ ] `Shift+G` → G-code エクスポートキャンセル
- [ ] `V` → 録画トグル（画面/HUD含む）
- [ ] `Shift+V` → 録画トグル（品質/ラインのみ）

インターフェース（案）
- [ ] `class KeyHandlers:`（または `TypedDict`）
  - [ ] `save_png_screen() -> None`
  - [ ] `save_png_quality() -> None`
  - [ ] `start_gcode() -> None`
  - [ ] `cancel_gcode() -> None`
  - [ ] `toggle_record_screen() -> None`
  - [ ] `toggle_record_quality() -> None`
  - [ ] `close_window() -> None`
- [ ] `def handle_key_event(sym: int, mods: int, key, handlers: KeyHandlers) -> None`
  - [ ] `key` は `pyglet.window.key` への参照（依存注入）
  - [ ] 分岐はここに集約（ユニットテストで網羅可能）

段階的作業（DoD: 変更ファイル限定で ruff/black/isort/mypy/pytest）
- [ ] Phase 1: スケルトンとテスト追加
  - [ ] `keymap.py` 追加（最小の型・関数定義）
  - [ ] `tests/api/sketch_runner/test_keymap.py` 追加（`handle_key_event` が各ハンドラを呼ぶこと）
- [ ] Phase 2: `sketch.py` からの委譲
  - [ ] `on_key_press` 内の分岐を `KeyHandlers` 構築＋ `handle_key_event` 呼び出しに置換
  - [ ] 既存ロジック（録画/G-code/PNG）は既存ヘルパ（export/recording）を呼ぶ
- [ ] Phase 3: 仕上げ
  - [ ] ログ/例外メッセージを軽く整える（既存表現踏襲）
  - [ ] architecture.md に `sketch_runner/keymap.py` を追記

テスト項目（ユニット）
- [ ] `handle_key_event` — 各キー/修飾子で対応するハンドラが 1 回だけ呼ばれる
- [ ] 未対応キー/修飾子では何もしない（例外なし）
- [ ] `KeyHandlers` の一部が未設定でも例外なくスキップ（必要なら no-op デフォルト）

非目標
- [ ] キーバインドの変更/拡張（今回の計画では行わない）
- [ ] デバウンス/状態機械の導入（シンプル優先のため行わない）

実行メモ
- [ ] 単体: `pytest -q tests/api/sketch_runner/test_keymap.py`
- [ ] 併用: `pytest -q -m smoke -k keymap`
