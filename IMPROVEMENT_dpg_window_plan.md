# DPG ParameterWindow 段階的実装改善計画（dpg_window.py）

目的
- `src/engine/ui/parameters/dpg_window.py` の実装を、より堅牢（スレッド安全/例外安全）かつ使い勝手の良いものへ、破壊的変更可の前提で段階的に改善する。

スコープ
- 対象: `src/engine/ui/parameters/dpg_window.py`（必要に応じて `window.py` など最小限の関連ファイル）。
- 非目標: 新規依存の追加、DPG 以外の UI 実装、プリセット保存/検索機能など大規模機能追加。

前提・制約
- 値は実値（float/int/bool/vector/enum）で受け取り、クランプは表示上のみ（Store では不実施）。
- ヘッドレス/未導入環境でも import/生成/破棄で落ちないこと（現仕様を維持）。

合意が必要な設計ポイント（確認待ち）
- [ ] RangeHint.step がある場合は「スライダ」ではなく「ドラッグ入力（drag_*）」へ切替でよいか。
- [ ] Store 更新は他スレッド（例: MIDI/ランタイム）から発生し得る前提でよいか。よい場合、UI スレッドへのマーシャリング（キュー導入）を実施。
- [ ] vector の軸ラベル（x/y/z/w）を最小限で表示する方針にするか（将来拡張でも可）。

---

## M0: 即効性のある小改善（1 PR）
目的
- 安全で影響範囲が小さい改善を先に当て、可読性と挙動の明確化を図る。

タスク（チェックリスト）
- [ ] `get_item_alias` の除去: 文字列タグをそのまま `parent` に使用する。
  - 変更箇所: `src/engine/ui/parameters/dpg_window.py:156-159`
- [ ] vector 入力の表示精度を統一: `input_float3/4` に `format="%.{value_precision}f"` を付与。
  - 変更箇所: `src/engine/ui/parameters/dpg_window.py:244-259`
- [ ] 可視制御の明確化: `set_visible(False)` で `minimize_viewport` ではなく `hide_viewport` を使用。
  - 変更箇所: `src/engine/ui/parameters/dpg_window.py:124-132`
- [ ] テーマ解除の統一: `bind_item_theme(pid, None)` へ一本化（バージョン差異の注記を残す）。
  - 変更箇所: `src/engine/ui/parameters/dpg_window.py:402-409`
- [ ] ビューポート参照保持の意図をコメントで補足（GC/明示破棄タイミング）。
  - 変更箇所: `src/engine/ui/parameters/dpg_window.py:91-94`

DoD（完了条件）
- 変更箇所に対し `ruff/black/isort/mypy` が緑。
- 既存の挙動（初期マウント/双方向同期/ハイライト）に変化がない。

---

## M1: RangeHint.step の反映（操作性）
目的
- 範囲・刻みのヒントを UI 操作に反映し、意図通りの入力粒度を提供する。

タスク
- [ ] `hint.step` がある場合はドラッグ系へ切替（`add_drag_float`/`add_drag_int`）。`speed` に `step` を反映。
  - 変更箇所: `src/engine/ui/parameters/dpg_window.py:261-286`
- [ ] `hint.step` がない場合は従来通り `slider_*` を使用。
- [ ] vector についても `step` があれば `input_float3/4` に `step`/`step_fast` を反映（DPG の API 範囲で可能な限り）。

DoD
- float/int/vector で `step` が UI 操作に反映される（見た目のフォーマットも維持）。
- 変更後も Store は実値保持（クランプなし）。

---

## M2: スレッド安全な更新（マーシャリング）
目的
- DPG のスレッド非安全性を踏まえ、UI スレッド以外からの Store 通知に安全に対応する。

タスク
- [ ] `queue.Queue[tuple[str, Any]]` を追加し、`_on_store_change()` は `(pid, value)` をキューに積むのみへ変更。
  - 変更箇所: `src/engine/ui/parameters/dpg_window.py:307-331`
- [ ] `_tick()` 内でキューをドレインし、`dpg.set_value()`/`_update_highlight()` を UI スレッドで実行。
  - 変更箇所: `src/engine/ui/parameters/dpg_window.py:336-347`
- [ ] 非 pyglet 経路のループを `start_dearpygui()` ではなく `while is_dearpygui_running(): drain→render` へ変更し、同様にキューをドレイン。
  - 変更箇所: `src/engine/ui/parameters/dpg_window.py:343-347`

DoD
- DPG への呼び出しが UI スレッド（`_tick`/自前ループ）に限定される。
- 並行での `set_override()` 多投下でもクラッシュ/フリーズしない。

---

## M3: 例外ハンドリング/ロギング最適化
目的
- 広い `except` を減らし、デバッグ時に追跡可能なログへ。

タスク
- [ ] 例外を想定箇所に限定（DPG 終了時/未存在時など）し、それ以外は `logging.debug` で 1 行記録（環境変数 `PXD_DPG_DEBUG` で有効化）。
- [ ] `close()` 経路の順序と多重呼び出し耐性を簡潔にコメント化。

DoD
- 通常運用時はログ増加なし、デバッグ時のみ詳細が出る。

---

## M4: UX 微調整（任意）
目的
- 視認性・使い勝手の向上（仕様は最小限維持）。

タスク
- [ ] `help_text` があればツールチップ表示（任意）。
- [ ] 長文ラベルの折返し/トリム（簡易）。
- [ ] カテゴリの `default_open` を設定で切替（大量カテゴリ時の初期展開抑制）。
- [ ] vector 軸ラベル（x/y/z/w）の最小表示（任意）。

DoD
- 既存機能を壊さないこと。テーマ/配色は現状維持。

---

## M5: 更新負荷・スケール対策（任意）
目的
- 大量パラメータ時の更新効率を確保。

タスク
- [ ] 同値更新の間引き（Store 側の通知設計と齟齬がない範囲で）。
- [ ] まとめ更新時は `stage/unstage` を活用（既存踏襲）。

DoD
- 更新ストーム時でも UI の応答が保たれる。

---

## M6: テスト/CI（最小）
目的
- 代表型の往復同期、headless/未導入環境での生成/破棄スモークを担保。

タスク
- [ ] DPG 未導入環境でのスタブ生成/メソッド呼び出しスモーク（skip 条件付き）。
- [ ] 代表型（float/int/bool/enum/vector）の UI→Store→UI round-trip を最小ケースで検証。

DoD
- `pytest -q -m smoke` で緑。変更ファイル単位の Lint/Type も緑。

---

## M7: ドキュメント整備
目的
- 実装とドキュメントの同期を取り、保守性を上げる。

タスク
- [ ] `architecture.md` に DPG に一本化した旨と更新方針（キュー駆動/step 反映）を追記。
- [ ] 本計画の進捗（完了/保留/要確認）を継続更新。

DoD
- 実装と記述の差分がない。

---

進め方
- 本計画の「合意が必要な設計ポイント」への回答をいただき次第、M0→M2 の順で着手（M1 は回答内容で分岐）。
- 各マイルストーンは小さめの PR に分割。変更ファイル限定で `ruff/black/isort/mypy` を実施し、必要に応じて最小テストを追加。
- 進捗/追加の確認事項は本ファイルに追記し、チェックボックスを更新する。

