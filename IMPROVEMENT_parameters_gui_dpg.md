# Parameters GUI: Dear PyGui 完全移行計画（最終案）

目的

- 既存の `pyglet` ベースのパラメータ GUI を Dear PyGui（DPG）へ完全移行し、見た目/操作性/保守性を向上する。
- 既存の `ParameterStore / ParameterDescriptor / RangeHint` はそのまま利用し、値は実値（float/int/bool/vector）で受け取る。`__param_meta__` の min/max/step は RangeHint として表示レンジに反映（クランプは表示上のみ）。

採用ライブラリ / 前提

- Dear PyGui（即時モード GUI, GPU アクセラレーション, 豊富なウィジェット, テーマ/ドッキング）
- 依存: `dearpygui`。CI/ヘッドレスでは import/起動ガードを実装し、落ちないこと。

方針（完全移行）

- 切替フラグや並存は行わない。`pyglet` 実装は削除し、DPG 実装へ一本化する。
- `ParameterStore` の購読/更新を唯一のデータフローとし、UI は即時モード前提の stateless 設計に寄せる。
- テスト/CI は DPG 依存箇所を headless でも import/生成/破棄できるよう最小限に留める（ビューポート非表示運用）。

対象範囲

- 本移行の対象は「パラメータ編集パネル」のみ。描画ウィンドウ（`engine.core.render_window`）は現状通り pyglet を継続利用する。

最終ファイル構成（予定）

- 新規: `src/engine/ui/parameters/dpg_window.py`
  - `ParameterWindow`（DPG 実装）: 初期化/マウント/イベント/破棄
  - Descriptor → DPG ウィジェット生成、購読/反映、テーマ適用
- 置換: `src/engine/ui/parameters/window.py`
  - 中身を DPG 実装へ差し替え（`ParameterWindow` をエクスポート）
- 削除: `src/engine/ui/parameters/panel.py`（pyglet ウィジェット群）
- 既存維持: `state.py`, `manager.py`, `controller.py`, `runtime.py`, `value_resolver.py`, `introspection.py`
- テスト: `tests/ui/parameters/test_dpg_window.py`（新規・最小）/ 既存の `panel` 依存テストは移行/削除

UI 仕様（DPG マッピング）

- float/int
  - `add_slider_float` / `add_slider_int`。`RangeHint.min/max/step` を反映。内部値は実値を保持（変換なし）。
  - 変更は `store.set_override(id, value)` に集約。
- bool
  - `add_checkbox`（見た目はテーマで最終調整）。
- enum
  - 選択肢 <= 5: `add_radio_button(items=choices)`、> 5: `add_combo(items=choices)`。
  - 値は文字列で保持し、`store.set_override` で反映。
- vector（x/y/z/w）
  - `add_input_float3/4` もしくは横並びスライダを使用。グループ見出し＋短ラベル（x/y/z/w）。
- グルーピング
  - `add_collapsing_header` を使用し、`scope.name#index` を見出しにして配下へ各 param を並べる。
- 変更マーカー / Reset
  - 現在値 ≠ 既定値ならラベル色を変える/● 表示。右クリック context menu に Reset。
- ツールチップ
  - `help_text` があれば `add_tooltip` で表示。
- ショートカット
  - `handler_registry` を用い、←/→ で enum 移動、1..9 でクイック選択（範囲外は無視）。
- テーマ
  - ダークテーマを既定とし、コントラスト/角丸/パディング/フォントを設定。

イベント/データフロー

- 初期化: `create_context` → ウィジェット定義 → `create_viewport`（非表示でも可） → `setup_dearpygui`。
- UI → Store: 各ウィジェットの callback で `store.set_override`。
- Store → UI: `store.subscribe` で差分のみ `dpg.set_value`。無限ループ回避のため同値更新は抑止。
- 終了: `destroy_context`。テストではビューポートを開かず import/生成/破棄のみ実施。

実装チェックリスト（完全移行）

1. DPG ウィンドウの土台

- [x] `dpg_window.py` 追加（`ParameterWindow`: `__init__/mount/set_visible/close`）。
- [x] DPG の context/viewport/lifecycle ユーティリティを内包（pyglet 連携 or スレッド駆動）。

2. 最小ウィジェット（float/int/bool/enum）

- [ ] Descriptor 群からウィジェットを生成（行レイアウト + collapsing header）。
- [x] float/int: `min/max/step` を RangeHint から反映、内部値は実値。
- [x] bool: `add_checkbox`。
- [x] enum: 候補数で radio/combo を自動選択。

3. Store 連携（双方向）

- [x] UI → Store: すべて callback で `set_override`。
- [x] Store → UI: `subscribe` で `set_value`（差分のみ）。
- [x] `param_id ↔ dpg_item_id` マップを保持。

4. 変更マーカー / Reset / ツールチップ

- [ ] 既定値との差分をラベル装飾で表示。
- [x] Reset: 取り止め（実装しない）。
- [x] ツールチップ（help_text）: 非表示（実装しない）。

5. レイアウト/テーマ

- [x] パディングのみ整備（WindowPadding/FramePadding/ItemSpacing）。
- [x] enum の選択時コントラスト最適化（Radio/Selectable の色のみ調整）。
- [ ] ダークテーマ/フォント/角丸は不採用。

6. ベクトル/複合

- [x] `input_float3/4` または横並びスライダで x/y/z/w を 1 行化。
- [ ] 軸ラベルの最小表示を実装。

7. ショートカット/操作性

- [ ] ←/→ で enum 移動、1..9 でクイック選択。

8. 安定化

- [ ] 大量パラメータ時の更新間引き（同値更新抑止）。
- [ ] 長文ラベルの折返し/ツールチップ化。
- [ ] 例外ガード（DPG 側の例外でも落とさない）。

9. 削除/置換（完全移行）

- [x] `src/engine/ui/parameters/window.py` の実装を DPG へ差し替え。
- [ ] `src/engine/ui/parameters/panel.py` を削除（参照テストも整理）。
- [ ] `tests/ui/parameters/test_slider_widget.py` 等の pyglet 依存テストを削除または DPG 版に移行。

10. テスト/CI

- [ ] headless で import/生成/破棄のみ行うスモークを追加（DPG 未導入環境は skip）。
- [ ] 代表型（float/int/bool/enum/vector）の round-trip（UI→Store→UI）テスト。

11. ドキュメント/整備

- [x] `README/AGENTS/architecture.md` に「パラメータ GUI は DPG」を明記し差分更新。
- [x] 変更ファイル限定で `ruff/black/isort/mypy` を通す。

変更対象一覧（最終）

- 追加: `src/engine/ui/parameters/dpg_window.py`
- 置換: `src/engine/ui/parameters/window.py`（DPG 実装に差し替え）
- 削除: `src/engine/ui/parameters/panel.py`
- 更新: `tests/ui/parameters/*`（pyglet 依存の削除/移行）
- 変更なし: `ParameterStore/ValueResolver/Introspector`（公開 API は維持）

リスクと緩和

- GPU/依存環境差: headless/CI ではビューポート非表示 + import ガードで回避。
- 互換性: 既存パラメータ仕様（実値/RangeHint）は不変。UI 層のみ変更。
- パフォーマンス: create/update はまとめて実行し、差分更新で負荷を低減。

受け入れ条件（DoD）

- 既存機能（float/int/bool/enum/vector/グループ/Reset/ツールチップ/ショートカット）を DPG で再現。
- 既定起動で DPG パラメータウィンドウが表示される（切替フラグ不要）。
- 変更ファイルに対する `ruff/black/isort/mypy` が緑。
- headless/CI でも import 失敗せずスモークが通る。

運用/起動

- 通常起動: `python main.py`（パラメータ GUI は DPG）。
- テスト環境（例）: ビューポートを開かずに `create_context` のみ実行して生成/破棄を確認。

メモ（後続拡張）

- Docking による「プロパティパネル + ライブプレビュー」レイアウト。
- プリセット保存/読込、フィルタ検索、キーバインドカスタム。
- テーマスイッチ（ライト/ダーク）。

---

本「完全移行」案で進めてよいか確認してください。承認後、ステップ 1 から着手します。
