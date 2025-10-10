# HUD 色コントロール（GUI）実装計画

目的
- 画面左下の HUD（テキスト/メータ）の色を Parameter GUI から即時に調整可能にする。
- 既存の色制御（背景/線色）と同じ UX（0–255 RGB ピッカー、保存/復帰、即時反映）に統一。

方針（統一ルール）
- GUI 表示: 0–255 の整数 RGB（no_alpha=True）。
- 内部保存・描画適用: 0–1 の RGBA（共通パーサで正規化）。
- 優先順位（適用順）: 引数 > 保存値 > config（今回の HUD は保存値 > config のみ、引数は導線なし）。
- 反映タイミング: 起動直後（初期適用）＋ GUI 操作時（即時）。

追加する runner.* パラメータ（保存/復帰対象）
- `runner.hud_text_color`（0–1 RGBA）: HUD テキスト色。既定は `configs/default.yaml: hud.text_color`。
- `runner.hud_meter_color`（0–1 RGBA）: HUD メータ前景色。既定は `configs/default.yaml: hud.meters.meter_color_fg`（RGB）に `a=1.0` 補完。

実装タスク（TODO）
- [x] Parameter 登録（初期化）
  - [x] `ParameterManager.initialize()` で `runner.hud_text_color` / `runner.hud_meter_color` を `vector(RGBA)` で登録。
  - [x] 既定は `configs/default.yaml` を読み、0–1 RGBA に正規化して設定。
- [x] GUI 構築（HUD セクション）
  - [x] `dpg_window.build_display_controls()` 内に HUD セクション（または `build_hud_controls()` を新設）。
  - [x] 2カラムテーブルで「左=ラベル（Text / Meter）」「右=ColorEdit（0–255 RGB, no_alpha=True）」の1行配置。
  - [x] 初期表示で `force_set_rgb_u8(tag, [R,G,B])` を呼び、整数 RGB を強制反映。
  - [x] 変更時は `store_rgb01(pid, app_data)` で 0–1 RGBA へ正規化して `ParameterStore` に保存。
- [x] 初期適用（起動直後）
  - [x] `api/sketch.py` の初期適用処理内（Overlay 作成後）で、Store から HUD の 2 色を取得して Overlay へ適用。
- [x] 実行時反映（購読）
  - [x] Store 変更イベント（subscribe ラッパ）で `runner.hud_text_color` / `runner.hud_meter_color` 変更時に Overlay に即時適用。
  - [x] GL/描画コンテキストの安全性のため、`pyglet.clock.schedule_once` 経由で適用（背景/線色と同様）。
- [x] Overlay への適用 API 追加
  - [x] `OverlayHUD.set_text_color(rgba01: tuple[float,float,float,float])` を追加。
  - [x] `OverlayHUD.set_meter_color(rgb01: tuple[float,float,float])`（または `set_meter_style(color_rgb01, alpha_fg_u8?)`）。今回は色のみ。
- [ ] ドキュメント更新
  - [ ] `docs/user_color_inputs.md`: HUD 色の GUI 操作を追記（表示は 0–255、内部は 0–1）。
  - [ ] `architecture.md`: 起動時の保存値 → Overlay 初期適用の記述を HUD にも拡張。

コード変更見込み（ファイル別）
- `src/engine/ui/parameters/manager.py`
  - HUD 用 runner.* の Descriptor 登録（vector RGBA）。
- `src/engine/ui/parameters/dpg_window.py`
  - HUD セクション追加（2カラム1行配置×2項目）。
  - 既存ヘルパ `force_set_rgb_u8` / `store_rgb01` の再利用。
  - `sync_display_from_store()` 対象に HUD を追加。
- `src/engine/ui/hud/overlay.py`
  - Setter 追加: `set_text_color()` / `set_meter_color()`（内部色フィールドを更新）。
- `src/api/sketch.py`
  - 初期適用 `_apply_initial_colors()` 内で HUD の 2 色を Overlay に適用。
  - Store 購読ハンドラで HUD の 2 色変更時に Overlay を更新（schedule_once）。
- `docs/user_color_inputs.md` / `architecture.md`
  - 仕様追記。

テスト（方針）
- 起動 → GUI を開かずに HUD が保存色で表示される（視覚確認/簡易ログ）。
- GUI で HUD テキスト/メータ色を変更 → 即時反映（視覚確認）。
- 終了 → 再起動で保存/復帰（JSON に runner.hud_* が保存される）。

受け入れ条件（DoD）
- HUD テキスト/メータの色が GUI から変更でき、描画に即時反映される。
- 保存→終了→再起動で GUI/描画とも前回色に復帰。
- 既存の背景/線色の制御に影響がない（退行なし）。

リスク/留意点
- Dear PyGui のバージョン差（引数名/表示の差異）は既存方針（getattr ベース）で回避。
- alpha の扱いは当面 GUI 非対応（no_alpha=True）。将来的な alpha 調整は別計画で。

作業順（推奨）
1) Overlay に setter を追加 → 2) ParameterManager で runner.hud_* を登録 → 3) dpg_window に HUD セクション追加 → 4) sketch の初期適用と購読で Overlay へ反映 → 5) ドキュメント更新。
