# Sketch Runner ユニットテスト計画（チェックリスト）

目的
- [ ] `api.sketch_runner` 各モジュールを単体でテスト可能にし、回帰検知と局所変更の安全性を高める
- [ ] 実 GL/ウィンドウ依存をモック化し、ヘッドレス環境で実行可能にする

スコープ/方針/配置
- [ ] スコープ: `src/api/sketch_runner/{utils,export,midi,render,params,recording}.py`
- [ ] 非スコープ: 実 GL・実ファイル出力のゴールデン差分（e2e に委譲）
- [ ] 方針: monkeypatch/スタブで依存差し替え、分岐・入出力・副作用を検証
- [ ] 配置: `tests/api/sketch_runner/test_*.py`

---

## utils.py（純粋関数）
- [ ] resolve_fps — 明示指定（None/正/0/負/文字/不正）→ 最終値は >=1
- [ ] resolve_fps — 設定読込（load_config を monkeypatch）→ 値反映/例外時フォールバック
- [ ] resolve_canvas_size — 既知キー/タプル/未知キー・不正タプルでの分岐
- [ ] build_projection — 形状/要素符号（Y反転）の確認
- [ ] hud_metrics_snapshot — 正常/例外時（ゼロ埋め）の戻り値

## export.py（保存ヘルパ）
- [ ] make_gcode_export_handlers — 実行中ガード（2連続 start→2回目はHUD通知）
- [ ] make_gcode_export_handlers — 対象なし（front None/is_empty）でHUD警告
- [ ] make_gcode_export_handlers — 正常遷移（completed/failed/cancelledでHUDとunschedule）
- [ ] make_gcode_export_handlers — submitがRuntimeErrorのフォールバック
- [ ] save_png_screen_or_offscreen — mode="screen" 引数透過の検証
- [ ] save_png_screen_or_offscreen — mode="quality" で mgl_context/draw 必須・引数透過
- [ ] save_png_screen_or_offscreen — 未知モードで ValueError

## midi.py（初期化フォールバック）
- [ ] setup_midi(False) — NullMidi（tick no-op、snapshot {}）
- [ ] setup_midi(True) — 正常経路（controllersあり/MidiServiceスタブでsnapshot転送）
- [ ] setup_midi(True) — 例外/未接続で警告ログ→NullMidi

## render.py（初期化/色決定）
- [ ] ModernGL/Window/Renderer の生成呼び出し（モックで記録）
- [ ] 背景色: 引数/設定/既定の適用
- [ ] 線色: 設定→自動（背景輝度）→引数の優先順
- [ ] ヘックス/タプル入力の正規化

## params.py（GUI連携）
- [ ] make_param_snapshot_fn — extract_overrides 呼び出し/例外時 None
- [ ] apply_initial_colors — 背景/線/HUD の初期適用
- [ ] subscribe_color_changes — schedule_once 経由で UI スレッド適用（各キー）

## recording.py（品質モード）
- [ ] enter_quality_mode — unschedule(frame), schedule_interval(quality_cb) が呼ばれる
- [ ] enter_quality_mode — quality_cb(dt) 実行で Tickable.tick(fixed_dt) 順に呼ばれる
- [ ] leave_quality_mode — unschedule(quality_cb), schedule_interval(frame) が呼ばれる
- [ ] leave_quality_mode — worker_count 反映の WorkerPool 再構築

---

DoD / 実行例
- [ ] 変更ファイル優先: `pytest -q -k 'sketch_runner and utils'`
- [ ] ユニット群: `pytest -q tests/api/sketch_runner`
- [ ] スモーク併用: `pytest -q -m smoke -k sketch_runner`

注意
- [ ] 実 GL/ウィンドウ生成は行わない（モック）
- [ ] 保存系は呼び出しと引数の検証中心（実ファイル出力なし）

