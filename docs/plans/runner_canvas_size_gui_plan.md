# 実装改善計画: Parameter GUI からキャンバスとウィンドウサイズを制御

目的: `src/api/sketch.py:106` の `run_sketch(..., canvas_size=...)` とウィンドウの幅/高さを Parameter GUI から操作可能にする。
- 起動時に GUI 側の値でキャンバスサイズ（mm）とウィンドウ寸法（px）を決定（必須）。
- プリセット（A5 など）はプルダウン選択（enum）で指定可能にする（必須）。
- 任意で、実行中の変更を取り込みウィンドウ/射影を動的に更新できるようにする（第二段階）。

## スコープ/前提
- スコープ: API ランナー層（`api.sketch` と `api.sketch_runner.*`）、Parameter GUI 層（`engine/ui/parameters/*`）、レンダラ最小変更（第二段階）。
- 既存の優先順位原則（shape/effect）: 「明示引数 > GUI > 既定値」。Runner 系は明示仕様がないため、本変更では「use_parameter_gui=True の場合は GUI で上書き可」を基本とする（詳細は設計要点）。
- GL 初期化回避: `init_only=True` でヘッドレス検証ができる現状を維持。

## 方針（段階的）
- 第1段階（必須・軽量）: 起動前に GUI 値で「キャンバス（mm）」と「ウィンドウ（px）」を解決し、ウィンドウ生成と射影行列に反映。
- 第2段階（任意・拡張）: 実行中に GUI 変更を購読して、ウィンドウサイズ（px）と射影行列（mm）を動的更新（再初期化なし）。

## 設計要点
- 表示項目（Runner グループ）
  - キャンバス（mm）
    - `runner.canvas_preset`（enum, choices=`util.constants.CANVAS_SIZES.keys()` + `CUSTOM`）
    - `runner.canvas_size_mm`（vector2, `(width_mm, height_mm)`）: `CUSTOM` 選択時のみ意味を持つ
  - ウィンドウ（px）
    - `runner.window_mode`（enum: `SCALE` | `PIXELS`）
    - `runner.render_scale`（float, px/mm）: `SCALE` のとき使用
    - `runner.window_size_px`（vector2, `(width_px, height_px)`）: `PIXELS` のとき使用
- 解決ロジック（起動時）
  1) キャンバス mm を決定（GUI > 引数 > 既定）。
  2) ウィンドウ px は `window_mode` で分岐。
     - SCALE: `window = canvas_mm * render_scale`（四捨五入・1px以上にクランプ）
     - PIXELS: `window = window_size_px`
  3) 射影行列は常に「キャンバス mm」に基づき `build_projection(canvas_mm)` を使用（ウィンドウ px とは独立）。
- 優先順位（Runner系）
  - use_parameter_gui=False: 既存通り引数のみ使用。
  - use_parameter_gui=True: GUI 側に override があればそれを採用。なければ引数値。
- 値域（GUI 表示ヒント）
  - `canvas_size_mm`: `min=(10,10)`, `max=(2000,2000)`, `step=(1,1)`（mm）
  - `render_scale`: `min=0.25`, `max=20.0`, `step=0.25`（px/mm）
  - `window_size_px`: `min=(64,64)`, `max=(8192,8192)`, `step=(1,1)`（px）
- アスペクト比の扱い（注意）
  - `PIXELS` でウィンドウ比がキャンバス比と異なる場合、描画は伸縮して見える（射影は mm 基準のため）。
  - 本実装では「警告/補正はしない」。将来の拡張でレターボックスや自動比率固定を検討し得る。
- 既存動作への影響
  - use_parameter_gui=False では差分なし。True でも override 無しなら差分なし。

## 実装タスク（チェックリスト）

[第1段階] 起動時 override（必須）
- [ ] ヘルパ追加: GUI ランナー項目登録
  - 追加: `src/api/sketch_runner/params.py` に `register_runner_canvas_window_params(store, defaults)` を新設
    - `runner.canvas_preset`（enum; choices=CANVAS_SIZES.keys()+CUSTOM; 既定=引数の型で決定）
    - `runner.canvas_size_mm`（vector2; 既定=(w,h)）
    - `runner.window_mode`（enum: SCALE|PIXELS; 既定=SCALE）
    - `runner.render_scale`（float; 既定=引数の `render_scale`）
    - `runner.window_size_px`（vector2; 既定=引数から計算した px）
- [ ] 初期化順序の見直し（軽微）
  - `src/api/sketch.py:158-181` 付近の Parameter GUI 初期化を「キャンバス解決前」に移動。
  - `ParameterManager` 生成直後に上記ヘルパで Runner パラメータを登録してから `initialize()` 実行。
- [ ] 実値解決フローの挿入
  - `ParameterManager.initialize()` 完了後、Store から `runner.canvas_* / runner.window_*` を読み取り、キャンバス mm とウィンドウ px を決定。
  - `resolve_canvas_size()` でキャンバス mm を正規化、`build_projection()` を生成。
  - `window_mode` に応じて `window_width/height` を決定（SCALE: mm×scale、PIXELS: 指定値）。
- [ ] ドキュメント最小更新
  - `src/api/sketch.py` のモジュール先頭ドキュメントに「Parameter GUI からキャンバスとウィンドウサイズを変更可能（use_parameter_gui=True 時）」の一文を追記。

[第2段階] 実行中の動的更新（任意）
- [ ] レンダラ API 拡張（小）
  - `src/engine/render/renderer.py` の `LineRenderer` に `set_projection(matrix)` を追加（uniform `projection` 更新）。
- [ ] 変更購読ヘルパ
  - 追加: `src/api/sketch_runner/params.py` に `subscribe_canvas_window_changes(parameter_manager, rendering_window, line_renderer, pyglet_mod)` を新設。
  - ストア変更で以下を検出し UI スレッドで適用：
    - キャンバス（preset/size_mm/render_scale or window_mode==SCALE）変更: 射影再計算 + 必要ならウィンドウ px 再計算/適用
    - ウィンドウ（window_mode==PIXELS / window_size_px）変更: `rendering_window.set_size(w,h)` のみ
  - 備考: Pyglet の `on_resize` により viewport は自動更新。射影は mm 変更時のみ更新。
- [ ] 線太さの扱い（据え置き）
  - 将来の拡張として「mm 基準の線幅固定」を検討可。当面は clip 空間値のまま。

## 変更予定ファイル
- src/api/sketch.py:106, 158-181, 161-167（初期化順/実値解決の挿入）
- src/api/sketch_runner/params.py（ヘルパ追加/購読）
- src/engine/ui/parameters/manager.py（変更なし予定。Runner 用 Descriptor はヘルパ経由で登録）
- src/engine/render/renderer.py（第二段階で `set_projection` 追加）
- src/api/sketch_runner/utils.py（既存の `resolve_canvas_size`, `build_projection` を再利用）

## テスト計画（編集ファイル優先）
- 単体（smoke）
  - [ ] Runner 項目の登録検証（Store に `runner.canvas_*`, `runner.window_*` が登録される）。
- 起動時 override
  - [ ] `run_sketch(..., init_only=True, use_parameter_gui=True)` 経路で GUI 値が採用されること（mm と px の両方）を検証（必要なら解決関数を分離）。
- 動的更新（任意）
  - [ ] `subscribe_canvas_window_changes` のモック検証（`set_size`/`set_projection` 呼び出し）。

## 安全性/リスク
- use_parameter_gui=False では一切の影響なし。
- Dear PyGui 未導入時は Parameter GUI 自体が無効だが、ヘルパは import 遅延とフェイルソフト設計（既存方針）を踏襲。
- 変更は API 互換（引数追加なし）。

## 確認事項（要指示）
- Runner の優先順位: 「use_parameter_gui=True なら GUI で上書き可」で問題ないか。
- プリセット名表示: `CANVAS_SIZES` のキーをそのまま（例: `A5_LANDSCAPE`）で問題ないか（日本語表記が必要なら指示ください）。
- ウィンドウモード: まずは `SCALE`/`PIXELS` の2択でよいか。既定は `SCALE` で問題ないか。
- 第二段階（実行中変更）まで実装するか。まずは第1段階のみでよいか。

---
以上の計画で進めてよいかご確認ください。修正要望があれば反映します。
