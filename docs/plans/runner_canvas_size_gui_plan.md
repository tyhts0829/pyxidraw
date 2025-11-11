# 実装改善計画: Parameter GUI から run_sketch(canvas_size) を制御

目的: `src/api/sketch.py:106` の `run_sketch(..., canvas_size=...)` を Parameter GUI から操作できるようにする。
- 起動時に GUI 側の値でキャンバスサイズを決定（既定の最小実装）。
- オプションで実行中のキャンバスサイズ変更（ウィンドウ/射影の動的更新）にも対応できる設計にする（第二段階）。

## スコープ/前提
- スコープ: API ランナー層（`api.sketch` と `api.sketch_runner.*`）、Parameter GUI 層（`engine/ui/parameters/*`）、レンダラ最小変更（第二段階）。
- 既存の優先順位原則（shape/effect）: 「明示引数 > GUI > 既定値」。Runner 系は明示仕様がないため、本変更では「use_parameter_gui=True の場合は GUI で上書き可」を基本とする（詳細は設計要点）。
- GL 初期化回避: `init_only=True` でヘッドレス検証ができる現状を維持。

## 方針（段階的）
- 第1段階（必須・軽量）: 起動前に GUI 値で `canvas_size` を解決し、ウィンドウ生成と射影行列に反映する。
- 第2段階（任意・拡張）: 実行中に GUI 変更を購読して、ウィンドウサイズと射影行列を動的更新（再初期化なし）。

## 設計要点
- 表示項目
  - プリセット選択（enum）: `util.constants.CANVAS_SIZES` のキー + `CUSTOM`。
  - カスタムサイズ（vector2）: `(width_mm, height_mm)`。`CUSTOM` 選択時のみ意味を持つ。
- 優先順位（Runner系）
  - use_parameter_gui=False: 既存通り引数のみ使用。
  - use_parameter_gui=True: GUI 側に override があればそれを採用。なければ引数値。
  - 必要であれば将来 `lock_canvas_size` のような引数で GUI 上書きを禁止できる余地を残す（今回は導入しない）。
- 値域
  - プリセット: `sorted(CANVAS_SIZES.keys()) + ["CUSTOM"]`
  - カスタムサイズ: `min=(10,10)`, `max=(2000,2000)`, `step=(1,1)`（mm 単位、実装は GUI 表示ヒントに留める）。
- 既存動作への影響
  - use_parameter_gui=False の場合は差分なし。
  - True でも override 無しなら差分なし。

## 実装タスク（チェックリスト）

[第1段階] 起動時 override（必須）
- [ ] ヘルパ追加: GUI ランナー項目登録
  - 追加: `src/api/sketch_runner/params.py` に `register_runner_canvas_params(store, default_canvas_size)` を新設
    - `runner.canvas_preset`（enum, choices=CANVAS_SIZES.keys()+CUSTOM, default=引数文字列ならそのまま／タプルなら CUSTOM）
    - `runner.canvas_size`（vector2, default=(w,h)）
- [ ] 初期化順序の見直し（軽微）
  - `src/api/sketch.py:158-181` 付近の Parameter GUI 初期化を「キャンバス解決前」に移動。
  - `ParameterManager` 生成直後に上記 `register_runner_canvas_params()` を呼ぶ（`initialize()` の前）。
- [ ] 実値解決フローの挿入
  - `ParameterManager.initialize()` 完了後、Store から `runner.canvas_preset/runner.canvas_size` を読み取り、実際に用いる `canvas_size` を決定。
  - `src/api/sketch_runner/utils.py:26` の `resolve_canvas_size()` を用いて `(width_mm, height_mm)` に正規化し、以降の `window_width/height` と射影行列に反映。
- [ ] ドキュメント最小更新
  - `src/api/sketch.py` のモジュール先頭ドキュメントに「Parameter GUI からキャンバスサイズを変更可能（use_parameter_gui=True 時）」の一文を追記。

[第2段階] 実行中の動的更新（任意）
- [ ] レンダラ API 拡張（小）
  - `src/engine/render/renderer.py` の `LineRenderer` に `set_projection(matrix)` を追加（シェーダ uniform `projection` を更新）。
- [ ] 変更購読ヘルパ
  - 追加: `src/api/sketch_runner/params.py` に `subscribe_canvas_size_changes(parameter_manager, rendering_window, line_renderer, pyglet_mod, render_scale)` を新設。
  - ストア変更で `runner.canvas_*` に関係があれば、`build_projection()` 再計算 + `rendering_window.set_size(px)` + `line_renderer.set_projection()` を `pyglet.clock.schedule_once` で UI スレッド適用。
- [ ] 線太さの扱い（今回の仕様では据え置き）
  - 将来の拡張として「mm 基準で線幅固定」を検討可能だが、当面は clip 空間値（指定値）を維持。

## 変更予定ファイル
- src/api/sketch.py:106, 158-181, 161-167（初期化順/実値解決の挿入）
- src/api/sketch_runner/params.py（ヘルパ2種の追加）
- src/engine/ui/parameters/manager.py（変更なし予定。Runner 用 Descriptor はヘルパ経由で登録）
- src/engine/render/renderer.py（第二段階で `set_projection` 追加）
- src/api/sketch_runner/utils.py（既存の `resolve_canvas_size`, `build_projection` を再利用）

## テスト計画（編集ファイル優先）
- 単体（smoke）
  - [ ] `tests/ui/parameters` に Runner 項目の登録を検証する軽いテストを追加（Store に `runner.canvas_*` が載る）。
- 起動時 override
  - [ ] `run_sketch(..., init_only=True, use_parameter_gui=True)` の経路で、GUI 側の override を与えたときに `resolve_canvas_size` に渡る値が置換されることを検証（必要なら分離関数化して単体テスト）。
- 動的更新（任意）
  - [ ] `subscribe_canvas_size_changes` のコールバック単体をモックで検証（`set_size`/`set_projection` が呼ばれる）。

## 安全性/リスク
- use_parameter_gui=False では一切の影響なし。
- Dear PyGui 未導入時は Parameter GUI 自体が無効だが、ヘルパは import 遅延とフェイルソフト設計（既存方針）を踏襲。
- 変更は API 互換（引数追加なし）。

## 確認事項（要指示）
- Runner 側の優先順位: 「use_parameter_gui=True なら GUI で上書き可」で問題ないか。
- プリセット名称: `CANVAS_SIZES` のキーをそのまま表示でよいか（例: `A5_LANDSCAPE`）。別名/日本語表記が必要か。
- 第二段階（実行中変更）まで実装するか。まずは第1段階のみでよいか。

---
以上の計画で進めてよいかご確認ください。修正要望があれば反映します。
