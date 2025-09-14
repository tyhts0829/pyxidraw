# MIDI コントローラ別 GUI 追加計画

目的: 旧 `previous_code/ui` にあった「コントローラごとの GUI 表示」を、現行アーキテクチャ（api/engine/ui/pipeline 構成）へ安全に移植し、非侵襲に統合する。

---

## ゴール / 非ゴール

- ゴール
  - 接続された各 MIDI コントローラに対応する簡易 GUI（フェーダー/ノブ/ボタン表示）をオーバーレイで描画。
  - 表示のオン/オフ切替、レイアウトはデバイス別に最小構成で用意。
  - UI は I/O へ直接依存しない（依存注入: `cc_snapshot: () -> Mapping[int, float]` と `DeviceSpec`）。
  - 既存のレンダリング/ワーカー/IO と競合しない（描画順と Tickable 統合）。
- 非ゴール
  - 旧実装の ModernGL 用 GUI シェーダ/描画コードの移植（初期段階は `pyglet` の軽量描画で実装）。
  - 完全な双方向操作（GUI から CC を送信）は後続フェーズ。まずは可視化と最低限のボタン反応まで。

---

## 追加コンポーネント（設計）

- `engine/ui/widgets.py`
  - `BaseWidget`: `tick(dt)`, `draw()`, `set_value(v: float)`, `label`。
  - `FaderWidget`, `KnobWidget`, `LedWidget`（`pyglet.shapes` + `Label` で実装）。
  - mm/px 換算はランナーから渡されるピクセル座標系を前提（現行はウィンドウピクセル座標）。

- `engine/ui/controllers/base.py`
  - `ControllerUI` 抽象: `name`, `widgets: list[BaseWidget]`, `layout(origin: tuple[int,int])`。
  - `apply_cc(cc: Mapping[int, float])`: 自身の `cc_map` に応じて `set_value` を反映。

- `engine/ui/controllers/{tx6,fb8n,fb16n,grid,arc}.py`
  - 旧 `previous_code/ui/controllers/*` を参考に、最小レイアウト（行/列/間隔）だけ定義。
  - CC 名/番号は `configs/default.yaml` の `midi_devices[].cc_map` を尊重。

- `engine/ui/controller_registry.py`
  - `register(name: str, ui_cls: type[ControllerUI])` / `create(name: str, spec: DeviceSpec) -> ControllerUI`。
  - 初期登録: `tx6`, `fb8n`, `fb16n`, `grid`, `arc`。

- `engine/ui/midi_panel.py`
  - `MidiControllersPanel`（`Tickable` + `draw()` 実装）。
  - 受け取り: `device_specs: list[DeviceSpec]`, `cc_snapshot: Callable[[], Mapping[int,float]]`。
  - 機能: `tick()` で `cc = cc_snapshot()` を取得 → 各 `ControllerUI.apply_cc(cc)` を呼ぶ。
  - 表示切替: 既定オフ、`U` キーでトグル（`api.runner` でバインド）。

- `api/runner.py`（統合）
  - MIDI 構築後に `device_specs` を抽出して `MidiControllersPanel` を生成。
  - `FrameClock` へ `panel` を追加し、`rendering_window.add_draw_callback(panel.draw)` を登録。
  - UI は I/O 型を import しない。`device_specs` はランナー側で `engine.io.manager` から最小情報を取り出して渡す。

---

## データフロー

`engine.io.MidiService.snapshot()` → `MidiControllersPanel.tick()` → `ControllerUI.apply_cc()` → `Widget.set_value()` → `Widget.draw()`

---

## レイアウト方針（初期）

- `tx6`: 6 フェーダー + 3×6 ノブ + 一部ボタンを行列で簡略表示。
- `fb8n`: 8 フェーダー（横並び）。
- `fb16n`: 16 フェーダー（2 段 × 8）。
- `grid`: 4×4 ノブ。
- `arc`: 4 エンコーダを円環ゲージで（初期は円弧ではなく円 + 進捗バーで代替）。

---

## 互換性と責務分離

- UI は `engine.io` を import しない（依存注入のみ）。
- 既存の `OverlayHUD` と共存（描画順は HUD の後ろ → HUD の前景テキストを上に）。
- `architecture.md` に「UI: MIDI パネル」を追記（導入後に差分同期）。

---

## 実装ステップ（チェックリスト）

- [ ] 0. 仕様合意（本ファイルの確認と修正点の合意）
- [ ] 1. UI プリミティブ実装（`widgets.py`: Fader/Knob/Led）
- [ ] 2. `ControllerUI` 基底とレイアウトユーティリティ（グリッド配置）
- [ ] 3. デバイス別 UI 実装（`tx6`, `fb8n`, `fb16n`, `grid`, `arc`）
- [ ] 4. レジストリ実装と初期登録（`controller_registry.py`）
- [ ] 5. `MidiControllersPanel` 実装（`tick/draw/toggle`）
- [ ] 6. `api/runner.py` 統合（デバイス検出→`DeviceSpec`抽出→パネル生成→描画登録）
- [ ] 7. 入力系イベント登録（`U` キーで表示切替）
- [ ] 8. 軽量スモークテスト（MIDI 無し/有りでエラー無）
- [ ] 9. ドキュメント更新（`architecture.md` 差分同期、`README.md` 使い方に追記）
- [ ] 10. 仕上げ（`ruff/black/isort/mypy` 変更ファイル、スクリーンショット `screenshots/` 追加任意）

---

## テスト計画

- 単体
  - Widget の座標計算・値→描画サイズ変換（純粋関数部）を pytest で検証。
- 結合（スモーク）
  - `run(..., use_midi=False)` でウィンドウ生成と描画コールバックがエラー無。
  - MIDI 実機接続時に例外が出ない（`midi_strict=False`）。
- スタブ
  - `cc_snapshot` に固定辞書（擬似 CC）を与え、GUI が想定のレイアウトで描画されることを目視確認。

---

## 追加の設計詳細

- 表示切替と描画順
  - `rendering_window.add_draw_callback(line_renderer.draw)` → `panel.draw` → `overlay.draw` の順。
  - `U` キーで `panel.visible` をトグル。
- 座標系
  - UI はウィンドウピクセル座標（左上原点）で扱う。既存ジオメトリ描画は正射影・中心座標だが干渉しない。
- パフォーマンス
  - `pyglet.graphics.Batch` を用いたバッチ描画。ウィジェット数は数百程度で十分軽量。
- エラーハンドリング
  - 未登録 `controller_name` は汎用 `FaderGrid` にフォールバック（`cc_map` の数だけ自動レイアウト）。

---

## 変更予定ファイル

- 追加
  - `src/engine/ui/widgets.py`
  - `src/engine/ui/controllers/base.py`
  - `src/engine/ui/controllers/{tx6,fb8n,fb16n,grid,arc}.py`
  - `src/engine/ui/controller_registry.py`
  - `src/engine/ui/midi_panel.py`
- 変更
  - `src/api/runner.py`（統合点の最小変更）
  - `architecture.md`（導入後に更新）

---

## オープンな確認事項（要ご指示）

- 表示トグルのデフォルト: 起動時オフで良いか（提案: オフ）。
- 旧 GUI の見た目再現度: 初期は簡易表示でよいか、どのコントローラを優先するか。
- `cc_map` の名称優先/番号優先: ラベルは名称（存在すれば）を優先でよいか。
- 将来の双方向化（GUI→MIDI 送信）: 当面スコープ外で問題ないか。

---

## 運用・品質ゲート

- 変更ファイルに対してのみ `ruff/black/isort/mypy` を実行。
- 公開 API 影響なしのためスタブ生成は不要（`api` 未変更想定）。
- PR ではスクリーンショットと簡単な動作説明を添付。

---

## 作業見積（ラフ）

- 実装/結合: 0.5–1.0 日
- テスト/調整: 0.5 日
- ドキュメント/整備: 0.5 日

---

更新履歴
- v0.1 初稿（計画・チェックリスト）

