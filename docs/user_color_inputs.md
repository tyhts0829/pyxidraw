どこで: プロジェクト全体（公開APIと設定）。
何を: ユーザーが指定できる「色」入力ポイントの一覧（型・適用範囲・参照実装）。
なぜ: 配色変更の入口を一目で把握し、迷いなく適切な箇所を変更できるようにするため。

**公開API（実行時）**
- `api.run` / `api.sketch.run_sketch` の引数で指定
  - `line_color`: 線色。
    - 受理型: RGBA タプル（各成分 0.0–1.0）、RGB タプル（α=1.0 補完）、ヘックス文字列（`#RRGGBB` / `#RRGGBBAA`、`0x`/接頭辞なしも可）。
    - 実装参照: `src/api/sketch.py:135`, `src/api/sketch.py:360`, `src/api/sketch.py:377`, `src/api/sketch.py:420`。
  - `background`: 背景色。
    - 受理型: RGBA タプル（0.0–1.0）、ヘックス文字列（`#RRGGBB` / `#RRGGBBAA`）。
    - 実装参照: `src/api/sketch.py:392`（正規化→`RenderWindow`へ適用）。
  - 利用例: `main.py:43` 以降（`background="222831"`, `line_color="EEEEEE"`）。

**設定ファイル（config.yaml / configs/default.yaml）**
- Parameter GUI（Dear PyGui）テーマ色
  - 入口: `parameter_gui.theme.colors.{key}`（RGBA 配列）。
  - 主なキー: `text`, `window_bg`, `frame_bg`, `frame_bg_hovered`, `frame_bg_active`, `header`, `header_hovered`, `header_active`, `accent`, `accent_active`。
  - 値の型: `[r,g,b,a]`（0–255 または 0.0–1.0 のいずれも可。内部で 0–255 RGBA に正規化）。
  - 実装参照: `configs/default.yaml:200` 以降の例、`src/engine/ui/parameters/manager.py:64-81`, `src/engine/ui/parameters/dpg_window.py:662-689`。

- HUD（オーバーレイ）メータの色/不透明度
  - 入口: `hud.meters.meter_color_fg`（RGB, 0–255）、`hud.meters.meter_alpha_fg`（0–255）、`hud.meters.meter_alpha_bg`（0–255）。
  - 効果: 右側のバー表示の前景色/不透明度および背景の不透明度を変更。
  - 実装参照: `configs/default.yaml:148-178`, `src/engine/ui/hud/overlay.py:118-146`。

**開発者向け（直接利用時のみ）**
- HUD テキスト色
  - 入口: `engine.ui.hud.OverlayHUD(..., color=(r,g,b,a))`（0–255）。
  - 備考: 公開API `api.run` 経由ではパラメータを露出していないため、直接コンポーネントを組み立てる場合のみ指定可能。
  - 実装参照: `src/engine/ui/hud/overlay.py:24`, `src/engine/ui/hud/overlay.py:36-45`。

**注意（現状未使用の設定）**
- `canvas.background_color`（`configs/default.yaml:8-12`）
  - 現実装のランナーはこのキーを参照せず、背景は `api.run(..., background=...)` 引数からのみ設定される。
- `status_manager.color`（`configs/default.yaml:19-41`）
  - 旧UI想定の設定。現在の HUD 実装からは参照されていない。

---

補足:
- シェーダの色は `LineRenderer` 初期化時に `line_color`（RGBA 0–1）として一括適用される（`src/engine/render/renderer.py:24-43`）。
- ヘックス文字列は `#`/`0x` あり・なしを許容し、`RRGGBB` または `RRGGBBAA` を受理（`src/api/sketch.py:360-374`）。
