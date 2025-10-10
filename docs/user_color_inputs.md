どこで: プロジェクト全体（公開APIと設定）。
何を: ユーザーが指定できる「色」入力ポイントの一覧（型・適用範囲・参照実装）。
 なぜ: 配色変更の入口を一目で把握し、迷いなく適切な箇所を変更できるようにするため。

基本方針（色指定）
- ユーザーが触る入口は2系統のみとする。
  - スケッチ描画用: 公開API引数（`line_color`, `background`）。
  - UI テーマ用: 設定 `parameter_gui.theme.colors.*`。
- HUD の色は開発者向け（直接利用時のみ）。一般ユーザーは触らない前提。
- 受理形式は「Hex と RGBA(0–1)」を採用し、0–255 配列は非推奨（内部互換のみ）。
  - Hex: `#RRGGBB` / `#RRGGBBAA` / `0x` 接頭辞可。
  - RGBA(0–1): `(r, g, b[, a])`。`a` 省略時は 1.0。
- 推奨の使い分け:
  - 設定・ドキュメント例: Hex を推奨。
  - コード内の計算・補間: RGBA(0–1) を推奨。
- 優先順位は「明示引数 > 設定 > 既定値」。
- 表示の正規形は Hex（ログやスナップショットでの表示は Hex を基本とする）。
- すべての色指定は `configs/default.yaml` からも指定できるようにし、明示引数が無い場合は設定値を採用する（本ファイルのキー一覧を参照）。

クイックリファレンス（OK/NG）
- OK: `"#112233"`, `"#112233CC"`, `"0x112233"`, `(0.12, 0.34, 0.56)`, `(0.12, 0.34, 0.56, 1.0)`
- NG: `"#123"`（短縮表記は未対応）, `(255, 0, 0[, 128])`（0–255 は新規指定では非推奨）

設定例（configs/default.yaml 抜粋）
```
canvas:
  background_color: "#222831"   # RGBA でも可（0–1 配列推奨）
  line_color: "#EEEEEE"         # 新規キー。線色の既定（API 未指定時に適用）

parameter_gui:
  theme:
    colors:
      text: [0.95, 0.95, 0.95, 1.0]
      window_bg: "#212121"       # 共通パーサ導入後に Hex 受理予定

hud:
  meters:
    meter_color_fg: "#333333"    # 既存: RGB(0–255) / 今後: Hex も受理
    meter_alpha_fg: 220
    meter_alpha_bg: 120
  text_color: "#0000009B"        # 新規キー（予定）。HUD テキスト色
```

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
- ランタイム描画（線色/背景）
  - 入口: `canvas.line_color`, `canvas.background_color`。
  - 値の型: Hex または RGBA 0–1（0–255 配列は互換/非推奨）。
  - 優先順位: API 引数が未指定の場合のみ設定値を採用。
  - 備考: `canvas.line_color` は新設。`canvas.background_color` は既存をフォールバックに昇格。
  - 実装予定: `src/api/sketch.py` の色解決で設定を参照（明示引数が優先）。
- Parameter GUI（Dear PyGui）テーマ色
  - 入口: `parameter_gui.theme.colors.{key}`（RGBA 配列）。
  - 主なキー: `text`, `window_bg`, `frame_bg`, `frame_bg_hovered`, `frame_bg_active`, `header`, `header_hovered`, `header_active`, `accent`, `accent_active`。
  - 値の型: `[r,g,b,a]`（0–255 または 0.0–1.0）。現状は配列のみ受理。Hex 入力は今後、共通パーサ導入後に対応予定。
  - 推奨: 0.0–1.0 配列を推奨（0–255 は互換/非推奨）。
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
- `status_manager.color`（`configs/default.yaml:19-41`）
  - 旧UI想定の設定。現在の HUD 実装からは参照されていない。

---

補足:
- シェーダの色は `LineRenderer` 初期化時に `line_color`（RGBA 0–1）として一括適用される（`src/engine/render/renderer.py:24-43`）。
- ヘックス文字列は `#`/`0x` あり・なしを許容し、`RRGGBB` または `RRGGBBAA` を受理（`src/api/sketch.py:360-374`）。

---

ユーザビリティ改善計画（色指定）

目的: 入口ごとの差異と迷いを解消し、即時の視覚フィードバックで試行錯誤を加速する。

適用範囲: 公開API（実行時）/ Parameter GUI / HUD・設定 / ドキュメント。

優先順位: 1) 受理形式の統一と共通パーサ化 → 2) UI プレビュー → 3) 設定の整理 → 4) 拡張（短縮表記/プリセット）。

短期（即日〜小規模）
- 入力形式の一本化と明記
  - 受理形式: Hex（`#RRGGBB` / `#RRGGBBAA` / `0x` 接頭辞可）と RGBA 0–1 タプル。
  - 0–255 配列は内部互換のみ（非推奨）。
  - ドキュメントへクイックリファレンス（OK/NG 例）を追加。
- 色パーサの共通化
  - `_parse_hex_color_str` と `_normalize_color_param` を `util.color.parse_color` に切り出し、全入口で共有。
  - 実装参照: `src/api/sketch.py:520-575`（現行の色正規化）。
- 優先順位ルールの徹底
  - 「明示引数 > 設定 > 既定値」を明文化。
  - 実装参照: `src/api/sketch.py:140` 近傍（引数解決/フォールバック）。
- 設定からの全色指定に対応
   - `canvas.line_color` を追加、`canvas.background_color` をフォールバックとして参照。
   - Parameter GUI / HUD の色も共通パーサで Hex を受理（既存の配列指定は互換維持）。
- バリデーション/エラーメッセージの改善
  - Hex 桁不正・文字不正時に例と対処（例: `#RGB` ではなく `#RRGGBB`）を含めて返す。
- Do/Don’t を本ファイルに追加
  - Do: `#RRGGBBAA`、RGBA 0–1。 Don’t: 0–255 と 0–1 の混在、未使用キーへの依存。

中期（UI 強化）
- ライブプレビュー（色ピッカー）
  - Parameter GUI に `line_color`/`background` のピッカーとミニプレビューを追加（相互に Hex/RGBA 表示）。
  - 実装参照: `src/engine/ui/parameters/dpg_window.py:620-760`（テーマ色適用）, `src/engine/ui/parameters/manager.py:1-120`（GUI 起動）。
- 自動既定線色
  - `line_color` 未指定時、背景輝度で黒/白を自動選択（最低限の可読性を確保）。

設定の整理
- `canvas.background_color`（未使用）
  - 背景の既定として採用（API 未指定時のフォールバック）。
- Dear PyGui テーマ色の入力統一
  - 0–1/0–255 両対応は維持しつつ、Hex 入力も許容（共通パーサ適用）。
  - 実装参照: `src/engine/ui/parameters/dpg_window.py:700` 近傍（RGBA 正規化）。
 - HUD テキスト色の設定キーを追加
   - `hud.text_color`（Hex / RGBA 0–1 / 0–255 互換）を導入し、OverlayHUD 初期化に適用。

上級者向け/拡張
- プリセットの用意
  - `light`/`dark`/`blueprint` などの配色セットを config に同梱。明示引数で上書き可能。
- ショートハンドの許容
  - `#RGB` / `#RGBA` を展開して受理（docs に受理範囲を明示）。
- テスト追加
  - パーサの単体テスト（Hex 全形態、境界/異常系、0–1/0–255 相互変換）。

ドキュメント更新方針（本ファイル）
- 受理形式の一覧と優先順位の図解または箇条書き。
- 具体例（OK/NG）と典型ミスの対処。
- 自動既定（背景輝度での黒/白選択）の記述。
- 設定キーの現状（利用/非利用）と方針の明記。

実装タスク（チェックリスト・ドラフト）
- [ ] 共通パーサ `util.color.parse_color` の新設（Hex/0–1/0–255 対応）
- [ ] API/GUI/HUD 呼び出し箇所を共通パーサに置換
- [ ] Parameter GUI に色ピッカーとプレビュー追加
- [ ] `line_color` 自動既定（背景輝度から黒/白選択）
- [ ] `canvas.line_color` を追加し、`canvas.background_color` と合わせて API 未指定時の解決に使用
- [ ] HUD テキスト色 `hud.text_color` を導入し、OverlayHUD で参照
- [ ] 本ドキュメントのクイックリファレンス/Do&Don’t 追記
- [ ] パーサ/境界/異常系のテスト追加
