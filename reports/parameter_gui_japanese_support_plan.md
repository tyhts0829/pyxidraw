# Parameter GUI 日本語入力対応（提案）

目的: Parameter GUI（Dear PyGui）の文字列入力で日本語（IME/表示）を扱えるようにする。

想定原因:
- DPG の既定フォントが CJK グリフを含まず、入力/描画で日本語が表示できない。

対応方針（最小・明確・可逆）
- DPG に日本語グリフを含むフォントを明示ロードして `bind_font` する。
- グリフ範囲ヒントに `Japanese` を追加（`add_font_range_hint(Japanese)`）。
- フォント指定は設定ファイル（`parameter_gui.font`）から受け取り、なければ代表的な日本語フォント名で探索（HackGen/Noto等）。
- .ttf/.otf を優先（.ttc は環境により非対応のため回避）。

設定案
- `configs/default.yaml` に以下を追加（ユーザーは `config.yaml` で上書き可）:
  - `parameter_gui.font.name: HackGenConsoleNF-Regular`（名称の部分一致で探索）
  - `parameter_gui.font.path: "/path/to/YourJPFont.ttf"`（パス優先）
  - `parameter_gui.font.size: 12`（ピクセル、未指定は layout.font_size）

スコープ外（今回やらない）
- `shapes.text` のフォント自動フォールバック（日本語字形欠落時の代替合成）。
  - 利用者は `G.text(font="Noto Sans CJK JP" など)` を引数で明示可能。
- OS へのフォント導入支援（同梱やネット導入）。

チェックリスト
- [ ] DPG フォント読み込み/バインド処理を `ParameterWindow` に追加
- [ ] 日本語グリフ範囲ヒントを付与（利用可能な API を自動検出）
- [ ] `configs/default.yaml` に `parameter_gui.font` セクションを追加
- [ ] `architecture.md` の Parameter GUI 節に「フォント読み込み対応」を追記
- [ ] 変更ファイルの ruff/black/isort/mypy を通す（対象限定）
- [ ] smoke 実行（必要最小のテスト範囲）

補足/リスク
- .ttc は DPG で読み込めない場合があるため、探索時に .ttf/.otf を優先。
- 日本語が表示できても、描画（`shapes.text`）で同じフォントを使わない限り、字形が出ない場合がある（UI 入力と描画は別系統）。

レビュー/確認事項
- 上記方針と設定キー名で問題ないか？
- 既定フォント名（HackGenConsoleNF-Regular）でよいか？（なければ Noto 系などに変更可能）

