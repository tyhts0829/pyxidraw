# Parameter GUI 外観設定（cfg対応）改善計画 ✅

目的: パラメータ GUI（Dear PyGui）の色・スライダー高さ等の見た目を cfg（YAML）から設定可能にする。現状既定スタイル最小のみ。既存の API/動作は変えず、設定が無い場合は従来の見た目で動作。

非目標: MIDI/cc の仕様変更、値変換レイヤ復活、個別パラメータごとの細粒度テーマ指定（将来拡張）。

基本方針:
- 既存構造への最小介入（ParameterLayoutConfig を中心に、必要最小の新設型を追加）。
- 設定が無い時は完全に既定（互換）動作。
- ヘッドレス/未導入環境でも安全（現在のスタブを維持）。
- cfg のマージは既存どおり“トップレベルのみ上書き”。`parameter_gui` キーは一塊で上書き（ディープマージは本計画では導入しない）。

想定 YAML スキーマ（案）:

```yaml
parameter_gui:
  window:
    width: 420
    height: 640
    title: "Parameters"
  layout:
    padding: 8          # 既存: mvStyleVar_Window/Frame/ItemSpacing に適用
    row_height: 28      # 目安。必要に応じて set_item_height or padding で近似
    font_size: 12
    value_precision: 6
  theme:
    style:              # DPG StyleVar
      frame_padding: [8, 4]
      item_spacing: [8, 4]
      window_padding: [8, 8]
      frame_rounding: 4
      grab_rounding: 4
      grab_min_size: 12
    colors:             # DPG ThemeCol（RGBA, 0.0..1.0）
      text: [0.95, 0.95, 0.95, 1.0]
      window_bg: [0.13, 0.13, 0.13, 1.0]
      frame_bg: [0.20, 0.20, 0.20, 1.0]
      frame_bg_hovered: [0.28, 0.28, 0.28, 1.0]
      frame_bg_active: [0.33, 0.33, 0.33, 1.0]
      header: [0.22, 0.22, 0.22, 1.0]
      header_hovered: [0.28, 0.28, 0.28, 1.0]
      header_active: [0.33, 0.33, 0.33, 1.0]
      accent: [0.35, 0.60, 0.98, 1.0]        # Grab/Slider へ反映
      accent_active: [0.20, 0.45, 0.90, 1.0]
```

実装タスク（チェックリスト）:

- [ ] 仕様確定: 上記スキーマで問題ないか確認（このファイルにフィードバックください）
- [ ] 既定値追加: `configs/default.yaml` に `parameter_gui` を追加
- [ ] 型定義追加: `src/engine/ui/parameters/state.py` に UI 設定型を追加
  - [ ] `ParameterThemeConfig`（style/colors を保持）
  - [ ] 必要なら `ParameterWindowConfig`（width/height/title）
  - [ ] 既存 `ParameterLayoutConfig` はそのまま or 最小拡張（value_precision 等は流用）
- [ ] 設定読込: `ParameterManager` 初期化時に `load_config()` から `parameter_gui` を取り出し
  - [ ] `ParameterLayoutConfig`/`ParameterThemeConfig`/`ParameterWindowConfig` を構築
  - [ ] `ParameterWindowController`/`ParameterWindow` に渡す
- [ ] DPG テーマ適用: `ParameterWindow._setup_theme()` を拡張
  - [ ] StyleVar（Window/Frame/ItemSpacing/FrameRounding/GrabRounding/GrabMinSize）適用
  - [ ] ThemeCol（Text/WindowBg/FrameBg/…/Header/Grab など最小セット）適用
  - [ ] 例外安全（try/except 維持）
- [ ] スライダー高さ: スタイルで近似（`frame_padding.y`, `grab_min_size`）。必要時 `set_item_height` を併用
- [ ] ウィンドウ寸法/タイトル: `ParameterWindow` 生成時に cfg を反映
- [ ] テスト追加（DPG 非導入前提の軽量テスト）
  - [ ] 設定→型へのマッピング単体テスト（`tests/ui/parameters`）
  - [ ] `ParameterWindow` の生成が設定ありでも例外なく通ること（既存 smoke を流用/拡張）
- [ ] ドキュメント
  - [ ] `architecture.md` に「Parameter GUI 外観設定（cfg）」を追記（実装参照箇所を明記）
  - [ ] `README.md` に最小の設定例を追加
- [ ] 仕上げ: 変更ファイル限定の Lint/Format/Type/Test を緑化
  - `ruff check --fix {changed}` / `black {changed} && isort {changed}` / `mypy {changed}` / `pytest -q tests/ui/parameters`

実装ポイント（要旨）:
- 色指定は RGBA float 0..1（既存の `canvas.background_color` と統一）。
- 「スライダー高さ」は DPG に明示 API が乏しいため、`FramePadding.y` と `GrabMinSize` でコントロール（近似）。
- `load_config()` はトップレベル shallow merge のため、`config.yaml` で `parameter_gui` を指定すると同キー以下は全置換。（必要なら将来ディープマージ検討）
- ヘッドレス環境ではテーマ適用は no-op（現在の try/except を踏襲）。

確認したいこと（要回答）:
- [ ] スキーマ命名（日本語/英語）はこの案で良いか（`parameter_gui.window/theme/layout`）
- [ ] 既定テーマは「ダーク基調」で良いか（配色は上記例）
- [ ] スライダー高さは「おおよその太さが増える」近似で十分か（厳密なピクセル高さは不可な場合あり）
- [ ] 将来的に per-parameter のハイライト/色分けニーズはあるか（今回範囲外に据え置き）

影響範囲と後方互換:
- 既存コードパスは cfg 不存在時に従来既定で動作。公開 API 変更なし。
- `configs/default.yaml` にキー追加のみ（既存テストへの影響は限定的）。

タイムライン目安:
- 0.5 日: 実装＋単体テスト
- 0.5 日: ドキュメント、微調整（配色/スタイルのデフォルト詰め）

備考（拡張余地）:
- テーマのプリセット切替（"light"/"dark"）
- `config.yaml` での部分上書き支援に限定ディープマージ導入
- パラメータカテゴリ単位での色リング（collapsing header ごと）

