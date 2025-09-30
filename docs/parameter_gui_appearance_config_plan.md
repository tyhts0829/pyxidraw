# Parameter GUI 外観設定（設定ファイル対応）改善計画（提案）

目的: Parameter GUI（Dear PyGui）の外観（色・ウィンドウ寸法・余白など）を設定ファイル（YAML）から変更可能にする。既定動作は維持し、設定未指定時は現状の見た目のまま動作する。

非目標: MIDI/cc の仕様変更、値変換レイヤの復活、パラメータ単位の個別テーマ、動的リロード。

設計方針:
- シンプル・可読性最優先。既存構造への最小介入。
- `configs/default.yaml` に `parameter_gui` を追加。`config.yaml` はトップレベル shallow merge の現仕様に従う（`parameter_gui` 以下は一塊で上書き）。
- UI 設定は `state.py` に小さな設定型（dataclass）を追加して明示化。
- DPG 未導入/ヘッドレス環境でも import/実行が安全（現行の try/except 方針を踏襲）。

想定 YAML スキーマ（案）:

```yaml
parameter_gui:
  window:
    width: 420
    height: 640
    title: "Parameters"
  layout:
    padding: 8           # 既存: Window/Frame/Item の padding/spacing に適用
    row_height: 28
    font_size: 12
    value_precision: 6
  theme:
    style:               # Dear PyGui StyleVar 相当
      window_padding: [8, 8]
      frame_padding: [8, 4]
      item_spacing: [8, 4]
      frame_rounding: 4
      grab_rounding: 4
      grab_min_size: 12
    colors:              # Dear PyGui ThemeCol 相当（RGBA, 0..255）
      text: [242, 242, 242, 255]
      window_bg: [33, 33, 33, 255]
      frame_bg: [51, 51, 51, 255]
      frame_bg_hovered: [71, 71, 71, 255]
      frame_bg_active: [84, 84, 84, 255]
      header: [56, 56, 56, 255]
      header_hovered: [71, 71, 71, 255]
      header_active: [84, 84, 84, 255]
      accent: [89, 153, 250, 255]  # Grab/Slider 等へ反映
      accent_active: [51, 115, 230, 255]
```

実装タスク（チェックリスト）:

- [ ] 仕様確定（この計画の確認）
  - [ ] スキーマ: `parameter_gui.window/layout/theme` 命名と階層で良いか
  - [ ] 既定テーマはダーク基調で良いか
  - [ ] スライダーの「太さ」は `frame_padding.y`/`grab_min_size` による近似で許容か
- [ ] 既定値追加: `configs/default.yaml` に `parameter_gui` を追加
- [ ] 型定義追加: `src/engine/ui/parameters/state.py`
  - [ ] `@dataclass ParameterWindowConfig(width:int, height:int, title:str)`
  - [ ] `@dataclass ParameterThemeConfig(style: dict[str, Any], colors: dict[str, list[int]])`
  - [ ] 既存 `ParameterLayoutConfig` は据え置き（必要なら docstring 追補）
- [ ] 設定読込: `ParameterManager` 初期化時に `util.utils.load_config()` から `parameter_gui` を取得
  - [ ] `ParameterWindowConfig`/`ParameterLayoutConfig`/`ParameterThemeConfig` を構築
  - [ ] `ParameterWindowController` に渡し、`ParameterWindow` 生成へ伝播
- [ ] ウィンドウ設定: `ParameterWindow` へ `width/height/title` を引数で供給
- [ ] テーマ適用: `src/engine/ui/parameters/dpg_window.py::_setup_theme()` を拡張
  - [ ] StyleVar マッピング（例: `window_padding`→`mvStyleVar_WindowPadding` 等）
  - [ ] Colors マッピング（例: `text`→`mvThemeCol_Text`, `window_bg`→`mvThemeCol_WindowBg` 等）
  - [ ] 例外安全（try/except 維持、欠損キーはスキップ）
- [ ] テスト
  - [ ] 設定→型マッピングの単体（DPG 非依存）: `tests/ui/parameters/` に追加
  - [ ] 既存 smoke: 設定有/無の双方で `ParameterWindow` 生成/可視切替/close が例外なく通ること
- [ ] ドキュメント
  - [ ] `architecture.md` に「Parameter GUI 外観設定（cfg）」の項を追記（実装参照箇所を明記）
  - [ ] `README.md` に最小の設定例を追加
- [ ] 仕上げ（変更ファイル限定の高速チェック）
  - [ ] `ruff check --fix {changed}`
  - [ ] `black {changed} && isort {changed}`
  - [ ] `mypy {changed}`
  - [ ] `pytest -q tests/ui/parameters`

受入条件（Definition of Done）:
- 設定未指定で従来の見た目（420x640、最小テーマ）で動作。
- 設定指定でウィンドウ寸法/タイトル/スタイル/色が反映される。
- DPG 未導入環境（スタブ）でも例外なく通る（既存テスト緑）。
- 変更ファイルに対する ruff/mypy/pytest が成功。必要時スタブ再生成は不要（公開 API 変更なし）。

補足/トレードオフ:
- `config.yaml` の shallow merge により、`parameter_gui` を部分上書きすると同キー以下は全置換となる（将来の限定ディープマージは要検討）。
- DPG は明示高さ API が限定的なため、スライダー高さは padding/grab サイズで近似制御する。
- カラー表現は RGBA 0..255（config 推奨）。0..1 float も互換で受け付け、内部で 0..255 に拡大。

確認事項（ご回答ください）:
- [ ] 上記スキーマ/命名で進めてよいか
- [ ] 既定テーマを「ダーク」で開始してよいか
- [ ] 優先実装項目（色/ウィンドウ寸法/余白）に抜けがないか

関連ファイル（実装時に更新予定）:
- `src/engine/ui/parameters/state.py`
- `src/engine/ui/parameters/manager.py`
- `src/engine/ui/parameters/controller.py`
- `src/engine/ui/parameters/dpg_window.py`
- `configs/default.yaml`
- `architecture.md`, `README.md`

注記:
- 既に類似の計画メモがある場合（例: `docs/parameter_gui_cfg_plan.md`）、本計画は色/ウィンドウ寸法に焦点を当てた最新版として扱い、実装時に内容を統合します。
