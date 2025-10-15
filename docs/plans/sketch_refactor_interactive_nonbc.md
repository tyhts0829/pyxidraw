# api.sketch リファクタリング計画（非互換・インタラクティブ重視）

本計画は、互換性を捨てて「美しい・直交・インタラクティブ最優先」の設計へ再構成するための実施計画。ユーザーが描画中に録画/PNG/G-code を連打し、GUIでパラメータを弄りながら即レコーディングできる体験を最重要要件とする。

## 目的（Why）
- 単一責務・直交APIにより理解/変更/テストを容易化。
- ランナーは「フレーム駆動＋アクション実行」に限定。録画/エクスポートは Job として並列管理。
- グローバル状態（例: api.cc）を排除し、`FrameContext` で明示的に入力を渡す。

## 公開API（新）
- `def run_sketch(user_draw: Callable[[FrameContext], Geometry], config: SketchConfig) -> AppHandle`
- `def render_video(user_draw, video: VideoConfig, config: SketchConfig) -> Path`
- `def export_gcode(draw_or_geom, gcode: GCodeConfig, config: SketchConfig) -> Path`

型（抜粋; dataclass/frozen）
- `FrameContext(t: float, cc: Mapping[int,float], params: Mapping[str,object], rng: Random)`
- `SketchConfig(CanvasConfig, RenderConfig, IOConfig, HUDConfig, ParamsConfig, RecordingConfig)`
- `AppHandle.toggle_recording(mode: Literal["screen","quality"]="screen")`
  - `save_png(mode: Literal["screen","quality"]="screen")`
  - `start_gcode()`, `cancel_gcode()`
  - `set_param(key: str, value: object)`

優先順位: 明示引数 > GUI > Config > 既定。

## ディレクトリ設計
```
src/
  api/
    __init__.py              # 公開 API/型を最小限 re-export
    app.py                   # オーケストレーション: run_sketch/render_video/export_gcode
    config.py                # SketchConfig/各種Config（@dataclass, frozen）
    context.py               # FrameContext/AppContext（Window/Renderer/Worker等のハンドル）
    jobs/
      job_manager.py         # Jobの状態機械/並列管理/デバウンス
      recording_job.py       # 録画ジョブ（screen/quality）
      gcode_job.py           # G-codeジョブ
      png_job.py             # PNGジョブ
    runtime/
      scheduler.py           # Scheduler抽象 + PygletScheduler
      worker_adapter.py      # Worker呼出し（inline/spawn切替）
    render/
      window.py              # WindowBackend抽象 + PygletWindow
      renderer.py            # Renderer抽象 + LineRendererGL（mm→clip変換）
      projection.py          # mmベース正射影（純粋関数）
    io/
      midi.py                # MidiProvider抽象 + Null/Rtmidi実装
    params/
      store.py               # ParameterStore（量子化/署名=__param_meta__に準拠）
      gui.py                 # ParameterGUI（任意）。UIスレッド適用/subscribe
    hud/
      metrics.py             # hud_metrics_snapshot（spawn安全）+ HIT/MISS判定
      overlay_adapter.py     # OverlayHUD用の薄いポート
    export/
      png.py                 # save_png_screen/save_png_offscreen
      gcode.py               # GcodeExporter + ExportJob基盤
      video.py               # Recorder（FrameSink 抽象により画面/FBO取り込み）
```

設計原則
- ワーカーに渡す関数はトップレベル（spawn安全）。
- GL操作はUIスレッド専有。FBO/読み出し含め同一スレッドで実行。
- JobManager が録画/エクスポートの連打・キャンセル・進捗を一元管理。

## 挙動仕様（インタラクティブ要件）
- キー割当（例）: `V`=録画(画面), `Shift+V`=録画(品質), `P`=PNG保存(画面), `Shift+P`=高解像度PNG, `G`=G-code開始, `Shift+G`=G-codeキャンセル。
- 連打耐性: アクションはデバウンス（~200ms）。録画はトグル、G-codeは逐次/キャンセル可、PNGは都度独立。
- HUD: トースト/メーター/RECインジケータを統一APIで更新。
- パラメータ: 変更は次フレームから反映。録画/G-code中も許可。

## 量子化/署名（Params）
- `__param_meta__['step']` を用いて「floatのみ量子化」。未指定は `1e-6`（`PXD_PIPELINE_QUANT_STEP` 上書き可）。
- ベクトルは成分ごとに適用し、step不足は末尾値で補完。int/boolは量子化しない。
- Effects は量子化後の値を実行引数にも渡す。Shapes は鍵のみ量子化。

## 実施フェーズ（チェックリスト）

DoD: 各フェーズで変更ファイル限定の ruff/black/isort/mypy/pytest(smoke or -k) 緑。

- [ ] Phase A: 骨格/型導入（ノーオペ）
  - [ ] `api/config.py` `api/context.py` `api/app.py` スケルトン
  - [ ] `FrameContext`/`SketchConfig`/`AppHandle` 定義（docstring/型）
  - [ ] `runtime/scheduler.py`（PygletScheduler）追加

- [ ] Phase B: Job基盤
  - [ ] `jobs/job_manager.py`（状態機械/デバウンス/イベント）
  - [ ] `jobs/png_job.py`（画面/高解像度）
  - [ ] `jobs/gcode_job.py`（進捗/キャンセル）
  - [ ] `jobs/recording_job.py`（screen/quality モード）

- [ ] Phase C: レンダ/ウィンドウ
  - [ ] `render/window.py`（PygletWindow）
  - [ ] `render/renderer.py`（LineRendererGL, mm→clip変換: thickness_clip ≈ 2*mm/H）
  - [ ] `render/projection.py`（純粋関数）

- [ ] Phase D: Worker統合
  - [ ] `runtime/worker_adapter.py`（inline/spawn, Queue分岐）
  - [ ] `user_draw(ctx)` へ統一。`FrameContext` を構成して渡す

- [ ] Phase E: MIDI/Params 統合
  - [ ] `io/midi.py`（Null/Rtmidi + Tickable）
  - [ ] `params/store.py`（量子化/署名）と `params/gui.py`（UIスレッド適用）

- [ ] Phase F: HUD/メトリクス
  - [ ] `hud/metrics.py`（snapshot + HIT/MISS判定）
  - [ ] `hud/overlay_adapter.py`（薄いポート）

- [ ] Phase G: App 結線/キー入力
  - [ ] `app.py` で Window/Renderer/Worker/JobManager/HUD/GUI を結線
  - [ ] Keymap→Action マップ/イベント購読/スケジューリング

- [ ] Phase H: 旧実装の置換/清掃
  - [ ] `api.sketch` を薄い互換ラッパにするか廃止（非互換リリース前提）
  - [ ] `architecture.md` を新構成に同期

## ビルド/テスト方針
- 単体: `projection`, `utils`, `job_manager` の状態遷移、量子化/署名。
- 疎通: `AppHandle` のトグル/キャンセル/PNG保存（ヘッドレスでも可能な範囲）。
- 可能なら `pytest -q -m smoke` と `-k (projection|jobs|params)` を増やす。

## リスク/緩和
- spawn安全: ワーカーへ渡す関数はトップレベルに限定。
- GLスレッド: UI専有とし、オフスクリーン/読み出しも同スレッド。
- 競合: JobManagerで一元管理（録画1本・G-code逐次・PNG独立）。
- 回帰: フェーズ毎に小さく通し、HUDで状態を可視化。

## 確認事項（要指示）
- `AppHandle` のメソッド名/引数（特に録画モード literal）
- `line_thickness` を mm 第一級にする仕様（既定値/上限）
- Keymap の初期割当（V/P/G + Shift の現行踏襲で良いか）
- `api.sketch` の扱い（廃止 or 薄い互換ラッパ一時提供）

---
本計画が妥当であれば Phase A から着手します。修正は小刻みに進め、各フェーズ完了ごとに本MDのチェックを更新します。
