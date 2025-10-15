# api.sketch 分割計画（提案）

本計画は、`src/api/sketch.py` の責務分割とディレクトリ設計を定義し、段階的に安全なリファクタリングを行うためのチェックリストを提供する。公開 API の互換を保ちつつ、API 層を薄く美しく保つことを目的とする。

## 目的（Why）
- 公開 API 層（`api.sketch`）の単純化と可読性向上。
- 変更理由の分離（レンダ、イベント、エクスポート、録画、MIDI、Parameter GUI 連携）。
- spawn 環境（macOS）における安全な並列（ピクル可能なトップレベル関数化）。
- テスト・型安全性の向上と局所的な検証ループの確立。

## 非目標（Out of scope）
- レンダラ実装（ModernGL シェーダ等）の最適化や機能追加は対象外。
- MIDI/Export/Video の仕様変更はしない（呼び出し位置の整理のみ）。
- 既存ユーザ API の破壊的変更（`run_sketch` シグネチャ変更）は行わない。

## 互換性ポリシー
- `api.sketch.run_sketch` のシグネチャ・挙動は維持。
- 既存の `from api.sketch import run_sketch` は引き続き有効。
- 内部構造は `api.sketch` から新規パッケージへ委譲（re-export/facade）。

## 現状確認の要点（抜粋）
- HUDメトリクス取得はトップレベル関数で spawn 安全（`src/api/sketch.py:100`）。
- FPS 解決と投影行列は純粋関数（`src/api/sketch.py:130`, `src/api/sketch.py:147`）。
- G-code エクスポートは関数化済みで分離しやすい（`src/api/sketch.py:161`）。
- Parameter GUI は `ParameterManager` を通じ、色適用と subscribe が UI スレッドで行われる（`src/api/sketch.py:487`, `src/api/sketch.py:592`）。
- 録画時に品質モードへ切替（インラインWorker・HUD非描画・専用tick）（`src/api/sketch.py:741`）。
- WorkerPool には picklable な関数のみを渡す必要があるのは metrics/apply_cc_snapshot 等（`src/engine/runtime/worker.py:20`, `src/engine/runtime/worker.py:66`）。`param_snapshot` は親側で評価されるため非ピクルでも良い（`src/engine/runtime/worker.py:152`）。

以上から、ピクル要件のある関数をトップレベルに保ちつつ、UI/録画/エクスポートをモジュール分割するのが安全。

## 提案ディレクトリ設計（最小分割）

新規 API 内部パッケージを導入し、`api.sketch` は薄いファサードにする。

```
src/
  api/
    sketch.py                # 既存: run_sketch を薄く保つ（内部へ委譲）
    sketch_runner/           # 新設（内部実装・非公開）
      __init__.py            # 内部モジュールのまとめ（外部公開はしない）
      utils.py               # 純粋関数: resolve_fps, build_projection, canvas_size解決, 色ユーティリティ
      midi.py                # MIDI 検出・Null 実装・snapshot 橋渡し（Tickable提供）
      hud.py                 # HUDConfig適用、メトリクススナップショット/ハンドラ生成
      render.py              # RenderWindow/ModernGL/LineRenderer 初期化と色決定
      export.py              # PNG/G-code 保存ハンドラ（トップレベル関数）
      params.py              # Parameter GUI 連携：初期適用・subscribe・snapshot抽出
      recording.py           # VideoRecorder と品質モード切替（enter/leave）
      app.py                 # オーケストレーション（キー入力・スケジューリング含む）

補助クラス（必要時）:
- `config.py`: `RunnerConfig`（引数の集合）
- `context.py`: `RunnerContext`（Window/GL/Renderer/HUD/Worker 等のハンドル集合）
```

備考:
- 実装の粒度は必要最小限で開始し、凝集度が下がる場合のみ更に分割する。
- `spawn` 安全性のため、ワーカーに渡すコールバックはトップレベル定義に集約（`utils.py`, `export.py` など）。

## モジュール境界と想定インターフェース（型は簡略）

- utils.py
  - `resolve_fps(requested: int|None, *, default=60) -> int`
  - `resolve_canvas_size(size: str|tuple[int,int]) -> tuple[int,int]`
  - `build_projection(w: float, h: float) -> np.ndarray`
  - （任意）`auto_line_color_for_bg(rgba: RGBA) -> RGBA`

- midi.py
  - `setup_midi(use_midi: bool) -> (manager|None, service: Tickable, snapshot: Callable[[], Mapping[int,float]])`
  - `NullMidi(Tickable)`

- hud.py
  - `hud_metrics_snapshot() -> dict[str, dict[str,int]]`  # picklable 必須
  - `init_hud(window, swap_buffer, hud_conf) -> (sampler|None, overlay|None)`
  - `make_on_metrics(sampler|None) -> Callable[[Mapping[str,str]], None] | None`

- render.py
  - `create_window_and_renderer(*, w_px, h_px, bg, line, cfg) -> (window, mgl_ctx, line_renderer, bg_rgba, line_rgba)`

- export.py
  - `make_gcode_export_handlers(export_service, swap_buffer, canvas_w_mm, canvas_h_mm, overlay, pyglet_mod) -> (start, cancel)`
  - `save_png_screen_or_offscreen(window, *, mode, mgl_ctx, draw, name_prefix, canvas_mm) -> Path`

- params.py
  - `prepare_parameter_gui(user_draw) -> ParameterManager`
  - `apply_initial_colors(pm, window, renderer, overlay) -> None`
  - `subscribe_color_changes(pm, overlay, renderer, window, pyglet_mod) -> None`
  - `make_param_snapshot_fn(pm, cc_snapshot) -> Callable[[], dict[str, object] | None]`

- recording.py
  - `enter_quality_mode(ctx) -> quality_tick_cb`
  - `leave_quality_mode(ctx, quality_tick_cb) -> None`
  - 備考: `ctx` は worker_pool/stream_receiver/frame_clock/overlay/sampler/fps 等を保持

- app.py
  - `run_sketch(user_draw, *, canvas_size, render_scale, line_thickness, line_color, fps, background, workers, use_midi, use_parameter_gui, hud_config) -> None`
  - 内部で: utils/midi/hud/render/export/params/recording を組み立て、`pyglet.app.run()` を駆動

インターフェースの原則:
- Worker サブプロセスに渡す関数（metrics/apply_cc_snapshot）はトップレベル定義。
- UI スレッド向け schedule コールバックはクロージャ可（プロセス間を越えない）。
- `run_sketch` は `app.run_sketch` 委譲で薄く保つ。

## 段階的移行プラン（チェックリスト）

段階ごとに「変更ファイルに限定した ruff/mypy/pytest」を通す。必要時のみスタブ再生成を行う。

- [ ] Phase 0: 受入条件の確認
  - [ ] 互換性方針（上記）の承認
  - [ ] ディレクトリ設計（上記）の承認

- [ ] Phase 1: パッケージ骨格＋最低限の純粋関数移設（ノーオペ挙動）
  - [ ] `src/api/sketch_runner/` と `__init__.py` を追加
  - [ ] `utils.py` に `resolve_fps`, `build_projection`, `resolve_canvas_size` を追加
  - [ ] `hud.py` に `hud_metrics_snapshot` を移設（`_hud_metrics_snapshot` の置換準備）
  - [ ] 変更ファイルの `ruff/mypy` 実行（型注釈・docstring 付与）

- [ ] Phase 2: G-code/PNG エクスポートの切り出し
  - [ ] `export.py` に `make_gcode_export_handlers` を移植（トップレベル関数維持）
  - [ ] PNG 保存の高解像度/画面コピーの分岐を `save_png_screen_or_offscreen` に整理
  - [ ] `sketch.py` の呼び出しを `sketch_runner.export` に付け替え
  - [ ] `ruff/mypy/pytest -q -k export`

- [ ] Phase 3: HUD/メトリクスの切り出し
  - [ ] `hud.py` に `make_on_metrics(sampler)` を実装（`sampler` を引数に取りクロージャ依存を明示）
  - [ ] `sketch.py` は HUDConfig の受け渡しと `init_hud` の呼び出しのみ担当
  - [ ] `ruff/mypy/pytest -q -k hud`

- [ ] Phase 4: MIDI 初期化の切り出し
  - [ ] `midi.py` に Null 実装と `setup_midi` を実装
  - [ ] `sketch.py` の `_setup_midi` を撤去して置換
  - [ ] `ruff/mypy/pytest -q -k midi`

- [ ] Phase 5: レンダ/ウィンドウ初期化の切り出し
  - [ ] `render.py` に `create_window_and_renderer` を実装（背景色・線色決定を包含）
  - [ ] `sketch.py` の ModernGL 初期化・色決定ロジックを移設
  - [ ] `ruff/mypy/pytest -q -k render`

- [ ] Phase 6: Parameter GUI ブリッジの切り出し
  - [ ] `params.py` に初期色適用・subscribe のロジックを移設
  - [ ] `make_param_snapshot_fn` を導入（親プロセス評価のため picklable 要件なし）
  - [ ] `ruff/mypy/pytest -q -k parameters`

- [ ] Phase 7: 録画/品質モードの切り出し
  - [ ] `recording.py` に品質モード切替・録画開始/停止の補助を実装
  - [ ] `RunnerContext`（候補）を導入してスケジューラ切替をモジュール内に閉じる
  - [ ] `ruff/mypy/pytest -q -k recording`

- [ ] Phase 8: `app.py` に統合（オーケストレーション）
  - [ ] SwapBuffer/WorkerPool/StreamReceiver の構築・結線
  - [ ] pyglet イベント（キー入力）・フレームドライバの設定
  - [ ] `api.sketch.run_sketch` は `sketch_runner.app.run_sketch` を呼ぶだけに縮小

- [ ] Phase 9: ドキュメント/スタブ更新
  - [ ] `architecture.md` の「ランナー」章を新構成へ更新（参照ファイルパスを明記）
  - [ ] 公開 API に変更がなければスタブ差分ゼロを確認（必要時のみ `python -m tools.gen_g_stubs`）

## 付随改善（安全・明確化のための最小変更）
- 入力検証（`src/api/sketch.py` に導入; Phase 1 で可）
  - `fps = max(1, int(fps))`（`pyglet.clock.schedule_interval` の 1/fps 防御）
  - `render_scale > 0` を検証し、`window_width/height = max(1, round(...))`
  - `canvas_size` 未知キーは `ValueError` で一覧提示
  - `workers = max(0, workers)` にクランプ
- ロギングの明確化
  - 例外握り潰し箇所は `logger.debug("context...", exc_info=True)` へ段階的置換
- docstring 整備
  - `api.sketch.run_sketch` を NumPy スタイルに更新（Parameters/Returns/Notes）。
  - `line_thickness` の単位と目安を簡潔に追記。

## ビルド/テスト（各フェーズの DoD）
- 変更ファイル限定で実行:
  - `ruff check --fix {changed_files}`
  - `black {changed_files} && isort {changed_files}`
  - `mypy {changed_files}`
  - `pytest -q -m smoke` または関連テストの選択実行（`-k`）
- 公開 API 影響時: スタブ再生成＋`tests/stubs/test_g_stub_sync.py` 緑化。

## リスクと緩和策
- 循環依存: 内部パッケージはエンジン層を参照するが、その逆はしない（API→Engine の一方向）。
- spawn 安全性: ワーカーへ渡す関数はトップレベルで定義し、クロージャ化を避ける。
- キーバインド/イベント拡散: イベントハンドラは `app.py` に集約して可視化。
- 回帰: 段階的に小さく分け、フェーズごとに動作確認とメトリクス（HUD）で監視。

## 追加の確認事項（要ご指示）
- ディレクトリ名: `sketch_runner`（提案）で問題ないか（`api.sketch` と明確に区別）。
- 分割粒度: 上記の 7 分割で開始し、必要時にモジュールを統合/再分割する方針で良いか。
- 例外処理方針: 現状の `except Exception: pass` は `logger.debug(..., exc_info=True)` へ段階的に置換してよいか。
- 入力検証の強化: `fps>=1`、`render_scale>0`、`canvas_size` 未知キーの `ValueError` 化などを最初のフェーズで導入してよいか。
 - `RunnerContext/RunnerConfig` の導入可否（導入する場合は app/recording で使用）。

---

承認後、Phase 1 から着手します。修正は小刻みに行い、各フェーズ完了ごとにチェックを更新します。
