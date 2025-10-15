# api.sketch リファクタリング計画（シンプル&直感重視 / 互換維持）

前提:

- 連打耐性（デバウンス等）は不要。簡潔さと直感性を優先。
- ユーザーは `from api import E, G, cc, lfo` の形式で直感的に利用可能であることを維持。
- `cc` はグローバル状態のままでよい。
- 現在の公開 API（`run_sketch(user_draw, ...)`/`run`/`E`/`G`/`lfo` 等）は維持。

目的（Why）:

- `src/api/sketch.py` の肥大化を軽減し、読みやすさと保守性を改善。
- ランナーは最小限のオーケストレーションに限定し、補助ロジックは小さなヘルパに分割。
- インタラクティブ性（録画/PNG/G-code/GUI 変更）をそのまま保ち、操作は即時実行。

互換性ポリシー:

- ユーザーコードは更新不要（`user_draw(t)->Geometry` 維持、`cc` グローバル維持）。
- キーバインド（P/Shift+P/G/Shift+G/V/Shift+V）と挙動は現行踏襲。

設計の要点（シンプル版）:

- Job 管理や状態機械は導入しない。現行の軽量トグル/単発実行を踏襲。
- G-code は既存 `ExportService` の単一実行中ガードのみ（多重起動は HUD で通知して無視）。
- 録画は現在のトグル動作と品質モード切替を維持（HUD の REC インジケータのみ）。
- PNG 保存は都度実行（画面コピー/オフスクリーンの 2 種）。

ディレクトリ設計（最小分割）:

```
src/
  api/
    sketch.py                # 入口（薄いオーケストレーション）。公開 API はここ
    sketch_runner/           # 内部ヘルパ（API 非公開）
      __init__.py
      utils.py               # resolve_fps, build_projection, canvas_size 解決
      midi.py                # MIDI 初期化（Null を含む）
      render.py              # Window/ModernGL/LineRenderer 準備と色決定
      export.py              # PNG/G-code ハンドラ生成（開始/キャンセル関数）
      params.py              # Parameter GUI 連携（初期色/subscribe）
      recording.py           # 品質モード enter/leave（小関数）
```

現状確認と移設対象（実装参照）:

- `resolve_fps`（fps解決）: `src/api/sketch.py:130` → `sketch_runner/utils.py`
- `build_projection`（正射影行列）: `src/api/sketch.py:147` → `sketch_runner/utils.py`
- `canvas_size` 解決/検証: `src/api/sketch.py:292`-`src/api/sketch.py:296` 付近 → `sketch_runner/utils.py` に関数化
- HUDメトリクス取得（spawn安全）: `_hud_metrics_snapshot` `src/api/sketch.py:100` → `sketch_runner/utils.py`（名称は `hud_metrics_snapshot`）
- MIDI 初期化（Null含む）: `_setup_midi` ローカル関数 `src/api/sketch.py:301` → `sketch_runner/midi.py`
- G-code開始/キャンセル: `make_gcode_export_handlers` `src/api/sketch.py:161` → `sketch_runner/export.py`
- PNG 保存分岐: `_handle_save_png` 内の分岐 `src/api/sketch.py:695` → `sketch_runner/export.py` に薄いラッパ
- ModernGL/ウィンドウ初期化と色決定: `src/api/sketch.py:427`-`src/api/sketch.py:486` → `sketch_runner/render.py`
- Parameter GUI: 初期色適用 `src/api/sketch.py:487`-`src/api/sketch.py:551`、subscribe `src/api/sketch.py:592`-`src/api/sketch.py:691` → `sketch_runner/params.py`
- 録画/品質モードの切替: `_enter_quality_mode` `src/api/sketch.py:741`、`_leave_quality_mode` `src/api/sketch.py:785` → `sketch_runner/recording.py`

インタラクティブ仕様（踏襲）:

- `P` → PNG 保存（HUD 含む）。`Shift+P` → 高解像度（HUD なし・ラインのみ）。
- `G` → G-code 変換開始（進捗 HUD）。`Shift+G` → キャンセル。
- `V` → 録画トグル（画面）。`Shift+V` → 録画トグル（品質：FBO/固定刻み）。
- GUI の色変更は UI スレッドで即時適用（背景/線/HUD 色）。

段階的移行（チェックリスト / 変更ファイル優先で検証）:

- [ ] Phase 1: 骨格＋小改善（互換維持のまま動作不変）
  - [ ] `src/api/sketch_runner/{__init__,utils.py}` を追加
  - [ ] `resolve_fps`/`build_projection`/`resolve_canvas_size`/`hud_metrics_snapshot` を `utils.py` に実装し、`sketch.py` は呼び替え
  - [ ] 小改善（このフェーズで一括導入）
    - [ ] `fps` を `max(1, int(fps))` でクランプ（`src/api/sketch.py:130` と `schedule_interval` 呼び出し部 `src/api/sketch.py:570`, `src/api/sketch.py:782`）
    - [ ] `render_scale > 0` を検証し、`window_width/height = max(1, round(...))`（`src/api/sketch.py:296`）
    - [ ] `workers = max(0, workers)` にクランプ（`src/api/sketch.py:337` or `src/api/sketch.py:807` 生成直前）
    - [ ] `canvas_size` 未知キーは `ValueError` with allowed keys（`src/api/sketch.py:293`）
    - [ ] 主要な `except Exception: pass` を `logger.debug(..., exc_info=True)` に変更（色適用・HUD・録画周辺）
    - [ ] `run_sketch` の docstring を NumPy スタイルへ簡潔整備（Parameters/Returns/Notes）
  - [ ] 変更ファイルに対して `ruff/black/isort/mypy` を実行
- [ ] Phase 2: エクスポート切り出し
  - [ ] `export.py` に `make_gcode_export_handlers` と PNG 保存ラッパを移設
  - [ ] `sketch.py` の呼び出し箇所を置換
- [ ] Phase 3: MIDI 初期化切り出し
  - [ ] `midi.py` に Null 実装と `setup_midi(use_midi)` を実装
  - [ ] `sketch.py` の `_setup_midi` を撤去
- [ ] Phase 4: レンダ初期化切り出し
  - [ ] `render.py` に `create_window_and_renderer` を実装（背景/線色決定を含む）
  - [ ] ModernGL 初期化と色自動決定を移設
- [ ] Phase 5: Parameter GUI 連携切り出し
  - [ ] `params.py` に初期色適用と subscribe による色反映を移設
- [ ] Phase 6: 録画補助切り出し
  - [ ] `recording.py` に品質モードの enter/leave（`pyglet.clock` 再スケジューリング）を移設
- [ ] Phase 7: sketch.py の簡素化
  - [ ] イベントハンドラはそのまま、各機能を `sketch_runner.*` へ委譲
- [ ] Phase 8: 文書同期
  - [ ] `architecture.md` のランナー図と参照ファイルを更新
  - [ ] 公開 API 影響なしを確認（必要時のみスタブ再生成）

付随の小改善（シンプルの範囲内で）:

- 入力検証の明確化（極小のガードのみ）
  - `fps = max(1, int(fps))`、`render_scale > 0`、`workers = max(0, workers)`
  - `canvas_size` 未知キーは `ValueError(f"invalid canvas_size: {name}; allowed={list(CANVAS_SIZES)}")`
- ログの粒度: 重要箇所は WARN/INFO、細部は DEBUG で簡潔に（握り潰しは最小化）。
- docstring を NumPy スタイルで簡潔に（公開 API のみ）。

テスト/チェック（各フェーズ DoD）:

- `ruff check --fix {changed}` / `black {changed} && isort {changed}` / `mypy {changed}`
- スモーク: `pytest -q -m smoke` または関連 `-k`
- スタブ同期（公開 API 影響時のみ）: `python -m tools.gen_g_stubs && git add src/api/__init__.pyi`

確認事項（済）:

- ディレクトリ名 `sketch_runner`: はい
- Phase 1 で小改善を一括導入: はい
