# リファクタ計画: api.sketch の小規模クリーンアップと堅牢化

対象: `src/api/sketch.py` と最小限の周辺（`src/api/sketch_runner/*` は参照のみ）。
目的: 不具合ゼロを維持しつつ、lint/type のノイズ低減と安全性の微修正を行う。挙動は不変。

## スコープ
- 実行フローや公開 API は変更しない（非破壊）。
- 変更は `src/api/sketch.py` のみを基本対象とし、必要時に関連 md の追従（任意）。

## 変更チェックリスト（提案）
- [x] 未使用戻り値の明示破棄で lint 抑制（F841 相当）
  - 行: `src/api/sketch.py:271` — `create_window_and_renderer` の戻り値（色）を未使用である旨を明示。
    - 具体案A: `rendering_window, mgl_ctx, line_renderer, _bg_rgba, _line_rgba = ...`
    - 具体案B: `rendering_window, mgl_ctx, line_renderer, _, _ = ...`（読みやすさ優先ならA）
- [x] `quality_tick_cb` の型注釈を明示（`Callable[[float], None] | None`）
  - 行: `src/api/sketch.py:350` — `type: ignore` を除去し型で表現。
- [x] 録画品質モード切替関数での重複 import の解消
  - 行: `src/api/sketch.py:402`, `src/api/sketch.py:429` — ネスト内の `import pyglet` を削除し外側参照を使用。
- [x] HUD メトリクスコールバックの Null セーフ化（まれな順序依存の予防）
  - 行: `src/api/sketch.py:247-262` — `sampler is None` 時は早期 return を追加。
- [ ] （任意）ModernGL/Window 初期化失敗時のユーザー向けログ強化
  - 範囲: `create_window_and_renderer(...)` 呼出し部を try/except で囲み、失敗時に `logger.error` と簡潔なガイドを出力。
  - 例: 「GL 初期化に失敗しました。ヘッドレス環境またはドライバ未設定の可能性…」
- [ ] （任意）モジュール冒頭 docstring の簡素化（詳細は docs に集約）
  - 先頭は「どこで/何を/なぜ」に絞り、長文の運用説明は既存の `docs/reviews/api_sketch_review.md` 等へ誘導。

## 非対象（今回行わない）
- `sketch_runner/render.py` の戻り値シグネチャ変更（色 RGBA を返さない方向）は広がるため今回見送り。
- パブリック API/挙動の変更（キー操作・録画・エクスポート・MIDI 連携）。

## 受け入れ基準（Definition of Done）
- 変更ファイルに限定したチェックがグリーン:
  - `ruff check --fix src/api/sketch.py`
  - `black src/api/sketch.py && isort src/api/sketch.py`
  - `mypy src/api/sketch.py`（型注釈の追加によりノイズ無し）
- 主要スモーク（必要時のみ）:
  - `pytest -q tests/api/test_sketch_init_only.py::test_run_sketch_init_only_returns_none_without_importing_heavy_deps`
  - `pytest -q tests/api/test_sketch_more.py::test_runner_midi_fallback_when_mido_missing`

## リスクと緩和
- 変更加減は最小で、挙動差分は無し（静的品質のみ向上）。
- HUD メトリクスの Null ガードは安全側（実行タイミングによる稀な NPE を未然防止）。

## 実施手順（概略）
1) 変更を実装（上記チェックリストの A 案を採用）。
2) ファイル単位の Lint/Format/Type を実行し調整。
3) スモーク対象のみ pytest 実行（CI 全体は不要）。
4) 必要に応じて `docs/reviews/api_sketch_review.md` に「対応済み」注記を追加（任意）。

## 確認事項（要ご判断）
- 未使用戻り値の扱いは A 案（`_bg_rgba`, `_line_rgba`）で問題ありませんか？
- 初期化失敗時のユーザーメッセージ強化（任意）を今回含めますか？
- 冒頭 docstring の簡素化は今回のスコープに含めますか？

承認いただければ、このチェックリストに従って段階的に変更を行い、完了項目へチェックしていきます。
