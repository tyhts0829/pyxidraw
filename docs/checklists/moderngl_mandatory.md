# moderngl 必須化 変更チェックリスト

目的
- optional 扱いになっている moderngl を「必須依存」に統一し、ホットパス/分岐の整理と挙動の一貫性を高める。
- init_only=True 経路では GL 依存を読み込まない現在の起動特性は維持。

方針
- モジュール内の「毎フレーム/関数内の try: import moderngl」を排除し、トップレベル import に集約（当該モジュールが読み込まれる時点で存在を前提）。
- 例外時の黙殺スキップはやめ、必要箇所は明示的に例外を上位へ伝搬（ユーザーが原因に気づけるようにする）。
- パッケージ定義（pyproject.toml）とドキュメントに「moderngl は必須」を反映。

変更チェックリスト（必要作業）

- [ ] パッケージ/セットアップ更新
  - [ ] `pyproject.toml: [project].dependencies` に `moderngl` を追加（バージョンピンは運用方針に合わせて設定）。`pyproject.toml:1`
  - [ ] `pyproject.toml: [project.optional-dependencies].dev` に重複追加しない（必須側で十分）。`pyproject.toml:29`
  - [ ] 開発手順の整合（`AGENTS.md`/README のセットアップ手順に「moderngl は必須」を明記）。`AGENTS.md:1`, `README.md:1`
  - [ ] （任意）`requirements.txt`/`req.txt` のピン整合を確認（既に `moderngl==5.12.0` が含まれる）。`requirements.txt:1`, `req.txt:1`

- [ ] コード（lazy import/optional ガードの整理）
  - [ ] `src/engine/render/renderer.py:103` `draw()` 内の毎フレーム `try: import moderngl as mgl` を削除し、モジュール先頭の `import moderngl as mgl` を使用。コメントの「optional 依存」表現を削除。`src/engine/render/renderer.py:106`
  - [ ] `src/api/sketch_runner/render.py:32` の関数内 `import moderngl` をトップレベルへ移動（本モジュールは `init_only=True` 経路では import されない設計のため互換）。
  - [ ] `src/engine/export/image.py:99` FBO 経路の `try: import moderngl` を削除し、トップレベル import へ移行（または import 自体を排し、ctx を前提にする）。
  - [ ] `src/engine/export/video.py:165` および `src/engine/export/video.py:210` の `try: import moderngl` を削除し、トップレベル import へ移行。重複する overlay=False 初期化ブロックの簡素化は別タスク（任意）。

- [ ] ドキュメント更新（表現/前提の整合）
  - [ ] `docs/reviews/api_sketch_review.md:8`/`:14` の「pyglet/ModernGL を遅延 import で optional 配慮」の文言を「ModernGL は必須（init_only=True 経路では import されない）」へ更新。
  - [ ] `README.md` のセットアップ手順に「ModernGL は必須依存」を追記（任意依存セクションからは除外）。
  - [ ] `AGENTS.md` の Build & Test 初期化手順に moderngl 必須の注記を追加（仮想環境手順は現状維持）。

- [ ] テスト調整（必要に応じて）
  - [ ] `tests/perf/test_renderer_utils_perf.py:12` の `pytest.importorskip("moderngl")` は必須化後は不要（環境に常備される前提）。削除または維持の方針を決定。
  - [ ] `tests/api/test_sketch_more.py:14` の `init_only` スモークはそのまま通る（`sketch_runner/render.py` が import されないため）。回帰を確認。

検証ステップ（変更後）
- [ ] 変更ファイルに限定して `ruff/black/isort/mypy` を実行。
- [ ] `pytest -q -m smoke` を実行（`init_only=True` の経路が落ちないことを確認）。
- [ ] 通常実行 `python main.py`（ウィンドウ起動〜描画まで）。
- [ ] 画像/動画保存（任意機能）: `S`（PNG）と `V`/`Shift+V`（録画）の最低限動作を確認。

備考/リスク
- CI/開発環境に ModernGL を追加することで、ヘッドレスランナーや macOS の OpenGL サポートに依存する点が増える。必要なら CI 用に `glcontext`/`osmesa` 相当のランタイム整備を検討。
- `init_only=True` の経路では引き続き GL 依存を import しない（`src/api/sketch.py:176` 以降で import する構造）。

このチェックリストで進めて問題ないか確認をお願いします。合意後、上から順に対応し、完了項目にチェックを付けていきます。

