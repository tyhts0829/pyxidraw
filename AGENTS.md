# Repository Guidelines

## プロジェクト構成 & モジュール配置

- `main.py`: 公開 API を使うエントリーポイント例。
- `api/`: 公開インターフェイス（`E`, `G`, `run`）。`api/__init__.pyi`（スタブ）を最新に維持。
- `engine/`: ジオメトリ中核、I/O、パイプライン、レンダリング、UI。
- `effects/`: エフェクト演算子。追加時は `effects/registry.py` に登録。
- `shapes/`: 形状プリミティブ。追加時は `shapes/registry.py` に登録。
- `common/`, `util/`: レジストリ、ロギング、数値/幾何ユーティリティ。
- `scripts/`: 保守・調整用ツール（例: `gen_g_stubs.py`）。
- `data/`（大きな生成物など）、`screenshots/`（プレビュー）。
- CI: `.github/workflows/verify-stubs.yml` がスタブとテスト整合を検証。

## ビルド・テスト・開発コマンド

- Python 3.10 推奨。初期化:
  - `python3.10 -m venv .venv && source .venv/bin/activate`
  - `pip install -U pip && pip install -e .[dev]`
- 実行: `python main.py`
- API スタブ更新: `PYTHONPATH=src python -m scripts.gen_g_stubs && git add src/api/__init__.pyi`
- 静的検査/整形/型: `ruff check . && black . && isort . && mypy .`
- テスト: `pytest -q`（`pytest.ini` で `tests/` を探索）。簡易チェック: `python -m scripts.run_unified_geometry_check`。

### 変更ごとのチェック手順（テンプレ）
- 整形/静的検査: `ruff check . && black . && isort .`
- 型チェック（段階導入）: `mypy`
- テスト（全体）: `pytest -q`
- テスト（目的別）:
  - 並行処理: `pytest -m integration -k worker -q`
  - スタブ同期: `pytest -q tests/test_g_stub_sync.py tests/test_pipeline_stub_sync.py`
  - e2e/perf: `pytest -m "e2e or perf" -q`

## コーディングスタイル & 命名規則

- インデント 4 スペース、行長 100、Python 3.10 型ヒント必須。
- 可能な限り純粋・決定的な関数（副作用を避ける）。
- ファイル/モジュールは `lower_snake_case.py`、クラスは `CamelCase`、関数/変数は `snake_case`。
- 新規 `effects`/`shapes` は各フォルダに配置し、対応する `registry.py` に追記。
- 全てのコメント、docstring は日本語とすること。

## テスト指針

- フレームワークは `pytest`。テスト名は `tests/test_*.py`、対象モジュールと対応させる。
- 幾何/ノイズ系は乱数固定などで再現性を確保。
- 公開 API 変更時はスタブ再生成し、スタブ同期テスト（`tests/test_g_stub_sync.py`, `tests/test_pipeline_stub_sync.py`）を更新。

## コミット & プルリクエスト

- Conventional Commits を推奨（例: `feat(effects): add ripple clamp`）。絵文字は任意（`🎨` 整形、`🚧` WIP）。
- PR には「何を/なぜ」を明記、関連 Issue をリンク。見た目の変更は `screenshots/` にスクリーンショットを追加。
- 送信前チェック: 初回 `pre-commit install`、以降 `pre-commit run -a`。CI グリーンかつ `api/__init__.pyi` が最新であること。
  - 受け入れ基準（本リポのDoD）: lint/format/type/test/stub同期がCIで緑、README/AGENTSが最新、未使用ファイル・コメントが削減。

## セキュリティ & 設定のヒント

- 機密情報をコミットしない。`config.yaml` はローカル用途—必要に応じて個人情報を避ける。
- 大容量/生成物は `data/` または `screenshots/` に置き、リポジトリを軽量に維持。
