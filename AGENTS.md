# Repository Guidelines (AGENTS.md)

本ファイルは、エージェント/人間が共通で参照する「短く・具体的・反復可」な運用規約。近接する AGENTS.md が優先される（ネスト時）。以下のチェックは“実施すべき”前提。

## 一般的な指示

- 必要十分な堅牢さ。複雑化を避け、可読性・シンプルさ・美しさを優先。
- このリポジトリはまだ配布しておらず、ユーザーは居ないので、破壊的変更でも構わないので美しいシンプルな実装を目指すこと。
- '報告して'や'どう思う？'や'提案して'といった指示は、聞かれたことだけ答え、コードを先回りして変更しないこと。
- ルート AGENTS.md の最小項目（Build/Test/Style/Safety/PR）を常に最新化
- ルート architecture.md は実装（src/ 配下）と同期しています。差分を見つけた場合は、該当コードの参照箇所とともに更新してください。
- コード改善を指示された場合、コード変更前に、改善アクションを細分化したチェックリストを新規.md ファイルとして保存し、私にそれでいいか確認してください。その後、私の返答に基づいて改善を行い、完了アイテムをチェックしていき、何が完了し、何が完了していないかを常に明確にして下さい。改善の中で気がついた、私に事前確認したほうがいいことや、さらなる改善提案があれば、その md ファイルに追記して報告してください。
- 回答は日本語。

## Build & Test（編集ファイル優先の高速ループ）

- 環境: Python 3.10。初期化: `python3.10 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e .[dev]`
- 実行: `python main.py`
- スタブ再生成: `PYTHONPATH=src python -m scripts.gen_g_stubs && git add src/api/__init__.pyi`
- プロジェクト全体チェック（明示要求時のみ）:
  - Lint/Format/Type: `ruff check . && black . && isort . && mypy .`
  - Test: `pytest -q -m "not optional"`
- ファイル単位（変更ファイルに限定して優先実行）:
  - Lint: `ruff check --fix {path}`
  - Format: `black {path} && isort {path}`
  - TypeCheck: `mypy {path}`
  - Test (file): `pytest -q {test_path}` または `pytest -q -k {expr}`
  - 例: `pytest -q tests/test_geometry_model.py::test_from_lines_normalizes_shapes`
- テストマーカーの活用:
  - `-m smoke`（最小確認）、`-m integration`、`-m e2e`、`-m perf`、`-m optional`
- 完了条件（変更単位）: 変更ファイルに対する ruff/mypy/pytest が成功し、必要時にスタブ再生成済み。

## Safety & Permissions（許可境界）

- Allow（事前承認不要）:
  - 読取/一覧（`rg`, `cat` 等）
  - Lint/Format/Type/Test の“単一/対象限定”実行（上記ファイル単位コマンド）
  - スタブ生成のドライラン/差分確認
- Ask-first（要承認）:
  - 依存追加/更新（`pip install`, `uv sync` 等のネットワーク）
  - 破壊的操作（`rm`、権限変更、大量移動/rename）
  - フルビルド/全体テストの長時間実行、CI 設定変更
  - スナップショット更新やベースライン生成などリポ内容を書き換える操作
  - `git push` / リリース作業

## Project Map（索引）

- 入口: `main.py`
- 公開 API: `src/api/__init__.py`（エクスポート）、`src/api/__init__.pyi`（自動生成スタブ）
- スタブ生成: `src/scripts/gen_g_stubs.py`（`python -m scripts.gen_g_stubs`）
- 形状/エフェクト登録: `src/shapes/registry.py`, `src/effects/registry.py`
- 中核ジオメトリ: `src/engine/core/geometry.py`
- レンダリング: `src/engine/render/`
- パイプライン/ワーカー: `src/engine/pipeline/`
- 共通型/ユーティリティ: `src/common/`, `src/util/`
- 設定: `config.yaml`, `configs/default.yaml`
- テスト: `tests/`（`smoke`/`integration`/`e2e`/`perf`/`optional`）
- CI: `.github/workflows/verify-stubs.yml`

## Testing / CI ルール

- 本ファイル記載のチェックは“実施すべき”。編集ファイル優先で高速に回す。
- PR/コミット前に最低限実施:
  - `ruff check --fix {changed_files}`、`black {changed_files} && isort {changed_files}`
  - `mypy {changed_files}`（段階導入設定。必要に応じ対象を拡大）
  - `pytest -q -m smoke` もしくは対象テストファイルを直接指定
  - 公開 API に影響時: スタブ再生成 + `tests/stubs/test_g_stub_sync.py` / `tests/stubs/test_pipeline_stub_sync.py` を緑化
- CI 成功条件（DoD）:
  - スタブ最新（生成後に差分ゼロ）
  - `ruff/black/isort/mypy` 成功
  - `pytest -q -m "not optional"` 緑（optional は別ジョブ）
  - README/AGENTS の整合、未使用ファイル/コメントの削減

## ドキュメンテーション

- 原則『What/How はコードと型, Why/Trade-off はコメント』
- すべての公開 API に NumPy スタイル docstring + 型ヒント。
  - docstring: 日本語の事実記述（主語省略・終止形、絵文字不可）。
  - 目的・設計意図・既知のトレードオフのみを短く記す。逐語説明や重複は避ける。
- 型ヒント: `dict[str, Any]` 等の組込みジェネリックで統一。`typing` 由来は最小限（`Callable`, `Mapping`, `Sequence`）。
- 影響大の判断は ADR（背景 → 決定 → 根拠 → 結果）。
- lint: ruff。型: mypy + pylance。

## コーディング規約

- インデント 4、行長 100、Python 3.10 型ヒント必須。
- できるだけ純粋・決定的（副作用を避ける）。
- 命名: ファイル/モジュール `lower_snake_case.py`、クラス `CamelCase`、関数/変数 `snake_case`。
- `effects`/`shapes` 追加時は各 `registry.py` に登録。
- コメント/docstring は日本語。

## テスト指針

- `pytest` 使用。`tests/test_*.py` を対象モジュールと対応させる。
- 乱数は固定し再現性を確保。
- 公開 API 変更時はスタブ再生成し、スタブ同期テストを更新。
- マーカー別の実行例:
  - 並行処理: `pytest -q -m integration -k worker`
  - スタブ同期: `pytest -q tests/stubs/test_g_stub_sync.py tests/stubs/test_pipeline_stub_sync.py`
  - e2e/perf: `pytest -q -m "e2e or perf"`

## Good / Bad 実例（本リポ内参照）

- Good:
  - 設計意図が短く端的な docstring と型注釈: `src/engine/core/geometry.py`
  - API スタブの自動生成とテスト整合: `src/scripts/gen_g_stubs.py` と `tests/stubs/test_g_stub_sync.py`
- Bad（避ける）:
  - 重い依存を無断で追加/輸入（optional は `tests/optional` に隔離し、Ask-first）
  - 公開 API 層に肥大ロジックを持ち込む（薄い再エクスポートを保つ）

## コミット & PR ルール

- タイトル: Conventional Commits（例: `feat(effects): add ripple clamp`）
- 必須: Lint/Format/Type/Test 緑、必要時スタブ更新。差分の要約と根拠（ログ/出力）を PR に記載。
- 提出前: 初回 `pre-commit install`、以降 `pre-commit run -a`。

## When Stuck（詰まったら）

- 短い実行計画の提示、確認質問、または Draft PR を作成して相談する。

## 相互運用（他ツール）

- 本 AGENTS.md を単一の真実源とする。未対応ツール用の `CLAUDE.md`/`GEMINI.md` 等からは「AGENTS.md を参照」と誘導する。

## モノレポ運用（将来拡張）

- ルートに最小 AGENTS.md を置き、パッケージ直下に AGENTS.md を追加可能。競合時は“近接優先”。

## セキュリティ & 設定のヒント

- 機密情報をコミットしない。`config.yaml` はローカル用途—個人情報を避ける。
- 大容量/生成物は `data/` または `screenshots/` へ。リポを軽量に維持。

---

維持運用チェックリスト（推奨）

- ルート AGENTS.md の最小項目（Build/Test/Style/Safety/PR）を常に最新化
- ルート architecture.md は実装（src/ 配下）と同期しています。差分を見つけた場合は、該当コードの参照箇所とともに更新してください。
- 変更が繰り返される事項は規約化（Do/Don’t を短文で追記）
- “編集ファイル限定”の高速チェック系コマンドを整備（型/整形/Lint/単体）
- Ask-first 操作（依存追加/破壊的/ネットワーク/フルビルド/push）を明記
- CI でスタブ同期と必須チェックを検証
