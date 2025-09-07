# プロジェクト掃除ロードマップ（CLEANUP PLAN）

> 目的: 直近の大規模変更後に生じた不要物の除去、構成の整理、ドキュメント刷新、テスト戦略の再設計を段階的に実行し、安定した開発サイクルに戻す。

## 1. 原則と目標
- 最小リスクで段階的に実施（大移動は後段フェーズ）。
- 動く状態を常に維持（小さな変更単位で進める）。
- CI での再現性・自動検証を強化（lint/型/テスト/スタブ同期）。
- ドキュメントは「参照先が1つ」を徹底（重複を排除）。

## 2. 現状メモ（観測）
- ルート直下に `api/`, `engine/`, `effects/`, `shapes/`, `common/`, `util/`, `scripts/` が混在。
- `tests/` は整備済み（ユニット/統合/e2e/perf/optional）。pytest マーカー運用も確立。
- フォーマッタ/リンタ/型（Black, isort, Ruff, MyPy）は設定済（`pyproject.toml`）。
- スタブ生成 `scripts/gen_g_stubs.py` と CI ワークフロー（`verify-stubs.yml`）あり。
- アセット類（`screenshots/`, `data/`）が大きくなりやすい。

## 3. フェーズ別チェックリスト（期間指定なし）

### フェーズ0: 監査と安全策（準備）
- [x] リポジトリ健全性チェック: `ruff/black/isort/mypy --version` 確認、`pytest -q -m smoke` 実行
- [x] API/スクリプト到達性: `tests/test_api_surface.py`、`tests/integration/test_scripts_entrypoint_importable.py`
- [x] 既知課題の棚卸し: `rg -n "TODO|FIXME|HACK"` 出力をIssue化（該当なし）
- [x] ベースライン取得: `PXD_UPDATE_SNAPSHOTS=1 pytest -m "e2e or perf"`
- 受け入れ: smoke/e2e/perf が緑（perfは警告可）、問題点がIssue化

### フェーズ1: 最小整理（移動なし）
- コード/構成
  - [x] 未使用コード/コメントの削除、命名/定数の微修正（副作用なし）（`main.py` の未使用 import を整理）
  - [x] `scripts/` 整理（`tune_*.py` → `scripts/tuning/`、互換の呼び出しを `README` に明記）（現状 `tune_*` は存在せず、`scripts/tuning/` を新設）
  - [x] `requirements-dev.txt` 追加、`pre-commit` に `ruff/black/isort/mypy/pytest -m smoke` を統合（`types-PyYAML` を追加）
- テストゲート（自動/手動）
  - [x] ユニット/統合: `pytest -q`
  - [x] 並行処理: `pytest -m integration -k worker`（最短経路/多重 close/no-op）
  - [x] スタブ同期: `tests/test_g_stub_sync.py`, `tests/test_pipeline_stub_sync.py`
  - [x] パフォーマンス・スモーク: `pytest -m perf`（警告のみ）
- 受け入れ: 全テスト緑（optional除く）。lint/format/typeはCIで通過

### フェーズ2: ドキュメント刷新
- コンテンツ
  - [x] `docs/architecture.md`（目的/データモデル/境界/フロー）
  - [x] `docs/pipeline.md`, `docs/effects.md`, `docs/shapes.md`, `docs/dev-setup.md`
  - [x] `README.md` 更新（目的/クイックスタート/リンク集）。`AGENTS.md` と相互リンク
- 品質
  - [x] 断片の重複排除（単一参照先の原則）
  - [x] コード例の実行確認（API整合）
- 受け入れ: `pytest -q` 緑、`README` から各docsへ辿れる

### フェーズ3: 構成変更（任意・段階導入）
- `src/` レイアウト移行（必要なら）
  - [x] `src/` 配下へパッケージ移動、起動スクリプト/エントリの調整（完了）
  - [x] `tests/integration/test_src_layout_imports.py` 緑（サブプロセスで検証）
- 設定ファイルの再配置
  - [x] `config.yaml` → `configs/default.yaml`、上書きは `config.yaml`（`util.utils.load_config` で併用）
  - [x] サンプル `configs/example.yaml` 追加、README に反映
- スクリプトの互換
  - [x] `scripts/gen_g_stubs.py` エントリ維持（importable/`main()`）
- 受け入れ: `pytest -q` 全緑、`-m integration -k src_layout` 緑、optional ジョブ導入（CI）

## 4. 目標ディレクトリ構成（段階移行）
- フェーズ1（移動なし、命名の整理のみ）
  - `api/`, `engine/`, `effects/`, `shapes/`, `common/`, `util/`
  - `scripts/`（`scripts/tuning/` を新設）
  - `tests/`, `docs/`, `screenshots/`, `data/`, `configs/`
- フェーズ3（任意）
  - `src/<各パッケージ>`（既存 `main.py` はルートに残す）

## 5. ドキュメント方針
- 単一入口: `README.md`（概要/クイックスタート/リンク集）。
- コントリビューション: `AGENTS.md`（更新済）。
- 詳細: `docs/` に体系化。重複禁止、リンクで集約。
- 記述ルール: 冒頭に「目的/前提/手順/検証/注意」を統一見出しで。

## 6. テスト戦略
- 階層
  - ユニット: `common/`, `util/`, 効果/形状の純粋関数（数値は許容誤差付き）。
  - スナップショット: 幾何ダイジェスト（座標列のハッシュや統計量）で回帰検知。
  - プロパティ: `hypothesis` による不変条件（例: 変換の合成で頂点数不変 など）。
  - スモーク: `python -m scripts.run_unified_geometry_check` 相当を `pytest -m smoke` 化。
- 命名/配置
  - `tests/test_*.py`、対象モジュールに対応。`@pytest.mark.smoke` を用意。
- カバレッジ
  - 初期 50% → 70% を目標（フェーズ2終了時点）。

## 7. CI/CD 強化
- `verify-stubs.yml` を拡張
  - 追加ステップ: `ruff check`, `black --check`, `isort --check-only`, `mypy`, `pytest -q`。
  - キャッシュ: `pip`, `pytest`。
- 変更レビュー要件
  - スクショ差分は `screenshots/` へ、説明に貼付。スタブ変更は生成コマンドを明記。

## 8. 依存管理
- `requirements.txt`（ランタイム最小）と `requirements-dev.txt`（開発: numpy, fonttools, pytest, hypothesis, black, isort, ruff, mypy, pre-commit）。
- 将来案: `pip-tools` または `uv` によるロック。

## 9. デプリケーション方針
- 公開APIの削除は2リリース猶予。`warnings.warn(..., DeprecationWarning)` を付与し、`CHANGELOG.md` に記載。

## 10. 成功指標（Definition of Done）
- CI グリーン（lint/型/テスト/スタブ同期）。
- `docs/` 整備完了、README/AGENTS が最新を指す。
- `tests/` が存在し、スモーク+主要ユニットが安定。
- 使われていないファイル/コメントが削減（`rg --unreachable` 的な監査をクリア）。

## 11. 実施タスク（フェーズ横断の横串）
- [x] 変更ごとのテスト実行テンプレ: `pytest -q` / `pytest -m integration -q` / `pytest -m "e2e or perf"`（AGENTS.md に追加）
- [x] CI ワークフロー拡張（lint/型/テスト/optional ジョブ、pip/pytest キャッシュ）
- [x] 既存Issue/PRのクローズ条件をこの計画の受け入れ基準に合わせて更新（AGENTS.md にDoDを明記）

## 12. コマンド例（抜粋）
- 環境構築: `python3.10 -m venv .venv && source .venv/bin/activate`
- 依存: `pip install -U pip && pip install -r requirements-dev.txt`
- 実行: `python main.py`
- 整形/静的解析: `ruff check . && black . && isort . && mypy .`
- テスト: `pytest -q`（スモークのみ: `pytest -q -m smoke`）
- スタブ更新: `python -m scripts.gen_g_stubs`（更新後に `api/__init__.pyi` を目視確認）

---
この計画はフェーズ完了ごとに見直します。最初の変更セットは「フェーズ0: 監査と最小テスト雛形追加」です。
