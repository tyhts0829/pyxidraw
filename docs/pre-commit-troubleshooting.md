# pre-commit トラブルシューティング チェックリスト

> 目的: pre-commit が失敗する原因を素早く特定し、修正して通過させる。
>
> 対象フック（このリポ）: 生成/テスト系 + フォーマッタ/リンタ + 型チェック
>
> - 生成/テスト系: `gen-g-stubs`, `test-g-stub-sync`, `test-pipeline-stub-sync`, `smoke-tests`
> - 整形/静的検査: `black`, `ruff --fix`, `isort`
> - 型チェック: `mypy`

---

## TL;DR（最短ルート）

- [ ] 仮想環境が有効（Python 3.10+）か確認: `which python` / `python -V`
- [ ] 依存を開発セットで導入: `pip install -e .[dev]`
- [ ] 一括実行で状況把握: `pre-commit run -a -v`
- [x] フォーマッタが変更したら再ステージ: `git add -A`
- [x] スタブを再生成＆同期テスト: `PYTHONPATH=src python -m scripts.gen_g_stubs && pytest -q tests/test_g_stub_sync.py tests/test_pipeline_stub_sync.py`
- [ ] 型チェックを通す: `mypy`
- [ ] もう一度: `pre-commit run -a`（成功するまで反復）

---

## 1) 失敗フックの特定

- [x] `pre-commit run -a -v` を実行し、失敗した Hook ID とログを記録する。
- [x] 設定を確認: `.pre-commit-config.yaml`（Hook の順序・除外・固定バージョン）。
- [x] 失敗の型を分類する:
  - 生成差分（ファイルが自動更新されコミット中断）
  - 整形差分（black/isort/ruff の自動修正）
  - テスト失敗（`pytest` 系）
  - 型エラー（`mypy`）
  - インポート/環境エラー（PYTHONPATH/依存不足）

---

## 2) 生成・同期系フックの対処

対象: `gen-g-stubs`, `test-g-stub-sync`, `test-pipeline-stub-sync`

- [x] まずスタブを生成: `PYTHONPATH=src python -m scripts.gen_g_stubs`
- [x] 差分を確認: `git diff -- src/api/__init__.pyi`
- [x] 差分がある場合はステージ: `git add src/api/__init__.pyi`
- [x] 同期テストを実行: `pytest -q tests/test_g_stub_sync.py tests/test_pipeline_stub_sync.py`
- [ ] よくある原因と対処:
  - 形状/エフェクトの追加・改名後にスタブ未更新 → 上記生成＋ステージで解消。
  - レジストリに登録漏れ（`effects/registry.py`, `shapes/registry.py`）→ 登録を追加し再生成。
  - 生成時 ImportError → `PYTHONPATH=src` を付ける／`pip install -e .[dev]` を実施。

---

## 3) 整形・静的検査（black/ruff/isort）

- [x] 一括修正: `ruff check . --fix && black . && isort .`
- [x] 再ステージ: `git add -A`
- [x] もう一度 `pre-commit run -a`。
- メモ: フォーマッタがファイルを変更した直後のコミットは拒否されます。必ず再ステージしてやり直す。

---

## 4) 型チェック（mypy）

- [x] `mypy` を単体実行し、エラー原文を確認。
- [ ] このリポ固有の注意点:
  - `pyproject.toml` の `[tool.mypy].files` が `util/utils.py` を指していますが、構成変更で `src/util/utils.py` に移動済みです。
-  - 今回はピンポイント修正（B）で解消:
      ```toml
      [tool.mypy]
      exclude = "...|^src/api/__init__\\.pyi$"
      files = ["src/util/utils.py"]
      ```
- [x] 変更後に再実行: `mypy`
- [ ] サードパーティ型不足は当面 `ignore_missing_imports = true` で抑制（既定）。必要に応じて `types-*` を追加。

---

## 5) スモークテスト（`-m smoke`）

対象: `smoke-tests`

- [ ] 開発依存導入済みか確認: `pip install -e .[dev]`
- [ ] 単体実行: `PYTHONPATH=src pytest -q -m smoke`
- [ ] 失敗時の主な原因:
  - ランタイムの import パス不整合 → `PYTHONPATH=src` を付ける。
  - `tests/_utils/dummies.py` の導入漏れ → テスト内で呼ばれているか確認。
  - 破壊的変更で API シグネチャがズレた → 変更箇所を `main.py`/エンジン側と合わせる。

---

## 6) キャッシュ・環境のリセット

- [ ] pre-commit キャッシュを消す: `pre-commit clean`
- [ ] pytest キャッシュを消す: `rm -rf .pytest_cache .mypy_cache .ruff_cache`
- [ ] 仮想環境が壊れていそうなら再作成。

---

## 7) 最終確認とコミット

- [ ] `pre-commit run -a` がオールグリーンである。
- [ ] 変更をステージしてコミット。必要なら `--no-verify` を外して通常コミットへ戻す。

---

## よくあるエラーメッセージと即応

- "mypy: can't read file 'util/utils.py'":
  - `pyproject.toml` の `files` を `src` へ更新（上記 §4）。
- `ModuleNotFoundError`（スタブ生成/テスト時）:
  - `PYTHONPATH=src` を付けて実行。`pip install -e .[dev]` で依存導入。
- フォーマッタ実行後に即コミットが失敗する:
  - 変更が発生したため。`git add -A` して再トライ。

---

## 参考: このリポの Hook 一覧（.pre-commit-config.yaml）

```yaml
repos:
  - repo: local
    hooks:
      - id: gen-g-stubs
        entry: python -m scripts.gen_g_stubs
        env: { PYTHONPATH: src }
      - id: test-g-stub-sync
        entry: python -m pytest -q tests/test_g_stub_sync.py
        env: { PYTHONPATH: src }
      - id: test-pipeline-stub-sync
        entry: python -m pytest -q tests/test_pipeline_stub_sync.py
        env: { PYTHONPATH: src }
      - id: smoke-tests
        entry: python -m pytest -q -m smoke
        env: { PYTHONPATH: src }
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks: [{ id: black, exclude: ^api/__init__\.pyi$ }]
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.5.6
    hooks: [{ id: ruff, args: ["--fix"] }]
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks: [{ id: isort }]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks: [{ id: mypy, additional_dependencies: [types-PyYAML] }]
```

---

## 付録: 相談時に貼ってほしい情報

- pre-commit の実行結果（失敗フックと末尾のエラー全文）
- `git diff`（自動生成/整形での差分）
- `pyproject.toml` の `[tool.mypy]` と `[tool.ruff*]` セクション
- 実行したコマンドと仮想環境の Python バージョン

---

このファイルはリポジトリの標準ワークフロー（README/AGENTS 参照）に合わせ、段階的に型検査を導入しつつ pre-commit をグリーンに保つことを目的としています。
