# 開発環境セットアップ

## 目的
- 同じ手順で lint/型/テスト/スタブ生成が再現できる状態を整える。

## 必要要件
- Python 3.10 以上（3.11 推奨）
- `pip`

## 初期化
```
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -e .[dev]
```

オプション依存を含める場合:
```
pip install -e .[dev,optional]
```

## よく使うコマンド
- 整形・静的検査: `ruff check . && black . && isort . && mypy .`
- テスト: `pytest -q`（スモークのみ: `pytest -q -m smoke`）
- スタブ生成/同期検証:
  - 生成: `python -m scripts.gen_g_stubs`
  - 同期テスト: `pytest -q tests/test_g_stub_sync.py tests/test_pipeline_stub_sync.py`

## pre-commit（推奨）
```
pre-commit install
pre-commit run -a
```

## トラブルシュート
- Python のバージョン差異で `|` Union が解釈できない → Python 3.10+ を使用。
- 省メモリ環境で `shapely/numba/fontTools` が無い → テストはダミー依存を自動挿入（`tests/_utils/dummies.py`）。
