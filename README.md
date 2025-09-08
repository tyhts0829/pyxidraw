# PyXidraw5

このリポジトリは、ラインベースの幾何生成とエフェクト処理を行い、リアルタイムに描画するための軽量フレームワークです。公開 API を通じて、形状 `G.<name>()` とパイプライン `E.pipeline.<effect>(...).build()` を組み合わせたスケッチを簡潔に記述できます。

- 目的/全体像: `docs/architecture.md`
- パイプライン/API: `docs/pipeline.md`
- エフェクト一覧: `docs/effects.md`
- シェイプ一覧: `docs/shapes.md`
- 開発環境セットアップ: `docs/dev-setup.md`
- コーディング規約/運用ガイド: `AGENTS.md`

## クイックスタート

1) Python 3.10+ を用意し、仮想環境を作成:
```
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -e .[dev]
```

2) スタブ生成とスモークテスト:
```
PYTHONPATH=src python -m scripts.gen_g_stubs
pytest -q -m smoke
```

3) 実行例（プレビュー）:
```
python main.py
```

## 任意依存（実機/高速化など）

- 追加インストール: `pip install -e .[optional]`
- 使用例: shapely, numba, mido, fonttools などを使う optional テストが有効になります。

## 設定ファイル

- 既定値: `configs/default.yaml`
- ローカル上書き: ルート `config.yaml`（存在すれば既定値にマージ）
- ローダ: `util.utils.load_config()`
- サンプル: `configs/example.yaml`

## リンク集

- スタブ検証 CI: `.github/workflows/verify-stubs.yml`

バグ報告/提案は Issue にお願いします。スタブ更新が必要な変更では `python -m scripts.gen_g_stubs` を実行し、`api/__init__.pyi` を更新してください。
