# src ディレクトリ別コードレビュー

## api

- **重大** `src/api/effects.py:231` `PipelineBuilder._add` が渡された `params` 辞書をそのまま保持しており、呼び出し後に参照元でミューテーションが起こるとパイプライン定義とキャッシュキーが乖離します。`pipeline = E.pipeline.displace(**opts).build(); opts["amplitude_mm"] = 1.0` のような操作で実際の処理内容が変わっても `_pipeline_key` が更新されず、誤ったキャッシュヒットや再現性崩壊に繋がります。`params` を `dict(params)` で複製して保存する必要があります。

## common

- 指摘なし（現時点で重大な問題は確認できませんでした）。

## effects

- **重大** `src/effects/__init__.py:5` 経由ですべてのエフェクトモジュールを即時 import していますが、`src/effects/displace.py:19` のように任意依存の `numba` を必須 import しているため、`pip install .`（optional extras 未指定）直後でも `ModuleNotFoundError: No module named 'numba'` で `import api` が失敗します。基盤依存に昇格させるか、エフェクト側で遅延 import とフォールバック処理を入れる必要があります。

## shapes

- **重大** `src/shapes/__init__.py:9-20` で `text` など全シェイプを即時 import しており、`src/shapes/text.py:8-11` で任意依存の `fontPens`, `fontTools`, `numba` を要求するため、optional extras を入れていない標準環境では `ModuleNotFoundError` で `from api import G` が壊れます。エフェクト同様、遅延 import か依存関係の整理が必要です。
- **重大** `src/shapes/text.py:130-190` でフォント検索パスとデフォルトフォントを macOS 固定（かつ開発者のホームディレクトリ `/Users/tyhts0829/...` を含む）にハードコードしています。Linux/Windows 環境では既定フォントが存在せず `TTFont(default_font, ...)` が `FileNotFoundError` を投げ、パッケージ import が失敗します。プラットフォームごとの検索パスを切り替えるか、存在確認してフォールバックする処理が必要です。

## util

- 指摘なし（現時点で重大な問題は確認できませんでした）。
