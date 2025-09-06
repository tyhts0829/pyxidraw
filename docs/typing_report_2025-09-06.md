# mypy 実行レポート（2025-09-06）

実行コマンド:

```
mypy . --hide-error-context --pretty --show-column-numbers
```

主な指摘カテゴリ（抜粋）

- 外部ライブラリのスタブ欠如/属性未定義
  - `yaml` のスタブ未導入（Hint: `pip install types-PyYAML`）
  - `numba` に `njit/types` 属性が無い扱い（型スタブ不足）。対処案:
    - 最短: `typings/numba/__init__.pyi` を追加し `def njit(*args, **kwargs): ...` などを定義。
    - 代替: 呼び出し側で `from typing import Any; njit: Any` による型回避。

- 返り値/未到達return
  - `engine/io/helpers.py::_is_toggle_key`、`engine/io/controller.py::calc_combined_value` などで return が不足。

- 値/変数注釈不足
  - `shapes/text.py` のキャッシュ辞書 `_fonts/_glyph_cache` などで注釈が必要。

- 型整合
  - `effects/extrude.py` の `new_offsets` を ndarray に変換後、そのまま `Geometry` へ渡す際の型ミスマッチ（list[int] 型注釈のまま）。
  - `effects/trimming.py` の numpy スカラーと Python float の不一致（`np.float32` 等）。

- スタブ不一致
  - `_GShapes` に `list_shapes/empty` を生やしていない呼び出し箇所（ツール/デモ向けユーティリティ）。対処案: Protocol を広げるか、呼び出し側を `ShapeFactory` 経由に変更。

- 既存修正（本PRの差分）
  - `api/pipeline.py` の重複検査/変数型衝突（`allowed_sorted`）を解消。

推奨アクション（段階）

1) 最小サプレッション
   - `typings/numba/__init__.pyi` を追加（`njit`, `types` ダミー定義）。
   - `types-PyYAML` を開発依存に追加。

2) 返り値と注釈の穴埋め
   - `engine/io/helpers.py`, `engine/io/controller.py` の return 追加/Optional 正規化。
   - `shapes/text.py` の辞書キャッシュに型注釈を付与。

3) 型整合の調整
   - ndarray/list の整合（`effects/extrude.py` など）を修正。

4) Protocol/スタブの整理
   - `_GShapes` にユーティリティを含めるか、使用側を `ShapeFactory`/`G` の公開APIに寄せる。

本レポートは型品質の全修正を目的とせず、優先度付けの材料として作成したものです。
