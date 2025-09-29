# demo/effect_grid.py リファクタリング計画（提案）

目的: 可読性を上げ、最小の変更で意図を明確にする。挙動は不変。

- [ ] 命名・コメントの明確化
  - `_build_params` を `effect_default_params` に改名（役割を直示）
  - ラベル関連のコメントを簡潔化（何/なぜに集中）
- [ ] マジックナンバーの定数化
  - ラベル用フォントサイズ `LABEL_FONT_SIZE` を導入（20 を除去）
- [ ] 例外・型注釈の微修正
  - `inspect._empty` 依存を `inspect.Parameter.empty` に変更し、`type: ignore` を削減
  - 関数の戻り値型の注釈を補強（戻り値の明確化）
- [ ] 小さな関数分割
  - ラベル Geometry 構築を `_make_label_geo(name, origin)` に切り出し
- [ ] グローバルキャッシュの整理（最小）
  - 3 変数（`_CELL_GEOMS`/`_CELL_CENTERS`/`_LABELS_GEO`）を `GridCache` のデータクラスに集約（初期化の原子性/見通し向上）
- [ ] レイアウト定数の型明示
  - `CELL_SIZE: tuple[float, float]` など、型ヒントを明確化
- [ ] 変更ファイル限定チェック
  - `ruff check --fix demo/effect_grid.py`
  - `black demo/effect_grid.py && isort demo/effect_grid.py`
  - `mypy demo/effect_grid.py`

補足:
- いずれも挙動変更を伴わない範囲で実施します（表示/出力は不変）。
- データクラス導入は 1 ファイル完結の最小追加で、グローバル変数のまとまりを可視化する狙いです。

この計画で進めてよいか確認お願いします。必要なら追加・削除の指示をください。

