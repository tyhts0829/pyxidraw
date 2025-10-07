# plan: fill の `mode` を廃止し `angle_sets`（整数）に統一、dots を削除

目的
- `mode=['lines','cross','dots']` をやめて、整数 `angle_sets` により方向数を指定する API にする。
- 例: `angle_sets=1`→単方向、`2`→90°クロス（現行 cross）、`3`→60°間隔の3方向。
- dots は仕様から削除する（線描画エンジン前提のため）。

実装タスク（チェックリスト）
- [x] `fill(g, *, angle_sets: int = 1, angle_rad: float, density: float, remove_boundary: bool)` にシグネチャ変更。
- [x] XY共平面パス: `_generate_line_fill_evenodd_multi(..., angle)` を `i=0..k-1` の `angle + i*(pi/k)` で合成。
- [x] 非共平面パス: `_fill_single_polygon(..., angle_sets=k)` で同様に `k` 方向を合成。
- [x] `__param_meta__` を更新（`angle_sets` を integer min=1,max=8）。
- [x] dots 系の生成関数を無効化/削除（互換のため空実装に変更）。
- [x] 既存テストの `mode` 参照を `angle_sets` に置換。dots を使う箇所は `angle_sets>=1` に置換。
- [x] スタブ再生成 `python -m tools.gen_g_stubs`。`tests/stubs` を緑化。
- [x] ドキュメント例（`docs/pipeline.md`）の `mode` を置換。

検証（変更ファイル優先）
- ruff/black/isort: 変更ファイルに対して実行。
- mypy: `src/effects/fill.py`。
- pytest: `tests/test_effect_fill_remove_boundary.py`, `tests/test_effect_fill_nonplanar_skip.py`, `tests/stubs/*`。

備考
- 既定の `angle_sets=1` は従来の `mode='lines'` に相当し、破壊的変更の影響を最小化。
- dots の需要が再び出た場合は別エフェクト（`stippling` 等）として独立に設計する。

