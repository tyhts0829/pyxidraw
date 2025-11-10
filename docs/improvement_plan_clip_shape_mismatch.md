# 改善計画: clip(planar+Shapely) 早期経路の配列形状不整合修正

## 目的
- `effects.clip` の共平面＋Shapely 経路における早期分岐（AABB 非重複／完全内外）の出力配列形状不整合を解消し、`transform_back` での行列積エラーを防止する。
- 実装を最小・明確に保ち、既存の規約（XY 配列は Z=0 を付与して (N,3) に統一）に合わせる。

## 背景 / 原因の要約
- 該当箇所: `src/effects/clip.py:1043-1115` 付近。
- 早期経路で `pl`（形状 (N,2)）をそのまま `out_lines_xy` に追加している。
  - 行: 1069, 1077, 1081（`out_lines_xy.append(pl)`）。
- 後段で `transform_back(arr, R_all, z_all)` を適用するが、`transform_back` は (N,3) 前提（`src/util/geom3d_ops.py`）。
- 結果として numba 経由の `np.dot` が (N,3)×(3,3) を期待するところに (N,2) が来て落ちる。

## 実装方針（最小修正）
- 早期経路で `pl` を `out_lines_xy` へ追加する際、Z=0 列を付与して (N,3) に正規化する。
  - `np.hstack([pl, np.zeros((pl.shape[0], 1), dtype=np.float32)])` を適用。
- 他経路（Shapely 結果／フォールバック）は既に (N,3) に統一済みのため変更不要。
- 余計な防御（`transform_back` の多形対応など）は行わず、原因箇所の是正に限定する。

## 変更ファイル（想定）
- `src/effects/clip.py`
  - 1069, 1077, 1081 行の `out_lines_xy.append(pl)` を、Z=0 を付与した (N,3) 配列に置換。

## テスト計画（最小）
- 新規: `tests/effects/test_clip_planar_shapely_early_branch.py`
  - 条件: 共平面、Shapely 利用可、対象ラインが完全内側／完全外側／AABB 非重複の各ケースを作る。
  - 期待: 例外発生なし。戻り値のジオメトリが空／同一／一部クリップのいずれかとして妥当。
- 既存テストへの影響: なし（挙動は仕様通りで、例外が消えるだけ）。

## 品質チェック（変更ファイル限定の高速ループ）
- Lint: `ruff check --fix src/effects/clip.py`
- Format: `black src/effects/clip.py && isort src/effects/clip.py`
- Type: `mypy src/effects/clip.py`
- Test (局所): `pytest -q tests/effects/test_clip_planar_shapely_early_branch.py`
- スモーク: 該当スケッチの再実行（描画ウィンドウが立ち上がり例外なし）

## 検証手順（再現 → 修正確認）
1. 修正前に再現（任意）
   - スケッチ: `python sketch/251110.py`
   - 期待: `ValueError: incompatible array sizes for np.dot(a, b)` が発生。
2. 実装
   - 上記 3 箇所を (N,3) 正規化に置換。
3. 変更ファイル限定チェック
   - `ruff/black/isort/mypy` を対象ファイルに対して実行。
4. テスト
   - 追加テストを実行。`pytest -q tests/effects/test_clip_planar_shapely_early_branch.py`
5. スモーク
   - `python main.py` あるいは対象スケッチを実行し、例外が出ないことを確認。

## 影響範囲 / リスク
- 変更は早期経路の配列形状統一のみ。描画内容への影響はなし（座標は同一で Z=0 を付与）。
- Shapely 無フォールバックや非平面経路には影響なし。

## ロールバック / フォールバック
- 問題があれば当該 3 箇所の置換をリバートすれば元に戻る。
- 万一ほかの呼び出し元でも同種の不整合が見つかった場合は、同様に (N,3) へ統一する方針で対処。

## 要確認事項（お願いします）
- この「最小修正」方針で進めてよいか。
- 追加テストファイルの導入（`tests/effects/...`）は問題ないか。
- もし `transform_back` に (N,2) 許容の多形対応を追加する案を採る場合は、方針変更の指示をください（今回は避ける想定）。

## 完了条件（変更単位）
- 変更箇所の `ruff/mypy/pytest` 緑。
- 対象スケッチ実行で例外が発生しない。

---

## 進捗チェックリスト（実施・記録）
- [x] 早期経路3箇所の (N,3) 正規化を実装（`src/effects/clip.py`）
- [x] 変更ファイル限定の Lint/Format 実行（ruff/black/isort）
- [x] 変更ファイル限定の型チェック実行（mypy）
- [ ] 局所テストの追加と実行（`tests/effects/test_clip_planar_shapely_early_branch.py`）
- [ ] スモーク（対象スケッチ/`main.py` 実行で異常なしを確認）
