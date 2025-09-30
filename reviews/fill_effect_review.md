# fill エフェクト コードレビュー（提案とチェックリスト）

- 対象: `src/effects/fill.py`
- 目的: 現実装の設計/挙動/性能/規約適合の観点でのレビューと、改善アクションの提示。
- 備考: 本ファイルはレビューのみ。コード変更は未実施。

## 概要（所見）
- 単一ポリゴンと XY 共平面の多輪郭（偶奇規則）を使い分け、XY 射影→生成→3D復元の流れが一貫。
- スキャンライン法（交点抽出）と偶奇規則の適用は妥当。穴（holes）を保持して塗り区間を構成。
- 交点/内外判定は Numba 化されホットパス最適化。純関数で Geometry を返し、Pipeline/キャッシュ設計と整合。

## 良い点
- 偶奇規則で多輪郭を一括処理し、ハッチ/クロス/ドットを生成する構成が明快。
- 交点判定は半開区間＆0除算回避で境界の二重カウントを抑制。
- `__param_meta__` を定義し GUI と連携点が明確。

## 気になった点 / 改善候補（抜粋）
- 密度 `density` は実質「本数/グリッド数」で `int(round(...))` の丸めを行うが、UI では number のまま。量子化/整数扱いの方針を揃えたい。
- 角度指定時、「間隔は未回転の高さで一定」「スキャン範囲は回転後」のため、角度により実本数が変動する。仕様として合理だが、ドキュメント補足があると親切。
- Numba 関数 `generate_line_intersections_batch` が Python の list[(float, ndarray)] を返すため、nopython モード維持が不安定になり得る（object モード落ち）。
- いくつか dtype/キャストの微最適化（`np.zeros` の dtype 明示など）。
- 未使用の補助関数が残存（`_find_line_intersections`, `_point_in_polygon`）。

## 事前確認したい点（仕様・方針）
- density を UI では「整数ステップ」で扱いますか（例: `type: integer` または `step: 1.0`）？
- `angle_rad` の量子化粒度の希望はありますか（キャッシュ効率目的で `step = π/180` など）？
- Numba 戻り型の厳格化（typed.List 等）に伴う実装複雑化の許容度は？

## 改善アクション（チェックリスト）
- [ ] ループ外キャスト: `_generate_dot_fill_evenodd_multi` で `offsets.astype(np.int32)` をループ外に移動（性能改善）。参照: `src/effects/fill.py:317-321`
- [ ] dtype 明示: 単一ポリゴン経路の `line_3d` やドットセグメント生成で `np.zeros(..., dtype=np.float32)` を明示（無駄な float64 化回避）。参照: `src/effects/fill.py:98`, `src/effects/fill.py:165`, `src/effects/fill.py:169`
- [ ] __param_meta__ 整合: `density` を整数扱いに揃える（`{"type": "integer"}` または `{"type": "number", "step": 1.0}`）。参照: `src/effects/fill.py:409-413`
- [ ] 角度量子化（任意）: `angle_rad` に `step` を設けるか検討（例: `step = π/180`）。GUI/体験とキャッシュ効率のバランス次第。参照: `src/effects/fill.py:409-413`
- [ ] Numba 戻り型の堅牢化: `generate_line_intersections_batch` を `numba.typed.List` + `Tuple` で型固定、または「y配列」と「交点値配列＋offsets」構造へ再設計（nopython 維持）。参照: `src/effects/fill.py:487-495`
- [ ] ドキュメント補足: 間隔は角度に依存せず一定だが、角度により実本数が変動する旨を docstring に追記。参照: `src/effects/fill.py:25`, `src/effects/fill.py:346`
- [ ] 未使用関数の整理: `_find_line_intersections` / `_point_in_polygon` を削除または用途注記を追記し保守性向上。参照: `src/effects/fill.py:333-343`
- [ ] （任意）重複計算の削減: `center` の算出や `cos/sin` をループ外に寄せる軽微な最適化（現状でも可読性優先で可）。

## 実施順序（提案）
1) ループ外キャスト、dtype 明示（安全かつ効果大）
2) __param_meta__ の整合（UI/量子化の一貫性）
3) ドキュメント補足（仕様の明文化）
4) 未使用関数の整理（削除 or 注記）
5) Numba 戻り型の堅牢化（nopython 維持のための再設計）

## 実施後の検証（変更ファイル限定）
- Lint/Format/Type: `ruff check --fix src/effects/fill.py && black src/effects/fill.py && isort src/effects/fill.py && mypy src/effects/fill.py`
- 既存最小動作確認（例）: `pytest -q -k fill`（該当テストがなければ smoke 程度）
- GUI パラメータ変更時の高速チェック（必要に応じて）: `pytest -q tests/ui/parameters`

---
本チェックリストで問題なければ、この順序で小さくパッチを切っていきます。必要があれば方針（特に density/angle の量子化と Numba 型設計）をご指定ください。

