# plan: fill 非平面ポリラインをスキップ（no-op）

目的
- 非XY共平面の入力でも各ポリラインを局所平面化して塗る現状仕様を見直し、
  「個々のポリラインが十分な平面性を満たさない場合は塗りを行わず、元の境界線のみ返す」挙動を導入する。
- これにより、3D的に歪んだ（真に非平面な）入力で発生する“境界と塗りの平面不一致”の違和感と無駄な計算を回避する。

背景（現状）
- XY共平面（全ジオメトリが同一Z）の場合は、偶奇規則で複数輪郭を一括塗り: `src/effects/fill.py:377-394`。
- 非共平面の場合はポリライン単位の個別処理（局所平面へ回転→2D塗り→3Dへ戻す）: `src/effects/fill.py:396-407`。
- 閉ループ判定はしておらず、スキャンライン/レイキャストで末尾→先頭を暗黙に接続: `src/effects/fill.py:463`, `:484`。

変更方針（概要）
- 非XY共平面パス内の「ポリライン個別処理」において、ポリラインの平面性を判定し、
  閾値を超える“非平面”と判断した場合は塗り生成をスキップして、元の境界線のみ返す。
- 3D形状の真に平面な面（例: polyhedron の各面）は従来どおり塗られる（閾値内で平面と判定）。

実装タスク（チェックリスト）
- [x] 平面性判定ヘルパを追加（`_is_polygon_planar(vertices, *, eps_abs, eps_rel) -> bool`）
  - 先に `transform_to_xy_plane(vertices)` を呼び、変換後の z 残差（`max(|z|)` あるいは `max(z)-min(z)`）を算出。
  - 閾値: `threshold = max(eps_abs, eps_rel * bbox_diag)`（`bbox_diag` は元座標の対角長）。
  - 既定値案: `eps_abs=1e-5`, `eps_rel=1e-4`。スケール不変性を確保。
- [x] `src/effects/fill.py` の `_fill_single_polygon(...)` 冒頭で平面性を判定し、非平面なら「塗りなし」で早期 return。
  - 返り値: `remove_boundary` に関わらず元頂点列 `[vertices]` を返す（下記「要確認」参照）。
- [x] 定数をファイル先頭へ追加（`NONPLANAR_EPS_ABS`, `NONPLANAR_EPS_REL`）。
- [x] `fill()` の docstring に非平面スキップの注記を追記。
- [x] `architecture.md` の該当セクション（Effects/Fill の挙動）へ「非平面ポリラインはスキップ」の補足を追加（参照: `src/effects/fill.py`）。

テスト計画
- [x] `tests/test_effect_fill_nonplanar_skip.py` を新規追加。
  - [x] 非平面四辺形（例: z が 0,0,0.1,0 の4点）に対し、`E.pipeline.fill(density>0)` の出力が入力と同一（coords/offsets が一致）であること。
  - [x] `remove_boundary=True` でも上記のケースで境界が保持されること（＝空にならない）。
  - [x] 平面な多角形（`G.polygon(n_sides=4)`）では既存のテストが引き続きパスすること（輪郭が先頭に残る）。

ビルド/検証（変更ファイル優先）
- ruff: `ruff check --fix src/effects/fill.py tests/test_effect_fill_nonplanar_skip.py`
- format: `black src/effects/fill.py tests/test_effect_fill_nonplanar_skip.py && isort src/effects/fill.py tests/test_effect_fill_nonplanar_skip.py`
- type: `mypy src/effects/fill.py`
- test: `pytest -q tests/test_effect_fill_nonplanar_skip.py` および既存の fill 近傍テスト。

設計ノート
- 判定単位は「ポリラインごと」。XY共平面一括処理（偶奇規則）は変更しない。
- 先頭3点が共線のとき `transform_to_xy_plane` は回転スキップ（z平行移動のみ）となるが、
  z 残差が大きければ非平面と判定されスキップされるため問題ない。
- dots/cross/lines いずれのモードでも一律にスキップ対象。

リスク/副作用
- 非平面ポリラインでこれまで「局所平面に載った塗り線」が生成されていた場面では、
  本変更により塗りが出なくなる（境界のみ維持）。ただし 3D面の塗りニーズは「真に平面」な面で満たされる想定。
- 平面性の閾値選定によっては過剰/過少スキップが起こり得るため、値は運用で調整可能にする（定数→環境変数化を将来検討）。

要確認（ご指示ください）
1) remove_boundary の扱い（非平面時）
   - [x] 案A: 常に無視して境界を残す（推奨: 空出力を避けるため）。
   - [ ] 案B: 指定通り境界も消す（結果が空になる可能性あり）。
2) 閾値の既定値
   - [x] `eps_abs=1e-5`, `eps_rel=1e-4` で開始。
3) ログ/計測
   - [x] デバッグ用途に「スキップ件数」を返す/記録する仕組みは今回は未対応。

完了条件（DoD）
- [x] 非平面サンプルで no-op（入力=出力）になることをテストで確認。
- [x] 平面サンプルで従来どおり塗りが生成され、既存テストが緑。
- [x] ruff/black/isort/mypy が成功し、`architecture.md` が更新されている。

スケジュール目安
- 実装: 1人日未満（変更箇所は `_fill_single_polygon` と小ヘルパ追加のみ）。
- テスト/調整: 0.5人日。

備考（将来拡張）
- `nonplanar_policy={'skip','local','planarize'}` の公開パラメータ化（既定は 'skip' か現状維持 'local'）を検討可能。
- PCA 最小二乗平面での“planarize”オプションはコスト O(N) を追加するが、見た目一貫性を高める。
