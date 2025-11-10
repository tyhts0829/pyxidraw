# clip: 非共平面は常にXY投影（`use_projection_fallback`/`projection_use_world_xy` 廃止）計画

目的: クリップ処理の前提を「非共平面が当たり前」に置き、非共平面時は常にXY投影で処理する仕様に固定する。これに伴い、`use_projection_fallback` と `projection_use_world_xy` を引数から削除し、API/実装/キャッシュを単純化する。

背景/現状:
- 現状は `if use_projection_fallback and projection_use_world_xy:` の二段スイッチで投影フォールバックの有無を切替。
- `projection_use_world_xy` は将来の座標系選択用の名残で、実装上は実質的に「OFFスイッチ」。
- 既定が True のため、実利用では常に投影するケースが大半。二つの引数は冗長で認知負荷がある。

---

変更内容（チェックリスト）

- [x] API 署名の破壊的変更
  - [x] `use_projection_fallback: bool` と `projection_use_world_xy: bool` を `clip` シグネチャから削除
  - [x] docstring の該当節を削除/更新（非共平面は常にXY投影と明記）
  - [x] `__param_meta__` から両エントリを削除

- [x] 分岐・制御の単純化
  - [x] 非共平面 or `_MODE_PIN == 'proj'` の場合は無条件で XY 投影経路を実行
  - [x] これに伴う no-op 経路（フォールバック無効時の早期 return）を削除
  - [x] 早期分岐（bounds/prepared）ロジックは現行のまま、`draw_inside` のみで判定

- [x] 結果キャッシュキーの更新
  - [x] `res_key` から `use_projection_fallback` と `projection_use_world_xy` を除外
  - [x] マスク準備キャッシュキーは現行の `(digest, 'proj'|'planar')` を維持

- [x] ドキュメント/注記の更新
  - [x] モジュール/関数 docstring に「非共平面は常にXY投影」「no-op切替は不可」を追記
  - [x] on-edge 差異の注記は維持

- [ ] オプション（デバッグ向け・後日）
  - [ ] グローバル設定（例: `CLIP_USE_PROJECTION=0`）で投影を一時的に無効化できる隠しスイッチを検討（デフォルト ON）

- [ ] 検証（編集ファイル限定・高速）
  - [ ] `ruff check --fix src/effects/clip.py`
  - [ ] `black src/effects/clip.py && isort src/effects/clip.py`
  - [ ] `mypy src/effects/clip.py`
  - [ ] `pytest -q -k clip` または `-m smoke`（あれば）

---

互換性/移行
- 破壊的変更: `use_projection_fallback`/`projection_use_world_xy` を指定していた呼び出しは削除が必要。
- 期待動作の差: これまで no-op を選べていた箇所は、非共平面時にXY投影クリップが行われる。
- 代替案（必要時）: 一時的に no-op を維持したい場合は、暫定で `MODE_PIN` を `planar` に固定するなどアプリ側で回避（将来的には設定スイッチ検討）。

リスク/デメリット
- 明示的 no-op の喪失（ただしデフォルト True 運用では影響小）。
- 常時投影により、非共平面が巨大な場合の計算コストが増える可能性（グリッド/キャッシュで緩和）。
- XY 投影による視覚的解釈の差（高さ情報は無視）。

タスク順序
1) シグネチャ・docstring から2引数削除 → `__param_meta__` 更新
2) フォールバック分岐・早期 return の削除 → 2経路（planar/Shapely, proj/XY）に集約
3) キャッシュキーから2引数のブールを除去
4) 注記更新（docstring）
5) 検証（ruff/black/isort/mypy/pytest）

完了条件
- 変更ファイル（src/effects/clip.py）に対する ruff/mypy/pytest が成功し、__param_meta__ と docstring の整合が取れていること。
