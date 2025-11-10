# clip: 描画モード簡素化（`draw_inside` 単一化）計画

目的: `draw_inside` と `draw_outside` の二重指定を廃止し、
`draw_inside: bool` のみで「内側/外側」を切り替える、シンプルで曖昧さのない API に統一する。
（`draw_inside=True`→内側、`False`→外側）。

背景/動機:
- `both`（内外同時）と `none`（どちらも描かない）は有用性が低く、誤用/計算無駄を招く。
- GUI/パイプラインでのパラメータ数を減らし、読み手の負担を軽減する。
- リポ方針上、破壊的変更は許容（美しいシンプル実装重視）。

影響範囲:
- 実装: `src/effects/clip.py`
- GUI/メタ: `__param_meta__`（同ファイル末尾）
- 呼び出し側（必要なら）: `draw_outside` を指定している箇所の置換
- ドキュメンテーション（必要最小限）

---

実施項目（チェックリスト）

- [x] 署名変更（破壊的）
  - [x] `clip(..., draw_inside: bool = True, draw_outside: bool = False, ...)` から `draw_outside` を削除
  - [x] docstring から `draw_outside` を削除し、切替仕様を明記（`draw_inside=True/False`）

- [x] 分岐ロジックの一本化
  - [x] Shapely 経路: `draw_inside` が True なら `intersection`、False なら `difference` のみ実行
  - [x] フォールバック経路: 「inside == draw_inside」を採択条件に統一
  - [x] 早期分岐: `bounds`/`prepared.contains/disjoint` も `draw_inside` に従って単純化

- [x] 結果キャッシュキーの更新
  - [x] `res_key` から `bool(draw_outside)` を削除（互換不要。キー縮減）
  - [x] プロジェクション・プラナーモード双方で同様に修正

- [x] UI/メタデータ
  - [x] `__param_meta__` の `draw_outside` エントリ削除
  - [ ]（必要なら）既存 GUI での表示確認（`draw_inside` のみ）

- [x] 呼び出し箇所の置換（必要時に限定）
  - [x] `draw_outside=True` を `draw_inside=False` に書き換え（該当なし）
  - [x] `draw_inside=True, draw_outside=True`（both）は `draw_inside=True` で十分（該当なし）
  - [x] `draw_inside=False, draw_outside=False`（none）は `draw_inside=False` に一本化（該当なし）

- [ ] ドキュメント（最小）
  - [ ] 本計画ファイルへの「移行ガイド」節を追記（コミット後）

- [ ] 検証（編集ファイル限定・高速ループ）
  - [ ] `ruff check --fix src/effects/clip.py`
  - [ ] `black src/effects/clip.py && isort src/effects/clip.py`
  - [ ] `mypy src/effects/clip.py`
  - [ ]（あれば）`pytest -q -k clip` または `-m smoke`

---

移行ノート（呼び出し側）
- 旧: `clip(..., draw_inside=False, draw_outside=True)` → 新: `clip(..., draw_inside=False)`
- 旧: `clip(..., draw_inside=True, draw_outside=False)` → 新: `clip(..., draw_inside=True)`
- 旧: both/none → 新: `draw_inside=True/False` のどちらか一方に統一（動作上の差異なし or no-op 削減）

確認事項
- 互換レイヤ（警告）なしの完全削除で確定（はい）。
- 既存ワークフロー/サンプルで `draw_outside` を指定していないか要確認（該当時は同時修正）。

承認後、上記の順で実装に着手し、項目を完了チェックしていきます。
