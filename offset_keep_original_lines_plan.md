# offset エフェクト: オフセット前のオリジナル線を残すオプション追加計画

目的: `effects.offset` に「オフセット前のオリジナル線も残す」ブール引数を追加し、既存スケッチとの互換性を保ちつつ、オフセット結果と元線を同時に扱えるようにする。

対象:

- 実装: `src/effects/offset.py`
- 公開 API スタブ: `src/api/__init__.pyi`（自動生成）
- テスト: `tests/effects/test_offset_minimal.py`, `tests/effects/test_offset_joins.py`（および必要なら追加テスト）
- ドキュメント: `architecture.md`（offset 記述があれば同期）

---

## 仕様案の整理

- パラメータ名案: `keep_original: bool = False`
  - 既定値 False で現行挙動と完全互換。
  - True のとき、「オフセット後の線 + 入力 Geometry の線」を 1 つの `Geometry` にまとめて返す。
- 挙動案:
  - `distance == 0` のとき:
    - 現状どおり no-op とし、`keep_original` の真偽に関わらず入力のコピーを返す（オフセットが存在しないため）。
  - `distance != 0` で Shapely `buffer` の結果が空のとき:
    - 現状どおり入力コピーを返す（`keep_original` は無視）。
  - `distance != 0` かつ `keep_original=False`:
    - 現状どおり「オフセット結果のみ」の Geometry を返す。
  - `distance != 0` かつ `keep_original=True`:
    - オフセット結果の各ポリラインに加えて、元の各ポリラインもそのまま追加する。
    - スケーリング補正 `_scaling` はオフセット結果のみに適用し、元線には一切かけない。
- 出力順序案:
  - オフセット結果を先に並べ、その後ろにオリジナル線を追加する（`[offset lines..., original lines...]`）。
    - 理由: 「offset エフェクトはオフセット結果を生成する」が主であり、「元線を残す」は付加的なオプションのため。
    - 後続エフェクトが「先頭数本のみを処理する」場合など、既存用途への影響を最小化することを優先。

---

## 実装タスク（チェックリスト）

### A. API / パラメータ定義

- [x] A-1. パラメータ名と既定値を確定する  
  - 現案: `keep_original: bool = False`（変更しない限り現行スケッチへ挙動影響なし）。
- [x] A-2. `PARAM_META` に `keep_original` の RangeHint を追加  
  - 例: `{"type": "bool"}` とし、量子化対象外（bool）として扱う。
- [x] A-3. `offset` 関数シグネチャに `keep_original` を追加し、docstring に説明を追記  
  - Parameters セクションに「bool, default False」「True で元線も残す」旨を明記。

### B. offset 本体ロジック拡張

- [x] B-1. `distance == 0` の early return 分岐での仕様を確認し、`keep_original` に依らず現状どおり `Geometry(coords.copy(), offsets.copy())` を返す方針を維持。
- [x] B-2. `_buffer` 呼び出し後に `keep_original` の真偽で処理を分岐  
  - `keep_original=False`: 現行コードと同じフロー（オフセット結果のみ）。
  - `keep_original=True`: `filtered_vertices_list` に加えて、入力 Geometry から復元した元ポリラインも追加。
- [x] B-3. 元ポリラインの追加方式を実装  
  - `coords, offsets = g.as_arrays(copy=False)` から各ポリラインを取り出し、`len(vertices) >= 1` のものを `np.float32` でコピーして追加。
  - 3D 姿勢は既に `Geometry` が表現しているものをそのまま用い、`transform_to_xy_plane` / `_buffer` 中のスケーリング補正は元線へ適用しない。
- [x] B-4. `new_offsets` 生成ロジックは現在のまま、`filtered_vertices_list` の要素数増加に自然に追従するようにする（元線追加分も同じループでカバー）。

### C. テスト追加・更新

- [x] C-1. `tests/effects/test_offset_minimal.py` に `keep_original=True` のテストを追加  
  - 例: 単一の直線入力に対して `distance>0` かつ `keep_original=True` で呼び、  
    - 出力のライン数が 2 本（オフセット線 + 元線）になっていること。  
    - うち 1 本は元入力と完全一致すること。
- [x] C-2. `tests/effects/test_offset_joins.py` に `keep_original=True` の smoke テストを追加  
  - join スタイルごとに `keep_original=True` を渡しても例外が出ず、offset 結果 + 元線の Geometry が返ることを確認。
- [x] C-3. `distance == 0` + `keep_original=True` の挙動を明示的にテスト  
  - 期待: 現状どおり入力コピー（ライン数・座標ともに完全一致）。
- [x] C-4. 編集ファイルに対して局所テストを実行  
  - `pytest -q tests/effects/test_offset_minimal.py`  
  - `pytest -q tests/effects/test_offset_joins.py`

### D. 公開 API スタブ / ドキュメント同期

- [x] D-1. `tools/gen_g_stubs.py` を用いて API スタブ `src/api/__init__.pyi` を再生成  
  - `offset` メソッドのシグネチャに `keep_original: bool = ...` が反映されていることを確認。
- [x] D-2. `tests/stubs/test_g_stub_sync.py` を実行し、スタブ同期テストを通す。
- [ ] D-3. `architecture.md` の effects/offset に関する記述があれば、  
  - 「オフセット結果 + 元線を残すオプションがある」ことを簡潔に追記し、参照コードとして `src/effects/offset.py` を明記。

---

## 事前に確認したいポイント / オプション

- [x] Q-1. パラメータ名と既定値  
  - 現案: `keep_original: bool = False`。  
  - 代案として `include_original` / `with_original` / `keep_input` などもあり得るが、特に好み・既存スケッチでの読みやすさの観点で希望があれば合わせたい。
- [x] Q-2. 出力ラインの順序  
  - 現案: `[offset lines..., original lines...]`。  
  - もし「元線を常に先頭にして欲しい（`[original..., offset...]`）」などの運用上の期待があれば、この段階で決めたい。
- [x] Q-3. 元線のフィルタリング  
  - 現案: 「頂点数 1 未満の線は無視、それ以外はそのまま追加」  
  - 短すぎる線やゼロ長線を積極的に間引きたいなどの要望があれば、簡易なフィルタを入れることも可能。

---

このチェックリストで問題なければ、あなたの了承後に `src/effects/offset.py` 等の実装変更に着手し、完了したタスクにチェックを入れながら進めます。
