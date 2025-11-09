% エフェクト `clip` 実装計画（同一平面の閉曲線で他ジオメトリを切り抜く）

目的
- 共平面にある閉曲線（リング）で定義される領域を「マスク（切り抜き）」として利用し、同一 Geometry 内の他ポリライン（例: グリッド線）を内側/外側でクリップする。
- `effects.fill` と同等の「共平面推定 → XY 整列 → 偶奇規則（外環+穴）」の設計を踏襲し、回転/スケールに対して安定な挙動を確保する。

非目標
- 3D で複数平面に跨る厳密クリップ（非共平面は安全側 no-op）。
- 形状の太さ/レンダリング属性変更（別エフェクト `style`）。
- 複数入力 API の導入は行わない。マスクは `outline` 引数で明示指定（必須）。

適用対象（確認）
- 要求スケッチ: `sketch/251109.py`
  - 例: `E.pipeline.affine().clip(outline=[ring], draw_outline=True, draw_inside=True)(grid)`
  - これに適合する API を第一優先で実装する。

仕様（案）
- 関数名/登録: `@effect()` により `effects.clip` を登録。
- シグネチャ:
  - `def clip(g: Geometry, *, outline: Geometry | list[Geometry], draw_outline: bool = False, draw_inside: bool = True, draw_outside: bool = False, eps_abs: float = 1e-5, eps_rel: float = 1e-4) -> Geometry:`
  - `outline` は必須: `g` は対象、`outline` はマスク（閉曲線のみを使用）。
  - `draw_inside`: マスクの内側部分を出力に含める（既定 True）。
  - `draw_outside`: マスクの外側部分を出力に含める（既定 False）。両方 False の場合は no-op に等価。
  - `draw_outline`: マスク輪郭線も出力に含める（既定 False）。
  - 非共平面: 安全側 no-op（`g` をそのまま返す。`draw_outline=True` のときは輪郭のみ連結して返す）。

アルゴリズム概要
1) 入力正規化
   - `outline` は必須。`outline`（単体/リスト）から `as_arrays()` を抽出。
   - マスク候補リングは「閉路（先頭≒末尾）かつ点数>=3」だけに限定。1 本も有効リングがなければ `ValueError`。
2) 共平面推定と XY 整列（重要）
   - `choose_coplanar_frame` を「対象+マスクの全頂点」に対して 1 回だけ実行。
   - `planar=False` なら no-op（`draw_outline=True` の場合は輪郭を連結）。
3) マスク領域の構築（偶奇）
   - XY 上の各リングから Shapely `Polygon` を生成し、`symmetric_difference` の反復適用で偶奇領域を構成（`effects.partition` と同様）。
   - Shapely 不在時はフォールバック: `build_evenodd_groups` でグループ化 → 各リングを耳切り三角化 → 三角重心で偶奇選別して近似領域（last resort）。
4) 対象ポリラインのクリップ（XY）
   - Shapely あり:
     - 各対象ポリラインを `LineString` 化し、`intersection(region)` で内側、`difference(region)` で外側を得る。
     - 返り値が `LineString`/`MultiLineString`/`GeometryCollection` の場合を分岐して線分配列へ変換。
   - Shapely 無し（フォールバック）:
     - 各線分をマスク各リングのセグメントと交差判定（Numba 実装のセグメント交差）→ 交点で分割。
     - セグメント中点を `point_in_polygon_njit`（util/polygon_grouping）で偶奇評価し、`draw_inside`/`draw_outside` に応じて採用。
5) 3D 復元と出力
   - 生成した 2D 線分群を `transform_back` で元姿勢へ戻し、`Geometry.from_lines`。
   - `draw_outline=True` の場合はマスク輪郭（元の 3D 座標）も連結する。

公開パラメータメタ（GUI/キャッシュ）
- `clip.__param_meta__` を設定:
  - `draw_inside`: `{type: "boolean"}`（非量子化）
  - `draw_outside`: `{type: "boolean"}`（非量子化）
  - `draw_outline`: `{type: "boolean"}`（非量子化）
  - `eps_abs`: `{type: "number", min: 1e-7, max: 1e-2, step: 1e-6}`
  - `eps_rel`: `{type: "number", min: 1e-7, max: 1e-2, step: 1e-6}`
- `outline` は UI 対象外（必須パラメータだが GUI で編集しない）。
- 量子化は float のみ適用（`AGENTS.md` 方針）。

実装ステップ（チェックリスト）

1) Effect 本体の追加と登録
- [x] 追加: `src/effects/clip.py`
  - [x] `@effect()` で `clip` を登録。
  - [x] docstring: 目的/非目標/共平面XY整列/偶奇/フォールバックを簡潔記述。
  - [x] `__param_meta__` の付与（上記）。
- [x] 追記: `src/effects/__init__.py` に `from . import clip  # noqa: F401`。

2) マスク抽出と平面整列
- [x] `outline` は必須。閉路のみを採用（先頭・末尾の距離 <= 1e-6、点数>=3）。0 本なら `ValueError`。
- [x] 対象ポリライン集合は `g` の全線（outline はマスク専用）。
- [x] `choose_coplanar_frame` を対象+マスクの連結配列に適用。`planar=False` の場合は no-op（`draw_outline` なら輪郭を連結）。

3) マスク領域の構築（Shapely 優先）
- [x] XY 上のマスクリングを `Polygon` に変換。無効/自己交差は `buffer(0)` で修正を試みる。
- [x] `symmetric_difference` を順次適用し偶奇領域を得る。空なら no-op。
- [x] Shapely が使えない場合は `build_evenodd_groups` + 耳切り三角化で近似領域を構築。

4) クリップ計算
- [x] Shapely あり: 各対象ポリラインで `intersection(region)` と `difference(region)` を評価し、`draw_inside`/`draw_outside` にしたがって収集。
- [x] Shapely 無し: セグメント交差（Numba 実装）→ 分割 → 中点の偶奇評価で採否判定。

5) 3D 復元と出力
- [x] 2D 線分を `transform_back` で 3D に戻す。
- [x] `draw_outline=True` の場合、マスク輪郭（元 3D）を `concat`。
- [x] `Geometry.from_lines` で返す。入力が空/領域が空の場合は冪等に `g` を返す。

6) テスト/検証
- [ ] 追加: `tests/effects/test_clip_basic.py`
  - [ ] 共平面: 正多角形マスク + グリッド → 内側のみ保持（線分数が減少、境界に跨る線は分割）。
  - [ ] `draw_outside=True` の場合: 内側除去の逆挙動。
  - [ ] `draw_outline=True`: 出力にマスク輪郭も含まれること。
  - [ ] 非共平面入力: no-op（`outline` も併用時は輪郭連結のみ）。
- [ ] 既存スケッチ `sketch/251109.py` で実行確認（目視）。

7) ドキュメント/整合
- [ ] `architecture.md` にエフェクト一覧へ `clip` を追記（簡素に）。
- [ ] `docs/spec/pipeline.md` に「外部ジオメトリを受け取るパラメータは GUI 対象外」の注記を補足（任意）。

実装上の注意
- パフォーマンス: Shapely がある環境では `intersection/difference` を優先。フォールバックは単純・安定を優先（Numba 実装）。
- 安定性: マスクリングは閉路化を強制（終端=先頭でなければ連結）。自己交差・退化は `buffer(0)` で可能なら修復。
- キャッシュ: `outline` はオブジェクト ID で署名化される（現仕様）。外部ジオメトリを差し替えた場合はパイプラインキャッシュが分離される。

オープン事項（要相談）
- 非共平面時の `draw_outline=True` の取り扱い（`g.concat(outline)` を返すのが自然か）。
- `invert` 的なショートハンド（`inside/outside` をひっくり返す）の追加要否。
- 将来の糖衣: 厳格条件付きの自動識別（「閉曲線がちょうど1つ」の場合のみ許容）を別オプションで検討可。

— 承認依頼 —
この計画に問題がなければ進めます。修正点や追加要件があれば指示してください。
