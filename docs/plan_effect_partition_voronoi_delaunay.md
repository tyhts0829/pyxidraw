# plan: 平面パーティション（Voronoi / Delaunay）— 設計/実装計画

目的
- 同一平面上の閉じた図形を、Voronoi 図または Delaunay 三角形分割で内部を分割し、各領域を「閉じた輪郭」の集合として出力する。
- 後段で `fill` を適用しても期待通りに機能するよう、出力は常に閉領域（クローズドポリライン）のみで構成する。

スコープ
- 対象: 新規エフェクト `src/effects/partition.py`（仮称）とレジストリ登録、`__param_meta__`、最小テスト、README/architecture への軽い言及。
- 非対象: 3D 非平面分割、パワー図/重み付き Voronoi、複雑な穴（holes）サポートの自動推定（初版では明示指定または非対応）。

仕様（提案）
- 関数名: `partition`
  - 署名（初版）:
    ```py
    @effect()
    def partition(
        g: Geometry,
        *,
        mode: str = "voronoi",           # 'voronoi' | 'delaunay'
        site_count: int = 12,             # 生成サイト数（site_source='random' のとき）
        site_source: str = "random",      # 'random' | 'grid' | 'vertices' | 'points_in_input'
        seed: int = 0,                    # RNG シード（再現性確保）
        jitter: float = 0.0,              # サイト位置の微小ランダム化（0..1, bbox 比）
        relax_iters: int = 0,             # Lloyd 緩和回数（Voronoi のみ、0 なら無効）
        group_by_plane: bool = False,     # 同一平面グループでまとめて分割（初版は False 推奨）
        keep_outline: bool = True,        # 元外周輪郭を結果に含めるか
    ) -> Geometry:
    ```
  - 入力解釈:
    - 「閉じた図形」を分割対象とする。閉路の判定は始点/終点の距離が閾値以下でクローズ扱い（必要なら自動クローズ）。
    - `site_source='points_in_input'` のとき、頂点数1のポリライン（点）群を「サイト」として使用し、それ以外のクローズ輪郭を分割領域とみなす。
    - 上記以外は各領域ごとに `site_count` を内部生成（`random` or `grid`。`vertices` は外周頂点をサイトに流用）。
  - 出力: 分割で得られた各セルを「閉じた輪郭（外周）」として返す。穴は初版では未対応（後続 `fill` の偶奇規則に依存しない形にする）。
  - 非平面: 頂点がほぼ平面でない場合は入力をそのまま返す（`fill` と同等の平面性判定を準用）。

UI メタ（RangeHint）
- `partition.__param_meta__`（量子化/GUI 用）:
  - `mode`: `{choices: ["voronoi", "delaunay"]}`
  - `site_count`: `{type: "integer", min: 1, max: 500, step: 1}`
  - `site_source`: `{choices: ["random", "grid", "vertices", "points_in_input"]}`
  - `seed`: `{type: "integer", min: 0, max: 2_147_483_647, step: 1}`
  - `jitter`: `{type: "number", min: 0.0, max: 0.5, step: 1e-3}`
  - `relax_iters`: `{type: "integer", min: 0, max: 10, step: 1}`
  - `group_by_plane`: `{type: "boolean"}`
  - `keep_outline`: `{type: "boolean"}`

アルゴリズム設計（概要）
- 共通前処理
  - 各ポリラインを候補領域（クローズ）とサイト（点）へ分類。
  - 各領域を `util.geom3d_ops.transform_to_xy_plane()` で XY 平面へ整列し、2D で処理。
  - サイト生成:
    - `random`: 領域ポリゴンの bbox に乱数 → ポリゴン内判定で採用。`seed` 固定で再現。
    - `grid`: bbox を均等グリッド分割し内点のみ採用。`jitter` で微少ランダム化。
    - `vertices`: 外周頂点をそのままサイトに使用。
    - `points_in_input`: 入力の点群（頂点数1の線）を使用。
- Voronoi モード
  - Shapely が利用可能なら `shapely.ops.voronoi_diagram(MultiPoint, extend_to=bounding_polygon)`（環境により `extent`/`envelope` 名が異なるためラッパ関数で吸収）。
  - 返る各セルポリゴンを分割対象ポリゴンで `intersection` し、非空/面積>εのみ採用。
  - 外周抽出は `poly.exterior.coords`。XY→元姿勢へ `transform_back()`。
- Delaunay モード
  - `shapely.ops.triangulate(Polygon or MultiPoint, tolerance=None, edges=False)` を用い、領域ポリゴンとの `intersection` を取り三角形セルを得る。
  - 線分だけでなくセル（ポリゴン）として出力し、必ず閉ループに整形。
- 仕上げ
  - 反復緩和（Lloyd）: `relax_iters>0` の場合、各セル重心へサイトを移動して再構築（Voronoi のみ）。
  - `keep_outline=False` の場合、元の外周輪郭は出力に含めない（`fill` と整合）。
  - すべての出力ループは先頭点を終端に複製して明示クローズ（ε=1e-6）。

失敗時/フォールバック
- Shapely 未導入 or `voronoi_diagram` 不在: `mode='voronoi'` 指定時は警告ログのうえ `delaunay` にフォールバックする（初版）。
- 平面性不足/サイト不足（<3）: 安全側で入力をそのまま返す。

実装タスク チェックリスト（編集単位）
- [ ] `src/effects/partition.py` 追加（docstring/型注釈/純関数）。
  - [ ] 平面性判定・XY 射影/復元（`util.geom3d_ops` 再利用）。
  - [ ] サイト生成ユーティリティ（random/grid/vertices/points_in_input）。
  - [ ] Voronoi パス（Shapely 利用、extent+clip、Lloyd 緩和0回で動作）。
  - [ ] Delaunay パス（`shapely.ops.triangulate` + clip）。
  - [ ] ループの明示クローズと数値安定化（重複点の除去、面積しきい値）。
  - [ ] `__param_meta__` 実装（量子化 step 設定）。
- [ ] レジストリ登録（`@effect()` デコレータで自動登録、`src/effects/__init__.py` は現状維持）。
- [ ] 最小テスト `tests/test_effect_partition.py`
  - [ ] `G.polygon(n_sides=4)` に `mode='voronoi', site_count=4, seed=1` → 4 セル（閉ループ）生成を確認。
  - [ ] 同上で `mode='delaunay', site_count=4` → 三角セル数と総ループクローズを確認。
  - [ ] 後段 `fill(density=10, remove_boundary=True)` がエラー無く走り、ライン数>0 を返すことを確認（スモーク）。
- [ ] ドキュメント
  - [ ] `docs/pipeline.md` に利用例（1節）を追記。
  - [ ] `architecture.md` にエフェクト一覧の差分を同期（該当コード参照を併記）。
- [ ] 公開スタブ再生成 `PYTHONPATH=src python -m tools.gen_g_stubs && git add src/api/__init__.pyi`
- [ ] 変更ファイルに限定したチェック: `ruff/black/isort/mypy/pytest`（ファイル単位）。

テスト計画（最小）
- Voronoi 正常系: 正方形領域 + 乱数サイト 4/9 点でセル数・クローズを確認（シード固定）。
- Delaunay 正常系: 三角セルの総面積が元領域面積±εに一致（clip 後）。
- フォールバック: `voronoi` → `delaunay` へ切替時も例外なし・クローズ保証。
- 非平面/空入力: 入力をそのまま返す（`coords/offsets` 同値）。

互換性/影響
- 既存パイプラインへの影響は限定的（新規関数）。
- 出力は常に閉ループで構成され、後段 `fill` の偶奇規則ベース塗りと相性が良い。
- 依存: Shapely は optional。未導入環境では Delaunay フォールバックで最低限の機能を提供。

オープン質問（要確認）
- デフォルト `mode` は `voronoi` で良いか。`delaunay` を既定にしたいか。
- 分割対象の単位: 1 領域（ポリゴン）単位で分割するか、`group_by_plane=True` で同一平面上の複数領域をまとめて分割するか（初版は前者）。
- 穴（holes）の扱い: 初版では非対応（外周のみ）で良いか。必要なら `points_in_input` で穴の外周を別入力とする運用にするか。
- 元外周の保持既定: `keep_outline=True` か `False` か。
- サイト上限/性能: 1 領域あたり `site_count<=500` 程度を上限にして良いか。

作業コマンド（編集ファイル優先の高速ループ）
- Lint: `ruff check --fix src/effects/partition.py`
- Format: `black src/effects/partition.py && isort src/effects/partition.py`
- Type: `mypy src/effects/partition.py`
- Test: `pytest -q tests/test_effect_partition.py::test_voronoi_square_closed`
- スタブ: `PYTHONPATH=src python -m tools.gen_g_stubs && git add src/api/__init__.pyi`

完了条件（DoD）
- 変更ファイルに対する `ruff/black/isort/mypy/pytest` が緑。
- `api/__init__.pyi` のスタブ差分ゼロ（再生成後）。
- 最小ドキュメントの追記と `architecture.md` の整合。

備考
- 量子化: `params_signature` 規約に合わせ、float は `__param_meta__['step']` を優先して丸め（未指定時は既定 1e-6）。
- 乱数は `seed` により再現性を担保。グリッド + `jitter` は 0..1 を bbox 比で解釈し、成分ごとに量子化 step を適用。

更新履歴
- 2025-10-07: 初版（計画・チェックリスト作成）。

