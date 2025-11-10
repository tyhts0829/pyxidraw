% `clip` 追加高速化（優先度A）実装計画

目的
- ユーザー側のスケッチを一切変更せず、`clip` の CPU 負荷をさらに削減する。
- 既存の内容ベース LRU と早期除外を活かしつつ、以下の3点を優先実装する。
  1) 共平面経路: Shapely Prepared Geometry による高速な空間述語・早期分岐
  2) 非共平面（投影）経路: 交差探索/偶奇判定の Numba JIT 化
  3) 共平面経路: ライン集合を MultiLineString 化して一括 `intersection/difference`（バルク化）

非目標
- API 変更や draw コードの改変。
- 複数スレッド/プロセス化。

背景（現状の最適化）
- 内容ダイジェスト LRU（マスク）により、投影系の XY リング/グリッド、共平面系の region（偶奇合成）を再利用。
- マスク union AABB/region.bounds による早期除外を導入済み。
- 投影系はグリッドで候補辺を絞ってから交差探索（まだ Python ループ中心）。

優先度Aの実装項目

1) 共平面: Prepared Geometry（高速な述語判定）
- 目的: `region_obj` 生成後に `prepared_region = shapely.prepared.prep(region_obj)` をキャッシュし、
  ラインごとの cheap 分岐を高速化。
- 分岐ポリシー（ライン `ls` に対し）:
  - `prepared_region.disjoint(ls)` なら outside 全採用（draw_outside=True）/inside 目的ならスキップ。
  - `prepared_region.contains(ls)` かつ draw_inside=True なら inside 全採用（difference 不要）。
  - 上記以外のみ `intersection/difference` を実行。
- 実装
  - region 作成箇所（キャッシュミス時）で `prepared_region` を生成し、LRU に {region, prepared, bounds} を保存。
  - ライン処理部で prepared を参照し、早期分岐→一括処理へ接続。
  - フェイルソフト: shapely.prepared が無い/失敗は無視して従来経路。

2) 投影: 交差・偶奇の Numba JIT 化
- 目的: Python ループのオーバーヘッドを減らし、候補辺交差 t 収集と偶奇判定を JIT 化。
- 変更点
  - マスク前処理の出力に「結合リング配列 + offsets」も追加（現在は list[np.ndarray]）。
  - `@njit(cache=True)` 関数を追加:
    - `intersections_with_candidates(A2,B2, rings_coords, rings_offsets, cand_pairs)` → t 値配列
    - `evenodd_membership(points, rings_coords, rings_offsets)` → inside/outside (XOR) 配列
  - `cand_pairs` は (ri, ei) の int32 Nx2。Python 側で candidates を配列化して渡す。
  - 既存のグリッドはそのまま。候補選別は Python で、交差/偶奇が Numba で高速化。
  - フェイルソフト: numba 例外時は従来の Python 版にフォールバック。

3) 共平面: バルク `intersection`/`difference`（MultiLineString）
- 目的: ラインごとの Shapely 呼び出し回数を減らす。
- 変更点
  - 対象ポリライン群を一旦 `MultiLineString` にまとめ、1 回の `intersection(region)`/`difference(region)` を実行。
  - 返り値（Multi/Mixed）から LineString 群を抽出して配列化。
  - 事前に prepared/bounds で外側（または内側）に落ちる群は一括でスキップ/全採用し、MLS の入力を減らす。
  - フェイルソフト: shapely 例外時は per-line 経路へフォールバック。

補助・環境変数（任意）
- `PXD_CLIP_NUMBA=0/1`（既定 1）: Numba JIT を無効/有効。
- `PXD_CLIP_BULK=0/1`（既定 1）: MLS による一括 `intersection/difference` を無効/有効。

受け入れ基準（DoD）
- 同一入力で描画結果が従来と一致（許容誤差内）。
- ZX/XY 回転/拡大縮小のケースで退行が無い。
- 代表ケース（傾いた大リング×高分解能グリッド）で CPU 時間の低減が確認できる。
- 例外時でもフェイルソフトで従来経路へ戻る。

実装ステップ（チェックリスト）

1) Prepared Geometry 導入
- [x] region 作成時に `prepared_region` を生成し LRU に格納。
- [x] ライン処理部で prepared を使い、disjoint/contains で早期分岐。
- [x] prepared 不在時/例外時は従来分岐。

2) 投影 JIT 化
- [x] マスク前処理で結合配列 + offsets を構築（既存 list も維持）。
- [x] `intersections_with_candidates`（njit）を実装し、候補辺の t 収集を JIT 化。
- [x] `evenodd_membership`（njit, point）を実装し、中点偶奇判定を高速化（スカラー）。
- [x] 例外時は従来の Python 版にフォールバック。

3) 共平面バルク処理
- [x] ライン群を `MultiLineString` にまとめて一括 `intersection/difference`。
- [x] prepared/bounds による外側/内側の全採用スキップと組み合わせ。
- [x] 例外時は従来 per-line 経路へフォールバック。

4) 軽量計測（任意）
- [ ] `PXD_DEBUG_CLIP_STATS=1` 時に、prepared/bulk/numba の利用回数・スキップ件数・候補辺合計などをログ出力。

スケジュール見積
- 1) Prepared: 0.5 日
- 2) Numba JIT（投影）: 0.5–1.0 日
- 3) バルク処理: 0.5 日
