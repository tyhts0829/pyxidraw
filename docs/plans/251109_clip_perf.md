% `clip` の計算を軽くする改善計画（ユーザー側変更なし）

前提
- ユーザー側のスケッチは変更しない（API/使い方を変えない）。
- 現状の高負荷要因は、毎フレームの重い幾何計算（Shapely intersection/difference または XY 投影でのセグメント×辺交差探索）と、アウトライン参照の id 由来でキャッシュキーが安定しないため再利用が効かないこと。

目標
- 共平面・非共平面の双方で、アウトラインに関する前処理（投影・偶奇領域構築・辺索引）をフレーム間で再利用可能にし、対象側のクリップ計算コストを削減。
- Shapely 経路・投影フォールバック経路の双方で早期除外や空間分割を導入し、O(#segments × #mask_edges) の実効コストを下げる。
- HUD の「キャッシュ有効」表示とは別に、幾何計算再利用のカウンタを導入（デバッグ/計測の透明性）。

非目標
- ユーザー API の変更（引数追加や呼び方の変更）。
- 厳密 3D クリップやマルチスレッド化（初回は着手しない）。

設計方針（概要）
1) アウトライン前処理のコンテンツキャッシュ（高効果・低リスク）
   - 「id ではなく内容」に基づくキー（digest）で、以下を LRU キャッシュ:
     - 共平面系: XY 整列後の `region_obj`（Shapely）/偶奇リング集合。
     - 投影系: XY 投影リング（閉路済み）とリング/辺の AABB 群。
   - digest 生成: `coords/offsets` を量子化（`param_utils.signature_tuple` を流用）→ blake2 でハッシュ。
   - キャッシュサイズは環境変数で制御（例: `PXD_CLIP_MASK_CACHE_MAXSIZE`）。

2) 空間インデックスによる交差探索の削減（投影系）
   - 現状: 各対象線分につき「全リングの全辺」に当てる→ AABB 早期除外のみ。
   - 追加: 2D グリッド（均等セル）でリング辺をバケット化し、候補辺のみ交差判定。
     - セルサイズは自動（マスク bounds と辺数から決定）/環境変数で上書き可。
   - 実装負荷が低い割に実効 O(N log N) 近傍まで削減可能。

3) Shapely 経路の `region` 準備と再利用（共平面系）
   - `Polygon.symmetric_difference` で構築した `region_obj` を mask digest で LRU 再利用。
   - 可能なら `shapely.prepared.prep(region_obj)` を保持し contains/intersection の内部判定を高速化（バージョン 2 系の互換に留意）。

4) ターゲット側の早期除外（共通）
   - マスクの union AABB を前処理で保持し、明らかに外側のポリライン/セグメントは丸ごと/早期にスキップ。
   - 共平面系では `LineString.is_empty`/`bounds` を用いた cheap check を併用。

5) 署名とキャッシュ適用範囲（コンテンツベース）
   - パイプライン署名は維持（互換）。幾何再利用キャッシュは effect 内部の LRU として独立管理。
   - キー: (mask_digest, planar_mode, projection_mode) + 必要に応じ transform 情報（共平面の整列行列）
   - 対象側 Geometry はキャッシュキーに含めない（結果のフルキャッシュではなく「マスクの前処理」だけを再利用）。

6) デバッグ/計測
   - `clip` 内部に簡易カウンタを持ち、mask_cache_hits/misses、grid_bucket_hits 等を集計。
   - HUD/ログへ最小限の出力（環境変数で ON/OFF）。

実装ステップ（チェックリスト）

1) マスク digest と前処理 LRU（共通）
- [ ] `effects/clip.py` に小さな LRU（OrderedDict）を実装（maxsize は env または既定 64）。
- [ ] `Geometry` → (coords, offsets) 抽出→量子化→blake2 で `mask_digest` を生成。
- [ ] `mask_digest` に対し、以下を保存/再利用:
  - 共平面: region_obj（Shapely）、整列用 R/z、XY リング。
  - 投影: XY リング（閉路）、union AABB、辺グリッド（bucket）。

2) 投影系: 辺グリッドの導入
- [ ] マスク bounds からセルサイズを自動決定（辺数/面積ベース）。
- [ ] 各リングの各辺をセルに登録（水平/垂直の端の二重カウント防止のルールは半開区間で統一）。
- [ ] 対象セグメントごとに、跨るセルのみの辺集合を候補にして交差計算。

3) 共平面系: region の準備と再利用
- [ ] `Polygon` 群→`symmetric_difference` の結果を `mask_digest` で LRU 再利用。
- [ ] Shapely 2 系では `shapely.prepared` を試し、存在すれば prepared geometry を保持（try/except でフォールバック）。

4) 早期除外
- [ ] マスク union AABB を保持し、対象ポリラインが完全に外側ならスキップ。
- [ ] 投影系ではセグメント vs マスク union AABB でも早期除外をかける（現状リング AABB の前に 1 段）。

5) カウンタとトグル
- [ ] 内部計測カウンタを追加（ヒット/ミス/バケット照会数など）。
- [ ] 環境変数 `PXD_DEBUG_CLIP_STATS=1` で HUD/ログに集計を露出（なければ無出力）。

6) 安全性と互換確認
- [ ] 既存共平面経路の結果が変わらないことをスナップテスト/目視で確認（許容誤差内）。
- [ ] 非共平面の投影結果も従来の project_xy と一致。
- [ ] 失敗時は静かに LRU をバイパス（フェイルソフト）。

追加メモ
- LRU はプロセス内のみ有効。ワーカ分散時は各プロセスで独立に保持。
- digest 量子化の粒度は `PXD_PIPELINE_QUANT_STEP` に追従（未指定は 1e-6）。

