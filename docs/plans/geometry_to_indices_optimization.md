# 251106: _geometry_to_vertices_indices() 最適化計画（設計・チェックリスト）

目的
- 描画直前のインデックス配列生成（Python ループ + per-line np.arange）と毎フレームの IBO 転送コストを削減する。
- displace 等で座標のみが変化し offsets（トポロジ）が不変なフレームでは、インデックスを再生成・再アップロードしない。

現状（要約）
- 関数: `src/engine/render/renderer.py:_geometry_to_vertices_indices()`
  - 各ラインに対して Python ループで `np.arange(start, end)` を生成 → 連結 → 区切りに `primitive_restart_index` を挿入。
  - 計算量は O(N + M) だが、per-line ループと小さな ndarray の多量生成がオーバーヘッド（N=頂点総数, M=ライン本数）。
- IBO（Index Buffer）は毎フレーム orphan+write で全量アップロード。

方針（段階的）
- Phase 1: ベクトル化と「前回オフセット一致時の IBO 再利用（スキップ）」の追加。
- Phase 2: インデックスの LRU キャッシュ（オフセット署名キー）と HUD/計測。
- Phase 3: GPU 転送の最小化（変更検出/部分更新 or フレーム間で IBO 固定）。

設計詳細

1) ベクトル化アルゴリズム（Python ループ排除）
- 目標: Python for なしで indices を O(N+M) のベクトル演算だけで構築。
- アルゴリズム案（mask 挿入法）:
  - `N = len(coords)`, `M = len(offsets) - 1`
  - `indices = np.empty(N + M, dtype=np.uint32)`
  - `restart_pos = offsets[1:] + np.arange(M, dtype=np.int64)`
  - `mask = np.zeros(N + M, dtype=bool); mask[restart_pos] = True`
  - `indices[~mask] = np.arange(N, dtype=np.uint32)`
  - `indices[mask] = primitive_restart_index`
- 性質: メモリアクセスは連続、Pythonコールベースの小配列生成が消え、ブロードキャストのみ。

2) IBO 再利用（オフセット不変時のアップロードスキップ）
- 目標: displace 等で `offsets` が不変の場合、IBO を再生成せず過去の IBO をそのまま再利用。
- 実装方針:
  - LineRenderer 側に「前回 offsets の署名」を保持（`blake2b-128(view(offsets).tobytes())` など）。
  - 同一署名なら indices 生成をスキップし、`LineMesh.upload()` では VBO のみ更新、IBO 書き込みをスキップ。
  - 署名計算がボトルネック化しないよう、N が十分大きく IBO 生成/アップロードコストより安いことを確認（デバッグカウンタを用意）。

3) LRU キャッシュ（Phase 2）
- 目標: 同一 offsets 署名に対して indices ndarray を再利用（生成もスキップ）。
- 実装方針:
  - プロセス内 LRU（`OrderedDict`）。キー: `(len(offsets), blake2b(offsets_bytes))`。
  - `PXD_INDICES_CACHE_MAXSIZE`（既定64）/ `PXD_INDICES_CACHE_ENABLED` で制御。
  - HUD/カウンタ: `hits/misses/stores/evicts` を集計して可視化（任意）。

4) GPU 転送の最小化（Phase 3）
- 目標: IBO が不変の連続フレームで IB の `write()` を行わない。さらに、変化時のみ再転送（サイズや内容が同じならスキップ）。
- 実装方針:
  - LineMesh.upload() に「IBO 更新不要」経路を追加（indices=None などでスキップ）。
  - あるいは LineRenderer 側で `self.gpu.update_vertices_only(verts)` を用意（軽量 API）。

受け入れ基準（DoD）
- [x] ベクトル化: Python ループを除去し、`_geometry_to_vertices_indices()` が O(N+M) の vectorized 実装に置き換わる。
- [x] オフセット不変時に IBO 書き込みがスキップされる（LineRenderer が offsets 署名一致で VBO のみ更新）。
- [ ] 10K+ ライン、100K+ 頂点のケースで per-frame のインデックス生成/転送時間が有意に短縮。

環境変数/設定（案）
- `PXD_INDICES_CACHE_ENABLED=1|0`：LRU の有効化
- `PXD_INDICES_CACHE_MAXSIZE=64`：LRU 上限
- `PXD_INDICES_DEBUG=1`：ベンチ/デバッグログ出力

リスク/注意点
- 署名計算（ハッシュ）のコスト: 小型データでは逆転の可能性。サイズ閾値で「小さい場合はスキップ」などの安全弁を検討。
- IBO スキップの正当性: offsets 不変（配列内容同一）であることを厳密に確認する。参照同一性では不十分。
- メモリ: LRU に indices を保持するため、サイズ上限・頂点数上限で制御する。

段階的実装ステップ（最小→拡張）
- Phase 1（最小）
  - [x] ベクトル化実装に置換。
  - [x] 「前回 offsets 署名一致なら IBO スキップ」を LineRenderer に追加（デバッグカウンタは未導入）。
- Phase 2（拡張）
  - [x] indices LRU を導入（署名キー、上限、カウンタ）。
  - [x] HUD へカウンタ追加（LineRenderer 経由で IBO/Indices カウンタを表示）。
- Phase 3（最適化）
  - [x] `upload()` の vertices-only 経路追加。
  - [x] IBO 固定化の実験（offsets 署名一致で indices 再生成/IBO 書き込みをスキップ。環境変数: `PXD_IBO_FREEZE_ENABLED`, `PXD_IBO_DEBUG`。カウンタ: `get_ibo_stats()`）。

備考
- offsets 不変なケース（displace/mirror/rotate など形状の連結構造を変えない効果）では横展開効果が大きい。
- Python ループ排除だけでもライン数が多いケースで顕著な短縮が見込める。
