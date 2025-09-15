# メモリ急増の疑い箇所レポート（main.py 実行時）

更新日: 2025-09-14

本書は、`python main.py` 実行時に観測される「メモリ使用量の急増／枯渇（MemoryError らしき症状）」について、コード読解ベースで特に怪しい箇所と切り分け観点を整理したもの。修正提案は記載するが、現時点ではコード変更は行っていない。

---

## 観測症状（提供情報に基づく）

- `main.py` 実行直後の RSS はおよそ 200MB。
- 画面左上（overlay の MEM 表示）が毎秒 ≈1MB のペースで増加（少なくとも 10 秒間継続）。

この「ゆるやかな直線的な増加（≈1MB/s）」は、フレーム毎に小さな一時オブジェクトが生成され、Python/GL のアロケータ高水位（high-water mark）として RSS が徐々に積み上がる挙動と整合的。大量バックログが一気に積み上がるケース（無制限キューの滞留）よりは、描画アップロード経路の一時確保や断片化の影響が疑わしい。

### 直近の切り分け結果（反映）
- `_geometry_to_vertices_indices: astype(np.float32, copy=False)` に変更 → MEM 勾配は「ほぼ変化なし」。
- `workers=1`、受信上限緩和 → MEM 勾配は「ほぼ変化なし」。
- `LineMesh.upload(): write(tobytes) → write(memoryview)` 置換 → MEM 勾配は「ほぼ変化なし」。


## 結論サマリ（優先度順・症状反映版）

 - 最有力: 「ジオメトリのハッシュ計算（digest）」＋「GPUアップロード前後の一時確保」が合算で RSS を押し上げている可能性が高い。
   - digest 側: `Geometry.digest` と `Pipeline._geometry_hash` が毎フレーム `coords/offsets` 全体に対し `np.ascontiguousarray(...).tobytes()` を実行（CPU ヒープで大きな bytes を都度生成）。
   - アップロード側: `indices=np.empty(...)` の新規確保、および `write(...)` によるドライバ側更新（bytes→memoryview 置換の効果は小）。
   - 備考: `astype(np.float32, copy=False)` では効果薄 → dtype コピーは主因ではない。
- 次点: 無制限の結果キュー（`multiprocessing.Queue`）と受信上限（2 件/ tick）による滞留。ただし「メインプロセス RSS が毎秒 1MB 前後で線形増加」という観測とは一致度が下がる（滞留は主に OS パイプやワーカ側ブロックで顕在化しやすいため）。
- 次点: icosphere 等の高細分形状生成でのピーク肥大（瞬間的な増加）。症状の「毎秒 1MB 増」とはややパターンが異なるが、細分化ノブの操作次第でピークは跳ね上がる。
- 補足: `Pipeline` の無制限キャッシュは設計上のリスクだが、main.py はフレーム毎に新規 `Pipeline` を生成しワーカ側でのみ使用するため、メインプロセス RSS の直線的増加要因にはなりにくい。

---

## ホットスポット詳細

### 1) ジオメトリの digest/ハッシュ計算（最有力の一翼）

- 位置:
  - `src/engine/core/geometry.py::Geometry._compute_digest`（`h.update(c.tobytes())` 等）
  - `src/api/effects.py::_geometry_hash`（digest 例外時フォールバックでも `tobytes()` 実行）
- 仕組みと症状適合:
  - `Pipeline.__call__` はキャッシュ有無に関係なく毎回 `key = (_geometry_hash(g), self._pipeline_key)` を先に計算するため、1 フレームにつき「入力 Geometry 全体」をバイト列化してハッシュ化する経路が必ず走る。
  - `PXD_DISABLE_GEOMETRY_DIGEST=1` を設定しても、`_geometry_hash` のフォールバックで `tobytes()` が発生（効果なし）。
  - 頂点数が多いほど 1 フレームあたりの `bytes` 化コストが増え、観測された線形増とも整合。
- 追加所見:
  - digest は Pipeline の単層キャッシュにのみ使用されるが、`main.py` は毎フレーム新規 Pipeline を生成するため、キャッシュの恩恵が無い一方で「毎回ハッシュするコストだけ発生」している。

### 2) レンダ前変換と GPU へのアップロード（最有力の一翼）

- 位置:
  - 変換: `src/engine/render/renderer.py::_geometry_to_vertices_indices`
  - 転送: `src/engine/render/line_mesh.py::upload`（`orphan()`→`write(tobytes())`）
 - 仕組みと症状適合:
  - 毎フレーム、`indices = np.empty(...)` を新規確保し、続いて `vertices.tobytes()` / `indices.tobytes()` が大きめの一時 `bytes` を生成。
  - `write(tobytes) → write(memoryview)` の置換効果は小。アップロード側単独よりも、digest 側と合算した総コストが RSS 勾配を作っている可能性が高い。
  - これらは都度解放対象だが、Python のメモリアロケータや GL ドライバ側の内部バッファが「高水位」まで確保を維持しやすく、RSS が少しずつ上がる挙動を取りやすい。
  - 観測の ≈1MB/s 増分は「フレーム単位の小さな一時確保の積み重ね」に整合（60fps 想定で ≈16KB/フレームの純増相当）。
- 参考コード:
  - `renderer._geometry_to_vertices_indices()` 新規 `np.ndarray` を作成
  - `LineMesh.upload()` で `orphan()` → `write(tobytes())` により大きな一時 `bytes` を生成

#### 追加テスト候補（未適用・実装案のみ）
- digest の `tobytes()` を避け、`hashlib.update(memoryview(ndarray_uint8))` を用いる（Python bytes を生成しない）。
  - 例: `Geometry._compute_digest()` と `Pipeline._geometry_hash()` の双方で `tobytes()` を `memoryview` に置換。
  - 併せて「キャッシュ無効時はハッシュを計算しない」分岐を `Pipeline.__call__` に導入（`_cache_maxsize == 0` や `None` の扱いの見直し）。

### 2) 無制限の結果キュー × 受信側スループット制限（次点）

- 位置:
  - 送信側: `src/engine/pipeline/worker.py` → `result_q: mp.Queue()`（上限なし）
  - 受信側: `src/engine/pipeline/receiver.py` → `StreamReceiver(..., max_packets_per_tick=2)`
- 仕組み:
  - ワーカは描画結果（`RenderPacket(Geometry, frame_id)`）を無制限キューへ投入。
  - 受信側は 1 tick あたり最大 2 件しか取り出さないため、UI 負荷増・例外・描画停滞時に結果が溜まり続ける。
  - 例外で受信側が止まると加速度的にメモリが膨張。
- 症状との適合性メモ:
  - メインプロセス RSS の「緩やかな線形増」よりは、滞留が臨界に達した段階での急増やワーカ側ブロックが出やすい。
  - ただし、受信側が恒常的に遅く（2 件/ tick が不足）わずかに追いつかない場合、メモリの純増が観測される可能性は残るため、要切り分け。

### 2) icosphere 高細分化での一時オブジェクト膨張（有力）

- 位置: `src/shapes/sphere.py::_sphere_icosphere`
- 仕組み:
  - 細分化レベルに応じて三角形分割が指数的に増え、`seen_edges`（Python の `set`）へ大量のタプルキーを保持。
  - 生成完了後は落ち着くが、ピークメモリが非常に大きくなる。
- 再現条件:
  - `main.py` の `G.sphere(subdivisions=c(1, 0.2), sphere_type=c(2, 0.0))` で、CC#2 を 0.6–0.8 付近（icosphere）、CC#1 を高値（細分化 ↑）にすると顕著。

（本項は 1) に統合）

### 4) Perlin 変位の一時配列確保（フレーム毎）

- 位置: `src/effects/displace.py`
- 仕組み:
  - `offset_coords = coords + t_offset`、`noise_offset`、`coords + noise_offset * intensity` など、`N×3` 配列の新規確保が重なる。
  - CC#6（`amplitude_mm` の元）や CC#7（`spatial_freq`）が高いと頂点数増や変位計算が重く、他の一時配列とピークが重なりやすい。

### 5) 設計上のリスク: `Pipeline` 無制限キャッシュ

- 位置: `src/api/effects.py`（`PipelineBuilder.cache(maxsize=...)` 未指定＝`None`＝無制限）
- 備考:
  - `pipeline_key` にステップのパラメータが入るため、時間 `t_sec` が含まれるとフレーム毎にキーが変わりキャッシュが増殖しうる。
  - ただし main.py は毎フレーム `build()` で新しい `Pipeline` を作り、1 回適用して破棄するため、今回の直接原因になりにくい。

### 6) ベースメモリ（上限を圧迫する要因）

- `workers=4` による `spawn` された Python プロセス（NumPy 読込済）で常駐メモリが大きめ（環境次第で数百 MB 規模）。
- ModernGL/pyglet 初期化および MSAA・VBO/IBO 再確保で GPU/CPU メモリを追加消費。

---

## 切り分け用チェックリスト（コード変更なし）

- レンダ経路の寄与確認（優先）
  - [あてはまる] FPS を下げる（例: `configs/default.yaml` の `canvas_controller.fps: 10`）→ 増加勾配が ≈1/6 に低下すれば、フレーム毎のアップロード経路が主因。
  - [あてはまる] `render_scale` や CC を固定したまま、数十秒放置で増加が継続するか確認（形状が一定でも増えるならアップロード経路濃厚）。
- 結果キュー滞留の切り分け
  - [あまり変わらない] `workers=1` で起動（生産=消費を近づける）。増加がほぼ止まる/減るなら、滞留寄与あり。
  - [あまり変わらない] `StreamReceiver.max_packets_per_tick` を一時的に大きく（例: 30 以上）して確認（要実装）。改善すれば滞留寄与あり。
- icosphere・一時ピークの切り分け
  - [ ] CC#2 を 0.6–0.8（icosphere）＋ CC#1（細分化）高値でピークの跳ねを観測（直線増とは別種のスパイクが出る）。
- キャッシュ/ハッシュの影響除外（注意）
  - [ ] `PXD_PIPELINE_CACHE_MAXSIZE=0` は「キャッシュ保存」を止めるだけで、現実装では「ハッシュ計算をスキップしない」点に注意（差は小さい見込み）。
  - [ ] 頂点数を 1/2, 1/4 に下げた時の勾配比を確認（digest/アップロードが主因ならほぼ比例）。
- 低コストのランタイム計測
  - [ ] `PYTHONTRACEMALLOC=1` で起動し、任意のシグナルや REPL から `tracemalloc.get_traced_memory()` を覗いて「Python ヒープ」寄与か判定（ドライバ側メモリは見えない）。

> メモ: HUD は `src/engine/monitor/sampler.py`（`psutil`）で `MEM` を出力しており、簡易観測に使える。

---

## 影響と見積もり（概算）

- レンダアップロード経路の一時配列（`astype`/`empty`/`tobytes`）はフレーム毎に数十〜数百 KB 程度になりうる。アロケータの高水位保持や断片化の影響で RSS が ≈1MB/s 規模で増えることはあり得る。
- `result_q` 滞留は条件次第で短時間に数百 MB へ膨張し得るが、今回の「緩やかな線形増」とはパターンが異なる。

---

## 修正アイデア（実装は未着手／要確認）

- 結果キューの制御（推奨）
  - [ ] `result_q` に `maxsize` を設定し、溢れたら古いフレームを捨てる／最新のみ残すポリシーを導入。
  - [ ] `StreamReceiver` の 1 tick 処理上限（`max_packets_per_tick`）を引き上げ、滞留を解消。
- 形状生成のピーク抑制
  - [ ] icosphere の重複辺検出で Python タプル／`set` を減らす（NumPy ベース／整列＋差分等）。
  - [ ] 高細分時の上限ガード（MIDI 入力を 0–1 → 0–3 段階にクリップ等）。
- レンダ前変換のコピー削減
  - [ ] `coords` の dtype が既に `float32` なら `astype(copy=False)` の徹底。
  - [ ] インデックス生成の方法見直し（大配列の都度確保を避ける）。
- キャッシュ
  - [ ] `PipelineBuilder.cache(maxsize=...)` の既定を小さめに設定、または環境変数 `PXD_PIPELINE_CACHE_MAXSIZE` を推奨値に。
- 並列度・JIT
  - [ ] `workers` 既定の見直し（環境に応じ 2 などへ）。

---

## 最小改変テスト（実施ログ）
- [実施済/効果小] `_geometry_to_vertices_indices: astype(np.float32, copy=False)` に変更 → MEM 勾配の改善は「ほぼなし」。
- [実施済/効果小] `LineMesh.upload(): tobytes() → memoryview()` 置換 → MEM 勾配は「ほぼ変化なし」。
- [未実施] `Geometry._compute_digest()` と `Pipeline._geometry_hash()` の `tobytes()` を `memoryview` に置換し、勾配変化を観測。

---

## 追加で見直した怪しい箇所（全体サーベイ）
- 毎フレーム Pipeline を新規生成（`main.py` の `draw()` 内）
  - 単層キャッシュの恩恵がないのに、毎回ハッシュ計算コスト（digest）だけが発生。
  - 対策案: Pipeline を外側で一度だけ構築し、`draw()` 内では再利用（時間パラメータはエフェクト引数に渡して処理）。
- `Geometry` 変換チェーンによる配列複製
  - `scale/rotate/translate` は毎回 `coords.copy()`/新規計算を行うため、フレームごとの一時配列が積み上がる。これは仕様上必要だが、digest/アップロードと合算してピークを作りやすい。
- icosphere 高細分（瞬間スパイク要因）
  - 今回の「線形増」とはパターンが異なるが、頂点数が多いほど digest/アップロードのコストが比例増となり、基線勾配を押し上げる。

---

## 参照箇所（抜粋）

- `src/engine/pipeline/worker.py`（`result_q` 無制限）
- `src/engine/pipeline/receiver.py`（`max_packets_per_tick=2`）
- `src/shapes/sphere.py`（`_sphere_icosphere`）
- `src/engine/render/renderer.py`（`_geometry_to_vertices_indices`）
- `src/effects/displace.py`（一時配列）
- `src/api/effects.py`（無制限キャッシュ）

---

このレポートに基づき、上記「修正アイデア」をタスク化して順に対応可能です。実装着手の前に、この内容で進めてよいかご確認ください。
