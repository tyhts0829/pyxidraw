# メモリ急増の疑い箇所レポート（main.py 実行時）

更新日: 2025-09-14

本書は、`python main.py` 実行時に観測される「メモリ使用量の急増／枯渇（MemoryError らしき症状）」について、コード読解ベースで特に怪しい箇所と切り分け観点を整理したもの。修正提案は記載するが、現時点ではコード変更は行っていない。

---

## 観測症状（提供情報に基づく）
- `main.py` 実行直後の RSS はおよそ 200MB。
- 画面左上（overlay の MEM 表示）が毎秒 ≈1MB のペースで増加（少なくとも 10 秒間継続）。

この「ゆるやかな直線的な増加（≈1MB/s）」は、フレーム毎に小さな一時オブジェクトが生成され、Python/GL のアロケータ高水位（high-water mark）として RSS が徐々に積み上がる挙動と整合的。大量バックログが一気に積み上がるケース（無制限キューの滞留）よりは、描画アップロード経路の一時確保や断片化の影響が疑わしい。

## 結論サマリ（優先度順・症状反映版）
- 最有力: レンダリングアップロード経路（`_geometry_to_vertices_indices` → `LineMesh.upload`）での毎フレーム一時確保（`np.astype`/`np.empty`/`.tobytes()`/moderngl バッファ更新）に伴うメモリアロケータの高水位化・断片化。
- 次点: 無制限の結果キュー（`multiprocessing.Queue`）と受信上限（2件/ tick）による滞留。ただし「メインプロセス RSS が毎秒 1MB 前後で線形増加」という観測とは一致度が下がる（滞留は主に OS パイプやワーカ側ブロックで顕在化しやすいため）。
- 次点: icosphere 等の高細分形状生成でのピーク肥大（瞬間的な増加）。症状の「毎秒 1MB 増」とはややパターンが異なるが、細分化ノブの操作次第でピークは跳ね上がる。
- 補足: `Pipeline` の無制限キャッシュは設計上のリスクだが、main.py はフレーム毎に新規 `Pipeline` を生成しワーカ側でのみ使用するため、メインプロセス RSS の直線的増加要因にはなりにくい。

---

## ホットスポット詳細

### 1) レンダ前変換と GPU へのアップロード（最有力）
- 位置:
  - 変換: `src/engine/render/renderer.py::_geometry_to_vertices_indices`
  - 転送: `src/engine/render/line_mesh.py::upload`（`orphan()`→`write(tobytes())`）
- 仕組みと症状適合:
  - 毎フレーム、`coords.astype(np.float32)` と `indices = np.empty(...)` を新規確保し、続いて `vertices.tobytes()` / `indices.tobytes()` が大きめの一時 `bytes` を生成する。
  - これらは都度解放対象だが、Python のメモリアロケータや GL ドライバ側の内部バッファが「高水位」まで確保を維持しやすく、RSS が少しずつ上がる挙動を取りやすい。
  - 観測の ≈1MB/s 増分は「フレーム単位の小さな一時確保の積み重ね」に整合（60fps 想定で ≈16KB/フレームの純増相当）。
- 参考コード:
  - `renderer._geometry_to_vertices_indices()` 新規 `np.ndarray` を作成
  - `LineMesh.upload()` で `orphan()` → `write(tobytes())` により大きな一時 `bytes` を生成

### 2) 無制限の結果キュー × 受信側スループット制限（次点）
- 位置:
  - 送信側: `src/engine/pipeline/worker.py` → `result_q: mp.Queue()`（上限なし）
  - 受信側: `src/engine/pipeline/receiver.py` → `StreamReceiver(..., max_packets_per_tick=2)`
- 仕組み:
  - ワーカは描画結果（`RenderPacket(Geometry, frame_id)`）を無制限キューへ投入。
  - 受信側は1 tick あたり最大2件しか取り出さないため、UI負荷増・例外・描画停滞時に結果が溜まり続ける。
  - 例外で受信側が止まると加速度的にメモリが膨張。
- 症状との適合性メモ:
  - メインプロセス RSS の「緩やかな線形増」よりは、滞留が臨界に達した段階での急増やワーカ側ブロックが出やすい。
  - ただし、受信側が恒常的に遅く（2件/ tick が不足）わずかに追いつかない場合、メモリの純増が観測される可能性は残るため、要切り分け。

### 2) icosphere 高細分化での一時オブジェクト膨張（有力）
- 位置: `src/shapes/sphere.py::_sphere_icosphere`
- 仕組み:
  - 細分化レベルに応じて三角形分割が指数的に増え、`seen_edges`（Python の `set`）へ大量のタプルキーを保持。
  - 生成完了後は落ち着くが、ピークメモリが非常に大きくなる。
- 再現条件:
  - `main.py` の `G.sphere(subdivisions=c(1, 0.2), sphere_type=c(2, 0.0))` で、CC#2 を 0.6–0.8 付近（icosphere）、CC#1 を高値（細分化↑）にすると顕著。

（本項は 1) に統合）

### 4) Perlin 変位の一時配列確保（フレーム毎）
- 位置: `src/effects/displace.py`
- 仕組み:
  - `offset_coords = coords + t_offset`、`noise_offset`、`coords + noise_offset * intensity` など、`N×3` 配列の新規確保が重なる。
  - CC#6（`amplitude_mm` の元）や CC#7（`spatial_freq`）が高いと頂点数増や変位計算が重く、他の一時配列とピークが重なりやすい。

### 5) 設計上のリスク: `Pipeline` 無制限キャッシュ
- 位置: `src/api/pipeline.py`（`PipelineBuilder.cache(maxsize=...)` 未指定＝`None`＝無制限）
- 備考:
  - `pipeline_key` にステップのパラメータが入るため、時間 `t_sec` が含まれるとフレーム毎にキーが変わりキャッシュが増殖しうる。
  - ただし main.py は毎フレーム `build()` で新しい `Pipeline` を作り、1回適用して破棄するため、今回の直接原因になりにくい。

### 6) ベースメモリ（上限を圧迫する要因）
- `workers=4` による `spawn` された Python プロセス（NumPy読込済）で常駐メモリが大きめ（環境次第で数百MB規模）。
- ModernGL/pyglet 初期化および MSAA・VBO/IBO 再確保でGPU/CPUメモリを追加消費。

---

## 切り分け用チェックリスト（コード変更なし）
- レンダ経路の寄与確認（優先）
  - [ ] FPS を下げる（例: `configs/default.yaml` の `canvas_controller.fps: 10`）→ 増加勾配が ≈1/6 に低下すれば、フレーム毎のアップロード経路が主因。
  - [ ] `render_scale` や CC を固定したまま、数十秒放置で増加が継続するか確認（形状が一定でも増えるならアップロード経路濃厚）。
- 結果キュー滞留の切り分け
  - [ ] `workers=1` で起動（生産=消費を近づける）。増加がほぼ止まる/減るなら、滞留寄与あり。
  - [ ] `StreamReceiver.max_packets_per_tick` を一時的に大きく（例: 30 以上）して確認（要実装）。改善すれば滞留寄与あり。
- icosphere・一時ピークの切り分け
  - [ ] CC#2 を 0.6–0.8（icosphere）＋ CC#1（細分化）高値でピークの跳ねを観測（直線増とは別種のスパイクが出る）。
- キャッシュの影響除外
  - [ ] `PXD_PIPELINE_CACHE_MAXSIZE=0` で起動し差を確認（今回の症状では差が小さい見込み）。
- 低コストのランタイム計測
  - [ ] `PYTHONTRACEMALLOC=1` で起動し、任意のシグナルやREPLから `tracemalloc.get_traced_memory()` を覗いて「Python ヒープ」寄与か判定（ドライバ側メモリは見えない）。

> メモ: HUD は `src/engine/monitor/sampler.py`（`psutil`）で `MEM` を出力しており、簡易観測に使える。

---

## 影響と見積もり（概算）
- レンダアップロード経路の一時配列（`astype`/`empty`/`tobytes`）はフレーム毎に数十〜数百 KB 程度になりうる。アロケータの高水位保持や断片化の影響で RSS が ≈1MB/s 規模で増えることはあり得る。
- `result_q` 滞留は条件次第で短時間に数百MBへ膨張し得るが、今回の「緩やかな線形増」とはパターンが異なる。

---

## 修正アイデア（実装は未着手／要確認）
- 結果キューの制御（推奨）
  - [ ] `result_q` に `maxsize` を設定し、溢れたら古いフレームを捨てる／最新のみ残すポリシーを導入。
  - [ ] `StreamReceiver` の1 tick処理上限（`max_packets_per_tick`）を引き上げ、滞留を解消。
- 形状生成のピーク抑制
  - [ ] icosphere の重複辺検出で Python タプル／`set` を減らす（NumPyベース／整列＋差分等）。
  - [ ] 高細分時の上限ガード（MIDI入力を 0–1 → 0–3 段階にクリップ等）。
- レンダ前変換のコピー削減
  - [ ] `coords` の dtype が既に `float32` なら `astype(copy=False)` の徹底。
  - [ ] インデックス生成の方法見直し（大配列の都度確保を避ける）。
- キャッシュ
  - [ ] `PipelineBuilder.cache(maxsize=...)` の既定を小さめに設定、または環境変数 `PXD_PIPELINE_CACHE_MAXSIZE` を推奨値に。
- 並列度・JIT
  - [ ] `workers` 既定の見直し（環境に応じ 2 などへ）。

---

## 参照箇所（抜粋）
- `src/engine/pipeline/worker.py`（`result_q` 無制限）
- `src/engine/pipeline/receiver.py`（`max_packets_per_tick=2`）
- `src/shapes/sphere.py`（`_sphere_icosphere`）
- `src/engine/render/renderer.py`（`_geometry_to_vertices_indices`）
- `src/effects/displace.py`（一時配列）
- `src/api/pipeline.py`（無制限キャッシュ）

---

このレポートに基づき、上記「修正アイデア」をタスク化して順に対応可能です。実装着手の前に、この内容で進めてよいかご確認ください。
