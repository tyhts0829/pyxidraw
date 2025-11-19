# draw() が LazyGeometry のリストを返す場合に「1つ目の shape/pipeline しか効かない」問題 調査メモと改善計画

目的: `sketch/251118.py` のように `draw()` が `LazyGeometry` を複数要素持つリストで返すケースで、「1つ目の shape/pipeline しか affine/fill が効いていないように見える」現象について、キャッシュ/グローバルオブジェクト周りを中心に現状実装を整理し、今後の改善方針をチェックリスト形式でまとめる。

## 現象の整理（想定シナリオ）

- スケッチ: `sketch/251118.py`（代表例）
  ```python
  def draw(t: float):
      N = int(cc[1] * 10) + 1
      geos = []
      for i in range(N):
          g = (
              G.polygon()
              .scale(4 * (i + 1), 4 * (i + 1), 4 * (i + 1))
              .translate(A5[0] / 2, A5[1] / 2, 0)
          )
          p = E.label(uid="polygons").affine().fill()
          geos.append(p(g))
      return geos
  ```
- ユーザー視点の症状（レポートからの推測を含む）
  - `draw()` が `LazyGeometry` のリストを返すとき、「先頭の shape/pipeline 以外では affine/fill の効果が無い（または GUI/CC から制御できない）」ように見える。
  - 特に `cc[1]` に応じて `N` を増やしたり、Parameter GUI から affine/fill のパラメータを触ったときに、「1つ目だけ期待どおり動く」印象になる。

## 関係する実装の整理

### 1. draw() の戻り値とレイヤー正規化

- `run_sketch()` → `WorkerPool` → `_execute_draw_to_packet()`（`src/engine/runtime/worker.py`）の流れ。
- `_execute_draw_to_packet()` 内で `draw_callback(t)` の戻り値を `_normalize_to_layers()` に渡す。
- `_normalize_to_layers()` の仕様:
  - `result` が `Sequence` かつ `Geometry`/`LazyGeometry` そのものではない場合、**常に「レイヤー列」として扱う**。
  - 各要素が `LazyGeometry` の場合:
    - plan から `__effect_kind__ == "style"` のエフェクト（`style`）だけを取り除き、残りのエフェクト（`affine`/`fill` など）はそのまま保持した `LazyGeometry` を新しく作って `StyledLayer.geometry` に入れる。
  - Renderer 側（`LineRenderer.tick()/draw()`）では、`StyledLayer.geometry` が `LazyGeometry` なら `realize()` を呼んだうえで GPU にアップロードする。
- ここでキャッシュ/シングルトンに依存する要素はなく、「リストで返したから 2 つ目以降が無視される」というロジックは存在しない。

### 2. Pipeline/E/G とグローバルキャッシュ

- `api.E` / `api.G` はモジュールレベルのシングルトンだが、**実際の Pipeline/Shape オブジェクトは毎回新規に生成される**。
  - `E.label(...).affine().fill()` → `PipelineBuilder` インスタンス → `.build()` → `Pipeline`。
  - `G.polygon()` → `LazyGeometry(base_kind="shape", base_payload=(shape_impl, params))`。
- `src/api/effects.py` の Pipeline キャッシュ:
  - `_GLOBAL_PIPELINES`: 生成された `Pipeline` の WeakSet（主にメトリクス用）。
  - `_GLOBAL_COMPILED`: key=`(steps_tuple, cache_maxsize)` に対して `Pipeline` を LRU で共有。
  - `steps_tuple` は `(effect_name, params_signature)` の列で、**pipeline_uid やレイヤー位置は含まれない**。
- `src/engine/core/lazy_geometry.py` のジオメトリキャッシュ:
  - `_SHAPE_CACHE`: `(shape_impl, params_signature)` → `Geometry`。
  - `_PREFIX_CACHE`: `(base_key, prefix_of_effect_sigs)` → 中間 `Geometry`。
  - いずれも「同じ shape/同じ effect パラメータの組み合わせは 1 度だけ計算し、複数の `LazyGeometry` から共有する」ためのもので、**リスト内の要素順や「1番目かどうか」には依存しない**。

### 3. Parameter GUI / Runtime 連携と pipeline UID

- Parameter GUI の基盤:
  - `ParameterRuntime`（親プロセス、GUI 用）: `user_draw` を 1 回トレースし、`ParameterStore` に `ParameterDescriptor` を登録。
  - `SnapshotRuntime`（ワーカープロセス側）: `extract_overrides()` から渡された override を適用しつつ、毎フレームの `user_draw` で同じインターフェース（`before_shape_call/before_effect_call`）を提供。
  - `pipeline_uid` は `ParameterRuntime/ SnapshotRuntime` が `next_pipeline_uid()` で `p0, p1, ...` を払い出し、`PipelineBuilder` が `self._uid` として保持する。
  - `ParameterContext.descriptor_prefix` は `effect@{pipeline_uid}.{effect_name}#{step_index}` となり、これが Parameter GUI 上の一意キーになる。
- `sketch/251118.py` のように `for i in range(N): E.label(...).affine().fill()` するケースでの挙動（実験で確認したもの）:
  - `cc[1]=0.5` 相当で `N=6` にした状態で `ParameterRuntime` 経由で `draw(0.0)` を呼ぶと、`p0..p5` それぞれに対して affine/fill の Descriptor が登録される（`effect@p0.fill#1.density` ～ `effect@p5.fill#1.density` など）。
  - その後 `SnapshotRuntime` + override スナップショットを用いて `draw()` を呼び直すと、
    - 各レイヤーの `LazyGeometry.plan` には `['_fx_scale', '_fx_translate', 'affine', 'fill']` が含まれ、
    - `fill` の `density` パラメータは `p0..p5` それぞれ異なる値に解決されていることを確認済み。
  - つまり **ParameterRuntime/SnapshotRuntime と caches/pipeline シングルトンの組み合わせ自体は、リスト返しでも正しく per-pipeline にパラメータを適用できる**。

## 原因候補の整理

調査の結果、「Sequence を返すこと」「G/E がシングルトンであること」「Pipeline のグローバルキャッシュ」が直接の原因である証拠は得られなかった。一方で、以下の設計上の制約がユーザー体験として「1つ目しか動いていない」ように見える可能性がある。

### 仮説 A: Parameter GUI の初回トレースと `N` の依存関係

- `ParameterManager.initialize()` は **起動時に 1 回だけ** `user_draw(0.0)` を呼び、その時点の `cc`/既定値に基づいて Shape/Effect の Descriptor を登録する。
- `sketch/251118.py` の `N` は `N = int(cc[1] * 10) + 1` なので、
  - 起動直後の `cc[1]` によって「GUI が知っている pipeline の数」が決まってしまう。
  - 例えば初期 `cc[1]=0.0` なら `N_init=1` であり、Parameter GUI/ParameterStore には `p0` の affine/fill だけが登録される。
- その後、実行中に CC や GUI の操作で `cc[1]` を変えて `N` を増やしても、
  - `SnapshotRuntime` が参照する override は「初回トレース時に存在した Descriptor 分だけ」になる。
  - つまり `p0` 以外の pipeline（`p1..p{N-1}`）には、GUI/CC 由来の override がなく、**常に実装のデフォルト値で動作**する。
  - ユーザーが「affine や fill の GUI スライダーを動かしても後続レイヤーが変化しない」場合、これを「1つ目しか効いていない」と感じる可能性がある。

### 仮説 B: GUI 側での pipeline 表示の集約による混乱

- Descriptor 上は `pipeline_uid` ごとに別 ID だが、GUI のカテゴリ表示は `category`（ここではラベル `"polygons"`）単位でまとめている。
- そのため、「見た目は 1 つの 'polygons' カテゴリだが、実際には pipeline ごとに別 ID がぶら下がっている」状態になる。
- 初回トレース時の `N` が 1 だった場合、そのカテゴリには `p0` 由来のパラメータだけが存在し、その後に増えた pipeline は GUI 上の操作対象にならない。
- これもユーザー視点では「リストで返して N を増やしても、追加分には affine/fill の GUI が効いていない」ように見える。

### 仮説 C: 「効いていない」の実感とデフォルト値の組み合わせ

- `affine` の既定値は「auto_center=True / scale=(0.5,0.5,0.5) / angles_rad=(0,0,0) / delta=(0,0,0)」。
- `fill` の既定値は「density=35, angle_rad=π/4, angle_sets=1, remove_boundary=False」。
- パイプライン/GUI の起動タイミングやデフォルトの CC 値によっては、
  - 最初のレイヤーだけが GUI override（あるいは CC バインド）を受けてパラメータが大きく変化し、
  - 後続レイヤーは常に「既定値の affine/fill」で描画される。
- その結果、「最初の 1 つだけ派手に動いて見える → 他は '効いていない' と感じる」状況が発生しうる。

## 改善方針（設計レベル）

ここでは「コードをまだ変更しない」前提で、今後行うべき改善アクションをチェックリストに分解する。

### 1. 再現性の高いテストベースの整備

- [ ] `tests/runtime` に、`draw()` が `Sequence[LazyGeometry]` を返すケースを明示的にテストするファイルを追加する。
  - [ ] `G.polygon()` + `E.label().affine().fill()` で複数レイヤーを生成し、`_normalize_to_layers()` → `LazyGeometry.realize()` の結果が全レイヤーで affine/fill を含むことを確認する。
  - [ ] `ParameterRuntime` + `SnapshotRuntime` + `extract_overrides()` を使って各 pipeline に別々の `fill.density` を適用し、plan 内の `density` が per-pipeline で異なることを確認する（スクリプトで行った検証をテスト化）。

### 2. Parameter GUI と動的 pipeline 数の整合性改善

- [ ] `ParameterManager.initialize()` が「初回の `user_draw(0)` に依存して Descriptor を固定している」ことを明文化する（ドキュメント/コメント）。
- [ ] `cc` や `t` に依存して pipeline/shape の個数が変わるスケッチ（今回の 251118 のようなもの）に対して、以下のいずれかの方針を検討する:
  - [ ] 方針 A-1: 「Parameter GUI がサポートするのは '構造（shape/effect の数）が固定のスケッチ' に限る」と仕様として割り切り、ガイドラインに追記する。
  - [ ] 方針 A-2: `initialize()` のトレース時に、「代表的な CC 状態（例: cc[1]=0.0 と cc[1]=1.0）」で複数回 `user_draw` を呼び、最大想定 `N` までの pipeline を Descriptor に起こす。
  - [ ] 方針 A-3: 実行中に「未知の pipeline_uid に対する before_effect_call」があった場合、ParameterRuntime/SnapshotRuntime 側で **その場で Descriptor を追加登録** できるようにする（GUI への増分通知を含む）。

### 3. GUI 側の pipeline 表示と識別性の改善

- [ ] `ParameterDescriptor` の `category` と `pipeline_uid` の扱いを整理し、「同じラベルの pipeline が複数ある場合でも、GUI 上で 'p0/p1/p2...' を識別できる UI 表現」を検討する。
  - 例: `polygons (p0)`, `polygons (p1)` のようなサブラベル追加。
- [ ] `ParameterWindowController` / `dpg_window_content.py` のカテゴリグルーピングロジックに、「pipeline_uid をキーとしたサブグループ化」を導入するかどうか設計する。

### 4. キャッシュとシングルトンの安全性検証

- [ ] `_GLOBAL_COMPILED`（Pipeline キャッシュ）について、「steps_tuple が同一でも pipeline_uid が異なるケース」で問題が起きないことをテストで明示的に検証する。
  - [ ] 同一 steps（affine/fill 同じ設定）の Pipeline を複数回生成・適用し、全レイヤーで `LazyGeometry.plan` に同じ effect 列が入ることを確認する（現在の挙動の再確認）。
- [ ] `_SHAPE_CACHE` / `_PREFIX_CACHE` について、「同一 shape/params + 同一 effect params を複数レイヤーで共有する」ことが仕様上問題ないことを整理し、必要なら architecture.md に追記する。
  - [ ] 一時的にキャッシュを無効化するテストモード（環境変数）を導入し、キャッシュ有無で Sequence 戻りの挙動が変わらないことを確認する計画を立てる。

### 5. ドキュメント/ガイドライン整備

- [ ] `architecture.md` または docs に、「`draw()` は `Geometry | LazyGeometry | Sequence[...]` を返せること」と「Parameter GUI は初回トレース時の構造を前提としている」旨を明記する。
- [ ] サンプルスケッチ群（`sketch/`）のうち、`draw()` がリストを返すものについて、
  - [ ] CC に依存して shape/pipeline の数が変わる場合は、その制約（GUI のサポート範囲）を冒頭コメントで説明する。
  - [ ] 特に 251118 のような「N を CC で増やす」サンプルでは、「Parameter GUI のパラメータは初期 N に対してのみ完全に対応する」など、現状の仕様をユーザーに伝える。

---

上記はあくまで「現時点のコードリーディングとローカル検証から導いた原因候補と改善方向性」であり、実際の GUI 上の挙動（特に dpg/pyglet 経由のランタイム UI 表示）はヘッドレス環境では完全には再現できていない。  
次のステップとしては、この計画に沿って **テストの追加と、小さな計測用ログ/トグルの導入** から始め、実機環境での再現確認を踏まえて原因仮説を絞り込むのが良いと思います。*** End Patch
