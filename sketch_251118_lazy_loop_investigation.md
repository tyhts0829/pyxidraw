# sketch/251118.py ループ + LazyGeometry パターン調査メモ（2025-11-19）

## 前提と現象

- 対象スケッチ: `sketch/251118.py`

  ```python
  from api import E, G, cc, run

  A5 = (148, 210)


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

- 現象（ユーザー報告）:
  - `draw()` 内で `for` ループを回し、`g` と `p` を作って `list` に追加して返すパターンにすると「うまく動作しない」。
  - `N` は `cc[1]` から決まるため、MIDI/CC に応じてポリゴンの個数が変わる。
  - LazyGeometry を前提にした実装（遅延パイプライン + GUI/ワーカー分離）が関係していそう、という仮説。

本メモでは:

- 「Geometry パスとして正しく解釈されているか」
- 「Lazy + Parameter GUI/Runtime + CC の構造上、どこに制約・落とし穴があるか」

を実装レベルで切り分けて整理する。

---

## 1. Geometry / LazyGeometry パスの挙動

### 1-1. `draw()` の素の戻り値

- `G.polygon()` は `api.shapes.G` からの呼び出しで、常に `LazyGeometry` を返す。
  - 実体: `src/api/shapes.py:ShapesAPI._lazy_shape`
  - `LazyGeometry(base_kind="shape", base_payload=(impl, params_dict))` を構築。
- `.scale(...).translate(...)` は `LazyGeometry` 上のメソッドで、plan に `_fx_scale` と `_fx_translate` を積む。
  - 実装: `src/engine/core/lazy_geometry.py:LazyGeometry.scale/translate`
  - `plan` は `[(impl, params), ...]` のリスト。
- `p = E.label(...).affine().fill()` は `PipelineBuilder` を組み立てる。
  - 実装: `src/api/effects.py:PipelineBuilder` / `_EffectsAPI`
  - `p(g)` は `PipelineBuilder.__call__ → build() → Pipeline.__call__` で、
    `g`（LazyGeometry）の plan に effect 実装（`affine`, `fill`）を追加した **新しい LazyGeometry** を返す。
- `draw()` の戻り値は `list[LazyGeometry]`。

実際に `api.cc.set_snapshot({1: 0.8})` をセットして `draw(0.0)` を直接呼ぶと:

- `N = int(0.8 * 10) + 1 = 9`
- `len(draw(0)) == 9`
- 各要素は `LazyGeometry(base_kind="shape", plan=[_fx_scale, _fx_translate, affine, fill])`
- `_fx_scale` の `scale` パラメータは `(4, 4, 4), (8, 8, 8), …, (36, 36, 36)` と i ごとに異なる。

→ `draw()` 自体は、ループでリストに積むパターンでも **正しく LazyGeometry を構築しており、バグはない**。

### 1-2. Worker 側での正規化 `_normalize_to_layers`

- ワーカーは `draw(t)` の戻り値を `engine.runtime.worker._normalize_to_layers` でレイヤー形式に変換する。
  - 実装: `src/engine/runtime/worker.py:_normalize_to_layers`
  - シグネチャ:
    - `result: Geometry | LazyGeometry | Sequence[Geometry | LazyGeometry]`
    - 戻り値: `(geometry: Geometry | LazyGeometry | None, layers: list[StyledLayer] | None)`

`draw()` が `list[LazyGeometry]` を返すケース:

- `isinstance(result, Sequence)` かつ `not isinstance(result, (Geometry, LazyGeometry))` → **Sequence 経路**に入る。
- 各要素 `item` について:
  - `LazyGeometry` なら:
    - `item.plan` から `style` エフェクトだけを取り除き（今回は `style` なしなので plan はそのまま）、
    - `StyledLayer(geometry=LazyGeometry(...), color=None, thickness=None)` を作る。
  - `Geometry` ならそのまま `StyledLayer(geometry=item, ...)`。
- `geometry=None, layers=[StyledLayer(...)] * N` を返す。

`sketch/251118.py` の戻り値に対して実際に `_normalize_to_layers` を当てると:

- `geometry is None`
- `len(layers) == N`（期待どおり）
- 各 `layer.geometry` は `LazyGeometry` で、`realize()` 後のバウンディングボックスは半径が順に大きくなっている。

→ 「リストで返すとレンダラーに届かない」「全部同じ形になる」といった **LazyGeometry レイヤリング起因の不具合は見当たらない**。

### 1-3. Renderer 側での LazyGeometry 実体化

- レイヤーは `RenderPacket.layers` として `SwapBuffer` 経由で `engine.render.renderer.LineRenderer` に渡される。
  - `tick()` 内で `front` が `list[StyledLayer]` なら `_frame_layers` に保存。
  - `draw()` 内で各レイヤーに対して `_upload_geometry(layer.geometry)` を呼び出す。
- `_upload_geometry` は `LazyGeometry` の場合に `geometry.realize()` を呼んでから GPU 転送する。
  - 実装: `src/engine/render/renderer.py:_upload_geometry`
  - Shape LRU と Prefix LRU は `LazyGeometry.realize()` 内で処理される。

`LazyGeometry.realize()` の中身:

- `base_kind == "shape"` の場合、`shape_impl` とパラメータから `Geometry` を生成し、`_SHAPE_CACHE` に LRU 保存。
- その後 `plan` の各ステップ（ここでは `_fx_scale`, `_fx_translate`, `affine`, `fill`）を順に適用。
  - Prefix LRU も `base_key`（shape spec）と `effect_sigs`（各ステップの署名）に基づいて働く。
  - `_fx_scale` の `scale` が i ごとに異なるので、署名も異なり、**各ポリゴンは個別に処理される**。

→ Geometry パスとしては、「ループ＋リスト返し」でも **Lazy/CACHE を含めて設計どおりの挙動をしている**。

---

## 2. Parameter GUI / Runtime / CC との相互作用

現象報告に「lazy であることが関係してそう」とあるため、Parameter GUI と CC の流れも整理しておく。

### 2-1. Parameter GUI 初期化タイミング

- `api.sketch.run_sketch` は `use_parameter_gui=True` のとき `ParameterManager` を生成し、初期化する。
  - 実装: `src/api/sketch.py`
  - 初期化フロー:
    1. `parameter_manager = ParameterManager(user_draw)` を構築。
    2. `parameter_manager.initialize()` を呼ぶ。
- `ParameterManager.initialize()` は内部で `ParameterRuntime` をアクティブにして `user_draw(0.0)` を **一度だけ**呼ぶ。
  - 実装: `src/engine/ui/parameters/manager.py:ParameterManager.initialize`
  - このとき:
    - `ParameterRuntime.begin_frame()` によりシェイプ/エフェクトの出現回数カウンタとパイプライン UID カウンタをリセット。
    - `runtime.set_inputs(0.0)` で「時刻 t=0」のみをランタイムに渡す。
    - `api.cc` のスナップショットは **まだ Worker から設定されておらず空** なので、`cc[1] == 0.0`。
    - 結果として `N = int(cc[1] * 10) + 1` は常に 1 になり、
      - 初期化時点では **ポリゴン 1 個分（ループ1回分）のパイプラインだけが ParameterRuntime に観測される**。

- この 1 回の `user_draw(0.0)` 呼び出し中に:
  - `G.polygon()`/`E.affine`/`E.fill` の各呼び出しが `ParameterRuntime.before_*_call` によってフックされ、
  - `ParameterStore` に Descriptor と初期値が登録される。

→ GUI が持っているパラメータのメタ情報（Descriptor 群）は、「**t=0, cc=0 の世界で 1 回だけ実行した結果**」に対応している。

### 2-2. Worker 側の適用と SnapshotRuntime

- 本番描画時、ワーカーは次の順で `draw(t)` を呼び出す。
  - 実装: `src/engine/runtime/worker.py:_execute_draw_to_packet`
  1. CC スナップショット適用:
     - `apply_cc_snapshot(cc_state)` → `api.cc.set_snapshot(...)`
     - ここで `cc[1]` が実際の MIDI 状態に更新される。
  2. Parameter GUI スナップショット適用:
     - `apply_param_snapshot(param_overrides, t)`。
     - 実体は `engine.ui.parameters.snapshot.apply_param_snapshot`。
     - `overrides` は `engine.ui.parameters.snapshot.extract_overrides(ParameterStore, cc_map)` の結果。
  3. `geometry_or_seq = draw_callback(t)`（ユーザーの `draw`）。
  4. `apply_param_snapshot(None, 0.0)` で SnapshotRuntime を解除。

`apply_param_snapshot` の挙動:

- `overrides is not None` のとき:
  - `SnapshotRuntime(overrides)` を生成し、`activate_runtime(rt)` でアクティブにする。
  - `SnapshotRuntime` は `ParameterRuntime` と同じインタフェース（`before_shape_call` / `before_effect_call`）を持つが、
    - メタ登録は行わず、
    - `overrides` に含まれるキー（`shape.<name>#<index>.<param>` / `effect@<pipeline>.<name>#<step>.<param>`）だけを見て、未指定引数に GUI/CC の値を注入する。
- `overrides is None` のときは `deactivate_runtime()` するだけ。

重要なのは:

- `extract_overrides()` は **ParameterStore に登録されている Descriptor 群** をベースに override を抽出する。
  - 実装: `src/engine/ui/parameters/snapshot.py:extract_overrides`
  - Descriptor の ID は `ParameterRuntime` 初期化時の 1 回の `user_draw(0.0)` のみから生成される。
  - つまり、初期化時に存在しなかったパイプライン（＝その時点では `N` が小さくてループに入らなかった分）には **Descriptor も override も存在しない**。

結果:

- `cc[1]` を動かして `N` を増やすと、ワーカー中の `draw(t)` では `for i in range(N)` が多く回るため、**新しいポリゴン用のパイプライン（p1, p2, ...）が実行される**。
- しかし Parameter GUI 側にはそれらに対応する Descriptor/override が存在しないため、
  - SnapshotRuntime は `effect@p1.fill#1.density` 等のキーを一切見つけられず、
  - **追加分のポリゴンに対しては常に「エフェクト実装側のデフォルト値」が使われる**。
- 初期化時に 1 個だけ存在したポリゴン（p0）の effect パラメータについては Descriptor があるため、
  - GUI からの変更や CC バインディングは p0 にだけ適用される。

→ 「ループで増やしたポリゴンには Parameter GUI の変更が反映されない」「一部のポリゴンだけ GUI と連動しない」という挙動が出るのは、この **初期化時 1 回だけの Lazy なパラメータトレース + SnapshotRuntime の設計** に起因する。

### 2-3. これは Geometry のバグか？

- Geometry/LazyGeometry/Worker/Renderer のパスだけを見ると、`list[LazyGeometry]` を返すパターンは仕様どおりに動いている。
  - 実体化してみると、全ポリゴンが正しくスケールされた Geometry として得られる。
-「うまく動作しない」部分は、**「GUI から見えるパイプライン」と「実際に実行されているパイプライン」の集合がずれる**ことにある。
  - 初期化時: `cc[1] == 0` で `N=1` → `p0` のみが ParameterRuntime に観測され GUI に登録される。
  - 本番時: `cc[1]` に応じて `p0..p(N-1)` が生成されるが、`p1..` には Descriptor が無い。
  - SnapshotRuntime は「知らないパイプライン」には override を適用できないので、追加分は常にデフォルト値で走る。

したがって:

- LazyGeometry 自体の遅延評価や `Pipeline` のキャッシュが「ループ＋リスト返し」を壊しているわけではない。
- 問題になるのは **ParameterRuntime が「初回フレームのトレース結果だけを永続化し、その後は SnapshotRuntime が override だけを当てる」という設計**と、
  - `N` を CC などランタイム入力に依存して決めているというスケッチ側の書き方の組み合わせ。

---

## 3. まとめ（なぜこのパターンが「うまく動作しない」と感じられるか）

1. `draw()` のループ + リスト返し自体は、LazyGeometry/Worker/Renderer の実装上は **正しい使い方** であり、ポリゴンは期待どおり生成・描画される。
2. しかし Parameter GUI は
   - 起動時に `ParameterRuntime` で `user_draw(0.0)` を **1 回だけ**トレースし、
   - そのときに観測された shape/effect 呼び出しだけを Descriptor として永続化する。
3. `sketch/251118.py` では `N = int(cc[1] * 10) + 1` なので、
   - 初期化時（`cc` 空 → `cc[1]==0`）には `N=1` で 1 つ分のパイプライン（p0）しか登録されない。
   - 実行時に `cc[1]` を上げて `N` を増やしても、追加分のパイプライン（p1..）には GUI の Descriptor/override が存在しない。
4. Worker 側では SnapshotRuntime が override を適用するが、
   - override は「初期化時に登録された Descriptor の ID」に対してのみ存在するため、
   - 結果として **GUI から操作できるのは初期 N 個分だけ**になり、後から増えたポリゴンはデフォルト値のまま動く。

このため、「for で `g` と `p` を作って list に積むと、Parameter GUI と連動しない形が出る」「lazy であることが怪しい」と感じられる。

実装レベルで見ると:

- LazyGeometry の遅延評価や `Pipeline` キャッシュ自体は正しく動いている。
- 問題の本質は **ParameterRuntime/ParameterStore のトレースが 1 回きりであり、ランタイム入力（CC）に依存して変化するパイプライン構造を追従しない**ことにある。

---

## 4. 補足: この制約を回避する書き方の方向性（実装変更はまだ行っていない）

※ここでは原因整理の一環として方針だけ記す。実際のコード変更は依頼があった時点で別途チェックリストを作成して対応する。

- 「ポリゴン数 N」を CC ではなく GUI パラメータとして表現し、`ParameterRuntime` の初期トレースで Descriptor を作成できるようにする。
  - 例: `E.label(uid="polygons").affine().fill()` のどこかに `count` 的な int パラメータを持たせる形に寄せる。
- あるいは、`repeat()` エフェクトのような **固定構造のパイプライン**を使い、ループではなく effect 側で繰り返し配置する。
  - そうすれば、初期トレース時に必要なパラメータがすべて観測される。
- Parameter GUI 側の仕様を変える場合は、
  - 「初期化後も定期的に ParameterRuntime で再トレースして Descriptor を更新する」
  - もしくは「SnapshotRuntime だけで新規パイプラインにも override を割り当てる」
  - といった設計変更が必要になる（現状はそのような機構は無い）。

