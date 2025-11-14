# G-code エクスポート時の AttributeError（front が list のケース）

現象
- G キー押下時に以下の例外でクラッシュする。
  - `AttributeError: 'list' object has no attribute 'is_empty'`
  - その後に `Glfw Error 65537: The GLFW library is not initialized` が続く（副次エラー）。

スタックトレース（要約）
- `src/api/sketch.py:494` の `on_key_press()` で `G` キーを検出し、`_start_gcode_export()` を呼び出す。
- `_start_gcode_export()` から `src/api/sketch_runner/export.py` の `_start()` が実行される。
- `_start()` 内で `front = swap_buffer.get_front()` を取得し、`if front is None or front.is_empty:` の評価中に例外が発生する。

原因（直接原因）
- 例外発生時、`SwapBuffer.get_front()` が返していたのは `Geometry` ではなく `list` だった。
- `make_gcode_export_handlers._start()` は `front` を `Geometry | LazyGeometry` 相当とみなし、`front.is_empty` や `front.as_arrays()` を直接呼び出している。
- `list` 型には `is_empty` / `as_arrays` プロパティが存在しないため、`AttributeError: 'list' object has no attribute 'is_empty'` が送出される。

原因（設計レベル）
- 現行のレンダリングパイプラインでは、`SwapBuffer` は「単一の Geometry」だけでなく「スタイル付きレイヤー列（StyledLayer[]）」も保持する。
  - `engine.runtime.worker._normalize_to_layers()` が `draw(t)` の戻り値を正規化し、必要に応じて `StyledLayer` に分解する。
  - `engine.runtime.receiver.StreamReceiver.tick()` は `packet.layers` が存在する場合、`SwapBuffer.push(list(layers))` を呼び出し、`SwapBuffer._front` に `list[StyledLayer]` を格納する。
  - `engine.render.renderer.LineRenderer.tick()` は `front` が `StyledLayer` の列であればレイヤー描画経路に乗せる（duck-typing で判定）。
- 一方、`api.sketch_runner.export.make_gcode_export_handlers()` は古い「Geometry-only」設計のままで、「`SwapBuffer.get_front()` は Geometry または LazyGeometry を返す」という前提で実装されている。
- その結果、「レンダリング側はレイヤー対応済みだが、G-code エクスポート側だけが Geometry 専用前提」という設計ずれが生じ、`front` が `list[StyledLayer]` になったケースで AttributeError が発生している。

副次的な 65537 エラーについて
- G-code エクスポート中の AttributeError により Pyglet のイベントハンドラが例外終了し、その後のウィンドウ/GLFW のクリーンアップ処理で `Glfw Error 65537: The GLFW library is not initialized` が出力されている。
- これは既知の「終了タイミング問題」と同系統の副作用であり、本件の一次原因ではない。
  - G-code ハンドラ側で例外を出さずに早期リターンできるようにすれば、今回のログは観測されなくなる見込み。

改善方針（全体像）
- G-code エクスポートは「レンダリングと同じ `SwapBuffer` を読む」が、「幾何情報だけを抽出してスナップショットする」責務。
- したがって `make_gcode_export_handlers` 側で `SwapBuffer.get_front()` の戻り値を
  - `Geometry` / `LazyGeometry`
  - `StyledLayer[]`（`list` / `tuple`）
  の両方に対応させ、最終的に 1 つの `Geometry` に正規化した上で `as_arrays(copy=True)` を呼び出す形に揃えるのが自然。

改善案（実装イメージ）
- `src/api/sketch_runner/export.py` 内に「front → Geometry」変換の小さなヘルパを追加する。
  - 例: `_normalize_front_to_geometry(front) -> Geometry | LazyGeometry | None`
- ロジック案（型別の扱い）:
  - `front is None` の場合
    - G-code エクスポート対象が存在しないため `None` を返す（既存メッセージ「ジオメトリ未生成」を出して早期 return）。
  - `front` が `Geometry` / `LazyGeometry` の場合
    - そのまま `Geometry | LazyGeometry` として返す。
    - `LazyGeometry` は `.is_empty` や `.as_arrays()` 経由で内部的に `realize()` されるため、呼び出し側で特別扱いする必要はない。
  - `front` が `list` / `tuple` の場合
    - 先頭要素が `StyledLayer`（または同等の duck-typing: `geometry` / `color` / `thickness` 属性を持つ）であるかを確認する。
    - 各レイヤーから `geometry` を取り出し、`LazyGeometry` であれば一度 `realize()` して `Geometry` にする。
    - 空でない `Geometry` を `Geometry.concat()`（または `+` 演算子）で 1 つにマージする。
    - マージ結果が空（全レイヤー空）であれば `None` を返す。
  - 上記以外の型（想定外）は安全側に倒して `None` 扱いとし、「G-code エクスポート対象なし（ジオメトリ未生成）」メッセージを出して何もしない。
- `_start()` 本体は `swap_buffer.get_front()` を直接触らず、このヘルパを経由してから空判定と `as_arrays()` を行う。
  - 例:
    - `geom = _normalize_front_to_geometry(swap_buffer.get_front())`
    - `if geom is None or geom.is_empty: ...`
    - `coords, offsets = geom.as_arrays(copy=True)`

設計上のメリット
- レンダリング（`LineRenderer`）と G-code エクスポートの両方が「`SwapBuffer` をソースにしつつ、用途に応じて Geometry を取り出す」という筋の通った責務分担になる。
- 変更が `api.sketch_runner.export` に閉じるため、`SwapBuffer` や `LineRenderer`、`WorkerPool` のシグネチャには手を入れずに済む。
- 将来 `SwapBuffer` に別種のペイロードが増えた場合も、正規化ヘルパ `_normalize_front_to_geometry` を 1 箇所拡張するだけで G-code 側の対応が完結する。

実装タスク（チェックリスト）
- [ ] `src/api/sketch_runner/export.py` に `SwapBuffer.get_front()` の戻り値を `Geometry` ベースに正規化するヘルパ `_normalize_front_to_geometry(front)` を追加する
- [ ] `_start()` から `front.is_empty` / `front.as_arrays()` の直接呼び出しをやめ、ヘルパ経由で `Geometry | LazyGeometry | None` を扱うように変更する
- [ ] `draw(t)` がレイヤー列（`Sequence[Geometry | LazyGeometry]`）を返すスケッチで G キーを押しても例外が発生せず、「対象なし」か正常に G-code 保存されることを確認する
- [ ] `docs/spec/export_image_spec.md` の「G-code 実装」セクションを、現状の実装（レイヤー経由の Snapshot 化・`y_down=True` 既定など）に合わせて更新する
- [ ] 可能であればユニットテストを追加し、擬似 `SwapBuffer` に `StyledLayer[]` を詰めた状態で `_start()` を呼んだときに `ExportService.submit_gcode_job()` が呼ばれること、また対象なしの場合は呼ばれないことを検証する
- [ ] 変更対象ファイルに対して `ruff/black/isort/mypy` を実行し、G-code 関連のスモークテスト（または手動確認）を含めてテストを緑にする

