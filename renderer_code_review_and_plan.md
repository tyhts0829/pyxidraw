# renderer.py 実装コードレビュー & 改善計画

対象: `src/engine/render/renderer.py`（ライン描画・Geometry→GPU 変換）

## レビュー結果

### 良い点

1. **責務が比較的明確で、役割がコメントで説明されている**

   - モジュール先頭のコメントと `LineRenderer` の docstring で、「SwapBuffer から Geometry を受け取り GPU に送る」「tick で受信し draw で描画」という大枠の責務が示されている。
   - HUD 連携用のカウンタや IBO 固定化（indices LRU を含む）など、付加機能についても属性名やコメントで目的が分かるようになっている。

2. **描画フローが tick/draw の 2 段階に分かれている**

   - `tick()` で SwapBuffer のフロントを取得し、`draw()` で実際の描画を行う構成になっており、更新と描画が分離されている。
   - レイヤー付きフレームと geometry-only フレームで処理を分けつつ、スナップショットによるフォールバック描画も用意されている点は柔軟。

3. **Geometry→VBO/IBO 変換とインデックス LRU がきちんと抽象化されている**

   - `_geometry_to_vertices_indices()` に Geometry→ 頂点/インデックス変換を切り出しており、ラインごとの primitive restart 挿入もベクトル化された実装になっている。
   - `_INDICES_CACHE` とそのカウンタ群で、インデックス配列の LRU キャッシュを実装しており、HUD から統計を参照できるようになっている。

4. **型ヒントと戻り値型がほぼ揃っている**
   - `LineRenderer` の公開メソッドやユーティリティ関数に型注釈が付いており、`get_last_counts()` / `get_ibo_stats()` / `get_indices_cache_counters()` なども型が明示されている。
   - Geometry・LazyGeometry まわりの型も `Geometry | LazyGeometry | None` などで分かりやすく表現されている。

### 改善したい点

1. **例外処理が広範かつ過度に防御的で、異常が見えづらい**

   - `draw()` のレイヤーループやスナップショット描画部分、`set_line_color` / `set_base_line_color` / `set_line_thickness` / `set_base_line_thickness` / `_upload_geometry` / `_geometry_to_vertices_indices` など、多くの箇所が `try/except Exception: pass` もしくは何もログを出さない形になっている。
   - AGENTS.md の方針（「必要十分な堅牢さ」「過度に防御的な実装は絶対にしない」）と比べると、クラッシュ回避を優先しすぎて観測性が落ちている。

2. **`__init__` 内の設定反映ロジックが読みにくく、実質的にデバッグフラグが無効化されている**

   - `common.settings.get()` から `_ibo_freeze_enabled` / `_ibo_debug` を読む処理の直後に、例外ハンドラ外で `self._ibo_debug = False` を上書きしており、設定値があっても必ず False になる。
   - 設定反映の責務とデフォルト値の決定が 1 つの try/except にまとまっていて読みづらい。

3. **`draw()` と `_upload_geometry()` の責務分担がやや曖昧**

   - `draw()` のレイヤーループで `LazyGeometry` を `realize()` してから `_upload_geometry(geometry)` を呼んでいる一方、`_upload_geometry()` の内部でも `LazyGeometry` を判定して `realize()` を呼び直している（`geometry` 経路と `frame.geometry` 経路の扱いが異なる）。
   - `LazyGeometry` の実体化タイミングと「どこで Geometry に正規化されるべきか」が分散しており、修正時に二重実体化や片側だけ直すリスクがある。

4. **`draw()` が 1 メソッド内で多くの関心事を扱っており、読みやすさに余裕がない**

   - フレームカウンタ更新、レイヤー有無分岐、レイヤー描画ループ、スナップショット構築と保存、HUD 用カウンタ更新、no-layers フレームでのスナップショット再描画、geometry-only 経路、粘着色の適用など、かなり多くの処理が 1 メソッドに詰め込まれている。
   - 例外処理も細かく入っているため、主要なフローを追うのに視線のジャンプが多くなる。

5. **HUD 用カウンタ周辺で `getattr(..., "_last_vertex_count", 0)` 等が多用されており、実装が冗長**

   - `__init__` で `_last_vertex_count` / `_last_line_count` を初期化しているため、`getattr(..., default)` ベースのアクセスや `# type: ignore[attr-defined]` は不要なはず。
   - HUD 用のカウンタ更新処理が `_upload_geometry` と `draw()` 内で重複しており、どちらが最新値を決めるかが直感的でない。

6. **インデックス LRU のグローバル状態がやや散漫**

   - `_INDICES_CACHE` とカウンタ (`_IND_HITS` など) がモジュールグローバルで定義され、増加ヘルパー関数も小さく分散しているため、慣れていないと全体像を掴みにくい。
   - `_INDICES_CACHE_ENABLED` / `_INDICES_CACHE_MAXSIZE` / `_INDICES_DEBUG` の設定読込とデフォルト決定も 1 つの try/except にまとまっており、読みやすさの面でやや重い。

7. **命名とコメントに小さなノイズがある**
   - `draw()` の docstring に `on_draw` という語が残っており、Window 側のコールバック名と混ざっているように見える。
   - 「実験」や「フォールバック」の挙動に関するコメントが要所にある一方で、どの条件で有効になるのか・どこまで正式サポートなのかが分かりづらい。

## 改善計画（チェックリスト）

コード変更前の計画（このファイルの項目はすべて未着手）です。実装の美しさ・可読性・分割を優先しつつ、必要十分な堅牢さを保つ方針で進める想定です。

### 描画フローと責務分割

- [x] `draw()` をレイヤー描画経路・スナップショット再描画経路・geometry-only 経路の 3 つの小さなヘルパー（例: `_draw_layers_frame` / `_draw_snapshot_layers` / `_draw_geometry_only`）に分割し、メインの `draw()` はフロー制御だけに絞る。
- [x] レイヤーごとの色・太さ適用処理を `_apply_layer_style(layer: Layer)` のような小さなメソッドに分離し、`draw()` からスタイル適用の詳細を隠す。
- [x] スナップショット構築と HUD カウンタ更新を、それぞれ `_draw_layers_frame` / `_upload_geometry` 内に集約し、`draw()` 本体から計数ロジックを切り離す。

### LazyGeometry とジオメトリ処理

- [x] `LazyGeometry` の実体化責務を 1 箇所（`draw()` または `_upload_geometry()` のどちらか）に集約し、二重 `realize()` が起こらないようにする（`Layer` スナップショットには実体化済み Geometry を渡す）。
- [x] `_upload_geometry()` の引数を `Geometry | None` のみに絞り、Lazy なものは事前に解決する（もしくは `_resolve_geometry()` のような小ヘルパーで `LazyGeometry`→`Geometry` を明確に行う）。
- [x] 空 Geometry（`is_empty`）の扱いを `_upload_geometry()` 冒頭で早期 return するスタイルに整理し、`_last_vertex_count` / `_last_line_count` 更新もこのパスで一元管理する。

### 例外処理・ログ方針

- [x] `try/except Exception: pass` となっていた箇所を洗い出し、`set_line_color` 系・IBO 固定化・indices LRU など「異常を検知したい箇所」については `logging.debug` を出すように変更した（描画自体は継続）。
- [x] 後者（異常検知したい箇所）については、少なくとも `logging.debug`/`logging.warning` を出すか、例外をそのまま伝播させるように変更し、観測可能性を確保する（indices LRU の lookup/store 失敗時に debug ログを出すよう変更）。
- [ ] ModernGL や numpy 由来の例外など、想定しうる例外型が限定される箇所では `Exception` ではなくより具体的な例外を捕捉する。
- [x] HUD 用カウンタ更新や snapshot 更新など、副作用が失敗しても描画全体は続行したい箇所は、例外を握りつぶさず通常の例外伝播とし、本質的なバグを早期に発見できるようにする（`_draw_layers_frame` / `_draw_snapshot_layers` 周辺）。

### IBO 固定化フラグと設定まわり

- [x] `__init__` 内の `_ibo_freeze_enabled` / `_ibo_debug` の設定読込を見直し、「設定値があればそれを採用し、例外時のみ明示的なデフォルトにフォールバックする」形に整理する。
- [x] 現状の `self._ibo_debug = False` による上書きを削除または条件付きにし、`common.settings` の `IBO_DEBUG` が正しく反映されるようにする。
- [x] IBO 固定化の挙動（どの条件でインデックスを再利用するか・freeze を解除する条件など）を `_upload_geometry` の docstring とコメントで説明し、「本番前提の最適化機構」として仕様を明確にする。

### インデックス LRU とヘルパー関数

- [ ] `_INDICES_CACHE` とカウンタ群を、小さな内部クラスまたは名前空間的ヘルパー（例: `_IndicesCacheState`）にまとめ、`_IND_HITS_INC()` 系の細かい関数をインスタンスメソッドに集約して見通しを良くする。
- [x] `_INDICES_CACHE_ENABLED` / `_INDICES_CACHE_MAXSIZE` / `_INDICES_DEBUG` の設定読込ロジックを `_load_indices_cache_settings()` などの関数に分離し、モジュールトップのグローバル初期化を簡潔にする。
- [x] `_geometry_to_vertices_indices()` 内の LRU 利用パスと通常パスのコメントを整理し、「キャッシュキーの構造」と「ヒット時の LRU 更新」「ミス時の保存処理」が一読で分かるようにする。

### HUD カウンタと属性アクセス

- [x] `_last_vertex_count` / `_last_line_count` 周辺の `getattr(..., default)` と `# type: ignore[attr-defined]` を廃止し、`__init__` での初期化を前提とした通常の属性アクセスに揃える。
- [x] HUD カウンタの更新タイミングを `_upload_geometry()` に一元化し、`draw()` 側では「直近アップロード結果を読むだけ」にして責務を分離する。
- [x] `get_last_counts()` / `get_ibo_stats()` / `get_indices_cache_counters()` の戻り値仕様を確認し、`get_indices_cache_counters()` は `dict[str, int]` として `enabled` も 0/1 の int を返すよう型と docstring を揃えた。

### 命名・コメント・docstring

- [x] `draw()` の docstring を実際の呼び出し元と責務に合わせ、「`RenderWindow.on_draw` から呼ばれ、受信は `tick` で行う」ことを明示した。
- [x] IBO 固定化と indices LRU について、「レンダリング性能を高めるための本番前提の最適化」であることと、その期待されるメリット（インデックス再生成の削減など）を `_upload_geometry` と `_geometry_to_vertices_indices` のコメントで示した。
- [x] LazyGeometry 実体化と snapshot 再利用の意図（「描画負荷を抑えつつ、no-layers フレームでも前フレームの線を維持する」等）を、`_draw_layers_frame` / `_draw_snapshot_layers` のコメントに明記した。

### テスト・検証

- [ ] `_geometry_to_vertices_indices()` に対する単体テストを追加し、複数ライン・primitive restart index・LRU キャッシュのヒット/ミス/Evict の挙動を確認する（ModernGL 依存なしでテスト）。
- [x] `_geometry_to_vertices_indices()` に対する単体テストを拡張し、LRU キャッシュのヒット/ミス/保存の挙動を確認する（`tests/render/test_renderer_utils.py`）。
- [x] `_upload_geometry()` の通常パスと IBO 固定化パスをモックした GPU オブジェクトでテストし、`index_count` や HUD カウンタが期待通りになることを確認する（`tests/render/test_renderer_upload_geometry.py`）。
- [ ] `LineRenderer` の `draw()` 分割後、ModernGL をモックまたはスタブ化した形でレイヤー描画フローのスモークテストを追加する（描画が呼ばれるか、例外が握りつぶされないかを確認）。
- [x] 変更後に `ruff` / `mypy` / 関連テスト（存在すれば renderer/geometry 関連）を対象ファイルに対して実行し、スタイルと型・挙動の整合性を確認する。

## 追加確認・相談したい点（ユーザー確認結果メモ）

- 例外処理ポリシー:
  - レンダリング中の例外については、HUD/統計更新失敗や indices LRU 関連などは「ログを残しつつ描画は継続する」方針でよい（致命的な初期化失敗などは別途検討）。
- IBO 固定化・indices LRU の扱い:
  - これらは本番前提で利用する最適化機構として扱い、削除ではなく「仕様を明確にしたうえでシンプルに保つ」方向で整備する。
- 構造の分割レベル:
  - `LineRenderer` はまずメソッド分割のみで整理し、必要になれば後から補助クラスへの切り出しを検討する。
