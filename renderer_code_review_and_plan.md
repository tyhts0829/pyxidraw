# renderer.py 実装コードレビュー & 改善計画

対象: `src/engine/render/renderer.py`（ライン描画・Geometry→GPU 変換）

## レビュー結果

### 良い点

1. **責務が比較的明確で、役割がコメントで説明されている**
   - モジュール先頭のコメントと `LineRenderer` の docstring で、「SwapBuffer から Geometry を受け取り GPU に送る」「tick で受信し draw で描画」という大枠の責務が示されている。
   - HUD 連携用のカウンタや IBO 固定化実験など、付加機能についても属性名やコメントで目的が分かるようになっている。

2. **描画フローが tick/draw の 2 段階に分かれている**
   - `tick()` で SwapBuffer のフロントを取得し、`draw()` で実際の描画を行う構成になっており、更新と描画が分離されている。
   - レイヤー付きフレームと geometry-only フレームで処理を分けつつ、スナップショットによるフォールバック描画も用意されている点は柔軟。

3. **Geometry→VBO/IBO 変換とインデックス LRU がきちんと抽象化されている**
   - `_geometry_to_vertices_indices()` に Geometry→頂点/インデックス変換を切り出しており、ラインごとの primitive restart 挿入もベクトル化された実装になっている。
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

- [ ] `draw()` をレイヤー描画経路・スナップショット再描画経路・geometry-only 経路の 3 つの小さなヘルパー（例: `_draw_layers_frame` / `_draw_snapshot_layers` / `_draw_geometry_only`）に分割し、メインの `draw()` はフロー制御だけに絞る。
- [ ] レイヤーごとの色・太さ適用処理を `_apply_layer_style(layer: Layer)` のような小さなメソッドに分離し、`draw()` からスタイル適用の詳細を隠す。
- [ ] スナップショット構築と HUD カウンタ更新を、それぞれ `_update_layers_snapshot` / `_update_counts_from_draw` のような専用ヘルパーに分け、`draw()` 本体から計数ロジックを切り離す。

### LazyGeometry とジオメトリ処理

- [ ] `LazyGeometry` の実体化責務を 1 箇所（`draw()` または `_upload_geometry()` のどちらか）に集約し、二重 `realize()` が起こらないようにする（`Layer` スナップショットには実体化済み Geometry を渡す）。
- [ ] `_upload_geometry()` の引数を `Geometry | None` のみに絞り、Lazy なものは事前に解決する（もしくは `_resolve_geometry()` のような小ヘルパーで `LazyGeometry`→`Geometry` を明確に行う）。
- [ ] 空 Geometry（`is_empty`）の扱いを `_upload_geometry()` 冒頭で早期 return するスタイルに整理し、`_last_vertex_count` / `_last_line_count` 更新もこのパスで一元管理する。

### 例外処理・ログ方針

- [ ] `try/except Exception: pass` となっている箇所を列挙し、「クラッシュ回避が最優先で沈黙させたい箇所」と「開発時に異常検知したい箇所」に分類する。
- [ ] 後者（異常検知したい箇所）については、少なくとも `logging.debug`/`logging.warning` を出すか、例外をそのまま伝播させるように変更し、観測可能性を確保する。
- [ ] ModernGL や numpy 由来の例外など、想定しうる例外型が限定される箇所では `Exception` ではなくより具体的な例外を捕捉する。
- [ ] HUD 用カウンタ更新や snapshot 更新など、副作用が失敗しても描画全体は続行したい箇所は、ログだけ出して描画処理自体は継続する方針に整理する。

### IBO 固定化フラグと設定まわり

- [ ] `__init__` 内の `_ibo_freeze_enabled` / `_ibo_debug` の設定読込を見直し、「設定値があればそれを採用し、例外時のみ明示的なデフォルトにフォールバックする」形に整理する。
- [ ] 現状の `self._ibo_debug = False` による上書きを削除または条件付きにし、`common.settings` の `IBO_DEBUG` が正しく反映されるようにする。
- [ ] IBO 固定化の挙動（どの条件でインデックスを再利用するか・freeze を解除する条件など）を短い docstring またはコメントで説明し、将来の保守者がコードから読み取れるようにする。

### インデックス LRU とヘルパー関数

- [ ] `_INDICES_CACHE` とカウンタ群を、小さな内部クラスまたは名前空間的ヘルパー（例: `_IndicesCacheState`）にまとめ、`_IND_HITS_INC()` 系の細かい関数をインスタンスメソッドに集約して見通しを良くする。
- [ ] `_INDICES_CACHE_ENABLED` / `_INDICES_CACHE_MAXSIZE` / `_INDICES_DEBUG` の設定読込ロジックを `_load_indices_cache_settings()` などの関数に分離し、モジュールトップのグローバル初期化を簡潔にする。
- [ ] `_geometry_to_vertices_indices()` 内の LRU 利用パスと通常パスのコメントを整理し、「キャッシュキーの構造」と「ヒット時の LRU 更新」「ミス時の保存処理」が一読で分かるようにする。

### HUD カウンタと属性アクセス

- [ ] `_last_vertex_count` / `_last_line_count` 周辺の `getattr(..., default)` と `# type: ignore[attr-defined]` を廃止し、`__init__` での初期化を前提とした通常の属性アクセスに揃える。
- [ ] HUD カウンタの更新タイミングを `_upload_geometry()` に一元化し、`draw()` 側では「直近アップロード結果を読むだけ」にして責務を分離する。
- [ ] `get_last_counts()` / `get_ibo_stats()` / `get_indices_cache_counters()` の戻り値仕様（特に `enabled` を bool にするか int にするか）を整理し、ドキュメントと実装を一致させる。

### 命名・コメント・docstring

- [ ] `draw()` の docstring 内で `on_draw` という表現を使っている箇所を、実際のメソッド名と責務に合わせて整理する（Window 側コールバック名との対応関係があれば短く追記する）。
- [ ] 「実験的」な挙動（IBO 固定化、indices LRU など）について、どの程度本番想定なのか・期待されるメリットは何かを 1 行程度のコメントにまとめる。
- [ ] LazyGeometry 実体化と snapshot 再利用の意図（「描画負荷を抑えつつ、no-layers フレームでも前フレームの線を維持する」等）を、レイヤー描画まわりのコメントに明記する。

### テスト・検証

- [ ] `_geometry_to_vertices_indices()` に対する単体テストを追加し、複数ライン・primitive restart index・LRU キャッシュのヒット/ミス/Evict の挙動を確認する（ModernGL 依存なしでテスト）。
- [ ] `_upload_geometry()` の空 Geometry / LazyGeometry / 通常 Geometry に対するパスをモックした GPU オブジェクトでテストし、`index_count` や HUD カウンタが期待通りになることを確認する。
- [ ] `LineRenderer` の `draw()` 分割後、ModernGL をモックまたはスタブ化した形でレイヤー描画フローのスモークテストを追加する（描画が呼ばれるか、例外が握りつぶされないかを確認）。
- [ ] 変更後に `ruff` / `mypy` / 関連テスト（存在すれば renderer/geometry 関連）を対象ファイルに対して実行し、スタイルと型・挙動の整合性を確認する。

## 追加確認・相談したい点（ユーザー確認用）

- 例外処理ポリシー:
  - レンダリング中の例外について、「ユーザーのスケッチ実行を止めないこと」と「異常を検知できること」のどちらをどの程度優先すべきか。特に HUD/統計更新失敗や indices LRU 関連の例外を、ログ + 描画継続にする線引きでよいか。；（要確認）
- IBO 固定化・indices LRU の扱い:
  - 現状「実験」的な扱いになっている機能を、どのくらい本番前提にするか。不要であれば思い切って削る/単純化する選択肢もあるが、パフォーマンス要件の観点からどこまで残したいか。；（要確認）
- 構造の分割レベル:
  - `LineRenderer` をあくまで 1 クラスのままメソッド分割に留めるか、indices LRU や HUD カウンタなどを小さな補助クラスに切り出してモジュール分割を進めるか。まずはメソッド分割のみで様子を見る方針でよいか。；（要確認）

