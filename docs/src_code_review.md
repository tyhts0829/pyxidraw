# src コードレビュー（アーキテクチャ / 可読性 / パフォーマンス）

作成日: 2025-11-14  
対象: `src/` 配下全体（特に `api/`, `common/`, `engine/core/`, `engine/runtime/`, `engine/render/`, `shapes/`, `effects/`, `util/`）

本ドキュメントは、現時点の実装に対して静的なコードレビューを行った結果のスナップショット。  
プロファイリングや実行ログに基づく詳細なボトルネック分析は含まない。

---

## 1. アーキテクチャ

### 1.1 全体評価（レイヤリング / 依存方向）

- `architecture.md` 記載のレイヤ構造（L0〜L3）と、`src/` 以下の実装は概ね整合している。
  - `common/`, `util/` は上位層への依存がなく、ユーティリティ層として独立している（例: `src/common/settings.py`, `src/common/param_utils.py`, `src/util/color.py`）。
  - `engine/*` から `api/*`, `effects/*`, `shapes/*` への依存は見当たらず、`src/engine/AGENTS.md` の禁止エッジを守っている。
  - `effects/`, `shapes/` から `engine/render/*`, `engine/runtime/*`, `engine/ui/*`, `engine/io/*` への依存もなく、数値変換と Geometry 生成に責務が限定されている。
- 登録と解決の責務分離が明確。
  - 共通レジストリ基盤: `src/common/base_registry.py`
  - shape レジストリ: `src/shapes/registry.py`
  - effect レジストリ: `src/effects/registry.py`
  - 高レベル API は `src/api/__init__.py`, `src/api/shapes.py`, `src/api/effects.py` に集約されており、ユーザー視点での入口がわかりやすい。
- Lazy 評価と Geometry 統一表現が、アーキテクチャの中心としてよく整理されている。
  - 統一 Geometry 型: `src/engine/core/geometry.py`
  - 遅延 spec/plan: `src/engine/core/lazy_geometry.py`
  - パイプライン API: `src/api/effects.py`（`Pipeline`, `PipelineBuilder`, `E.pipeline`）

### 1.2 良い点

- **責務と依存の境界が AGENTS + architecture.md で明示され、実装がそれに従っている。**
  - `src/api/AGENTS.md`, `src/effects/AGENTS.md`, `src/shapes/AGENTS.md`, `src/engine/AGENTS.md` など、ディレクトリごとの役割が短く具体的に書かれており、コードと整合している。
  - 例: `src/api/shapes.py` は shape レジストリの薄いファサードと LazyGeometry の spec 構築に限定され、実際の変換/加工は `Geometry` と `E.pipeline` 側に委譲されている。
- **Geometry / LazyGeometry / パイプラインの分離が明確で、拡張にも強い設計。**
  - `Geometry` は純粋変換メソッド（`translate/scale/rotate/concat`）のみを持ち、表現と変換の責務がシンプルにまとまっている（`src/engine/core/geometry.py`）。
  - `LazyGeometry` は shape 実装と effect 実装の「参照 + パラメータ列」を保持するだけで、registries には依存しない（`src/engine/core/lazy_geometry.py`）。これにより API 層から関数参照を注入する設計が素直。
  - `Pipeline` / `PipelineBuilder` は `effects.registry` と Runtime (`engine.ui.parameters`) にのみ依存し、UI/パラメータ GUI とパイプラインが緩やかに結合されている（`src/api/effects.py`）。
- **実行系（runtime/render）と API の結合がゆるく、ワーカー / レンダラーが独立している。**
  - ワーカー: `src/engine/runtime/worker.py`（`WorkerPool`, `_WorkerProcess`, `_execute_draw_to_packet`）
  - レンダラー: `src/engine/render/renderer.py`（`LineRenderer` + indices LRU）
  - ランナー: `src/api/sketch.py` が、WorkerPool / StreamReceiver / RenderWindow / MIDI / HUD をまとめるオーケストレータとして薄く機能している。

### 1.3 気になる点 / 改善アイデア（アーキテクチャ）

実装変更は伴わない観点ベースの指摘。

- **キャッシュ機構が複数箇所に分散しており、全体像の把握にやや負荷がある。**
  - 形状生成 LRU: `LazyGeometry` 内の `_SHAPE_CACHE`（`src/engine/core/lazy_geometry.py`）
  - prefix LRU（途中結果再利用）: `_PREFIX_CACHE`（同上）
  - エフェクトパイプラインの compiled cache: `_GLOBAL_COMPILED`（`src/api/effects.py`）
  - shape spec キャッシュ（統計のみ）: `_spec_seen`（`src/api/shapes.py`）
  - indices LRU: `_INDICES_CACHE`（`src/engine/render/renderer.py`）
  - それぞれ役割は明確だが、キャッシュの責務単位と削除ポリシーがファイル単位で散らばっているため、将来トラブルシュートやチューニングを行う際には「どの層のキャッシュか」をドキュメント等で一覧化しておくと理解コストが下がりそう。
- **署名生成のフィルタリングロジックが、`LazyGeometry` 内で二重に存在する。**
  - `LazyGeometry.realize()` 内のローカル関数 `_filter_sig_params`（`src/engine/core/lazy_geometry.py`）と、モジュール末尾の `_filter_sig_params_global` がほぼ同じ役割を持っている。
  - 現状でも動作上の問題はないが、「ObjectRef や *_ref を署名から除外する」というポリシーを 1 箇所に集約できると、将来の仕様変更時の漏れを減らせそう。
- **テキスト形状モジュールが、フォント探索ロジックと形状生成ロジックを同一ファイル内に持っており、責務がやや広い。**
  - `src/shapes/text.py` は、フォント探索 (`TextRenderer.get_font_path_list`), TTFont ロード, glyph flatten (`FlattenPen`), グリフコマンド LRU など、多機能なコンポーネントになっている。
  - アーキテクチャ的には `util.fonts` / `util.utils` を介した依存で境界は守られているが、「フォント列挙・キャッシュ」と「Geometry 生成」を薄く分けておくと、将来フォント周りだけ差し替えたいケースでの柔軟性が増す。

---

## 2. 可読性

### 2.1 全体評価

- 日本語のモジュール docstring（「どこで / 何を / なぜ」）と NumPy スタイルの関数 docstring が広く徹底されており、役割の理解がしやすい。
- 型ヒントが全体的に行き届いており、`Geometry`, `LazyGeometry`, `Pipeline`, `WorkerPool` など中核クラスのインターフェースが明快。
- AGENTS.md で示されている「短く・具体的・反復可」の方針に概ね沿っており、長い関数でもセクションコメントで区切られているため追いやすい。

### 2.2 良い点

- **中核型のドキュメントが充実しており、API 仕様をコードから直接把握できる。**
  - `Geometry` クラスのモジュール docstring とクラス docstring が丁寧に書かれており、`coords/offsets` の不変条件と設計意図が明確（`src/engine/core/geometry.py`）。
  - `LazyGeometry` も冒頭で設計要点がまとめられており、「base は shape spec or Geometry」「plan は effect 列」「realize は 1 度だけ評価」という契約が読み取りやすい（`src/engine/core/lazy_geometry.py`）。
  - `api` 側は、`src/api/__init__.py`, `src/api/shapes.py`, `src/api/effects.py`, `src/api/sketch.py` にそれぞれ高レベルな用途と設計意図が書かれており、ユーザー視点と内部構造の両方から理解できる。
- **命名と責務の粒度が揃っており、読む側の予想と実装が一致しやすい。**
  - `BaseRegistry.register/get/list_all/...`（`src/common/base_registry.py`）や、`effects.registry` / `shapes.registry` の API は対称で、初見でも挙動が予想しやすい。
  - `WorkerPool`, `StreamReceiver`, `SwapBuffer`, `LineRenderer` など、runtime/render 周りのクラス名が役割に直結している。
- **パラメータ処理と署名生成が共通化されており、可読性と一貫性に寄与している。**
  - `params_signature`, `quantize_params`, `signature_tuple`（`src/common/param_utils.py`）が shapes/effects/api 間で共通利用されており、「float のみ量子化 / int/bool はそのまま」というルールが 1 箇所に集約されている。
  - `__param_meta__` と docstring の同期ルールが AGENTS に明示されており、レビュー時にも整合性をチェックしやすい。

### 2.3 気になる点 / 改善アイデア（可読性）

- **例外を黙って握りつぶすパターンがいくつか存在し、デバッグ時の手掛かりが減る可能性がある。**
  - 例: `_GLOBAL_PIPELINES.add(self)` での例外を完全に無視（`src/api/effects.py:Pipeline.__post_init__`）。
  - 例: 各種キャッシュ初期化や設定読み込みで `except Exception: pass` 的な処理が散見される（`src/api/effects.py`, `src/engine/core/lazy_geometry.py`, `src/engine/render/renderer.py` 等）。
  - 方針として「フェイルソフト」が必要な箇所も多いが、ログレベル DEBUG かつ 1 行だけでも残しておくと、想定外の環境差異や設定ミスを追いやすくなる。
- **一部のモジュールは機能が多く、1 ファイル内での責務がやや広い。**
  - `src/api/sketch.py` はランナーとしての責務をよく果たしているが、ウィンドウ・ワーカー・MIDI・HUD・エクスポートなど多くのコンポーネントを組み合わせており、関数内のセクションも長い。
  - `src/shapes/text.py` も前述の通り、フォント検出・TTFont ロード・グリフ flatten・Geometry 生成までを単一ファイルで扱っている。
  - 現状でもコメントと docstring により読める範囲だが、将来的に機能追加が続く場合は「フォント I/O と Geometry 生成ヘルパ」の分割を検討すると見通しがさらに良くなりそう。
- **署名フィルタ関数の重複が、少しだけ読み手の混乱を招きうる。**
  - アーキテクチャ欄でも触れたが、`_filter_sig_params` と `_filter_sig_params_global` が同一モジュール内にあり、前者は `LazyGeometry.realize()` 内ローカル関数として定義されている。
  - 名前が似ており範囲も重なるため、「どちらが正式な仕様なのか」をコメントで補足するか、どちらかに統合すると読み手の負荷が下がる。

---

## 3. パフォーマンス

### 3.1 現状設計の特徴

- Geometry とパイプラインは **コピー最小化とベクトル化** を意識して実装されている。
  - `Geometry` は生成時に `float32` / `int32` / C 連続配列へ正規化しており、その後の変換メソッドは余計な dtype 変換を行わない（`src/engine/core/geometry.py`）。
  - `translate/scale/rotate/concat` は NumPy ベクトル化で実装されており、大量頂点でもループのオーバーヘッドが少ない。
- 主要なホットパスには **専用キャッシュと numba** が投入されている。
  - shape 生成 LRU + prefix LRU: `LazyGeometry`（`src/engine/core/lazy_geometry.py`）
  - effect パイプラインの事前解決 / compiled cache: `PipelineBuilder.build()` / `Pipeline`（`src/api/effects.py`）
  - indices LRU + IBO freeze 実験: `_geometry_to_vertices_indices`, `_INDICES_CACHE`（`src/engine/render/renderer.py`）
  - Perlin ノイズ系: `displace` エフェクトの `@njit` 化（`src/effects/displace.py`）と numba ベースの 3D 変換関数（`src/util/geom3d_ops.py`）。
- ランタイム側では、**WorkerPool + SwapBuffer + LineRenderer** による非同期パイプラインが構築されている。
  - CPU 側の `user_draw(t)` 実行は `WorkerPool`（`src/engine/runtime/worker.py`）に委譲され、メインスレッドはレンダリングと UI に専念する。
  - `WorkerPool.tick()` は Queue が詰まっている場合に `Full` を握りつぶしてスキップする設計で、描画ループ側の安定性を優先している。

### 3.2 良い点

- **Geometry 操作と GPU 転送の経路がシンプルで、ボトルネックが掴みやすい構造になっている。**
  - Geometry → VBO/IBO 変換は `_geometry_to_vertices_indices` に集約されており、1 パス O(N) の処理で indices を構築している（`src/engine/render/renderer.py`）。
  - offsets のハッシュによる indices LRU もローカルに完結しており、キー定義も `(primitive_restart_index, total_verts, hash(offsets))` と素直。
- **キャッシュ関連の設定は `common.settings` に集約されており、環境変数から一括制御できる。**
  - `PIPELINE_QUANT_STEP`, `SHAPE_CACHE_MAXSIZE`, `PREFIX_CACHE_MAXSIZE`, `INDICES_CACHE_MAXSIZE`, `COMPILED_CACHE_MAXSIZE` などが `src/common/settings.py` で管理されており、プロファイル結果に応じてチューニングしやすい。
  - 失敗時のフォールバックも `env_*` ヘルパ経由で行われており、異常な環境値に対する下限丸めが入っている。
- **Numba / numpy の適用範囲が明確で、純粋計算に限定されている。**
  - `util.geom3d_ops.transform_to_xy_plane/transform_back` は numba + float64 で精度と速度のバランスを取っている（`src/util/geom3d_ops.py`）。
  - `effects/displace.py` の Perlin ノイズ実装も numba 化されており、頂点数に比例した線形コストで処理される。

### 3.3 気になる点 / 改善アイデア（パフォーマンス）

静的レビュー時点では明確なボトルネックは見えないが、設計上のトレードオフとして以下のような点がある。

- **「常に新しい Geometry インスタンスを返す」方針によるコピーコスト。**
  - `Geometry.translate/scale/rotate/concat` は空ジオメトリや no-op（回転角 0 など）の場合でも、新しい `Geometry` を生成し `coords/offsets` をコピーしている（`src/engine/core/geometry.py`）。
  - 純粋関数としての一貫性とエイリアシング回避という観点では非常にわかりやすいが、大量のパイプラインを重ねるケースではコピーコストが効いてくる可能性がある。
  - 実測でボトルネックになるようであれば、「no-op 条件時のみインスタンスを再利用する」「LazyGeometry 内での prefix キャッシュをより積極的に使う」などの緩和策を検討する余地がある。
- **キャッシュの多重構造が、ワークロードによってはメモリ使用量や複雑さに影響しうる。**
  - shape LRU, prefix LRU, compiled cache, indices LRU がそれぞれ独立に存在するため、極端に大きなシーンや多種類のパラメータ組み合わせを扱う場合、合計メモリフットプリントが増える可能性がある。
  - 現状でも `*_MAXSIZE` が適切に設けられているため実用上問題になる場面は少ないと考えられるが、将来的にさらにキャッシュが増える場合は「キャッシュ種別ごとの役割と優先度」を短く整理しておくとチューニングがしやすくなる。
- **WorkerPool の Queue 深さと FPS の組み合わせは、実測前提での調整が必要。**
  - `WorkerPool` はタスク Queue の `maxsize` を `2 * num_workers` に固定しており、`Full` 発生時は静かにフレーム発行をスキップする設計になっている（`src/engine/runtime/worker.py`）。
  - 一般的には UI 側のフリーズを避けるための妥当な設計だが、FPS を高く設定したり `user_draw` が重い場合には「フレーム落ち」が起きる。
  - 実際のスケッチで問題になる場合は、HUD やログと組み合わせたチューニング（Queue 深さや worker 数の調整）が必要になる。

---

## 4. まとめ

- アーキテクチャ面では、`architecture.md` と AGENTS の設計方針が実装とよく同期しており、層間依存や責務分離が明快で、拡張・保守に強い構造になっている。
- 可読性は、中核モジュールの docstring / 型注釈 / 命名が整っており、特に Geometry / LazyGeometry / パイプライン / runtime/render 周りは初見でも把握しやすい。
- パフォーマンス面では、NumPy / numba / 各種 LRU キャッシュが要所に投入されており、設計として十分に「高性能を狙った上でのシンプルさ」が保たれている。
- 改善の余地があるとすれば、主に「キャッシュ関連のポリシーと署名フィルタの集約」「例外のフェイルソフト箇所でのデバッグ支援」「テキスト周りの責務分割」などであり、いずれも設計全体を崩さずに局所的に見直せる範囲に収まっている。
