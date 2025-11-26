# worker.py 実装コードレビュー & 改善計画

対象: `src/engine/runtime/worker.py`（描画タスク実行ワーカ／ワーカープール）

## レビュー結果

### 良い点

1. **責務の分離と構造が素直**

   - `WorkerPool` が「タスク生成とワーカープール管理」、`_WorkerProcess` が「子プロセス側でのタスク処理」、`_execute_draw_to_packet` が「1 フレーム分の draw パイプライン」のみを担当しており、役割の境界が明確。
   - インライン実行とマルチプロセス実行の切り替えも `num_workers < 1` という単純な条件で扱われており、呼び出し側 API がシンプル。

2. **マルチプロセスを意識した例外・シリアライズ設計**

   - `WorkerTaskError` がフレーム ID と元例外を保持しつつ、`__reduce__` を実装して「メッセージのみ」で復元できるようにしており、spawn ベースの `multiprocessing` でも扱いやすい。
   - 子プロセスでは `_execute_draw_to_packet` の戻り値をそのまま Queue に流すだけにし、親側で例外を一元的に扱える構成になっている。

3. **draw 戻り値の柔軟な受け入れが 1 箇所に集約されている**

   - `_normalize_to_layers` で `Geometry` / `LazyGeometry` / `Layer` / それらの `Sequence` を一括で受け、`geometry` と `layers` に正規化しているため、`draw_callback` の実装自由度とワーカー側の責務分離のバランスがよい。
   - `_apply_layer_overrides` により、パラメータ GUI からのレイヤー単位 override を Worker 側に閉じ込めている。

4. **キューとクローズ処理の扱いが冪等で安全**

   - `WorkerPool.close()` が `_closed` フラグで多重呼び出しを許容しつつ、`None` sentinel でワーカーに停止を伝える構造になっている。
   - inline 実行時は `queue.Queue` を使ってシリアライズを避けるなど、パフォーマンスとシンプルさのバランスが取れている。

### 改善したい点（美しさ・可読性・分割）

1. **色正規化ロジックが分散・重複しており、挙動が分かりにくい**

   - `_normalize_to_layers` 内の `_to_rgba` と `_apply_layer_overrides` 内の `_norm_color` 呼び出しで、似たような色正規化処理が二重に存在している。
   - `util.color.normalize_color` 例外を握りつぶしてフォールバックに落とすため、異常な色指定が静かに無視される（`layer_code_review.md` で指摘済みのパターンと同系）。
   - 色の扱いを一箇所に集約し、「どこで例外を許容するか/伝播させるか」が読み取りづらい。

2. **`_execute_draw_to_packet` が 1 関数で多責務になっている**

   - CC/パラメータスナップショット適用、メトリクスの before/after 取得、`draw_callback` 実行と結果正規化、レイヤー override、メトリクス差分からのフラグ計算、`RenderPacket` 生成までを 1 関数で行っており、行数とネストが多い。
   - メトリクス差分部分（hits/misses 比較）は一読で意図を理解しづらく、テストしにくい塊になっている。
   - 例外捕捉範囲が広く、「どこからの例外が `WorkerTaskError` に包まれているのか」がやや曖昧。

3. **`WorkerPool` のインターフェースと内部状態の関係がやや不透明**

   - `fps` を受け取って `_fps` に保持しているものの、`tick()` 内では `_elapsed_time` インクリメントにしか使っておらず、フレーム生成レートの制御には利用されていない。
   - `cc_snapshot` は型注釈がなく、`RenderTask` に渡される `cc_state` の形がコードからは読み取りにくい（docstring か型で明示されていると安心）。
   - inline / multiprocess モードで `_result_q` の型が変わる（`queue.Queue` vs `mp.Queue`）ため、利用側から見たインターフェースの期待値が少し分かりづらい。

4. **例外の握り方が少し防御的で、AGENTS の方針とギャップがある**

   - スナップショット適用やメトリクス取得の周辺で `try/except Exception: pass` が多用されており、開発時に異常が観測しづらい。
   - 「ユーザーの draw() からの例外は `WorkerTaskError` に包む」という明確な方針がある一方で、内部の補助的処理まで沈黙させているため、ログの出方に一貫性が欠ける。
   - AGENTS の「必要十分な堅牢さ・過度に防御的にしない」に対して、もう少しシンプルな例外処理ポリシーに寄せられそう。

5. **小さなスタイル・型注釈上の改善余地**

   - `WorkerTaskError.__init__` の `frame_id` は `int | None` と `str` の両用途を担っており、署名と実際の使われ方に少しギャップがある（Unpickle 経路用のパスが読み取りづらい）。
   - `_DEBUG` など未使用のローカル定数が残っており、読み手を少しだけ迷わせる。
   - `WorkerPool.result_q` の戻り値の型が `Queue | mp.Queue` のユニオンになっていて、IDE/型チェッカが補完しにくい（薄いラッパーや type alias で隠すとすっきりする可能性がある）。

## 実装改善計画（チェックリスト）

### 1. 色・レイヤー正規化ロジックの整理

- [ ] `_normalize_to_layers` から色変換部分を抽出し、ファイルローカルヘルパー（例: `_normalize_rgba(value) -> tuple[float, float, float, float]`）に統合する。
- [ ] `_apply_layer_overrides` でも同じ `_normalize_rgba` を利用するように変更し、色正規化の挙動を 1 箇所に集約する。
- [ ] `util.color.normalize_color` の例外扱いを `layer_code_review.md` / `layer_fixes_plan.md` の方針に合わせて見直し、必要以上に黙殺しない（例: そのまま例外を上げるか、最低限のログを出す）。

### 2. `_execute_draw_to_packet` の分割と単純化

- [ ] CC/パラメータスナップショットの適用とクリーンアップを、小さなヘルパー（例: `_apply_snapshots()` / `_clear_param_runtime()`）に切り出し、`_execute_draw_to_packet` 本体からネストと `try/except` を減らす。
- [ ] メトリクス before/after の取得と差分からのフラグ生成を、別ヘルパー（例: `_compute_cache_flags(before, after) -> dict[str, str] | None`）に分離し、テストしやすくする。
- [ ] `_execute_draw_to_packet` は「draw 実行 → 結果正規化 → レイヤー override → `RenderPacket` 生成」に集中させ、例外ハンドリングも `WorkerTaskError` 生成のみに絞る。

### 3. `WorkerPool` インターフェースの明確化

- [ ] `WorkerPool` から `fps` 引数と内部フィールドを削除し、「時間（`dt/t`）は呼び出し側フレームクロックが完全に管理する」という前提に実装と docstring を揃える。
- [ ] `cc_snapshot` / `param_snapshot` / `metrics_snapshot` などの引数に明示的な型注釈を付け、返り値の構造をコードから把握しやすくする。
- [ ] `result_q` の戻り値をラップする薄いアクセサ（もしくは type alias）を導入し、inline/マルチプロセスでの実体差を隠蔽して API を単純にする。

### 4. 例外処理ポリシーの整理

- [ ] スナップショット適用・メトリクス取得部分の `try/except Exception: pass` を棚卸しし、「本当に黙殺したいもの」と「ログに残すべきもの」を分類する。
- [ ] 黙殺しないと決めた箇所には `logging.debug` / `logging.warning` など最小限のログを追加し、`WorkerTaskError` でラップされる経路とそうでない経路を docstring などで明示する。
- [ ] `_WorkerProcess.run` / inline 実行パスの例外経路が同じ振る舞いになるように確認し、差分があれば揃える。

### 5. 型注釈・スタイルの微修正

- [ ] `WorkerTaskError.__init__` のシグネチャを実際の利用パターンに合わせて整理する（例: `frame_id: int | None` と `message: str | None` に分離し、Unpickle 経路用のクラスメソッドを用意するなど）。
- [ ] 未使用のローカル定数（`_DEBUG` など）や不要な `getattr(..., default)` 呼び出しを削除し、読みやすさを優先する。
- [ ] 内部関数やヘルパーの引数に足りていない型注釈があれば補い、`mypy` の `no-untyped-def` 回避のための例外ルールを減らす。

### 6. 動作確認・テスト

- [ ] `worker.py` のリファクタ後に、`WorkerPool` を利用するパス（`sketch.py` など）で簡単なスケッチを実行し、inline/マルチプロセス両方で描画が成立することを確認する。
- [ ] 可能であれば `_execute_draw_to_packet` と `_compute_cache_flags`（導入した場合）に対する小さなユニットテストを追加し、メトリクス差分やレイヤー override の挙動が変わっていないことを検証する。
- [ ] 変更対象ファイルに対して `ruff` / `mypy` を実行し、スタイル・型チェックを通す。

## 確認したいこと

- 色正規化エラーの扱いについて:
  - `layer_code_review.md` の方針どおり、「異常な色指定は例外として検知できる」実装（ただし Worker 経由では `WorkerTaskError` に包む）に寄せてよいかどうかを確認したいです。；それでいいよ、
- inline 実行とマルチプロセス実行の優先度:
  - デフォルトでどちらを推奨するか（例: macOS の spawn 問題を避けるため inline を優先する等）に合わせて、API や docstring のトーンを微調整してもよいかを相談したいです。；マルチプロセス実効がデフォルト。ユーザーは主にそちらを使うからね。それでトーン調整していいよ。

---

メモ:
- 時間軸はフレームクロックが決定し、`WorkerPool` は「渡された `dt` を積算して `t` をタスクに埋め込むだけ」の立場にする。
- `WorkerPool` の API から `fps` を取り除き、「マルチプロセス実行がデフォルト」という前提で docstring やコメントのトーンを整える。
