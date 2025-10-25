HUD: p0 のキャッシュ表示が誤って「効いている」になる件（原因調査）

概要
- 症状: 同一フレームで `p0` のパイプラインはキャッシュ無効（MISS）なのに、HUD の「CACHE/EFFECT」表示が「HIT」になる。
- 対象: 複数パイプライン（例: `p0`, `p1`）を同一 `draw()` 内で実行するスケッチ（`sketch/251022.py` など）。

結論（原因）
1) HUD のエフェクトキャッシュ判定は「フレーム内に1つでもエフェクトキャッシュの HIT があったか」を見るグローバル集計であり、パイプライン別ではない。
   - Worker 側でフレーム前後の累計ヒット数（全エフェクト合算）を比較し、増えていれば `HIT` とする実装。
   - src/engine/runtime/worker.py:236
2) エフェクトのキャッシュ統計は「コンパイル済みパイプライン」単位でグローバル集約されており、`p0` と `p1` が同一構成なら同じコンパイル体を共有する。
   - 共有カウンタにより、`p1` 側でヒットしてもフレーム集計としては `HIT` 扱いになる。
   - src/api/effects.py:200 付近（`global_cache_counters()` は全 compiled pipelines の `_hits/_misses` を合計）

作用の流れ（コード参照）
- HUD メトリクス取得関数
  - `hud_metrics_snapshot()` は shapes/effects のキャッシュ統計を取得（spawn 安全なトップレベル関数）。
    - src/api/sketch_runner/utils.py:76
  - effects 側は `global_cache_counters()` を呼び、compiled pipelines 全体の `hits/misses` 累計を返す。
    - src/api/sketch_runner/utils.py:88
- Worker 側でのフラグ決定
  - 描画前に `before = hud_metrics_snapshot()`、描画後に `after = hud_metrics_snapshot()` を取得し、
    `after.effect.hits > before.effect.hits` を満たすと `CACHE/EFFECT = HIT` にする（MISS 増分は考慮しない）。
    - src/engine/runtime/worker.py:236-255
- コンパイル済みパイプラインとカウンタ
  - `Pipeline._ensure_compiled()` は、同じ実行署名（ステップ名列＋量子化済みパラメータ列）のパイプライン間で compiled をグローバルに共有する。
    - src/api/effects.py:212-241
  - `global_cache_counters()` は全 compiled の `_hits/_misses` を合計する（パイプライン別ではない）。
    - src/api/effects.py:164-196

結果として起きること
- `p0` が MISS、`p1` が HIT のフレームでは、フレーム後の `hits` 増分が 0→1 となるため HUD は `HIT` を表示。
- HUD の「CACHE/EFFECT」は「フレーム内のいずれかのエフェクトでヒットがあったか」を表し、
  個別パイプライン（`p0`/`p1`）のヒット/ミスを判定しない。

補足（形状側）
- shapes は `functools.lru_cache` の単一グローバルに対する累計で、同様にフレーム内の「いずれかの HIT」を見ている。
  - src/api/shapes.py:239-247

改善の方向性（提案のみ、未実装）
- パイプライン別のインジケータを出すには、以下のいずれかが必要:
  1) Pipeline.__call__ 内で compiled.cache_info() の差分を計測し、`RenderPacket.cache_flags` に `effect@{uid}` 単位で埋め込む。
     - Worker 側の `flags` を dict で拡張し、HUD へそのまま渡す（現状は `{"shape": ..., "effect": ...}`）。
  2) `global_cache_counters()` を `pipeline_uid` ごとのカウンタへ拡張し、`hud_metrics_snapshot()` が dict[pipeline_uid] を返すよう仕様変更。
     - ただし compiled の共有設計（同一署名でパイプラインをまたいで再利用）と相反するため、集計のキー設計が課題。
  3) 現状仕様のままラベルを「FRAME/EFFECT HIT（any）」のように明示し、誤解を減らす（表示上の対処）。

備考
- 現在のインジケータは「フレーム全体での any-hit」を示すため、`p0` 単独の MISS を見分ける用途には不向き。
- パイプライン UID 導入（`effect@{uid}`）は GUI 側の識別・上書き分離に寄与するが、HUD の集計設計は別問題。

以上。

---

変更方針と実装計画（HIT/MISS の優先順位修正）

要件
- フレーム内に1つでも MISS があれば、HUD の判定は MISS（HIT より MISS を優先）。
- 形状（shapes）・エフェクト（effects）ともに同様の判定に統一。

設計
- 現状の Worker 側ロジックは「hits の増分のみ」を見て HIT/MISS を決めている。
- これを「misses の増分を先に確認し、増えていれば MISS。増えていなければ hits の増分があれば HIT。どちらも増えなければ MISS」とする。
  - 疑似コード:
    - `shape_miss = after.shape.misses > before.shape.misses`
    - `shape_hit  = after.shape.hits    > before.shape.hits`
    - `shape_flag = "MISS" if shape_miss else ("HIT" if shape_hit else "MISS")`
    - effects も同様に `effect_flag` を算出。

変更箇所（両方の経路を修正）
- src/engine/runtime/worker.py（インライン実行経路）
  - 現状: 236-255 行付近
    - `s_hit = after.shape.hits > before.shape.hits`
    - `e_hit = after.effect.hits > before.effect.hits`
  - 変更: misses 差分を評価し、MISS を優先。
- src/engine/runtime/worker.py の `_WorkerProcess.run()`（サブプロセス実行経路）
  - 同様の前後差分ロジックがあるため、同じ変更を適用。

テスト計画
- 単体テスト（擬似スナップショットでの判定）
  - before: {hits:0, misses:0} → after: {hits:1, misses:1} の場合 → 期待: MISS。
  - before: {hits:0, misses:0} → after: {hits:1, misses:0} の場合 → 期待: HIT。
  - before: {hits:5, misses:2} → after: {hits:5, misses:2} の場合 → 期待: MISS（変化なし）。
- 統合テスト（任意）
  - キャッシュONで1フレーム内に「同一 compiled へ MISS→HIT」が混在するケースを作り、HUD の flags が MISS になることを確認。

影響範囲/互換性
- HUD の「CACHE/SHAPE」「CACHE/EFFECT」の定義が、これまでの any-hit から any-miss-then-miss に変わる（より厳密）。
- 既存の表示に対し MISS が増える可能性があるが、要件に合致。

実装所要
- 片道10行程度の条件分岐の追加で、依存/インタフェースには影響しない（メトリクス取得関数は既に misses を返す）。

