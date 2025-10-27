# Pipeline 中間結果キャッシュ（Prefix LRU）実装計画

どこで: `src/api/effects.py`（Pipeline 実行・キャッシュ層）、`src/api/sketch_runner/utils.py`（HUD メトリクス）
何を: パイプライン内の中間結果（各ステップ適用後の `Geometry`）を、入力ジオメトリとパイプラインの共通接頭辞（prefix）単位でキャッシュして再計算を削減する。
なぜ: 現状は終端結果のみをキャッシュするため、後段ステップのパラメータ変更でも前段を再計算してしまう。共通接頭辞の再利用でフレーム遷移や GUI 操作時の応答性を向上する。

## スコープと非目標
- スコープ: Pipeline 実行時の中間結果キャッシュ（メモリ内 LRU）。終端結果キャッシュは現状維持。
- 非目標: ディスク永続化、分散キャッシュ、効果関数の自動融合やアルゴリズム最適化。

## 要件（機能/非機能）
- 正しさ: 入力ジオメトリ `g.digest` と量子化後パラメータから導出する「prefix キー」に完全依存し、不正な再利用をしない。
- 再利用: 最長一致する接頭辞が見つかれば、以降のステップのみを実行して終端へ到達する。
- メモリ上限: LRU で上限制御（env と API で設定）。無効化可能。
- 低侵襲: 既存の終端結果キャッシュと HUD 表示を壊さない。段階導入が可能。
- スレッド安全: 既存と同様 RLock による保護。

## 設計概要
- 新規: グローバル「ステップ（prefix）キャッシュ」
  - 形: `OrderedDict[(bytes16 geometry_digest, bytes16 prefix_key) -> Geometry]`
  - `prefix_key`: 既存の `pipeline_key` と同一の計算規則で、ステップ i までの累積ハッシュ（blake2b 16B）。
  - LRU 管理: `PXD_PIPELINE_STEP_CACHE_MAXSIZE`（env）で上限、`0` で無効、`None` で無制限。
  - 統計: `step_hits/step_misses/step_evicts` を保持。HUD はオプションで表示拡張。
- `_CompiledPipeline` 拡張
  - 初期化時に各ステップの `prefix_key[i]` を前方累積で計算・保持。
  - `__call__` フロー:
    1) 現行の終端キャッシュ（(g.digest, pipeline_key)）を先に参照（完全ヒット高速化）。
    2) ミス時、`i = n-1..0` の降順で `(g.digest, prefix_key[i])` を走査し、最長一致の中間結果を取り出す。
    3) 見つかった i から後続ステップのみ実行。実行の都度 `(g.digest, prefix_key[k])` に中間結果を保存（有効時）。
    4) 最後に終端結果を既存キャッシュへ保存。
  - フォールバック: 中間キャッシュ無効時は現行実装と同等（全段実行）。
- API（任意拡張）
  - `PipelineBuilder.intermediate_cache(maxsize: int | None)` を追加（ビルダー単位で上限/無効を上書き）。未指定は env 既定に従う。

## 互換性とデフォルト
- 既定（安全側）: `PXD_PIPELINE_STEP_CACHE_MAXSIZE` 未設定時は `0`（無効）。メモリインパクトを抑え、明示有効化で段階導入。
- 既存 API/既存挙動は維持。終端キャッシュと `cache_info()` は破壊しない。

## 実装タスク（チェックリスト）

- [ ] 型/定数の追加（`src/api/effects.py`）
  - [ ] `_GLOBAL_STEP_CACHE: OrderedDict[tuple[bytes, bytes], Geometry]` と `_STEP_LOCK` を追加
  - [ ] `_STEP_MAXSIZE` を env `PXD_PIPELINE_STEP_CACHE_MAXSIZE` から解決（負値は 0 として扱う）
  - [ ] 統計カウンタ `step_hits/step_misses/step_evicts` を追加

- [ ] `_CompiledPipeline` の拡張
  - [ ] `__init__`: `self._prefix_keys: list[bytes]` を計算（各ステップ追加ごとに blake2b へ `name`, `fn_version`, `params_digest` を順次 update し `digest()` を保存）
  - [ ] `__call__`: 終端キャッシュのミス時に prefix 走査→最長一致を取得→後続のみ実行→各段でステップキャッシュへ保存
  - [ ] ステップキャッシュ無効時は保存/走査をスキップ

- [ ] PipelineBuilder/API の拡張
  - [ ] `intermediate_cache(maxsize: int | None)` を追加し、`Pipeline` → `_CompiledPipeline` へ有効/上限を伝搬

- [ ] HUD/メトリクス（任意・小さく）
  - [ ] `global_cache_counters()` を拡張し `step_hits/step_misses` を含めて返す（キーは後方互換のため追加のみ）
  - [ ] `api/sketch_runner/utils.py:hud_metrics_snapshot()` で effect セクションに `step_hits/step_misses` を統合
  - [ ] Worker の HIT/MISS 判定は現状維持（累計差分方式）。必要なら将来 `effect_step` を別フラグに分離

- [ ] テスト
  - [ ] 単体: 疑似エフェクト（副作用カウンタ付）を 3 段 pipeline に組み、後段のみパラメータ変更時に前段のカウンタが増えないことを確認
  - [ ] 回帰: 終端キャッシュ HIT/MISS の統計が従来と整合すること
  - [ ] 無効時: `PXD_PIPELINE_STEP_CACHE_MAXSIZE=0` で現行挙動と等価（前段も再実行）

- [ ] ドキュメント/設計同期
  - [ ] `architecture.md` に「エフェクトキャッシュ（終端/中間）」の節を追加し、キー構成・上限・既定値を明記
  - [ ] ルート `AGENTS.md` の最小項目（Build/Test/Style/Safety/PR）へ本機能の env とチェックを追加

## 疑似コード（`_CompiledPipeline.__call__` 差分イメージ）
```
# 1) 完全一致（終端）
key_final = (g_digest, self._pipeline_key)
if key_final in final_cache: return hit

# 2) 最長 prefix 探索（降順）
best_i = -1; out = g
for i in reversed(range(n)):
    k = (g_digest, self._prefix_keys[i])
    if k in step_cache: out = step_cache[k]; best_i = i; break

# 3) best_i+1 から末尾まで実行（保存しながら）
for j in range(best_i+1, n):
    out = fn_j(out, **params_j)
    if step_cache_enabled: step_cache[(g_digest, self._prefix_keys[j])] = out

# 4) 終端保存
final_cache[key_final] = out
return out
```

## リスクと軽減策
- メモリ使用量の増加: 既定を無効化（0）し、API/env で opt-in。LRU で上限超過時に追い出し。
- 競合/ロック粒度: グローバル step-cache は RLock で短時間保持（pop→push）。必要に応じて分割（将来）。
- キー衝突: 16B blake2b の実用上リスクは低い。必要なら拡張余地あり。

## 検証計画
- ベンチ: 既存 `tests/perf/test_pipeline_perf.py` に「後段のみ変更」のケースを追加し、ミス時も CPU 時間が改善することを確認。
- 機能: 単体テストでカウンタにより前段再実行が抑制されることを確認。

## 完了条件（DoD）
- 変更ファイルに対して `ruff/black/isort/mypy/pytest -q -k pipeline` 緑。
- `architecture.md` と `AGENTS.md` 更新済み（差分ゼロ前提はなし）。
- 既存 HUD 表示は後方互換で動作（step 統計は追加情報）。

---
確認事項（要回答）
- 既定値: 中間キャッシュは「無効（0）」で開始し、必要プロジェクトのみ有効化でよいか？
- API: `intermediate_cache(maxsize)` を追加する方針で問題ないか？（環境変数のみでも可）
- HUD: まずは統計だけ追加（表示は現状維持）でよいか？

上記で問題なければ、実装に着手します。

