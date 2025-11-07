# 251106: Effect Prefix Cache（中間結果 LRU）設計・導入計画（チェックリスト）

目的

- ユーザーコードを変更せず、エフェクトチェーンの「静的前段」までをフレーム間で再利用する。
- 例: `affine/scale/translate/fill/subdivide` までを 1 回だけ計算し、毎フレームは `displace(t)/mirror` 等の動的後段のみを再評価。

非機能目標

- 評価は Worker 側（サブプロセス/インライン）で完結させ、描画スレッドに戻さない。
- キャッシュは有界（LRU + サイズ閾値）で、メモリ使用を制御。

---

基本方針（高レベル）

- キー: `(shape_signature, prefix_steps_signature)` を中間結果（Geometry）キャッシュの鍵にする。
  - 量子化済みの `params_signature()` を用いて、各ステップの署名を決定的に生成。
  - 形状部は shape 名 + 量子化済みパラメータで署名。
  - prefix は plan の先頭から i 番目までの効果（名前 + 量子化済み params）列。
- 凍結点: `LazyGeometry.realize()` の評価ループで「各ステップの直後」に凍結。
  - 直前までの prefix がヒットしたら、その Geometry を起点に再開し、以降の残ステップのみ評価。
- 位置: `LazyGeometry.realize()` 内（Worker 側で実行されるため描画スレッドは軽いまま）。

キャッシュ対象の判定（署名変化ベース）

- 判定は t に特化せず、「量子化済みパラメータ署名（params_signature）」の変化で行う。
- 各フレームで plan の先頭から順にステップ署名を比較し、直前フレームと同一の連続区間を「最長一致プレフィックス」とみなす。
  - そのプレフィックスの中間結果 Geometry を LRU から取得できれば HIT、以降の残ステップのみ評価する。
  - プレフィックスが短く（または 0）なった場合は、その位置から再評価。
- 制限（保存側の基準）:
  - Geometry の頂点数が `PXD_PREFIX_CACHE_MAX_VERTS` を超える場合は保存しない（メモリ保護）。
  - エフェクト実装が明示的に「非キャッシュ」を宣言した場合（`__effect_cacheable__ = False` を将来導入）も対象外。

LRU ポリシー

- プロセス内 LRU（`OrderedDict`）。
- 代表設定（環境変数で上書き可）:
  - `PXD_PREFIX_CACHE_MAXSIZE`（既定 128）: 登録件数の上限。
  - `PXD_PREFIX_CACHE_MAX_VERTS`（既定 10_000_000）: 保存対象の頂点数上限。
  - `PXD_PREFIX_CACHE_ENABLED`（既定 1）: 有効/無効。
- Eviction は FIFO（`popitem(last=False)`）。

キー生成の詳細

- shape: `shape_name` + `params_signature(shape_impl, params_dict)`
- prefix: 計算済み `[(effect_name, params_signature(effect_impl, params_dict)), ...]` の先頭 i 要素
- キー形式（例）:
  - `("prefix", blake2b-128(shape_sig + step_sig_0 + ... + step_sig_i))`
- 量子化: 既存の規約（float のみ量子化、ベクトルは成分ごと、既定 step=1e-6 / `PXD_PIPELINE_QUANT_STEP`）を使用。

計測/可観測性

- グローバルカウンタ（プロセス内集計）: `hits`, `misses`, `stores`, `evicts`。
- デバッグ出力（環境変数で制御）: `PXD_DEBUG_PREFIX_CACHE=1` のときに HIT/MISS を 1 行ログ。
- HUD 拡張（Phase 2）: 既存の `global_cache_counters()` に prefix-cache の集計を加える。

スコープ/非スコープ

- スコープ:
  - `LazyGeometry.realize()` に prefix LRU 実装を追加（Worker 側で機能）。
  - 既存 API/ユーザーコードの変更は不要。
- 非スコープ（今回はやらない）:
  - 効果の合成/最適化（例: アフィン合成）。
  - 分散共有（プロセス間でのキャッシュ共有）。
  - 署名の高速化（メモ化）や、データ軽量化のための圧縮。

既存実装との整合

- 形状 LRU（実装済み）との相互作用: shape 実体化の負荷は既に低減。prefix LRU はその後段を対象。
- Worker 側実体化（実装済み）: prefix LRU は Worker 内に持つためメインスレッドへの回帰なし。
- HUD: 現行では Renderer の実測計数を参照しているため、prefix LRU 追加に伴う変更は不要（Phase 2 で計数追加のみ）。

リスクと対策

- メモリ使用増: `MAXSIZE`/`MAX_VERTS` の二重制限で抑制。HUD/ログで観測。
- 誤キャッシュ（外部状態依存）: 効果は純関数（外部状態直参照禁止）。cc/t は Runtime により引数として解決され、署名に反映される設計にする。
- スラッシング: GUI で前段パラメータが頻繁に変わる場合、ミスが増える。量子化ステップと LRU に任せ、必要に応じしきい値/サイズ調整。

段階導入（タスク）

- Phase 1（最小実装）
  - [x] `LazyGeometry.realize()` に prefix LRU 実装（キー生成・HIT→ スキップ・MISS→ 保存）。
  - [x] 環境変数による制御（ENABLED/MAXSIZE/MAX_VERTS/DEBUG）。
  - [x] HIT/MISS/STORE/EVICT のカウンタ実装（プロセス内）。
  - [x] しきい値: `MAX_VERTS=10_000_000`（既定）で保存抑制。
- Phase 2（堅牢化）
  - [ ] 効果実装に `__effect_cacheable__`（明示非キャッシュ）を導入（任意、既定は自動判定）。
  - [ ] `api.effects.global_cache_counters()` に prefix-cache の値を追加して HUD 連携。
  - [ ] 基本スモーク/負荷テスト（HIT 率、計算/IPC/描画時間の比較）。
- Phase 3（高度化）
  - [ ] 最長静的プレフィックス検出に最適化（1 plan につき 1 本だけ保持）。
  - [ ] 署名生成のメモ化（効果 impl ごとに param->signature キャッシュ）。

受け入れ基準（DoD）

- [ ] `displace(t)` の前段（`affine/translate/scale/fill/subdivide`）が一定のフレームで `HIT` し、前段の計算時間がフレーム間でほぼゼロになる。
- [ ] Worker 側実体化ログに `HIT` が出る（Renderer 側からの realize 呼び出しは存在しない）。
- [ ] LRU が上限に達した際に `evicts` が増加し、落ちずに動作継続。

影響ファイル（予定）

- `src/engine/core/lazy_geometry.py`（prefix LRU 本体: 生成/参照/保存/計測）
- `src/api/lazy_signature.py`（必要なら補助: 部分的な plan 署名の共通化）
- `src/api/effects.py`（HUD 集計に prefix-cache を追加する場合のみ）

運用/設定

- 既定は有効（ENABLED=1, MAXSIZE=128, MAX_VERTS=10_000_000）。
- 大規模データ検証時は `PXD_PREFIX_CACHE_ENABLED=0` で容易に切替可能。

デバッグ/ログ

- `PXD_DEBUG_PREFIX_CACHE=1` のとき: `HIT/MISS (store)/EVICT` を 1 行ログで出力（Worker 側）。

備考

- 初期から「署名変化検出」に一本化。t に特化した除外ルールは使用しない。

---

承認いただければ Phase 1 を実装し、計測ログと HUD（オプション）で効果を確認します。
