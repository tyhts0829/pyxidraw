# 251106: PrefixCache 動的末尾の保存抑制ガード（設計・実装計画）

課題
- PrefixCache は「最長一致プレフィックス」を HIT できているが、HIT より後ろ（動的側）のステップ出力を毎フレーム STORE して即座に EVICT するスラッシングが発生。
- 末尾は `t` 等で毎フレーム署名が変わるため、保存しても再利用されず、辞書操作/メモリ確保/GC のオーバーヘッドだけが残る。

目的
- 「HIT より後ろ（動的側）」の中間結果を保存しない（=STORE を抑制）ことで、スラッシングを解消。
- 既存の HIT 旨味（静的前段の再利用）は維持。

方針（コアルール）
- 長さ L のプレフィックスが HIT したフレームでは、長さ > L のプレフィックス（=動的側）の STORE を禁止する（既定）。
- 初回や MISS（L=0）のフレームでは、計算中に得られる各段のプレフィックスを保存してプリウォームする。

詳細設計
- 変数定義
  - `prefix_hit_len = L`（本フレームでの最長一致プレフィックス長）
  - `i` は 1..N のプレフィックス長（N=総ステップ数）
- 保存条件
  - if `L > 0`: STORE するのは `i <= L` のみ。ただし本フレームは `i <= L` を再評価していないため、通常 STORE なし（=実質全スキップ）。
  - if `L == 0`: MISS プリウォームとして `i <= STORE_ON_MISS_UP_TO` の範囲のみ STORE（上限でメモリ/時間を制御）。
- 例（steps=7, L=5 のケース）
  - 既存: i=6,7 を毎回 STORE → 直後に EVICT（キーが毎回変わる）
  - 変更後: i>5 の STORE を禁止 → STORE 無し、HIT だけ活かす

設定/制御（環境変数）
- `PXD_PREFIX_CACHE_STORE_TAIL`（既定: 0）
  - 0: L>0 のフレームは i>L を STORE しない（推奨）
  - 1: 従来互換（すべて STORE、非推奨）
- `PXD_PREFIX_CACHE_STORE_ON_MISS_UP_TO`（既定: 16）
  - L==0 のとき、プリウォームで STORE する最大プレフィックス長（無制限は <=0）

可観測性（デバッグ）
- カウンタ追加: `tail_skipped`（i>L に対する STORE 抑止回数）
- ログ（`PXD_DEBUG_PREFIX_CACHE=1`）
  - `HIT i=… / MISS …`（既存）
  - `SKIP-TAIL i=… (>L=…)`（追加）
  - `STORE i=…` / `EVICT`（既存）

実装ステップ（最小）
- [x] `LazyGeometry.realize()` 内の STORE 条件に `i <= prefix_hit_len` を加える（`PXD_PREFIX_CACHE_STORE_TAIL=0` の場合のみ有効）。
- [x] MISS 時のプリウォーム: `i <= PXD_PREFIX_CACHE_STORE_ON_MISS_UP_TO` を満たすときのみ STORE。
- [x] カウンタ `tail_skipped` とデバッグログ `SKIP-TAIL` を追加。
- [x] 既定値/ENV の取得とテスト（軽い自己診断ログ）。

将来拡張（任意）
- プレフィックスの「次フレーム安定性」を軽量に推定し、`i==L+1` だけ例外的に保存（改善余地）。
- HUD へ prefix-cache メトリクス（hits/misses/stores/evicts/tail_skipped）を追加。

DoD（受け入れ基準）
- [ ] `PXD_DEBUG_PREFIX_CACHE=1` で、HIT フレームにおける `STORE` が消え、代わりに `SKIP-TAIL` が観測できる。
- [ ] Evict の連発が解消（少なくとも 10× 以上の削減）。
- [ ] 体感/計測でフレーム時間が改善し、Renderer 側は引き続き実体化ゼロ。

影響範囲
- `src/engine/core/lazy_geometry.py`（STORE 条件の分岐追加、ENV 取得、カウンタ/ログ）

リスク/回避
- キャッシュウォームまでの 1 フレームだけ若干重いが、以降は HIT で相殺。
- 互換: `PXD_PREFIX_CACHE_STORE_TAIL=1` で従来挙動に戻せる。

備考
- 「末尾は毎フレーム署名が変わる」前提の設計。GUI 等で末尾も安定するケースは `STORE_TAIL=1` によってオプトイン可能。
