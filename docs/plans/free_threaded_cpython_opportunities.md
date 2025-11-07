# 251106: Free‑Threaded CPython（GIL 無効ビルド）を活用したシステムレベル最適化計画

目的
- GIL 無効（free‑threaded）版の CPython（例: 3.13/3.14 系の `--disable-gil`／`PYTHON_GIL=0` 起動）で、現行の multiprocess 依存を減らし、スレッド並列とゼロコピー共有でフレーム時間を短縮する。
- 個別エフェクトの実装最適化ではなく、システム全体のアーキテクチャ改善で恩恵を最大化する。

前提と制約
- OpenGL/ModernGL の文脈は “メインスレッド専有” を維持（GL API は一般にスレッド非安全）。GPU 操作は主スレッドに限定。
- Numpy/BLAS は既に多くのケースで GIL を解放しているが、「Python 側の段取り（署名計算/計画/配列管理）」は free‑threaded で並列化の恩恵が出やすい。
- 互換性: GIL 有効ビルドでは現行動作（multiprocess）を維持し、free‑threaded 環境では thread‑first パスを優先する二重系で設計。

---

適用候補（優先度順）

1) WorkerPool の thread‑first 化（ゼロコピー + グローバル共有）
- ねらい: multiprocess の IPC/シリアライズを回避し、`Geometry`/キャッシュ/LRU をスレッド間で共有。
- 方式:
  - `ThreadWorkerPool` を追加し、free‑threaded 検出時は thread 実行に切替（ENV/自動判定）。
  - `SwapBuffer` は SPSC（single‑producer single‑consumer）の lock‑light 実装に差し替え（既存 API は維持）。
- 効果:
  - IPC/ピクルのゼロ化、キャッシュの統合（_SHAPE/_PREFIX/indices LRU/_GLOBAL_COMPILED などが共有）。
- リスク/対策:
  - 共有 LRU/辞書にロックが必要。シャーディング（分割ロック）か RCU 風に統計のみ lock‑free 更新を採用。

2) 署名計算・参照解決のスレッド並列化 + メモ化
- ねらい: `params_signature` とレジストリ参照（shape/effect 解決）をスレッドで先回り計算し、PrefixCache の探索準備を短縮。
- 方式:
  - `ThreadPoolExecutor` で「次フレーム候補」のステップ署名をプリフェッチ。
  - 小型 LRU（関数 ID + 正規化後 dict → 署名タプル）で命中時は O(1)。
- 効果:
  - 長い plan、GUI 安定時に前処理 CPU を削減。
- リスク/対策:
  - 過剰プリフェッチの無駄を LRU サイズ/頻度で抑制。GIL 無効なら小コストで並列化効果が得やすい。

3) 効果パイプラインの段階並列（Frame‐N Pipeline）
- ねらい: 1 フレーム内の順序依存は維持しつつ、「前フレームの後段」と「次フレームの前段」を重ねる（stage 並列）。
- 方式:
  - ステージ: [A] 署名/計画 → [B] 前段（静的）→ [C] 動的後段 → [D] indices → [E] GPU。
  - スレッドパイプラインでステージを跨いでオーバーラップ（GL は E のみメイン）。
- 効果:
  - 1 フレーム中の CPU 峰を分散。平均/分散（p95）が縮む。
- リスク/対策:
  - フレーム越し共有（Geometry の寿命管理）を厳密化。参照カウント/世代タグで破棄安全を担保。

4) LRU/カウンタのスレッド安全化と競合最小化
- ねらい: free‑threaded で複数スレッドが同一 LRU を叩く競合を緩和。
- 方式:
  - Sharded LRU（キーのハッシュで分割）+ shard 毎の Lock。統計は集約時に合算。
  - 読取専用パスは RCU 的に lock‑free、書込（挿入/evict）のみ lock。
- 効果: 複数スレッドが同時に cache を参照するワークロードでスケールする。
- リスク: 実装複雑度は中。既存 OrderedDict ベースからの置換は段階導入にする。

5) メモリアロケータ/バッファのスレッド向け最適化
- ねらい: 大規模配列の多重アロケーション/フラグメントを抑制。
- 方式:
  - Numpy 配列のプール/再利用（shape/dtype ごとの slab）。
  - Thread‑local scratch バッファ（効果/indices 生成で使い回し）。
- 効果: アロケーションオーバーヘッドと GC プレッシャを低減。
- リスク: 複雑度は低〜中。API は内部実装に閉じる。

6) ログ/計測の非同期集約（QueueLogger）
- ねらい: 多スレッド下での print/ロガー I/O を一本化し、フレームを阻害しない。
- 方式:
  - 非同期ロガー（単一コンシューマ）へ各スレッドがイベントを push。描画スレッドで pull＆HUD 表示。
- 効果: デバッグ/計測を常時有効にしてもフレーム影響が限定的。

---

検出と切替（ランタイム）
- 判定の考え方（候補）:
  - ENV: `PYTHON_GIL=0`（free‑threaded 起動時に設定される可能性）
  - sysconfig のビルドフラグ（`Py_GIL_DISABLED` 相当）や `platform.python_implementation()` の派生判定
  - 専用フラグ: `PXD_FREE_THREADED=1`（強制上書き）
- 切替方針:
  - デフォルト: GIL 有効 → 既存 multiprocess パス / GIL 無効 → thread‑first パス
  - ENV で明示的にオフ（`PXD_FORCE_MP=1`）も可。

段階導入ステップ（提案）
- Phase 1（安全起点）
  - [ ] `ThreadWorkerPool` を追加し、ENV で opt‑in（デフォルトは現状維持）。
  - [ ] 共有 LRU（_SHAPE/_PREFIX/indices/_GLOBAL_COMPILED）に Lock を付与（読み書き最小ロック）。
  - [ ] 署名メモ化 LRU を導入（スレッド安全）。
- Phase 2（拡張）
  - [ ] ステージパイプライン（A→E の重ね合わせ）を thread 実装。
  - [ ] SwapBuffer を SPSC リングへ差し替え（イベント駆動 + ノンブロッキング get/put）。
- Phase 3（最適化）
  - [ ] Sharded LRU へ置換（競合の低減）。
  - [ ] バッファプール/スレッドローカル scratch の導入。

DoD（各フェーズ）
- [ ] 代表シーンで thread‑first が mp‑first より p95 フレーム時間/平均fps/CPU%で優位。
- [ ] 共有 LRU の競合がボトルネックにならない（プロファイラで lock 待ちが支配的でない）。
- [ ] 例外時は安全に mp‑first へフォールバック（ENV/自動）。

リスク/論点まとめ
- GL コンテキストのスレッド制約 → メインに固定。upload/draw は主スレッドで実行。
- Data Race → LRU/カウンタ/グローバル状態に適切なロック/シャーディングが必要。
- バックプレッシャ → パイプライン段数/バッファ深さの調整が必要（輻輳回避）。

備考
- free‑threaded 環境では「mp→thread 化」のリターンが最も大きい（IPCゼロ化 + 共有キャッシュ）。
- その上で「段階並列（A→E）」と「署名メモ化」を組み合わせると、CPU 側の尾をさらに詰められる見込み。

