# 251106: 非同期化でさらに伸ばす余地（検討・設計計画）

目的
- 既存の高速化（lazy / cache / multiprocess / IBO freeze / indices LRU / ベクトル化）に加え、非同期化でフレーム時間のばらつきとピーク負荷をさらに抑える。
- 「計算・転送・描画」の重なりを最大化し、ブロッキング区間を縮める。

現状の非同期化（おさらい）
- WorkerPool（プロセス/インライン）で `draw(t)` を非同期実行、SwapBuffer 経由でメインへ受け渡し。
- ExportService（スレッド）で G-code 保存を非ブロック化。
- Video Recorder は PBO リングで GPU→CPU 読み出しの一部を非同期化（実装済み）。
- Renderer は VBO/IBO への orphan+write、IBO 固定化（offsets 署名一致時）、indices LRU で前処理負荷を低減。

---

非同期化の追加候補（優先度順）

1) GL 転送/描画の更なる非同期化（フレーム境界のオーバーラップ）
- 狙い: VBO/IBO 更新と描画の依存を緩め、メインスレッドの同期点を縮める。
- 案:
  - Persistent Mapping（GL_ARB_buffer_storage）で VBO/IBO を永続マップし、`GL_MAP_UNSYNCHRONIZED_BIT` を使って CPU 書き込みを非同期化。ModernGL の対応要確認。
  - glFenceSync + 2～3本リングバッファで「前フレームの GPU アクセス完了」をフェンスで確認しながら交互に使用（現行 orphan は多くのケースで十分だが、長尺描画ではフェンスがより安定）。
- DoD:
  - 60fps/100K+頂点で、描画呼び出しの分散（p50/95）と平均フレーム時間が改善。
- リスク: ModernGL API の制約。未サポート時は no-op（既存 orphan 経路を温存）。

2) Worker 内での並列チャンキング（効果ステップのスレッド並列）
- 狙い: `displace/subdivide/fill` のような頂点数比例の重い効果を、1 プロセス内で追加並列化。
- 案:
  - 効果ごとに「チャンク分割 + ThreadPool」で行配列を分担（NumPyベース処理はGIL解放が多く、スレッドで伸びやすい）。
  - チャンク結合のオーバーヘッドを抑えるため、in-place/ビュー参照を活用（可能な範囲で）。
- DoD:
  - 1ワーカー（num_workers=1）時でも CPU マルチコアを有効利用し、効果の処理時間が `~(1/threads)` に近似的に短縮。
- リスク: エフェクト実装の純粋性確保（副作用禁止）とメモリ帯域競合。段階導入（まず displace のみなど）を推奨。

3) Renderer 前処理の完全非同期化（indices 生成のワーカ移動 or 事前計算）
- 狙い: メインスレッドでの `indices` 生成（不変時はスキップ済み）を、変化時にも非同期化。
- 案（2択）:
  - A) Worker 側で offsets に応じた `indices` を生成して同梱（IPC コスト↑だが、メインスレッドの CPU 負荷はゼロに）。
  - B) メイン側での生成継続（現状ベクトル化済み）だが、`indices` 生成要求を別スレッドに投げ、結果準備後に GPU 書き込み（同期点を短縮）。
- DoD:
  - offsets 変更フレームでもメインの CPU 突発が抑制される。
- リスク: A は IPC 体積増（N + M から N + M + (N+M) へ）。B は実装複雑化（描画タイミング同期）。

4) 署名計算・レジストリ解決の非同期プリフェッチ
- 狙い: PrefixCache の最長一致探索時に必要な `params_signature` / エフェクト関数解決などの準備をフレーム間で事前計算。
- 案:
  - 「前フレームの plan」と「GUI/ccの変動傾向」から、次フレームでも再利用しそうなステップの署名をバックグラウンドでプリ計算。
  - 署名結果のメモ（関数id, パラメータdict→署名タプル）LRU をプロセス内に持ち、`params_signature` を O(1) 化。
- DoD:
  - plan 長が大きいケースで PrefixCache の探索/準備が軽くなる。
- リスク: 過剰プリフェッチ（外した場合は CPU 無駄）。ENV ガード（`PXD_SIGNATURE_PREFETCH=0/1`）とLRUで抑制。

5) PNG 保存・G-code 書き出しの I/O パイプライン化
- 狙い: ファイル出力の分割・圧縮をスレッドでパイプライン化し、I/O 待ちを隠蔽（ExportService で骨組みあり）。
- 案:
  - G-code: ライン塊ごとにテキスト生成→キュー→ライターが chunk 単位で flush（現在の擬似ジョブ実装を実体化）。
  - PNG: 画面キャプチャ（screen or FBO）後、PNG エンコードをワーカーへ委譲（既に Video はPBOリング有）。
- DoD:
  - 保存操作中も描画・操作が滑らか。
- リスク: 依存追加（png/codec）やメモリ消費。既存 ExportService に段階統合。

6) HUD/メトリクスの非同期集約
- 狙い: HUD 用集計（cache counters など）を描画パスから切り離し、スレッドで低頻度に更新。
- 案:
  - 現行は sampler.tick() 内で取得（低頻度）。より重くなる場合に備え、バックグラウンドスレッドで集計→atomicに置換。
- DoD:
  - HUD 表示の有無でフレーム時間が影響しない。

---

段階導入ステップ（提案）
- Phase A（低リスク）
  - [ ] Worker 内スレッド並列（displace 限定）
  - [ ] 署名メモ化（プロセス内 LRU）
  - [ ] HUD 追加メトリクスの更新頻度を固定（既定 0.5s のまま、必要時にスレッド化）
- Phase B（中リスク）
  - [ ] Indices 生成の非同期化（B 案: メイン側スレッド）
  - [ ] PNG/G-code 出力のパイプライン化（ExportService統合）
- Phase C（条件付き）
  - [ ] Persistent Mapping + フェンス（ModernGL対応時のみ）
  - [ ] Worker 内スレッド並列の対象拡大（fill/subdivide など）

計測/DoD（各フェーズ共通）
- [ ] 代表シーン（N, M 大）の p50/p95 フレーム時間・fps・CPU%/MEM を before/after で比較。
- [ ] 非同期化による揺らぎ増大がない（stutter/tear 無し）。
- [ ] エラー時は安全にフォールバック（ENV で全OFF可能）。

環境変数（案）
- `PXD_SIGNATURE_PREFETCH=0|1`（署名プリフェッチの有効化）
- `PXD_EFFECT_THREADING={off,displace,all}`（Worker 内スレッド並列の対象）
- `PXD_INDICES_ASYNC=0|1`（メイン側スレッドでの indices 生成）
- `PXD_PERSISTENT_MAPPING=0|1`（対応時のみ有効）

リスクと回避
- データ競合/同期ミス: 「受け渡し境界の不変条件」を明文化し、テストを追加（例: offsets 不変時の freeze を前提）。
- 追加メモリ: バッファリング・リング構造は上限を設け、ENV で調整可能に。
- API/依存: 非サポート環境では自動フォールバックして継続可能にする。

備考
- 現状の現場最適は「Heavyエフェクトの並列化 + GL 側の同期点短縮」。既存の Worker/Renderer 分離は良い土台になっており、非同期拡張の収まりは良い。

