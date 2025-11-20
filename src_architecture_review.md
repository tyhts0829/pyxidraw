# src モジュール アーキテクチャレビュー（effects/shapes 除外）

- どこで: `src/` 配下の API/engine/common/util（`effects/` と `shapes/` は対象外）
- 何を: アーキテクチャとクラス設計の観点で、結合度・凝集度と理解しやすさを阻害する箇所を点検
- なぜ: ランタイム/GUI/描画が密結合になりやすい構造をほぐし、責務を明確化するため

## 主要な懸念点

1. **ParameterRuntime がグローバル状態に依存し呼び出し元と密結合**
   - `engine/ui/parameters/runtime.py:23-38` で `_ACTIVE_RUNTIME` をグローバルに保持し、`api/effects.py:120-167`, `api/shapes.py:145-164` から暗黙参照している。複数ランナーや並列実行時に影響範囲が読み取りづらく、テストでもモックがばらける。
   - 提案: `G`/`E` にコンテキストを明示注入（参照を持つファクトリやコンテキストマネージャ）し、グローバルは廃止または最小限のフォールバックにとどめる。ワーカー側へも同じインスタンスをシリアライズ不要なハンドルで渡せるよう、依存注入の経路を一本化する。

2. **LazyGeometry が「spec+caching+簡易エフェクト」を抱え込み凝集が弱い**
   - `engine/core/lazy_geometry.py:315-374` で affine 系の実装を独自に持ち、同ファイルが spec 管理・キャッシュ・幾何操作を同時に担う。effects 層の affine と動機がずれるリスクがあり、キャッシュ／変換の責務境界が曖昧。
   - 提案: 「spec とキャッシュ管理」と「幾何操作ヘルパ」を分離し、変換ロジックは effects 層の関数を再利用するか、共通ユーティリティへ退避する。`LazyGeometry` は plan と実体化に専念し、キャッシュは専用マネージャに委譲すると挙動が追いやすい。

3. **PipelineBuilder が UI 介在・キャッシュ制御・ラベル管理まで一手に抱える**
   - `api/effects.py:113-205` で steps 管理と同時に runtime 介在・バイパス判定・UID/ラベル管理・グローバル LRU を扱う。Pipeline のキャッシュ `_GLOBAL_COMPILED` も効果実装の更新や設定変更に対する失効条件を持たず、ライフサイクルが不透明。
   - 提案: 「UI 介在（ParameterRuntime）」と「Pipeline コンパイル＋キャッシュ」を別オブジェクトに分け、Builder は宣言的に steps を集めるだけにする。キャッシュキーに impl_id と設定バージョン（quantize step など）を含め、`run_sketch` 起動/終了で明示的にクリアできるようにする。

4. **描画経路のメッセージ形状が曖昧で renderer 側が duck-typing**
   - `engine/runtime/buffer.py:17-45` は `Geometry` と `StyledLayer` のどちらも保持でき、`engine/render/renderer.py:112-134` で「tuple で geometry/color/thickness を持つか」を都度判定している。新しいペイロード追加時に renderer の判定漏れで無視/クラッシュする危険があり、データ契約が明示されていない。
   - 提案: `SwapBuffer` は単一の明示的メッセージ型（例: `RenderFrame{geometry|layers}`）に限定し、renderer は型で分岐するだけにするか、レイヤー専用の別バッファに分離する。`StreamReceiver` も同じ型を流すことで契約を一箇所に固定する。

5. **`run_sketch` がオーケストレーションを一関数に集約し肥大化**
   - 設定解決、MIDI 初期化、Parameter GUI 準備、ワーカ/バッファ連携、HUD/エクスポート、描画コールバック登録までを単一関数内で行っており（`api/sketch.py:164-359` 付近）、依存やエラーパスが線形に増える。個別機能の差し替えやヘッドレス確認時にテストしにくい。
   - 提案: 「設定解決」「IO 初期化」「並行実行」「描画組み立て」を担当する小さなビルディングブロック（クラス or 関数）へ分解し、`run_sketch` は組み立てとエラーハンドリングだけに絞る。各ブロックを個別にテストできるようにし、依存注入（時計、ログ、エクスポート先）を明示する。
