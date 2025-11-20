# src モジュール改善計画（effects/shapes 除外）

本計画は `src_architecture_review.md` の指摘に基づく改善チェックリストです。作業着手前に内容を確認してください。各項目は進行に応じてチェックを入れていきます。

## チェックリスト

- [x] ParameterRuntime の依存注入方式を明示化する
  - [x] `_ACTIVE_RUNTIME` の利用箇所（`api/effects.py`, `api/shapes.py`, Parameter GUI 関連）を洗い出す
  - [x] コンテキスト注入/オブジェクト渡しの方針を決定（グローバル廃止 or 最小フォールバック）
  - [x] ワーカ側への伝播方法（シリアライズ不要なハンドル or スナップショット）を設計
  - [x] 公開 API 互換ポリシーを決め、必要に応じてマイグレーション階層を用意

- [x] LazyGeometry の責務分割と affine ヘルパ統合
  - [x] `engine/core/lazy_geometry.py` で抱えている spec/caching/軽量エフェクトの役割を整理
  - [x] affine 系の実装を effects 側または共通ユーティリティへ集約する設計を決める
  - [ ] キャッシュ管理（shape/prefix）が単機能クラスに分離できるか検討し、API 影響を確認

- [ ] PipelineBuilder/キャッシュの分離と失効条件の明確化
  - [ ] UI 介在、ラベル管理、キャッシュ設定を分離した責務分担案をまとめる
  - [x] `_GLOBAL_COMPILED` のキーに含めるべき構成要素（impl_id/設定バージョン等）を定義
  - [x] `run_sketch` 起動・終了などライフサイクルでのキャッシュクリア手順を設計
  - [ ] 互換のためのインターフェース（現行チェーン API）維持策を決める

- [x] SwapBuffer/Renderer 間のメッセージ型を固定化
  - [x] レイヤー付き/単体 geometry を一つの型（例: `RenderFrame`）に統一する案を作成
  - [x] `StreamReceiver`/`SwapBuffer`/`LineRenderer` の契約を再定義し、duck-typing 判定を除去する変更手順を洗い出す
  - [x] 既存 HUD/Recorder への影響範囲とテスト追加ポイントを整理

- [x] `run_sketch` のオーケストレーション分割
  - [x] 設定解決、I/O 初期化、並行実行、描画構築のサブコンポーネント化案をまとめる
  - [x] 既存引数の挙動互換を確保するための移行ステップ（薄いラッパ等）を設計
  - [ ] 分割後の単体テスト可能な境界と依存注入（時計、ログ、エクスポート先）を定義

- [ ] ドキュメント/設計同期
  - [ ] `architecture.md` および関係 AGENTS.md に必要な更新箇所を特定
  - [ ] 変更内容に合わせてテスト方針・禁止エッジを更新する案を用意

進めてよければ、この計画をベースに着手します。修正や優先順位の入れ替えがあれば教えてください。
