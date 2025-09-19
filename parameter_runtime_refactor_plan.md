# ParameterRuntime 責務分割計画

## 背景

`ParameterRuntime` がランタイム管理からメタ解析・値変換・範囲決定までを一極集中で担当しており、変更影響が大きくテスト性も低い。責務を分割し、UI 層との疎結合を目指す。

## アクションチェックリスト

- [x] 現状調査: `ParameterRuntime` 内部の機能ブロック（活性管理・シグネチャ解析・メタキャッシュ・値正規化・ストア同期）を洗い出し、依存関係を整理する。
- [x] 設計案作成: 機能ごとの責務分割方針（例: `ParameterDocResolver`, `ParameterValueNormalizer`, `ParameterStoreSync`）を検討し、インターフェースを定義する。
  - `FunctionIntrospector`: docstring・シグネチャ・メタ情報をキャッシュし、`resolve(kind, name, fn)` で `FunctionInfo` を返す。
  - `ParameterValueResolver`: ParameterStore と連携し、デフォルト統合・ベクタ分解・レンジ決定・descriptor 登録を一手に担う。
  - `ParameterContext` dataclass で shape/effect 呼び出しの識別情報（scope/name/index）を受け渡し、ラベル生成を集中管理する。
- [x] メタ解析/ドキュメント抽出の切り出し: シグネチャ解析・`__param_meta__` 処理を専用クラスへ移し、キャッシュ戦略をそこに集約する。
- [x] 値・レンジ決定ロジックのモジュール化: ベクタ分解、レンジヒューリスティック、デフォルト値統合をまとめたユーティリティを独立させ、単体テスト可能にする。
- [x] ParameterRuntime の再構成: 既存クラスを上記コンポーネントのオーケストレーション役へ縮小し、外部 API を維持しつつ内部結合を緩める。
- [x] テスト整備: 新設コンポーネント用の単体テスト、ランタイム全体の挙動を確認する統合テストを追加する。
- [x] ドキュメント更新: `architecture.md` および `src/engine/ui/parameters/AGENTS.md` に新構成を反映する。
- [x] 移行確認: 既存利用箇所（Runtime manager, GUI, tests）が新 API に適合するか検証し、後方互換性の有無を整理する。
