# PipelineBuilder.label 実装計画（GUI ヘッダー名の指定）

目的
- `E.pipeline....label(uid="MyGroup")` で Parameter GUI のカテゴリ（ヘッダー）名を任意指定できるようにする。
- 既存のパイプライン UID（内部一意識別子）による衝突回避とキャッシュ/整列ロジックは維持する。

背景（現状の問題）
- スタブには `label(self, uid: str) -> _PipelineBuilder` が存在するが、実装が無く `__getattr__` により「未登録エフェクト label」扱いになっている。
- GUI のカテゴリは現在 `ParameterRuntime` で `pipeline_uid`（例: `p0`, `p1`）を用いて決まる。
- UID 文字列をラベルとして使い回すと、一意性と表示名の責務が混ざり、同名ラベルで ID 衝突リスクがある。

スコープ / 非目標
- スコープ: PipelineBuilder と ParameterRuntime/SnapshotRuntime の連携最小変更。DPG レイヤ（表示）は既存の `Descriptor.category` をそのまま使用。
- 非目標: ラベルの永続化、複雑な重複検出/自動リネーム、外部シリアライズ。

設計方針（採用案 B: 表示ラベルと内部 UID の分離）
- PipelineBuilder に「表示ラベル（display label）」を保持する新規フィールドを追加（例: `_label_display: str | None`）。
- `label(uid: str)` はステップを追加せず、ビルダーの表示ラベルを設定して `self` を返す（チェーン継続）。
- PipelineBuilder がランタイムへステップ登録する際、`before_effect_call(...)` に `pipeline_uid`（内部 UID: 例 `p0`）と併せて `pipeline_label`（表示ラベル）を渡す。
- ParameterRuntime/SnapshotRuntime 側で `pipeline_label` があれば `ParameterDescriptor.category` をそれに置き換え、ID/並び順には従来どおり `pipeline_uid` を用いる。
  - ID: `effect@{pipeline_uid}.{effect_name}#{index}`（変更なし）
  - カテゴリ（GUI ヘッダー表示）: `pipeline_label or pipeline_uid or scope`
- これにより、同一ラベル（表示名）が複数あっても ID は `p0/p1/...` に基づくため衝突しない。

他案（参考、採用しない）
- 案A: `pipeline_uid = label` に置換するだけの簡易実装。衝突時に ID が競合するため却下。

詳細仕様
- API:
  - `PipelineBuilder.label(uid: str) -> PipelineBuilder`
    - 役割: 表示ラベルの設定（ビルダ内部に保持）。
    - ステップに影響しない（パイプラインハッシュ/キャッシュキーに影響しない）。
- ランタイム:
  - `ParameterRuntime.before_effect_call(..., pipeline_uid: str = "", pipeline_label: str | None = None)`
  - `SnapshotRuntime.before_effect_call` も同シグネチャへ拡張。
  - `category = pipeline_label or (pipeline_uid or scope)` に変更し、`ParameterDescriptor.pipeline_uid` は従来の `pipeline_uid` を設定。
- 表示/整列:
  - `src/engine/ui/parameters/dpg_window.py` のカテゴリ表示は `desc.category` をそのまま使用（変更不要）。
  - 並び順のキーは `pipeline_uid/step_index/param_order` で従来通り（表示ラベルによる順序の変化なし）。

互換性
- 既存スケッチはそのまま動作。`label(...)` を追加してもパイプライン定義（steps）/キャッシュ署名に影響しない。
- スナップショット実行（`SnapshotRuntime`）も同等の挙動を提供するため、テスト/ベンチの整合が保たれる。

テスト計画
- ユニット（DPG 非依存、ランタイムのみ）
  - 2 本のパイプラインに同一 `label("L")` を付けた場合
    - それぞれの Descriptor.id が `effect@p0.*` と `effect@p1.*` のように異なること
    - それぞれの Descriptor.category が "L" になること
  - `label` 未指定時にカテゴリが `pN` になる既存挙動の保持
  - `bypass` ディスクリプタも同じカテゴリ配下になること
- スモーク（任意）
  - `sketch/251113.py` が `label(uid=...)` で GUI 初期化時に例外を出さないこと（ヘッドレスは `init_only=True` でも確認可）。

ドキュメント
- `docs/spec/pipeline.md`: 基本 API に `label(uid=...)` の説明と注意（表示名であり内部 UID は別）を追記。
- `architecture.md`: Parameter GUI セクションに「カテゴリ: pipeline_label 優先、無ければ pipeline_uid」のルールを明記。
- `README.md`: 簡単な使用例を追加（任意）。

実装タスク（チェックリスト）
- [x] PipelineBuilder: `_label_display` を追加し `label(uid)` 実装（steps には影響しない）
- [x] PipelineBuilder: ランタイム呼び出しに `pipeline_label` を付与
- [x] ParameterRuntime: `before_effect_call` に `pipeline_label` を追加しカテゴリ決定を変更
- [x] SnapshotRuntime: 同上
- [x] 既存呼び出し箇所のシグネチャ更新（型/スタブへの影響なし）
- [ ] テスト: 新規 `tests/ui/parameters/test_pipeline_label.py` を追加
- [x] ドキュメント更新（spec/architecture/README の該当箇所）
- [x] 変更ファイルに対して ruff/black/isort/mypy を通す

リスクと対応
- 同名ラベルを多数使うと GUI 上で同一ヘッダーに集約される（意図通り）。ID は `pipeline_uid` ベースで衝突しない。
- 既存スナップショット/ツールが `before_effect_call` のシグネチャに依存している場合は合わせて更新（本リポ内は SnapshotRuntime のみ）。
- ラベルは表示専用のため、キャッシュ/署名/パイプライン等価性に影響しないことをテストで担保。

完了条件（DoD）
- 変更ファイルに対する `ruff/black/isort/mypy` 緑
- 新規テスト緑（最低: ユニット）
- `sketch/251113.py` で `label(uid=...)` を使って初期化エラーが出ない
- `docs/spec/pipeline.md` と `architecture.md` を更新済み

開発メモ / 実行コマンド（編集ファイル限定）
- Lint: `ruff check --fix {changed_files}`
- Format: `black {changed_files} && isort {changed_files}`
- Type: `mypy {changed_files}`
- Test: `pytest -q -k pipeline_label`（ユニットのみに限定）

質問/確認事項（確定）
- 表示ラベルの重複は許容: yes（ID は `pipeline_uid` ベースで衝突しない）
- `label` を HUD 等で参照: No（現状スコープ外）

以上、問題なければこの計画に沿って実装を進めます。
