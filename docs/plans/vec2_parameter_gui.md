# Vec2 パラメータ型 + Parameter GUI 対応 改善計画（提案）

目的: 2 次元ベクトル（例: `grid.subdivisions=(nx, ny)`）を Parameter GUI で正しく 2 成分として扱い、既存の「3 成分固定」挙動に起因する不整合（例外や保存値の寸法ズレ）を解消する。

背景 / 問題
- 現在の GUI 実装はベクトル次元を実質 3（または 4）に固定しており、2 成分の既定値でも 3 成分スライダー/保存になりやすい。
  - 例: `shape.grid#0.subdivisions` が `[1.0, 20.0, 0.0]` として保存され、実行時に `nx_raw, ny_raw = subdivisions` で崩れる。
- 関連箇所（参考）:
  - 2D を 3 成分に丸めている UI: `src/engine/ui/parameters/dpg_window.py`（dim 決定）
  - スナップショットの次元決定が 3 最小: `src/engine/ui/parameters/snapshot.py`
  - grid 実装: `src/shapes/grid.py`（2 要素を期待）

達成したい状態（Done の定義）
- Parameter GUI が 2 成分のベクトルを 2 本のスライダーとして表示・編集できる。
- 既存の 3 成分で保存された override に対する互換処理は行わない（エラー許容）。
- `grid.__param_meta__` の `subdivisions` は 2 次元で一貫し、署名量子化/保存の両方で期待通りに扱われる。
- 変更ファイルに対する ruff/mypy/pytest が通る。

設計方針（シンプル優先）
- 型エイリアスを `common` に集約: `src/common/types.py:1` に `Vec2 = tuple[float, float]` を追加し、`Vec3` と同列の公開型として使用する（実体は `tuple`）。
- 実行時の次元解決は `param_meta` を優先: `meta.type` に `"vec2"/"vec3"/"vec4"` を追加（優先度: 明示 `type` > `vector_hint` 次元 > `default_value` 長さ）。
- GUI の次元決定は「2..4 にクランプ」へ変更（`len(value)` を尊重）。
- 既存保存データ（JSON）の寸法ズレはロード時に調整（切り詰め/ゼロ補完）。

スコープ / 影響範囲
- common: `src/common/types.py`（`Vec2` 追加）
- engine.ui.parameters: value_resolver / snapshot / dpg_window / persistence（2..4 次元可変対応）
- shapes: grid の `__param_meta__` 更新 + 型注釈 `subdivisions: Vec2` への置換
- docs: shapes のパラメータ仕様、AGENTS/architecture の整合

タスク一覧（チェックリスト）
- [x] 仕様確定
  - [x] `meta.type` に `"vec2"/"vec3"/"vec4"` を追加採用（`vec2` を使用）。
  - [x] 次元の解決順序: `meta.type` 明示 > `vector_hint` の長さ > `default_value` の長さ。
  - [x] 量子化ステップは現行ルールを踏襲（float のみ `step` で量子化、ベクトルは成分ごと）。

- [x] 共通型の追加（`src/common/types.py`）
  - [x] `Vec2 = tuple[float, float]` を追加し、`__all__` に含める。
  - [x] 既存の `Vec3` と同列の軽量エイリアスとして公開。

- [x] ValueResolver の拡張（`src/engine/ui/parameters/value_resolver.py`）
  - [x] `type: "vec2"/"vec3"/"vec4"` は既存ロジック（`"vec"` 含有で vector 判定）で対応済みのため変更不要。
  - [x] `vector_hint` の次元決定は現行のメタ/既定値に基づき動作（追加変更なし）。

- [x] SnapshotRuntime の次元下限を 2 に（`src/engine/ui/parameters/snapshot.py`）
  - [x] ベースベクトルの次元決定を `dim = clamp(len(vec), 2, 4)` に修正。

- [x] DPG Window（GUI）の次元処理修正（`src/engine/ui/parameters/dpg_window.py`）
  - [x] `dim` 決定を `dim = clamp(len(vec), 2, 4)` へ。
  - [x] バー生成/CC 入力/イベントハンドリングを `dim` に連動（x,y,z,w のうち先頭 `dim` のみ）。

- [ ] Persistence の互換処理（`src/engine/ui/parameters/persistence.py`）
  - 実施しない（既存 3 成分保存値への互換は行わず、エラー許容）。

- [x] Shapes 側メタ/型注釈（`src/shapes/grid.py`）
  - [x] `grid.__param_meta__["subdivisions"].type = "vec2"` を明示化（範囲/step は現状を踏襲）。
  - [x] 関数シグネチャを `subdivisions: Vec2` に置換（`from common.types import Vec2`）。

- [ ] ドキュメント更新
  - [ ] `docs/spec/shapes.md` に Vec2 表示ルール（2 成分スライダー）と `meta.type: vec2` を追記。
  - [ ] ルート `AGENTS.md`/`architecture.md` の整合チェック（差分があれば参照箇所込みで更新）。

- [ ] 検証 / テスト
  - [ ] 単体: ValueResolver が `vec2` を 2 成分で返すこと。
  - [ ] 手動: grid の Parameter GUI が 2 本スライダ表示、保存→再起動で例外なし（新規保存値が 2 成分であること）。
  - [ ] 最小スモーク: 変更ファイルに対して `ruff/mypy/pytest -q -m smoke` を通す。

互換性 / 移行
- 既存の 3 成分保存データについて互換調整は行わない（ユーザー意向）。
- 保存フォーマットは従来通り。Vec2 導入により今後の保存値は 2 成分で安定化。

リスクと軽減
- リスク: 既存 UI ロジックとの食い違いで GUI レイアウト崩れ。
  - 軽減: `dim` を中央関数で決定し（2..4 クランプ）、バー/CC の両方で同じ決定を使用。
- リスク: 他の 3D 系エフェクトが `vec3/vec4` 前提である場合への影響。
  - 軽減: 既存デフォルト/メタの長さを優先し、明示 `type` 無しでは現状維持（後方互換）。

代替案（要確認）
- A) 型エイリアス + `meta.type="vec2"`（推奨、実装簡潔・互換維持）
- B) `Vec2` の軽量クラス（`tuple` サブクラス）導入
  - メリット: 型で誤用を早期検出しやすい
  - デメリット: シリアライズ/比較/署名生成/JSON 互換の考慮が増え、複雑化

完了条件（変更単位）
- 変更ファイル対して `ruff check --fix {changed}`、`black {changed} && isort {changed}`、`mypy {changed}` が成功。
- スモークテスト（必要最小）にパスし、grid を含む手動確認で例外が発生しない。

実施順序（目安）
1) 仕様確定（type/次元決定の優先順位）
2) ValueResolver → Snapshot → DPG Window の順で 2 成分対応
3) Persistence の互換レイヤ追加
4) grid メタ更新 + ドキュ更新
5) 検証（スモーク/手動）

確認したい事項（ご回答ください）
1) `src/common/types.py` に `Vec2` を追加し、型注釈として広く採用する方針で問題ありませんか？（実装済み）
2) 実行時の次元判定は `param_meta.type`（vec2/vec3/vec4）を優先で進めますか？（現設計通り）
3) 既存の 3 成分保存値は互換調整せず、エラーで良いという方針で確定していますか？（反映済み）
4) `vec3/vec4` の一般化は別タスクで段階導入で良いですか？

承認後、上記チェックリストに沿って実装を進め、各項目の完了状況を本ファイルに反映していきます。
