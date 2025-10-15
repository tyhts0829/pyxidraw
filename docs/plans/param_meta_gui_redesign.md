# param_meta / GUI 再設計（条件付き可視 + 動的レンジ + 動的選択肢）

本ドキュメントは、`__param_meta__` と Parameter GUI の表現力/UX を高めるための再設計案。既存の方針（シンプル・宣言的・副作用なし）を維持しつつ、以下を段階導入する。

- 条件付き可視化（visible_if / enable_if / group / help）
- 動的レンジ（range_if / range_from）
- 動的選択肢（choices_from）
- 量子化 step の UI 切り離し（ui.step）と安定署名の維持（quant.step）

参考: `docs/plans/mirror3d_ui_meta_plan.md`（条件付き可視）、`docs/improvements/font_index_gui_dynamic.md`（font_index の動的レンジ）

## 目的（Goals）
- モード依存や相互排他のパラメータを GUI に正しく反映し、不要操作を排除する。
- 実行環境・入力に応じて UI レンジ/選択肢を自動調整し、エラー/空振りを防ぐ。
- 署名（キャッシュ鍵）や実行値の決定性を損なわず、後方互換を保つ。

## スコープ
- 主に `engine.ui.parameters` 層（introspection/value_resolver/state/dpg_window）。
- 既存の `__param_meta__` に後方互換な拡張キーを追加。
- shapes/effects の本体は非依存。必要に応じて `__param_meta__` のみ加筆。

## 設計方針（シンプル/宣言的）
- 「宣言（__param_meta__）」→「解釈（ValueResolver）」→「表示（DPG Window）」の単純な一方向。
- 条件ロジックは原則「等値判定（AND）」に限定。複雑な式/DSL は導入しない。
- 動的情報は Provider インタフェースを経由して取得し、関数本体へは流さない。

## 提案スキーマ（後方互換）
各パラメータの `__param_meta__[name]` に、以下の任意キーを追加できる。

共通（既存）
- `type: "bool"|"int"|"float"|"vector"|"string"|"enum"`（省略時は推定）
- `min, max, step`（RangeHint: 表示クランプのみ。既存仕様を維持）
- `choices: list[str|int|... ]`（enum 用）

UI メタ
- `ui.visible_if: {param: [value, ...], ...}`
  - すべて満たす時のみ表示（AND）。未指定は常時表示。
- `ui.enable_if: {param: [value, ...], ...}`
  - 表示はするが操作不可にする条件（AND）。未指定は常時有効。
- `ui.group: str`
  - セクション見出し（例: "Azimuth" / "Polyhedral"）。
- `ui.order: int`（任意）
  - 同一セクション内の表示順（小→大）。
- `ui.help: str`
  - ツールチップ/補足文。
- `ui.control: "slider"|"select"|"checkbox"|"textbox"`（任意・既定は自動）
- `ui.step: float | tuple`（UI スライダの見た目刻み。量子化とは分離）

動的レンジ/選択肢
- `ui.range_if: [ {when: {param: [value,...]}, min?:num|max?:num|step?:num}, ... ]`
  - 条件一致時に `min/max/step` を上書き（複数該当時は後勝ち）。
- `ui.range_from: {provider: str, key: str, args?: {k:v,...}, map?: {min?:str|max?:str|step?:str} }`
  - 外部 Provider の数値からレンジを算出。
  - 例: `{provider:"fonts", key:"face_count", args:{font_param:"font"}, map:{min:"0", max:"N-1", step:"1"}}`
- `ui.choices_from: {provider: str, key: str, args?:{..}, value_field?:str, label_field?:str}`
  - 外部 Provider の列挙から選択肢を生成（`enum`/`string`向け）。

量子化（署名）
- `quant.step: float | tuple`（任意）
  - 署名生成と実行引数に用いる量子化刻み。未指定は従来の `step` → 未指定は `PXD_PIPELINE_QUANT_STEP`。
  - UI の `ui.step` と分離し、UI 見た目の刻み変更で署名が不安定化しないようにする。

備考
- 条件の評価は「現在の解決済み値（明示引数 > GUI > 既定）」に対して行う。
- Provider は UI 層の責務。取得不能時はフェイルソフト（静的メタにフォールバック）。

## ランタイム拡張（UI 層）
- ValueResolver: `ui.*` を解釈し、`ParameterDescriptor` に反映。
  - `visible/enable` の判定を行い、Descriptor に付与。
  - レンジは `min/max/step` に `ui.range_if`/`ui.range_from` を順に適用した結果を `RangeHint` として注入。
  - `choices_from` があれば取得し、`choices` を上書き。
- ParameterDescriptor: 以下を追加
  - `visible: bool = True`, `enabled: bool = True`
  - `group: str | None = None`, `order: int | None = None`
- ParameterStore:
  - `update_descriptor(id, **attrs)` を追加（visible/enabled/range/choices の更新反映）。
  - 更新時に `_notify({id})` を発火。
- DPG Window:
  - `mount()` で group/order に従うセクション描画。
  - 更新通知で `visible/enabled` を `dpg.show_item/hide_item` および `configure_item(..., enabled=...)` に反映。
  - Range/Choices 変更は `configure_item(min_value=..., max_value=...)`／コンボ再構築。

## Provider インタフェース（最小）
```python
class UIPMetaProvider(Protocol):
    name: str
    def range(self, key: str, *, args: Mapping[str, Any]) -> dict[str, float] | None: ...
    def choices(self, key: str, *, args: Mapping[str, Any]) -> list[Mapping[str, Any]] | None: ...
```
- 既定実装: `fonts`
  - `range("face_count", args:{font_param:"font", context}) -> {N: int}` を内部的に解釈し、`{min:0, max:N-1, step:1}` を返す。
  - `choices("faces", ...) -> [{value:"FaceName", label:"..."}, ...]`（将来）
- Provider は `engine.ui.parameters` に登録（辞書管理）。取得失敗時は None を返す。

## 具体例
1) mirror3d（条件付き可視）
```python
mirror3d.__param_meta__.update({
  "n_azimuth": {"min":1, "max":64, "step":1, "ui": {"visible_if": {"mode":["azimuth"]}, "group":"Azimuth"}},
  "group": {"choices":["T","O","I"], "ui": {"visible_if": {"mode":["polyhedral"]}, "group":"Polyhedral"}},
  "mirror_equator": {"type":"bool", "ui": {"visible_if": {"mode":["azimuth"]}}},
  "source_side": {"type":"bool", "ui": {"visible_if": {"mode":["azimuth"], "mirror_equator":[True]}}},
})
```

2) text.font_index（動的レンジ）
```python
text.__param_meta__["font_index"] = {
  "type":"int", "min":0, "max":32, "step":1,
  "ui": {
    "range_from": {"provider":"fonts", "key":"face_count", "args": {"font_param":"font"}, "map": {"min":"0", "max":"N-1", "step":"1"}},
    "help": ".ttc のフェイス数に応じて 0..N-1 に自動調整"
  },
  "quant.step": 1  # 署名用の刻み（UI step とは独立）
}
```

## 署名/量子化の扱い
- 署名生成は従来通り `__param_meta__['step']`（または `quant.step` 指定時はそれ）を参照し、float のみ量子化。
- `ui.step` は UI 表示専用とし、署名に影響しない。
- 動的レンジ（min/max）の変化は UI 表示にのみ影響し、値そのものはクランプしない（既存方針の維持）。

## 段階導入計画（MVP → 拡張）
- P1: 条件付き可視 + group/order/help + ui.step（GUI 非表示既定、Disable 表示は設定で選択可）
  - mirror3d に適用。確認テスト整備。
- P2: 動的レンジ（range_if / range_from）、Minimal Provider（fonts.face_count）
  - text.font_index に適用。`docs/improvements/font_index_gui_dynamic.md` の方針を置換（一般化）。
- P3: 動的選択肢（choices_from）とライブ更新（update_descriptor + DPG configure）
  - fonts.faces のコンボ生成、将来のデバイス/ポート選択等に展開。

## 実装ステップ（チェックリスト）
- [ ] S-1 スキーマ: `engine/ui/parameters/AGENTS.md` に ui/quant 拡張キーを追記
- [ ] S-2 型追加: `ParameterDescriptor` に `visible/enabled/group/order` を追加
- [ ] S-3 Store API: `update_descriptor()` とレンジ/choices更新の通知を追加
- [ ] S-4 Resolver: `ui.visible_if/enable_if/group/order/help/ui.step` を解釈（条件評価は AND のみ）
- [ ] S-5 Resolver: `ui.range_if` 適用 / `ui.range_from` Provider 呼出を実装（フェイルソフト）
- [ ] S-6 Resolver: `ui.choices_from` を実装（値/ラベルは `choices` に反映）
- [ ] S-7 DPG: `visible/enabled` と range/choices の `configure_item` 反映、セクション/順序描画
- [ ] S-8 Provider: `fonts` 実装（face_count）し、`text.font_index` で動作確認
- [ ] S-9 mirror3d の `__param_meta__` を更新（ui.visible_if 等を適用）
- [ ] S-10 テスト: 表示/非表示・注入抑止・レンジ動的化・choices 反映の単体/結合テスト
- [ ] S-11 ドキュメント: `architecture.md` と `src/engine/ui/parameters/AGENTS.md` を更新

## テスト計画（最小）
- T-1 mirror3d: mode 切替で対象パラメータが非表示/表示に切替わる（store の Descriptor.visible が変化）。
- T-2 mirror3d: 非表示時、当該パラメータは kwargs 注入されない（Runtime 経由で検証）。
- T-3 text.font_index: Provider が N=6 を返すとき、スライダの RangeHint が 0..5 になる。
- T-4 range_if: `mirror_equator=True` のときだけ `source_side` が有効化される（enable_if でも可）。
- T-5 choices_from: ダミー Provider により選択肢が更新される（コンボの entries が変化）。

## 互換性 / 移行
- 既存の `__param_meta__` はそのまま有効。拡張キーは無視されても害がない。
- 署名生成は従来仕様を維持（`quant.step` 追加は後方互換）。

## リスク / トレードオフ
- 動的更新に伴う UI のちらつき → Provider 呼出の debounce/cache、必要時のみ `configure_item`。
- 依存逆流の懸念 → Provider 層に閉じる（関数本体へ渡さない）。
- 条件式の複雑化 → 等値 AND のみに限定。必要なら後続で `in_range` など追加検討。

## オープン事項（要確認）
- Q1 非表示 vs Disable の既定（提案: 非表示）。
- Q2 Provider の登録場所/設定（モジュール登録 or `engine.ui.parameters` 内の固定辞書）。
- Q3 `quant.step` の導入有無（提案: 導入、既定は従来 `step`）。
- Q4 `choices_from` の値種別（index vs label）。提案: 値は label（安定）を推奨。

## DoD（完了条件）
- mirror3d の UX: モード外スライダが出ない/Disable される。
- text.font_index の UX: 実フォントに応じてスライダ上限が適切化。
- 変更範囲に対する `ruff/mypy/pytest` 緑。`architecture.md`/AGENTS の更新済み。

