# param_meta / GUI 再設計（条件付き可視 + 動的レンジ/選択肢 + 量子化分離）

本ドキュメントは、`__param_meta__` と Parameter GUI の表現力/UX を高める再設計案。既存方針（シンプル・宣言的・副作用なし）を維持し、以下を段階導入する。

- 条件付き可視・無効化: `ui.visible_if` / `ui.enable_if` / `ui.group` / `ui.order` / `ui.help`
- 動的レンジ/選択肢: `ui.range_if` / `ui.range_from` / `ui.choices_from`
- UI.step と 量子化（署名）step の分離: `ui.step` と `quant.step`

参考: `docs/plans/mirror3d_ui_meta_plan.md`（条件付き可視）, `docs/improvements/font_index_gui_dynamic.md`（font_index の動的レンジ）

## 目的（Goals）
- モード依存・相互排他のパラメータを GUI に正しく反映し、無効操作を排除。
- 実行環境・入力に応じて UI のレンジ/選択肢を自動調整し、エラー/空振りを防止。
- 署名（キャッシュ鍵）や実行値の決定性を損なわず、後方互換を保つ。

## スコープ
- 主に `engine.ui.parameters` 層（introspection/value_resolver/state/dpg_window）。
- `__param_meta__` に後方互換な拡張キーを追加。
- shapes/effects の本体は非依存（必要に応じ `__param_meta__` のみ加筆）。

非スコープ（当面）
- 複雑な条件式/DSL（等値 AND のみ）。
- グループの「動的な再配置」（初回マウント順のみ適用）。

## 設計方針（シンプル/宣言的）
- 「宣言（__param_meta__）」→「解釈（Resolver）」→「表示（DPG Window）」の一方向。
- 条件ロジックは等値 AND に限定。外部情報は Provider から取得し UI 層に閉じる。

## 提案スキーマ（後方互換）

共通（既存）
- `type: "bool"|"int"|"float"|"vector"|"string"|"enum"`（省略時は推定）
- `min, max, step`（RangeHint: 表示クランプのみ。既存仕様を維持）
- `choices: list[str|int|... ]`（enum）

UI メタ
- `ui.visible_if: {param: [value, ...], ...}`
  - すべて満たす時のみ表示（AND）。未指定は常時表示。
- `ui.enable_if: {param: [value, ...], ...}`
  - 表示はするが操作不可（AND）。未指定は常時有効。
- `ui.group: str`（セクション見出し、例: "Azimuth" / "Polyhedral"）
- `ui.order: int`（同一グループ内の表示順・小→大）
- `ui.help: str`（ツールチップ/補足文）
- `ui.control: "slider"|"select"|"checkbox"|"textbox"`（任意・既定は自動）
- `ui.step: float | tuple`（UI スライダ見た目の刻み。量子化とは分離）

動的レンジ/選択肢
- `ui.range_if: [ {when: {param: [value,...]}, min?:num|max?:num|step?:num}, ... ]`
  - 条件一致時に `min/max/step` を上書き（複数該当時は後勝ち）。
- `ui.range_from: {provider: str, key: str, args?: {k:v,...}, map?: {min?:str|max?:str|step?:str} }`
  - Provider の数値からレンジを算出。
  - 例: `{provider:"fonts", key:"face_count", args:{font_param:"font"}, map:{min:"0", max:"N-1", step:"1"}}`
- `ui.choices_from: {provider: str, key: str, args?:{..}, value_field?:str, label_field?:str}`
  - Provider の列挙から選択肢を生成（enum/string向け）。

量子化（署名）
- `quant.step: float | tuple`（任意）
  - 署名生成・実行引数に用いる量子化刻み。未指定は従来 `step` → 未指定は `PXD_PIPELINE_QUANT_STEP`。
  - UI の `ui.step` と分離し、UI 変更で署名が揺れないようにする。

備考
- 条件評価は「解決済み値（明示引数 > GUI > 既定）」に対して行う。
- Provider は UI 層の責務。取得不能時はフェイルソフト（静的メタにフォールバック）。
- 注入規則（重要）:
  - `visible=False` のパラメータは kwargs に注入しない（値は Store に保持）。
  - `enabled=False` は表示はするが注入は行う（UI 操作不可、実値は保持・適用）。
  - 復帰時は前回の保持値をそのまま適用（モード往復で再入力不要）。
- DPG 制約メモ（UI.step）:
  - Dear PyGui の slider は見た目刻み/表示精度（`format`）中心で、厳密な入力量子化ではない。
  - 入力の決定的量子化は `quant.step`（署名/実行引数）で担保し、UI 側はヒントとして扱う。

## ランタイム拡張（UI 層）
- ValueResolver: `ui.*` を解釈して `ParameterDescriptor` に反映。
  - `visible/enabled` を判定して付与。
  - `min/max/step` は `ui.range_if` → `ui.range_from` の順に適用した結果を `RangeHint` に注入。
  - `choices_from` があれば取得し `choices` を上書き。
  - `visible=False` の項目は `updated` から除外（注入抑止）。
- ParameterDescriptor 追加フィールド
  - `visible: bool = True`, `enabled: bool = True`
  - `group: str | None = None`, `order: int | None = None`
- ParameterStore 拡張
  - `update_descriptor(id, **attrs)` を追加（`visible/enabled/range_hint/vector_hint/choices/group/order` を更新）。
  - 更新時に `_notify({id})` を発火。
- DPG Window
  - 初期 `mount()` は group/order に従うセクション描画（Phase 1 は「カテゴリ=scope」維持でも可）。
  - `_on_store_change` で `show_item/hide_item` と `configure_item(enabled=...)` を反映。
  - 数値レンジは `configure_item(min_value=..., max_value=...)` で更新（未対応の場合は再構成）。
  - enum は `configure_item(items=[...])` で更新（未対応の場合は再構成）。
  - group/order の「動的」変更はスコープ外（初回マウントのみ）。

## Provider インタフェース（最小）
```python
from typing import Protocol, Mapping, Any

class UIPMetaProvider(Protocol):
    """UI メタの外部情報を提供する簡素なプロバイダ。"""

    name: str  # 例: "fonts"

    # 数値レンジの提供（必要なものだけ返す）
    # 例: {"min": 0, "max": N-1, "step": 1}
    def range(self, key: str, *, args: Mapping[str, Any]) -> dict[str, float] | None:
        ...

    # 列挙の提供（ラベル/値は同一でよい）
    def choices(self, key: str, *, args: Mapping[str, Any]) -> list[str] | None:
        ...

# Provider の登録は UI 層内の固定辞書（設定/DI は将来検討）
PROVIDERS: dict[str, UIPMetaProvider] = {"fonts": FontsProvider()}
```

実例（最小）
- `fonts.face_count`: `args={"font_param": "font"}` を参照してフォント名を取り出し、フェイス数 `N` を返す（失敗時 `N=1`）。
- `fonts.faces`: `choices()` でフェイス名を列挙（任意・後続）。

## 実装ステップ（段階導入）

Phase 1（可視制御 + 注入抑止 + mirror3d 最小）
- [ ] S-1 スキーマ: `engine/ui/parameters/AGENTS.md` に `ui.visible_if/ui.enable_if/ui.group/ui.order/ui.help` を追記
- [ ] S-2 型追加: `ParameterDescriptor` に `visible/enabled/group/order` を追加
- [ ] S-3 Store API: `update_descriptor()` を追加（visible/enabled 更新の通知）
- [ ] S-4 Resolver: `ui.visible_if/enable_if/group/order/help` を解釈（等値 AND）。`visible=False` は kwargs 注入を抑止
- [ ] S-5 DPG: `show_item/hide_item` と `configure_item(enabled=...)` を `_on_store_change` に実装
- [ ] S-6 mirror3d: `__param_meta__` を更新（mode 連動、`source_side` 条件）

Phase 2（動的レンジ: font_index 最小）
- [ ] S-7 Provider: `fonts` 実装（`face_count`）。UI 層内の固定辞書で登録
- [ ] S-8 Resolver: `ui.range_from` を実装（フェイルソフト）。`text.font_index` に適用
- [ ] S-9 DPG: 数値 `min/max` の `configure_item` 反映を追加（未対応 UI は再構成で回避）

Phase 3（choices/range_if）
- [ ] S-10 Resolver: `ui.range_if` を実装（等値 AND で上書き）
- [ ] S-11 Resolver: `ui.choices_from` を実装し、コンボに `configure_item(items=...)` を反映

Phase 4（量子化分離/文書同期）
- [ ] S-12 量子化: `quant.step` を `param_utils` に導入（既定は従来 `step` → 未指定は環境）
- [ ] S-13 ドキュメント: `architecture.md`/AGENTS を更新（UI.step と quant.step の役割分離）
- [ ] S-14 整合: Shapes 実行引数の量子化方針を `architecture.md` と実装で一致させる（鍵のみ量子化 or 実行も量子化のどちらかに統一）

## テスト計画（最小）
- T-1 mirror3d: mode 切替で `n_azimuth/phi0_deg/mirror_equator` 表示、`group/use_reflection` 非表示（往復で値保持）。
- T-2 mirror3d: 非表示項目はエフェクト呼出 kwargs に含まれない（Resolver/Runtime 経由で検証）。
- T-3 text.font_index: Provider が N=6 を返すとき、スライダの RangeHint が 0..5 になる。
- T-4 range_if: `mirror_equator=True` のときだけ `source_side` が有効化（enable_if でも可）。
- T-5 choices_from: ダミー Provider により選択肢が更新される（コンボの `items` が変化）。

補足
- Window 側 `configure_item` が未対応の場合、最小スモークとして「再マウントで期待レンジ/選択肢になる」ことを確認。
- DPG 実行環境が無い CI では、Resolver/Store レベルの単体テストを優先。

## 互換性 / 移行
- 既存の `__param_meta__` はそのまま有効。拡張キーは無視されても害がない。
- 署名生成は従来仕様を維持（`quant.step` 追加は後方互換）。
- `ui.group/order` の「動的」変更は当面非対応（初回マウントのみ）。

## リスク / トレードオフ
- 動的更新に伴う UI のちらつき → Provider 呼出の debounce/cache、必要時のみ `configure_item`。
- 依存逆流の懸念 → Provider 層に閉じる（関数本体へ渡さない）。
- 条件式の複雑化 → 等値 AND のみに限定。必要なら後続で `in_range` など追加検討。
- DPG API 制約 → `configure_item` 未対応項目は再構成で回避。最初は数値/enum のみ動的更新対象。

## オープン事項（要確認と提案）
- Q1 既定の不可視挙動: 非表示 vs Disable
  - 提案: 既定は「非表示」。`enabled` は説明目的で使い、注入は継続。
- Q2 Provider の登録場所/設定
  - 提案: 当面は UI 層内の固定辞書に登録（設定/DI は後続）。
- Q3 `quant.step` の導入
  - 提案: 導入。既定は従来 `step`、未指定は `PXD_PIPELINE_QUANT_STEP`/1e-6。
- Q4 `choices_from` の値種別（index vs label）
  - 提案: 値=label の同一化を推奨（安定・実装簡素）。必要なら index は将来拡張。
- Q5 Shapes の実行値の量子化
  - 提案: 設計（鍵のみ量子化）に合わせるか、現行実装に寄せるかを決定し、`architecture.md` を同期。

## DoD（完了条件）
- mirror3d の UX: モード外スライダが出ない/Disable される。
- text.font_index の UX: 実フォントに応じてスライダ上限が適切化。
- 変更範囲に対する `ruff/mypy/pytest` 緑。`architecture.md`/AGENTS の更新済み。

## 付記（関連ドキュメントとの関係）
- `docs/plans/mirror3d_ui_meta_plan.md`: 本計画の Phase 1 と整合。mirror3d の `__param_meta__` 更新で達成。
- `docs/improvements/font_index_gui_dynamic.md`: Resolver 直参照案から Provider 案へ整理。実装は本計画 Phase 2 で実現。
- `architecture.md`: RangeHint/量子化/署名生成の説明を `ui.step`/`quant.step` 分離に合わせて更新。

