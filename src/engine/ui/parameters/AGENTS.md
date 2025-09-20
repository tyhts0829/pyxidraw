この AGENTS.md は `src/engine/ui/parameters/` 配下に適用されます。

## Overview（役割対応表）
- state.py: ParameterStore/Descriptor/RangeHint のコア型とレジストリ
- normalization.py: 0..1 正規化と実レンジの写像ヘルパ（clamp/normalize/denormalize）
- introspection.py: 関数の doc/signature/param_meta を解決（キャッシュ付き）
- value_resolver.py: 値解決の本体（merge → resolve scalar/vector → register → denormalize）
- runtime.py: Shapes/Effects 呼び出しフック（ParameterRuntime）。Introspector/Resolver を束ねる
- manager.py: `user_draw` をラップして Runtime/Window を起動・寿命管理
- controller.py/window.py/panel.py: GUI 層（ParameterStore の内容を表示・編集）

## フロー（最短要約）
1) Runtime が関数メタを `Introspector` で取得（doc/signature/param_meta）
2) `ValueResolver.resolve()` がパラメータをマージし、種別ごとに正規化/登録/逆変換
3) 実レンジ値の辞書を関数へ渡す（`g` は skip）

## 実装ガイド
- 0..1 正規化を入口、RangeHint の mapped_min/max/step へ実レンジを記述する
- vector は x/y/z/w に分割して Descriptor を発行し、group を `vector_group` に設定
- 既定値がない数値は Range 推定（中心±span）で安全側レンジを作る

## Do/Don't
- Do: UI/非UI の双方で `resolve_without_runtime` を使えるよう、副作用を持たない
- Don't: 形状/エフェクト本体へ UI 依存を逆流させない（ここで遮断）

