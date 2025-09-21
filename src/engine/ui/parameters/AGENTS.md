この AGENTS.md は `src/engine/ui/parameters/` 配下に適用されます。

## Overview（役割対応表）
- state.py: ParameterStore/Descriptor/RangeHint のコア型とレジストリ
- normalization.py: 廃止（正規化レイヤは撤去）
- introspection.py: 関数の doc/signature/param_meta を解決（キャッシュ付き）
- value_resolver.py: 値解決の本体（merge → resolve scalar/vector → register → denormalize）
- runtime.py: Shapes/Effects 呼び出しフック（ParameterRuntime）。Introspector/Resolver を束ねる
- manager.py: `user_draw` をラップして Runtime/Window を起動・寿命管理
- controller.py/window.py/panel.py: GUI 層（ParameterStore の内容を表示・編集）

## フロー（最短要約）
1) Runtime が関数メタを `Introspector` で取得（doc/signature/param_meta）
2) `ValueResolver.resolve()` がパラメータをマージし、種別ごとに Descriptor 登録/override 適用
   - 入力は実値のみ（float/int/bool/vector）。
   - クランプは行わない（UI 表示上のみ比率をクランプ）。
3) 実値の辞書を関数へ渡す（`g` は skip）

## 実装ガイド
- 実値のみを扱い、RangeHint は実レンジ（min/max/step）を示す。
- vector は x/y/z/w に分割して Descriptor を発行し、group を `vector_group` に設定。
- 既定値がない数値は Range 推定（中心±span）で安全側レンジを作る。
- 入力値のクランプは行わない。UI 表示（バー/トラック）に限り比率を 0..1 へクランプして描画してよい。

## Do/Don't
- Do: UI/非UI の双方で `resolve_without_runtime` を使えるよう、副作用を持たない
- Don't: 形状/エフェクト本体へ UI 依存を逆流させない（ここで遮断）
