この AGENTS.md は `src/engine/ui/parameters/` 配下に適用されます。

## Overview（役割対応表）
- state.py: ParameterStore/Descriptor/RangeHint のコア型とレジストリ
- normalization.py: 廃止（正規化レイヤは撤去）
- introspection.py: 関数の doc/signature/param_meta を解決（キャッシュ付き）
- value_resolver.py: 値解決の本体（merge → resolve scalar/vector → register）
- runtime.py: Shapes/Effects 呼び出しフック（ParameterRuntime）。Introspector/Resolver を束ねる
- manager.py: `user_draw` をラップして Runtime/Window を起動・寿命管理
- controller.py/window.py: GUI 層（ParameterStore の内容を表示・編集）
- dpg_window.py: 公開エントリ（ParameterWindow 本体・DPG ドライバ制御）
- dpg_window_theme.py: テーマ/フォント管理（ParameterWindowThemeManager）
- dpg_window_content.py: Dear PyGui レイアウトと値連携（ParameterWindowContentBuilder）

## フロー（最短要約）
1) Runtime が関数メタを `Introspector` で取得（doc/signature/param_meta）
2) `ValueResolver.resolve()` がパラメータをマージし、種別ごとに Descriptor 登録/override 適用
   - 入力は実値のみ（float/int/bool/vector）。
   - vector は親 Descriptor（1 件）として登録し、値は tuple とする。
   - クランプは行わない（UI 表示上のみ比率をクランプ）。
3) 実値の辞書を関数へ渡す（`g` は skip）

## 実装ガイド
- 実値のみを扱い、RangeHint は実レンジ（min/max/step）を示す。
- vector は親 Descriptor（1 件）で表現し、GUI 層で x/y/z/w の 3/4 スライダとして水平表示する。
- Range 推定は行わない。`__param_meta__` に min/max/step がある場合は `VectorRangeHint`/`RangeHint` を設定し、無い場合は UI 側で 0–1 既定レンジ（表示クランプのみ）を使う。
- 入力値のクランプは行わない。UI 表示（バー/トラック）に限り比率を 0..1 へクランプして描画してよい。
- enum 判定は `__param_meta__` の `choices` 有無で行う（`type: "string"` は自由入力テキストのヒント）。
- ParameterDescriptor.category_kind はヘッダ/テーマ用のカテゴリ種別とし、`\"shape\"`/`\"pipeline\"`/`\"hud\"`/`\"display\"` などを取る。
  - shape 由来の Descriptor は `\"shape\"`、effect 由来は `\"pipeline\"` を基本とする。
  - HUD/Display 用は `\"hud\"`/`\"display\"` を用い、Dear PyGui 側では `(category_kind, category)` の組でグルーピングされる。
  - `parameter_gui.theme.categories` のキーはこの `category_kind` と対応させる。

## Do/Don't
- Do: UI/非UI の双方で `resolve_without_runtime` を使えるよう、副作用を持たない
- Don't: 形状/エフェクト本体へ UI 依存を逆流させない（ここで遮断）
