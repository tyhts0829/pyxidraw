# 251101: 形状呼び出し単位の線色/太さ（Parameter GUI 対応）簡素計画（boldify + colorize）

目的
- 各 Shape「呼び出し単位」で線の色（RGBA）と見かけの太さを指定可能にする。
- 既定採用時は Parameter GUI で色/太さを操作できるようにする。
- 既存の Shape 実装を修正せず、共通の Effect で実現（太さは既存 boldify を活用）。

前提/制約
- per-line ではなく per-shape（呼び出し箇所）で十分とする。
- コードは必要最小限。ランタイムは大きく変えず、レンダパスは理解しやすく。
- Parameter GUI 仕様準拠（「未指定＝GUIに出る」）。RangeHint は `__param_meta__` を用いる。

設計方針（採用）
- 太さ: 既存 `effects.boldify(boldness=...)` を使う（新規ロジックを作らない）。
- 色: 新規 Effect `colorize(rgba=(r,g,b,a))` を追加し、呼び出し単位の色を与える。
- レンダリング: 「レイヤー分割（色ごとに draw する）」で実現。
  - Worker からは「複数レイヤー（Geometry, line_color）」のリストを返し、
    メイン側で色ごとにシェーダの uniform を切り替えて順に描画。
  - これにより Shader/VBO 形式や Geometry 型自体を変えず、単純さを維持。

実装ステップ（チェックリスト）
1) Colorize エフェクトの追加（共通構造）
   - [ ] `src/effects/colorize.py` を追加。
        - シグネチャ: `@effect() def colorize(g: Geometry, *, rgba: tuple[float,float,float,float] = (0,0,0,1)) -> Geometry:`
        - 挙動: 幾何は変更せずメタ（色）をタグ付けする「レイヤー化」のマーカーとして扱う。
        - GUI: `__param_meta__` で RGBA 0..1 の RangeHint（vector, min/max/step）。

2) レイヤー描画の薄い導入（最小変更）
   - [ ] `engine.runtime.packet.RenderPacket` を拡張: `geometry: Geometry | None`, `layers: list[RenderLayer] | None` を許容。
   - [ ] `RenderLayer`（dataclass）を新設: `geometry: Geometry`, `rgba: tuple[float,float,float,float]`。
   - [ ] `engine.runtime.buffer.SwapBuffer` を拡張: フロントに `Geometry | list[RenderLayer]` を保持できるよう型/実装を緩和。
   - [ ] `engine.runtime.receiver.StreamReceiver` は `RenderPacket.layers` があればそれをそのまま `SwapBuffer.push()`。
   - [ ] `engine.render.renderer.LineRenderer.draw()` を拡張:
        - 受け取ったフロントが `Geometry` → 従来どおり 1 回描画。
        - `list[RenderLayer]` → 各レイヤーについて `set_line_color(rgba)` → `upload(layer.geometry)` → `draw` を順次実行。
        - 既存の `shader.color` uniform/描画パスは流用（Shader 変更不要）。

3) パイプラインからのレイヤー出力（共通構造）
   - [ ] `api.effects` の Pipeline ビルダーに小拡張: chain 内に `colorize` が含まれる場合、
        `build()` 適用時に「色でグルーピングされた `list[RenderLayer]`」を返すヘルパを用意。
        - 具体案（シンプル）: ユーザーは複数の Pipeline を個別 shape に適用し、`draw()` は `[layer1, layer2, ...]` を返す。
        - 既存の `Geometry` 単体返却も引き続き有効（後方互換）。

4) Parameter GUI 連携
   - [ ] `colorize.__param_meta__` で RGBA を vector として提示（0..1、step=0.01）。
   - [ ] `boldify.__param_meta__` は既存のまま（boldness: 0..10）。
   - [ ] GUI の表示は「未指定＝既定を採用したステップのみ」が出る前提を満たすため、
        スケッチ側では `colorize()`/`boldify()` の引数を未指定で呼ぶ例を推奨。

5) sketch/251101.py の利用スタイル（例）
   - [ ] 各 Shape 呼び出しに対し `E.pipeline.boldify().colorize()` を適用し、
        `draw()` は `list[RenderLayer]` を返す（Parameter GUI から各呼び出しの色/太さを操作できる）。

6) 最小テスト/検証
   - [ ] Packet/Buffer/Renderer が `list[RenderLayer]` を安全に受け流し・順描画できることを単体で確認。
   - [ ] Parameter GUI が `colorize/boldify` の既定パラメータを Descriptor 登録することを確認。

7) ドキュメント/アーキ更新
   - [ ] `architecture.md` に「レイヤー描画（複数ジオメトリ + 色）」の流れとコード参照を追記。

優先順位/競合ルール
- 色/太さの優先度: 「Shape 呼び出しの Effect 指定 > Parameter GUI override > Runner 既定色」。
- 既存の `runner.line_color` は「レイヤー無し（素の Geometry）」受信時のみ適用。

選定理由（この案がシンプルな点）
- Geometry/Shader の形式を変えず、Shader 変更ゼロ。
- 太さは boldify に委譲するため新規幾何ロジック不要。
- 色は Effect で宣言 → レイヤーで分割描画するだけの明快さ。

ToDo（ユーザー確認）
- `draw()` が `Geometry` に加えて `list[RenderLayer]` を返す拡張で問題ないか。
- 色は RGBA(0..1) 固定で良いか（Hex 入力は別途検討）。
- 251101 はまず 2〜3 形状での利用例（レイヤー2–3本）で良いか。

完了条件（DoD）
- `colorize` エフェクトが追加され、GUI で RGBA を操作可能。
- `draw()` が `list[RenderLayer]` を返した場合、色ごとに順描画される。
- 太さは `boldify` の `boldness` で per-shape に設定・GUI から調整可能。
