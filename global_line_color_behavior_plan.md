# グローバル line color とレイヤー描画の設計改善計画

目的: `run(..., line_color=...)` と Parameter GUI (`runner.line_color`) の「グローバル線色」を、レイヤー描画（`StyledLayer`）ときれいに統合し、直感的な優先順位で反映されるようにする。特に、`return p1(g1), p2(g2)` のように Sequence を返した場合でも、`.style` 未指定レイヤーにはグローバル line color が効くようにする。

---

## 現状整理

### 線色の適用経路

- 起動時の初期適用:
  - `run(..., line_color=...)` 引数、または `configs/default.yaml` / `config.yaml` の `canvas.line_color` から初期線色が決まる。
  - Parameter GUI 有効時は `ParameterManager.initialize()` が `runner.line_color` Descriptor を登録し、`apply_initial_colors()` が
    - `store.current_value("runner.line_color") or original_value(...)` を取り出して
    - `LineRenderer.set_line_color(_norm(...))` を 1 回だけ呼ぶ。
- ランタイム更新:
  - Parameter GUI から `runner.line_color` を変更すると
    - `dpg_window_content.store_rgb01("runner.line_color", app_data)` が `store.set_override(...)` を呼ぶ。
    - `ParameterStore._notify()` → `subscribe_color_changes()` が購読。
    - `"runner.line_color"` が含まれていれば `line_renderer.set_line_color(_norm(raw_val))` を呼ぶ。

### レイヤー描画側の挙動

- Worker 側（`engine.runtime.worker._normalize_result`）:
  - `draw(t)` が Sequence（`(Geometry | LazyGeometry)...`）を返すと、必ず `StyledLayer` 列に正規化する。
    - `.style` 付き LazyGeometry は `StyledLayer(color=...)` になる。
    - `.style` が無い Geometry/LazyGeometry も `StyledLayer(color=None)` としてレイヤー化される。
  - `LazyGeometry` 単体の場合は、`.style` が無ければレイヤー化せずにそのまま返す。
- Renderer 側（`LineRenderer.draw`）:
  - `_frame_layers` がある場合（レイヤー描画モード）:
    - `layer.color is not None` → `set_line_color(layer.color)` し、`_sticky_color` に保存。
    - `layer.color is None` → `set_line_color(_base_line_color)` を呼び、ベース色に戻す。
  - `_frame_layers` が無い場合（Geometry 直描画）:
    - 直前に `set_line_color(...)` された色がそのまま使われる。

### いま起きている問題

- `return p1(g1) + p2(g2)`:
  - 1 つの LazyGeometry（`result`）として扱われ、`.style` 無しならレイヤー化されない。
  - よって Parameter GUI → `set_line_color(...)` の更新がそのまま描画に効く。
- `return p1(g1), p2(g2)`:
  - 2 レイヤー（`StyledLayer(color=None)`×2）として扱われる。
  - レンダラは毎フレーム `set_line_color(_base_line_color)` に戻すため、
    - GUI からの `set_line_color(...)` は直後の描画で上書きされてしまい、見た目に反映されない。
- `.style(color=...)` を使ったレイヤーについては、明示色が優先される設計は問題ないが、
  - `.style` 未指定のレイヤーにグローバル line color が効いてほしい、という直感とズレる。

---

## 目標と優先順位

### 目標挙動

- グローバル線色の源:
  - 優先順位: 「`run(..., line_color=...)` 引数 > Parameter GUI の `runner.line_color` override > config (`canvas.line_color`)」。
  - Parameter GUI が有効な場合、`runner.line_color` は「実行時に変更可能なグローバル線色」として扱う。
- レイヤー描画時の色決定ルール:
  - `StyledLayer.color is not None`:
    - 明示スタイルが最優先。Parameter GUI のグローバル線色より強い。
  - `StyledLayer.color is None`:
    - 常に「最新のグローバル線色」を使う。
    - グローバル線色は `runner.line_color`（GUI）や `run(..., line_color=...)` から更新される。
- 非レイヤー描画（Geometry 直描画）の挙動:
  - 現行どおり `set_line_color(...)` の最新値が適用される。
  - グローバル線色更新も `set_line_color(...)` を通じて反映。

### 非目標（維持したい点）

- `.style(color=...)` の意味を変えない:
  - 各レイヤーの見た目は、style で明示された通りに描画される。
  - グローバル線色は「style が無いレイヤー向けのデフォルト」として扱う。
- API 変更は最小限:
  - 既存の公開 API（`run`, `LineRenderer.set_line_color`）を壊さずに拡張する。
  - Parameter GUI を使わない場合の挙動（`run(..., line_color=...)` だけで完結）は維持する。

---

## 設計方針（高レベル）

1. **`LineRenderer` に「ベースライン色更新 API」を追加する**
   - 例: 
     - `set_base_line_color(rgba: Sequence[float]) -> None` を新設し、
       - 内部で `normalize_color` → `_base_line_color` を更新。
       - （必要なら）`set_line_color` も呼んで「現在色」と揃える。
     - または `set_line_color(rgba, *, as_base: bool = False)` のようにフラグで兼用する案も検討。
   - レイヤー描画の `layer.color is None` パスは、従来どおり `_base_line_color` を参照。

2. **初期 line color の適用を「ベース色更新」に寄せる**
   - `apply_initial_colors()`:
     - これまでの `line_renderer.set_line_color(_norm(ln_val))` に加え（または置き換え）、
       - 新 API（`set_base_line_color`）で `_base_line_color` も更新する。
     - `run(..., line_color=...)` 引数が存在する場合は、config よりこちらを優先。

3. **Parameter GUI からの line color 更新でベース色も更新する**
   - `subscribe_color_changes()`:
     - `"runner.line_color"` に対しては
       - `line_renderer.set_base_line_color(_norm(val))` を呼ぶことで `_base_line_color` を更新。
       - 可能なら同時に `set_line_color` も呼んで即時反映（非レイヤー描画時）。
     - 背景色変更時の「自動 line color 推定」（`runner.background` による自動黒/白）との兼ね合い:
       - 既存ロジック: `runner.background` 更新時、`runner.line_color` override が無い場合のみ自動色を適用。
       - ここは維持しつつ、「GUI で一度 `runner.line_color` を明示したら、以降は自動推定に戻さない」というポリシーを明確化。

4. **StyledLayer 側の仕様は維持**
   - `worker._normalize_result()`:
     - `.style(color=...)` のある LazyGeometry は、引き続き `StyledLayer(color=rgba)` に変換。
     - `.style` 無しの Geometry/LazyGeometry は `StyledLayer(color=None)` または geometry 直描画。
   - Renderer は
     - `layer.color is not None` → 明示色
     - `layer.color is None` → `_base_line_color`（＝グローバル線色）
     を使うだけで、構造はそのまま。

5. **優先順位の明文化と architecture.md 反映**
   - `architecture.md` の Parameter GUI / ライン色の節に、
     - グローバル線色とレイヤー色の優先順位（run 引数 / GUI / config / style）の表現を追記。
   - 特に「Sequence で返した場合の挙動」の記述を追加し、`p1(g1), p2(g2)` でも直感通りの動きになることを明示。

---

## 具体タスク（チェックリスト）

### 1. 振る舞い仕様の整理・ドキュメント

- [ ] `architecture.md` に、グローバル line color と StyledLayer の色決定ルールを追記する。
- [ ] `run(..., line_color=...)` / config / GUI / `.style(color=...)` の優先順位を短い箇条書きで明文化する。

### 2. Renderer API 設計

- [ ] `LineRenderer` にベース色更新のための API を追加する（`set_base_line_color` など）。
- [ ] 追加 API で `_base_line_color` を更新し、必要に応じて `set_line_color` も呼ぶかどうかを決定する。
- [ ] 既存の `set_line_color` 呼び出し（初期適用やテスト）との互換性を確認する。

### 3. 初期 line color 適用ロジックの調整

- [ ] `apply_initial_colors()` を見直し、初期 line color を「ベース色」として更新するように変更する。
- [ ] `run(..., line_color=...)` 引数がある場合は、それを `_base_line_color` の初期値として最優先に適用する。
- [ ] config (`canvas.line_color`) とのフォールバック順を確認・テストケース化する。

### 4. Parameter GUI からの更新経路の変更

- [ ] `subscribe_color_changes()` の `"runner.line_color"` ハンドリングを、「ベース色更新 API」を使う形に変更する。
- [ ] 背景変更による自動 line color 推定との兼ね合い（`runner.background` のハンドラ）を再確認し、「GUI override があるかどうか」で分岐するよう整理する。
- [ ] Parameter GUI 無効時（`parameter_manager is None`）との挙動差分を確認する（実質従来どおりかを確認）。

### 5. レイヤー挙動の確認

- [ ] `worker._normalize_result()` の現行仕様を再確認し、`.style` 無し Sequence（`p1(g1), p2(g2)`）が `StyledLayer(color=None)` になる前提を確認する。
- [ ] Renderer 側で「`layer.color is None` → `_base_line_color`」の経路が整合しているかを確認する。
- [ ] `.style` を混在させたケース（片方のみ style 指定、両方 style 指定）で、期待通り「明示スタイルが優先される」ことを仕様として整理する。

### 6. テスト計画

- [ ] 単一 Geometry （非レイヤー）＋ GUI line color 更新で色が変わるテストを追加/維持する。
- [ ] `return p1(g1), p2(g2)`（どちらも style 無し）のスケッチで、GUI line color が両レイヤーに反映されるテストを追加する。
- [ ] 片方だけ `.style(color=...)` を持つ `return p1(g1), p2(g2)` で、「style 指定レイヤーは固定色、style 無しレイヤーは GUI line color に追従」するテストを追加する。
- [ ] `return p1(g1) + p2(g2)` の挙動が現状と変わらないこと（グローバル line color が効くこと）を確認する。

### 7. 互換性と移行

- [ ] Parameter GUI 無効で `run(..., line_color=...)` のみを使うケースで、従来と同じ線色になることを確認する。
- [ ] 既存スケッチで `.style` を多用しているものがあれば、挙動差分（特に Sequence 返し）を spot check する。

---

この計画に沿って実装すれば、

- Sequence 返し（`return p1(g1), p2(g2)`）でも、`.style` 未指定レイヤーはグローバル line color に追従し、
- `.style` 指定レイヤーは指定通りの色で描画される、

という直感的な挙動に揃えられる見込みです。

この内容で問題なければ、次のステップとして具体的なコード変更案（`LineRenderer` の API 変更案と `subscribe_color_changes` の改修案）に落とし込みます。  
気になる点や「ここはこうしてほしい」という希望があれば、この md に追記する形で調整します。

