# palette 自動適用（背景/線色/レイヤー）計画（ドラフト）

目的: Parameter GUI の palette セクションで計算したパレットを、描画の背景色・グローバル線色・レイヤーごとの線色に自動適用できるようにし、ユーザーが palette 系パラメータだけを触ってザッピング的に配色バリエーションを試せるようにする。自動適用は明示的にオフにできる。

---

## 仕様案（ユーザー視点）

- palette セクションに「パレット自動適用」のトグルを追加する。
  - 例: `palette.auto_apply_mode`（enum）  
    - `"off"`: 既存挙動（palette はプレビューと `api.C` 用のみで、背景/線色には触れない）
    - `"bg_global_and_layers"`: 現在の背景色を基準に、グローバル線色 (`runner.line_color`) と `layer.*.color` を自動適用
  - UI ラベル例:  
    - Label: `Apply palette to colors`  
    - Choices: `Off / BG+Global / BG+Global+Layers`
- palette のパラメータ（`palette.L/C/h`, `palette.type`, `palette.style`, `palette.n_colors`）を動かすと、以下がリアルタイムに変わる。
  - 背景色: キャンバス背景 (`runner.background`)
  - グローバル線色: `runner.line_color`（レイヤー色未指定時の既定線色）
  - レイヤー線色: Parameter GUI で生成される `layer.{key}.color` 群
- パレット自動適用が「オン」の間は、palette パラメータ変更のたびに色が更新され、描画結果がザッピング的に切り替わる。
- パレット自動適用を「off」に戻すと、それ以降の palette 変更では `runner.*`/`layer.*` の色は更新されない（既に適用済みの値はそのまま残る）。
- 手動の色指定との優先順位:
  - パラメータ系の優先度は現状維持:  
    「draw の明示引数 > Parameter GUI override > default」
  - `Layer(color=...)` で draw 側が色を明示している場合でも、`layer.{key}.color` に GUI override が入れば、それが優先される（現状どおり）。  
    → palette 自動適用は「レイヤー色を GUI 指定で上書きする」機能として動く。
- 永続化:
  - palette から自動で書き込まれた `runner.*` / `layer.*` の override も、他の GUI override と同様に JSON に保存・復元される。
  - `palette.auto_apply_mode` 自体も override として永続化し、次回起動時の既定挙動を保つ。

---

## デザイン方針（内部仕様）

### 1. 適用トリガー

- 既存の palette 更新トリガーをそのまま利用する。
  - `src/engine/ui/parameters/dpg_window_content.py:on_store_change` で  
    `pid.startswith("palette.") or pid == "runner.line_color"` のとき `palette_dirty=True`。
  - フラグが立つと `_refresh_palette_preview()` が呼ばれる。
- palette 自動適用ロジックは `_refresh_palette_preview()` 内に追加する。
  - palette オブジェクトを計算し `util.palette_state.set_palette(palette_obj)` した直後に、
    `palette.auto_apply_mode` の値を見て `runner.*` / `layer.*` に色を流し込む。

### 2. 色マッピング戦略（背景固定版）

- palette オブジェクトから `colors` を取得し、OKLCH と sRGB の両方を使って「線のコントラスト」だけを最適化する。
- 基本方針:
  - 背景色: 既存のキャンバス設定 (`runner.background`) を尊重し、自動では変更しない。
  - グローバル線色: 背景との輝度差が最大になるパレット色を 1 つ選び、`runner.line_color` へ書き込む。
  - レイヤー線色: 背景との輝度差が大きい順にパレット色を並べ、`layer.{key}.color` へ順に割り当てる。
- 具体的アルゴリズム案:
  - 背景取得:
    - `ParameterStore` から `runner.background` の現在値（なければ original）を取り、`util.color.normalize_color` で sRGB(0..1) に正規化。
    - sRGB から相対輝度 `Y_bg = 0.2126*r + 0.7152*g + 0.0722*b` を計算。
  - 線色候補選択:
    - palette.colors の各色について sRGB を取得し、同様に輝度 `Y` を求める。
    - `abs(Y - Y_bg)` が最大の色をグローバル線色候補とする。
    - それでも `abs(Y - Y_bg)` が小さい場合（例: `< 0.15`）は、背景とのコントラストが明確になるよう黒/白を選ぶ（Y_bg に応じて「暗い線/明るい線」を決定）。
  - レイヤー線色:
    - `ParameterStore.descriptors()` から `id` が `layer.` で始まり `.color` で終わるものを列挙。
    - 各パレット色について `abs(Y - Y_bg)` を計算し、これが大きい順に並べたリストを作る。
    - レイヤー ID をソートし、このリストを循環しながら `layer.{key}.color` に割り当てる。
    - こうすることで、レイヤーごとに背景とのコントラストが高めの色が順番に付与される。

### 3. ParameterStore への書き込み

- `_refresh_palette_preview()` で palette_obj を計算後、`palette.auto_apply_mode` を取得する。
  - `mode == "off"` の場合は従来どおりプレビューのみ更新し、色は変更しない。
  - `mode != "off"` の場合のみ、以下の override を設定する。
- 背景/グローバル線色:
  - `runner.background`, `runner.line_color` の descriptor が存在する場合のみ `store.set_override(...)` を呼ぶ。
  - これにより
    - 初期設定時: `apply_initial_colors` が `ParameterStore` の original/current 値からレンダラーへ反映。
    - 変更時: `subscribe_color_changes` が `ParameterStore.subscribe()` 経由で更新を拾い、UI スレッドで `set_background_color` や `set_base_line_color` を呼ぶ。
- レイヤー線色:
  - `store.descriptors()` から `layer.{key}.color` を列挙し、順に override を入れる。
  - これにより
    - GUI 側: `_apply_layer_overrides` が `store.current_value(...)` を拾って、ユーザー draw の `Layer` に GUI 色を反映。
    - ワーカー側: `engine.runtime.worker._apply_layer_overrides` が snapshot 経由の `overrides` から `layer.{key}.color` を適用。
- 再帰の防止:
  - `set_override("runner.background", ...)` などで `on_store_change` が再度呼ばれるが、
    - `palette_dirty` は `palette.*` または `"runner.line_color"` でのみ立つ。
    - `runner.background` / `layer.*.color` はプレビュー再計算をトリガーしない。
  - そのため無限ループにはならない（設計上の前提として明記）。

### 4. 既存挙動との整合

- 現在の「背景色変更時に line_color 未指定なら自動で黒/白を選ぶ」ロジックは維持する。
  - palette 自動適用モードでは、通常 `runner.line_color` も同時に上書きされるため、この自動決定はほとんど発動しない。
  - palette 自動適用を Off にして背景だけ変更した場合は、現行どおりの自動線色が動く。
- `api.palette.C`/`util.palette_state` はそのまま利用し、palette 自動適用は純粋に「ParameterStore の override を更新する」だけにとどめる。

---

## 実装計画（チェックリスト）

### 1. パラメータ/GUI デザイン

- [x] `src/engine/ui/parameters/manager.py` の `_register_palette_descriptors` に `palette.auto_apply_mode` の Descriptor を追加  
  - `id="palette.auto_apply_mode"`, `value_type="enum"`, `category="Palette"`, `category_kind="palette"`  
  - `choices=["off", "bg_global_and_layers"]`, `default_value="bg_global_and_layers"`（初期案: palette を使うときはデフォルトで効かせる）
- [x] `src/engine/ui/parameters/dpg_window_content.py` の `build_palette_controls` に `palette.auto_apply_mode` の UI を追加  
  - `value_type="enum"` なので既存ロジックで `radio_button` or `combo` が自動選択される（必要なら palette セクション内だけ `radio_button` を強制）。

### 2. palette → 色マッピングヘルパー

- [x] `src/engine/ui/palette/helpers.py` か `src/engine/ui/parameters/dpg_window_content.py` 内に、「palette と layer ID 一覧から背景/線色/レイヤー色を決める純粋関数」を切り出す。
  - 入力: `Palette` オブジェクト、`apply_mode`（enum 値）、`layer_color_ids`（`["layer.layer0.color", ...]`）
  - 出力: `bg_rgba | None`, `line_rgba | None`, `layer_color_map: dict[str, tuple[float,float,float,float]]`
- [x] 上記ヘルパーで OKLCH L ベースの簡易アルゴリズムを実装（背景=最大L, 線=最大コントラスト, レイヤー=残りを循環）。
- [x] 将来の拡張用に、ヘルパー関数を単体テストしやすい形に保つ（UI 依存を持たない）。

### 3. `_refresh_palette_preview` 拡張

- [x] 既存の palette 計算 + `util.palette_state.set_palette(palette_obj)` の直後に、自動適用ロジックを追加。
- [x] `store.current_value("palette.auto_apply_mode")`（無ければ default）を取得し、`"off"` 以外なら:
  - [x] `layer_color_ids` を `self._store.descriptors()` から組み立てる。
  - [x] マッピングヘルパーを呼び出して色割り当てを決定。
  - [x] `runner.background` / `runner.line_color` / `layer.*.color` に対して `set_override` を呼ぶ（存在チェック付き）。
- [x] 既存のスウォッチ描画ロジックはそのまま保持し、palette_obj が None のときは何もしない（現在の仕様を踏襲）。

### 4. ランタイム連携確認

- [x] `src/api/sketch_runner/params.py` の `apply_initial_colors` / `subscribe_color_changes` が、`runner.background` / `runner.line_color` の変更をすでに拾っていることを確認（コードレビューのみ）。
- [x] レイヤー色については、`ParameterManager._apply_layer_overrides` と `engine.runtime.worker._apply_layer_overrides` が `layer.*.color` の override を適用していることを確認し、追加の変更が不要であることをメモする。

### 5. テスト方針

- [ ] palette 自動適用用の最小単体テストを追加（例: `tests/ui/parameters/test_palette_auto_apply.py`）。
  - [ ] 疑似 `ParameterStore` とダミー palette を用意し、`apply_mode` ごとに `runner.*` / `layer.*` への割り当てが期待通りになることを確認。
  - [ ] `apply_mode="off"` のときに `runner.*` / `layer.*` に一切 override が入らないことを確認。
- [ ] 既存の palette UI テスト（あれば）に影響していないことを確認。

### 6. ドキュメント/アーキテクチャ更新

- [ ] `architecture.md` に「palette セクション → ParameterStore → runner/layer 色への自動適用」フローを追記し、`util.palette_state` / `api.palette.C` との関係も簡潔に整理する。
- [ ] README か docs 内の Parameter GUI 説明に「palette 自動適用」の一文と簡単な使い方を追加する。

---

## メモ・相談ポイント

- `palette.auto_apply_mode` の default を `"bg_global_and_layers"` にするか `"off"` にするかは、既存スケッチとの互換性と「palette を開いた瞬間から気持ちよく動く」体験のバランスで決めたい。
  - 現状ユーザー数は少なく破壊的変更も許容されるため、「palette を有効にしたらデフォルトで BG+Global+Layers に効く」方が UX は良さそう。
- 背景を暗色系にしたいケース向けに、「背景は最小 L を選ぶ」モードを将来的に追加する余地を残しておく（今回は仕様をシンプルに保つため見送り）。
- レイヤー色は `layer.*.color` による GUI override が強いので、「レイヤー側で明示的に color を指定している場合は palette 自動適用をスキップする」ような高度な判定も考えられるが、初回はシンプルに「Parameter GUI のレイヤー色が常に最終優先」という方針で実装する。
