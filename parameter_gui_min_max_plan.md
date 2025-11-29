# parameter_gui スライダー min/max 入力追加計画（ドラフト）

目的: Parameter GUI 上の数値スライダーについて、GUI から直接 min/max を調整できる入力ボックスを追加し、パラメータの探索性と操作性を向上させる。

## 想定仕様（ドラフト）

- 対象: `value_type == "int" | "float"` のスカラーパラメータ（Style/Palette/shape/effect すべてを含む）。
- 表示位置: カテゴリ別テーブルの Bars 列にあるスライダーの直下に、小さな `min` / `max` 入力ボックスを横並びで配置する（vector は当面対象外）。
- 入力種別: Dear PyGui の `input_float` / `input_int` 相当の数値入力（型に応じて使い分ける）。
- 初期値: `desc.range_hint` もしくは `ParameterLayoutConfig.derive_range()` で得た min/max をそのままコピーする。
- 動作:
  - min/max を書き換えると、対応するスライダーの `min_value` / `max_value` を即時に更新する。
  - min >= max など明らかな不正値は採用せず、直前の有効な min/max を維持する（UI 側でのみガードし、例外は出さない）。
  - 実際の値（`ParameterStore` 内の original/override）は変更しない。Range はあくまで UI 表示用ヒントとして扱う。
- 永続化:
  - 第一段階ではセッション内のみ有効（アプリ終了でリセット）。
  - 余力があれば `persistence.py` の JSON フォーマットを拡張して min/max を永続化する（後述）。

## やること（チェックリスト）

### 1. 設計/仕様の確定

- [ ] 仕様ドラフトの確定（対象パラメータと UI レイアウトを明文化する）。
- [ ] 永続化の範囲を決める（セッションのみ / JSON 保存まで含めるか）。
- [ ] vector パラメータへの適用有無（当面は対象外とするか）を決める。

### 2. 状態管理の拡張

- [ ] `ParameterWindowContentBuilder` に「UI 用 range オーバーライド」を保持する構造（例: `dict[str, tuple[float, float]]`）を追加する。
- [ ] スライダー作成時に「ベース RangeHint」と「オーバーライド値」をマージするヘルパ関数（例: `_effective_range(desc)`）を実装する。
- [ ] min/max 入力変更時に上記 dict を更新し、`dpg.configure_item` 経由で該当スライダーの `min_value` / `max_value` を更新するフローを実装する。

### 3. Bars 列（通常パラメータ）の UI 変更

- [ ] `src/engine/ui/parameters/dpg_window_content.py` の `_create_bars` 内、int/float 分岐を「スライダー + min/max 入力」構成に変更する。
  - [ ] 1 行目: 既存どおり、スライダーを幅いっぱいに配置する。
  - [ ] 2 行目: 小さな `min` / `max` 入力ボックスを横並びで配置する（ラベルはプレースホルダや tooltip で簡潔に表示）。
- [ ] min/max 入力の `callback` から「入力文字列のパース → 妥当性チェック → range dict 更新 → 対応スライダーの再設定」を呼び出す。
- [ ] CC 列 / CC バインディング（`_create_cc_inputs`）には影響を与えない。

### 4. その他スライダー利用箇所の追従

- [ ] Style セクションの `Global Thickness` スライダー（`build_style_controls` 内）に min/max 入力を追加するかどうか決める。
  - [ ] 追加する場合は、共通ヘルパ（スライダー + min/max 入力）で構築できるように整理する。
- [ ] Palette セクション（`build_palette_controls` / `_create_row_3cols` 経由）の int/float スライダーも同じ仕組みで range オーバーライドに対応させる。

### 5. 永続化対応（オプション）

- [ ] `src/engine/ui/parameters/persistence.py` に UI range オーバーライド保存用のフィールド（例: `"ranges": { "<id>": {"min": ..., "max": ...}, ... }`）を追加する。
  - [ ] `save_overrides` で range dict を受け取れるよう API を拡張するか、Parameter GUI 側から別ヘルパ経由で書き込むかを決める。
  - [ ] `load_overrides` 側で range 情報を読み出し、Parameter GUI 初期化時に `ParameterWindowContentBuilder` へ適用するフックを用意する。
- [ ] JSON `version` の扱い（`1` → `2` など）と後方互換の方針を決める。
- [ ] `tests/ui/parameters/test_persistence.py` に min/max 保存/復元のテストケースを追加する。

### 6. ドキュメント/アーキテクチャ反映

- [ ] `architecture.md` の Parameter GUI セクションに「スライダーの UI range は GUI から変更可能であり、実値はクランプしない」旨を追記する。
- [ ] 必要であれば、関連する `AGENTS.md` に range オーバーライドが UI 専用であることを簡潔に記す。

### 7. 動作確認

- [ ] 代表的な sketch（shape/effect が複数あるもの）で Parameter GUI を起動し、min/max 入力からスライダー範囲が動的に切り替わることを目視確認する。
- [ ] 異常系（min >= max、非数値入力、極端に大きい/小さい値）でクラッシュしないことを確認する。
- [ ] 変更ファイルに対して `ruff` / `black` / `isort` / `mypy` / `pytest -q tests/ui/parameters/test_persistence.py` を実行し、緑であることを確認する。

## 確認したいポイント

- min/max の永続化（JSON 保存）まで今回のタスクに含めるかどうか。
- vector パラメータについても、将来的に per-component range を GUI から調整したいか（今回スコープ外なら明示的に除外する）。
- min/max 入力の UI（Bars 列内の 2 行構成）で問題ないか、別案があれば教えてほしい。

