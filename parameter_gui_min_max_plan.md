# parameter_gui スライダー min/max 入力追加計画（ドラフト）

目的: Parameter GUI 上の数値スライダーについて、GUI から直接 min/max を調整できる入力ボックスを追加し、パラメータの探索性と操作性を向上させる。

## 想定仕様（ドラフト）

- 対象:
  - `value_type == "int" | "float"` のスカラーパラメータ（Style/Palette/shape/effect すべてを含む）。
  - `value_type == "vector"` のベクトルパラメータ（min/max は全成分共通で適用する）。
- 表示位置:
  - カテゴリ別テーブルは「ラベル / Bars / MinMax / CC」の 4 列構成とする。
  - Bars 列にはこれまで通りスライダー（scalar/vector）を配置する。
  - MinMax 列に `min` / `max` 入力ボックスを横並びで配置する（vector も同じ min/max を全成分に適用）。
- 入力種別: Dear PyGui の `input_float` / `input_int` 相当の数値入力（型に応じて使い分ける）。
- 初期値: `desc.range_hint` もしくは `ParameterLayoutConfig.derive_range()` で得た min/max をそのままコピーする。
- 動作:
  - min/max を書き換えると、対応するスライダーの `min_value` / `max_value` を即時に更新する。
  - min >= max など明らかな不正値は採用せず、直前の有効な min/max を維持する（UI 側でのみガードし、例外は出さない）。
  - 実際の値（`ParameterStore` 内の original/override）は変更しない。Range はあくまで UI 表示用ヒントとして扱う。
- 永続化:
  - `persistence.py` の JSON フォーマットを拡張し、パラメータごとの min/max を保存/復元する。
  - 旧フォーマット（version 1）のファイルは従来通り override 値のみ読み込み、新フォーマット（version 2）では range 情報も復元する。

## やること（チェックリスト）

### 1. 設計/仕様の確定

- [x] 仕様ドラフトの確定（対象パラメータと UI レイアウトを明文化する）。
- [x] 永続化の範囲を決める（min/max も JSON 保存に含める）。
- [x] vector パラメータも min/max 指定可能とし、min/max は全成分共通のレンジとして適用する。

### 2. 状態管理の拡張

- [x] `ParameterStore` に「UI 用 range オーバーライド」を保持する構造（`range_override` / `all_range_overrides` など）を追加する（scalar/vector 共通）。
- [x] スライダー作成時に「ベース RangeHint」と「オーバーライド値」をマージするヘルパ関数（`_effective_range(desc)`）を `ParameterWindowContentBuilder` に実装する。
- [x] min/max 入力変更時に Store のレンジオーバーライドを更新し、scalar/vector 双方について `dpg.configure_item` 経由で該当スライダーの `min_value` / `max_value` を更新するフローを実装する。

### 3. Bars 列（通常パラメータ）の UI 変更

- [x] `src/engine/ui/parameters/dpg_window_content.py` の `_create_row_3cols` / `_create_bars` / `_create_cc_inputs` を「ラベル / Bars / MinMax / CC」の 4 列構成にリファクタリングする。
  - [x] Bars 列: 既存どおり、スライダー（scalar/vector）を幅いっぱいに配置する。
  - [x] MinMax 列: 各パラメータに対応する `min` / `max` 入力ボックスを横並びで配置する（vector も 1 組の min/max を全成分に共有する）。
  - [x] CC 列: 既存の CC 入力 UI を維持しつつ、MinMax 列との位置関係のみ変更する。
- [x] min/max 入力の `callback` から「入力値のパース → 妥当性チェック → range オーバーライド更新 → 対応スライダー（scalar/vector）の再設定」を呼び出す。

### 4. その他スライダー利用箇所の追従

- [x] Style セクションの `Global Thickness` スライダー（`build_style_controls` 内）も、共通ヘルパ（スライダー + min/max 入力）を使って min/max 調整可能にする。
- [x] Palette セクション（`build_palette_controls` / `_create_row_3cols` 経由）の int/float/vector スライダーも同じ仕組みで range オーバーライドに対応させる。

### 5. 永続化対応

- [x] `src/engine/ui/parameters/persistence.py` に UI range オーバーライド保存用のフィールド（例: `"ranges": { "<id>": {"min": ..., "max": ...}, ... }`）を追加する。
  - [x] `save_overrides` から Store の range オーバーライドを取得して JSON に書き出す設計にする。
  - [x] `load_overrides` 側で range 情報を読み出し、起動時に `ParameterStore.set_range_override` を通じて復元する。
- [x] JSON `version` の扱い（`1` → `2` など）と後方互換の方針を決める（version 1 は overrides のみ、version 2 では ranges も扱う）。
- [x] `tests/ui/parameters/test_persistence.py` に min/max 保存/復元のテストケースを追加する。

### 6. ドキュメント/アーキテクチャ反映

- [x] `architecture.md` の Parameter GUI セクションに「スライダーの UI range は GUI から変更可能であり、実値はクランプしない」旨を追記する。
- [ ] 必要であれば、関連する `AGENTS.md` に range オーバーライドが UI 専用であることを簡潔に記す。

### 7. 動作確認

- [ ] 代表的な sketch（shape/effect が複数あるもの）で Parameter GUI を起動し、min/max 入力からスライダー範囲が動的に切り替わることを目視確認する。
- [ ] 異常系（min >= max、非数値入力、極端に大きい/小さい値）でクラッシュしないことを確認する。
- [x] 変更ファイルに対して `pytest -q tests/ui/parameters/test_persistence.py` を実行し、緑であることを確認する（他の DPG 依存テストは環境により実行できない場合がある）。
