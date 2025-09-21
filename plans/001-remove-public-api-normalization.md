# 公開APIの正規化（0..1 入力）完全撤廃 — 実装変更計画（最終方針）

目的: 公開 API（`api.shapes` / `api.effects`）における 0..1 正規化入力を完全廃止し、shape/effect 関数へ実レンジの引数をそのまま渡す。後方互換は一切不要とし、正規化仕様に関するコード・コメント・テスト・ドキュメントをすべて削除する。

注意: 幾何の「曲線パラメータ t ∈ [0,1]」のようなドメイン固有の 0..1 は存続（例: `trim(start_param, end_param)`）。撤廃対象は「UI/公開APIのための一律 0..1 → 実レンジ変換」仕様およびその痕跡（コード/コメント/テスト/文書）。

---

## 現状の把握（要点）

- 正規化レイヤ:
  - `src/engine/ui/parameters/normalization.py`（正規化/逆正規化/クランプ）
  - `src/engine/ui/parameters/value_resolver.py`（正規化値の登録と実レンジ化）
  - `src/engine/ui/parameters/runtime.py`（呼び出し前変換・オフライン解決）
- 公開 API 側の言及:
  - `src/api/shapes.py` の docstring に「UI 経由で 0..1 → 実レンジ変換」記述
  - `src/api/effects.py` のモジュール解説に「パラメータ正規化結果でハッシュ」等の言及（ハッシュは値のシリアライズ正規化であり 0..1 変換とは別物）
- UI 表示:
  - `src/engine/ui/parameters/panel.py` の `SliderWidget` が「正規化値」を保持し、表示時/実表示値は `denormalize_scalar()` を用いる
- メタデータ:
  - 各関数の `__param_meta__` は min/max/step（実レンジ）を保持し、`RangeHint` に投影
- テスト依存（例）:
  - `tests/ui/parameters/test_value_resolver.py`（正規化入力→実レンジの期待）
  - `tests/ui/parameters/test_slider_widget.py`（正規化値と逆変換の整合）

---

## 変更方針（互換性なし・全面削除）

1) 正規化関連の物理削除
   - `src/engine/ui/parameters/normalization.py` を削除（モジュール自体を廃止）。
   - すべての呼び出し元（import/参照）を削除または置換し、残骸を残さない。

2) UI/ランタイムは実値のみを扱う
   - `ParameterRuntime`/`ParameterValueResolver` は実値の登録・既定値マージ・型判定のみ（変換ロジックは一切持たない）。
   - `ParameterStore` は実値（float/int/bool/ベクトル）を保持。override も実値。
   - `RangeHint` は実レンジのみ（min/max/step）。正規化に関連する概念を削る。
   - `SliderWidget` は実値→表示比率 `(value - min)/(max - min)` のみで描画。表示クランプはUIのみで、内部値はクランプしない。

3) 正規化ユーティリティの完全撤去
   - 互換スタブは作らない。存在痕跡（コメント・docstring・型・テスト）も含め削除。

4) ドキュメント/規約の一掃
   - `src/api/shapes.py`/`src/api/effects.py` を含む全ファイルから「0..1 正規化」や「正規化値/逆正規化」への言及を削除。
   - ルート `AGENTS.md` の該当規定（0.0〜1.0 正規化入力義務）を削除/改定。
   - `engine/ui/parameters/AGENTS.md` など内製ドキュメントも全面更新。
   - `architecture.md` を現実装に同期。

5) テストの一掃/置換
   - 正規化を前提にしたテストを削除または書き換え（pass-through/実値前提へ）。
   - `tests/ui/parameters/test_value_resolver.py` から「正規化→実レンジ」期待を削除し、登録/既定値/型判定/ベクトルのパススルーのみ検証。
   - `tests/ui/parameters/test_slider_widget.py` から `denormalize_scalar` 等の依存を削除し、実値→比率/ラベル表示/ドラッグ挙動を検証。
   - `tests/ui/parameters/test_runtime.py` は前処理のパススルーを検証。

---

## 実施チェックリスト（タスク分解）

- [x] API 層の記述更新（正規化言及の削除）
  - [x] `src/api/shapes.py:1` 冒頭の Notes から「UI 経由は 0..1 正規化」記述を削除・改稿
  - [x] `src/api/effects.py:1` モジュール解説から「正規化」関連の文言を削除（ハッシュ説明は「直列化向け整形」等の語に置換）

- [x] 変換レイヤの撤去/パススルー化
  - [x] `src/engine/ui/parameters/normalization.py` を削除（物理削除）
  - [x] `src/engine/ui/parameters/value_resolver.py` から `normalize_scalar`/`denormalize_scalar`/`clamp_normalized` を全廃
  - [x] `src/engine/ui/parameters/runtime.py` の `resolve_without_runtime` をパススルー化。`before_*` でも変換を行わない

- [x] UI スライダー/トグルの実値化
  - [x] `src/engine/ui/parameters/panel.py` から `normalize_scalar`/`denormalize_scalar`/`clamp_normalized` の依存を除去
  - [x] 実値を `ParameterStore` に保持。進捗バー比率の計算を `(value - hint.min_value)/(hint.max_value - hint.min_value)` に変更
  - [x] `ToggleWidget` の既定値・override を実値（bool）に統一

- [x] RangeHint/Descriptor 整理（正規化痕跡の削除）
  - [x] `src/engine/ui/parameters/state.py` から `RangeHint.mapped_min/mapped_max/mapped_step` を削除し、参照箇所を全置換
  - [x] `ParameterDescriptor.default_value` は実値（型に応じた素の値）を保持

- [x] `__param_meta__` 利用の見直し
  - [x] 既存の `min`/`max`/`step` は実レンジヒントとしてのみ利用（正規化用途の文言を削除）
  - [x] メタ未定義時のヒューリスティック（`ParameterLayoutConfig.derive_range`）は据え置き

- [x] テスト更新（正規化関連の一掃）
  - [x] `tests/ui/parameters/test_value_resolver.py` の正規化期待・オーバースケール期待を削除し、実値パススルーのみ検証
  - [x] `tests/ui/parameters/test_slider_widget.py` から `denormalize_scalar` 等の依存を削除し、実値→比率の整合を検証
  - [x] `tests/ui/parameters/test_runtime.py` は前処理のパススルーを検証
  - [x] `tests/ui/parameters/test_parameter_store.py` は据え置き（内部は実値保持）

- [x] ドキュメント/規約更新（正規化文言の削除）
  - [x] ルート `AGENTS.md` から「公開パラメータは 0.0〜1.0 正規化入力」の規定を撤廃し、「実値をそのまま受け取る」へ改定
  - [x] `engine/ui/parameters/AGENTS.md` を同趣旨で更新
  - [x] `architecture.md` の該当節を実装と同期

- [ ] スタブ/型/整形
  - [ ] 公開 API のスタブ再生成（今回の変更では不要想定）
  - [x] 変更ファイルに限定した `ruff/black/isort/mypy/pytest` を順次実行（UI パラメータ系 15 テスト緑）

---

## 互換性・移行

- 後方互換なし。正規化仕様に依存するコードは削除/改修し、実値を直接渡す。
- `__param_meta__` は UI 表示用ヒントに限定。変換は行わない。

---

## 完了条件（DoD）

- API レイヤ/ランタイム/UI/テスト/文書から 0..1 正規化のコード/依存/文言が完全に除去され、実値で統一
- ルート `AGENTS.md`/`architecture.md`/各モジュール docstring の整合
- 変更対象ファイルに対する `ruff/black/isort/mypy/pytest` 緑
- スタブ生成後の差分が最小であること（意図した記述変更のみ）

---

## リスクと留意点

- UI スライダーの比率計算に `min==max` の退避処理が必要（現行の異常系ガードを実値版でも維持）
- 一部エフェクト/シェイプの docstring にある「0..1」は幾何学的パラメータ（例: `trim`）であり、撤廃対象ではない点を明示
- 既存テストの意図（表示上のみクランプ・内部はクランプしない）を実値ベースで再実装

---

## 正規化仕様の痕跡（削除対象）ガイド

rg 例（参考）:
- 関数/識別子: `denormalize_scalar|normalize_scalar|clamp_normalized|parameters\.normalization`
- 文言: `正規化`、`0\.\.?1`、`0..1`、`normalized`、`denormalize`、`normalize`、`RangeHint mapped`
- コメント: 「0..1 正規化ではありません」「UI/正規化のためのメタ情報」等

対象ファイル例:
- `src/engine/ui/parameters/normalization.py`（削除済み）
- `src/engine/ui/parameters/value_resolver.py`（置換・簡素化済み）
- `src/engine/ui/parameters/runtime.py`（置換済み）
- `src/engine/ui/parameters/panel.py`（置換済み）
- `src/engine/ui/parameters/state.py`（RangeHint から mapped_* を削除済み）
- `src/api/shapes.py` / `src/api/effects.py`（docstring 修正）
- `src/effects/*` / `src/shapes/*`（コメントから正規化言及を削除。幾何パラメータの 0..1 は維持）
- `tests/ui/parameters/*`（正規化依存の期待を削除/置換）
- `demo/shape_grid.py`（正規化ユーティリティ依存の削除/置換）
- `engine/ui/parameters/AGENTS.md` / ルート `AGENTS.md` / `architecture.md`（文言更新済み）

---

以上に従い、正規化仕様関連のコード・コメント・テスト・文書を全面的に削除し、実値ベースへ移行します。
