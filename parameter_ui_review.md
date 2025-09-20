# Parameter UI モジュールコードレビュー詳細

## 対象範囲

- `src/engine/ui/parameters/panel.py`
- 参照仕様: `architecture.md` 63 行目付近（`RangeHint` の正規化/実レンジ分離）

## 主な所見

### 1. int パラメータスライダーが実レンジへ到達できない（重大）

- 該当コード: `src/engine/ui/parameters/panel.py:71-116`
- 現状の挙動:
  - `SliderWidget._range()` が `RangeHint.min_value / max_value` を直接返却。
  - これらの値は正規化域（0.0〜1.0）であり、`drag_to()` 内で `lo + normalized * (hi - lo)` を評価すると 0.0〜1.0 の範囲に固定される。
  - `value_type == "int"` の場合、最後に `int(round(value))` を実行するため、GUI から生成可能なのは 0 または 1 のみ。
- 期待仕様:
  - architecture.md に従い `RangeHint.min_value / max_value` は正規化域、実レンジは `RangeHint.mapped_min / mapped_max / mapped_step` に格納される。
  - GUI でも正規化値を `denormalize_scalar()` で実レンジへ変換してから整数化する必要がある。
- 影響:
  - `polygon.__param_meta__` のように min/max/step を実レンジで宣言している int パラメータは、GUI から実際の値（例: 3〜120）を指定できない。
  - MIDI/GU I を跨いだ一貫性が崩れ、`ParameterRuntime` での正規化ロジックと矛盾する。
- 修正方向:
  - `_range()` では `hint.mapped_min / mapped_max` を参照し、fallback を正規化域へ。
  - `drag_to()` では正規化値 → 実値変換に `normalization.denormalize_scalar()` を利用し、整数化は変換後に行う。
  - 逆変換（現在値 → 表示バー）時は `normalize_scalar()` を使って一致を取る。

### 2. 表示値が正規化域のままで実際の値が確認できない（中）

- 該当コード: `src/engine/ui/parameters/panel.py:190-201`
- 現状の挙動:
  - `_update_value()` で `current_value()` をそのままラベル表示。
  - `current_value()` は `ParameterStore.current_value()` を返すが、`SliderWidget._normalized()` と同じく正規化域（0.0〜1.0）の値を扱っている。
- 期待仕様:
  - GUI に表示される値は利用者にとって意味のある実レンジの値。
  - `ParameterRuntime`, CLI パラメータ, スタブ生成など他経路と同じく `denormalize_scalar()` を通すことで一貫性を保つ。
- 影響:
  - 0.0〜1.0 の数字が並び、意図した単位（例: 角度、セグメント数、ミリ秒など）が分からない。
  - 実レンジで `mapped_step` を設定しても GUI 表示が追従しない。
- 修正方向:
  - `_update_value()` で `RangeHint` を確認し、`normalization.denormalize_scalar()` を呼び出して実値へ変換。
  - int/bool の場合は変換後を `int` / `bool` 化し、フォーマット文字列で単位を表現する拡張も検討。

## 共通フォローアップ

- `normalization.normalize_scalar()` / `denormalize_scalar()` の導入箇所では、既存ユニットテスト（例: `tests/ui/parameters/test_parameter_store.py`）に加えて GUI ロジックのテストを追加して退行を防ぐ。
- 実装修正後は `architecture.md` の仕様との整合を再確認し、必要に応じてドキュメント更新を推奨。

## 修正計画（チェックリスト: 指摘 1「int パラメータスライダーが実レンジへ到達できない」）

- [x] `SliderWidget` から `normalization` モジュールを参照するための依存関係を整理（循環輸入の確認含む）。
- [x] `SliderWidget._range()`／`_normalized()`／`drag_to()` が `RangeHint.mapped_min/max/step` を用いて実レンジとの往復変換を行うよう改修。
- [x] `drag_to()` での値決定を `denormalize_scalar()` に委譲し、その後で `value_type == "int"` の丸め処理を適用。
- [x] `_normalized()` 側で `normalize_scalar()` を利用し、GUI 表示とバー位置が実レンジ基準で一致するか確認。
- [x] 既存 int/float パラメータの挙動をカバーするユニットテスト（例: `tests/ui/parameters` 配下）を追加し、正規化ラウンドトリップの精度を検証。
