# Thickness Parameter GUI 改善計画

目的: パラメータ GUI 上の thickness を 0.0001〜0.001 の範囲で、量子化の影響を受けにくく滑らかに制御できるようにする。ただし公開パラメータとしての意味は「実太さ」のまま保つ。

## チェックリスト

- [x] 現状仕様の確認
  - [x] `runner.line_thickness` / `layer.*.thickness` の RangeHint と step の確認（`src/engine/ui/parameters/manager.py`）
  - [x] Parameter GUI の thickness 表示フォーマット（value_precision / format）の適用箇所確認（`src/engine/ui/parameters/dpg_window_content.py`）
- [x] RangeHint.step の細分化
  - [x] グローバル thickness の RangeHint.step を 1e-5〜1e-6 の範囲で決定し変更する
  - [x] レイヤー thickness の RangeHint.step をグローバル thickness と同じ粒度に揃える
  - [x] RangeHint.min/max を 0.0001〜0.001 程度に調整するかどうか決定する
- [x] thickness 用の表示桁数の増加
  - [x] Global thickness スライダに thickness 専用の表示フォーマット（例: 小数 6 桁）を適用する
  - [x] レイヤー thickness スライダにも同じフォーマットを適用する
  - [x] 既存の `ParameterLayoutConfig.value_precision` との役割分担を整理し、thickness 専用ロジックに閉じ込める
- [x] 量子化・永続化への影響確認
  - [x] `src/engine/ui/parameters/persistence.py` の override 量子化が RangeHint.step の変更だけで期待通りになるか確認する
  - [x] `src/common/param_utils.py` による pipeline 量子化との整合性を確認し、必要であれば対象パラメータの `__param_meta__['step']` を調整する
- [x] ドキュメント更新
  - [x] `architecture.md` の thickness 関連記述に変更が必要か確認し、差分があれば更新する
- [ ] 動作確認
  - [ ] Parameter GUI 上で thickness を 0.0001〜0.001 の範囲で操作し、描画結果と操作感を確認する
  - [x] GUI override の保存・再起動後の復元が期待通りか確認する
  - [ ] 他のエフェクトパラメータの量子化・表示が悪化していないか確認する

## 追加で相談したい点

- [ ] thickness の UI レンジを「0.0001〜0.001 固定」とするか、Min/Max 入力で任意に広げられる前提にするか
- [ ] 将来的に、同様の「極小レンジが欲しいパラメータ」にも thickness と同じパターン（細かい RangeHint.step + 専用表示桁）を適用するか
