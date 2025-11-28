# palette base color L/C/h UI 置き換え計画（ドラフト）

目的: Parameter GUI の Palette セクションにある `Base Color` を、現在の RGB カラーピッカーではなく「Lightness / Chroma / Hue の 3 行スライダ」で操作できるようにする。そのうえで `api.C` および内部の Palette 生成に正しく反映させる。

## 前提 / 方針

- ベースカラーの内部表現は OKLCH ベースで扱う（L: 0–100, C: >=0, h: 0–360）。
- Parameter GUI のスライダは L/C/h の実値を直接いじる構成にする。
- 既存の `palette.base_color` RGBA ベクタは段階的に廃止し、L/C/h 用の 3 パラメータに置き換える（互換が必要なら一時的に併存させる）。
- Palette 生成ロジック（`engine.ui.palette.helpers.build_palette_from_values`）側で L/C/h → `ColorInput` に変換する経路を追加する。
- runtime/Worker 側への伝搬は既存の param snapshot 経路（`engine.ui.parameters.snapshot`）をそのまま利用する。

## やること（チェックリスト）

### 1. スキーマ/パラメータ設計

- [x] ベースカラー L/C/h の ID と Range を決める  
  - 候補: `palette.L`, `palette.C`, `palette.h`  
  - Range:  
    - L: 0.0–100.0, step=0.1  
    - C: 0.0–0.4 or 0.0–0.5 程度（後から調整可）  
    - h: 0.0–360.0, step=1.0
- [x] `ParameterDescriptor` の定義方針を決定  
  - `value_type="float"` / `category="Palette"` / `category_kind="palette"`  
  - RangeHint を明示設定してスライダ範囲を固定。
- [x] 既存 `palette.base_color`（RGBA vec4）の扱い方針  
  - GUI/ランタイムともに使用せず、L/C/h のみをベースカラーとして扱う（`snapshot.py` も `palette.L/C/h` に更新済み）。

### 2. ParameterManager 側: Descriptor 登録

- [x] `ParameterManager._register_palette_descriptors` を拡張し、L/C/h 用 Descriptor を追加  
  - 既定値の決め方:  
    - 1) 設定ファイルから line_color を読んで OKLCH に変換し、L/C/h を初期値にする  
    - 2) 変換に失敗した場合は (L=60, C=0.1, h=0) などの安全側既定値。
- [x] 既存 `palette.base_color` Descriptor の扱いを整理  
  - [x] Descriptor 登録を削除し、L/C/h のみを登録。  
  - [x] Snapshot/ランタイムは `palette.L/C/h` を利用するよう更新。

### 3. ParameterWindowContentBuilder 側: Palette ヘッダ UI

- [x] `build_palette_controls` から `Base Color` カラーピッカーを削除
- [x] 代わりに L/C/h 3 行スライダを追加  
  - 行構成例:  
    - `Lightness` / スライダ [0–100]  
    - `Chroma` / スライダ [0–0.5]  
    - `Hue` / スライダ [0–360]
- [x] L/C/h スライダの変更が Store に float 実値として保存されることを確認  
  - `store.set_override("palette.L", <float>)` 等。
- [ ] Palette ヘッダ内に小さなプレビュー（例: 現在ベースカラーの色サンプル）を残すかどうか決める  
  - 残す場合: L/C/h → sRGB に変換して 1 つの color_edit/色サンプルとして表示（編集不可）。

### 4. Palette 生成ロジックの更新

- [x] `engine.ui.palette.helpers.build_palette_from_values` に L/C/h 経路を追加  
  - 優先順位:  
    - L/C/h 値が揃っていればそれを優先  
    - なければ従来の `base_color_value`（RGBA/HEX）経路。
- [ ] L/C/h → OKLCH → `ColorInput` を構築する関数を実装  
  - 可能なら `palette.ColorInput.from_oklch` を利用。  
  - L/C/h のレンジチェックとクランプはここで行う。
- [x] `snapshot._update_palette_from_overrides`（worker 側）も L/C/h 対応  
  - overrides から `palette.L/C/h` を拾って `build_palette_from_values` に渡す。

### 5. API `C` 側の期待値確認

- [x] `api.palette.PaletteAPI` は内部 Palette オブジェクトにしか依存していないため、L/C/h 化で挙動が変わらないことを確認  
  - `C[0]` が L/C/h スライダで変えたベースカラーに従って更新されるかをテスト。

### 6. テスト & デバッグ

- [ ] 単体テスト追加（必要最小限）  
  - `tests/test_palette_api.py` に L/C/h スナップショットから Palette が生成されるケースを追加。  
  - GUI 依存部分はロジックレベル（`build_palette_from_values`/`_update_palette_from_overrides`）に限定。
- [ ] 手動確認  
  - Parameter GUI を起動し、Palette セクションの L/C/h スライダを動かして `C[0]` の色が変わることを確認。  
  - 既存 Style/HUD の挙動に影響が出ていないことを確認。

### 7. ドキュメント/メモ更新

- [ ] `palette_integration_plan.md` に L/C/h ベースカラー UI への変更を追記。  
- [ ] `README.md` に 「Palette セクションで L/C/h スライダを操作してカラーを決める」旨の説明を追加（必要であれば）。  
- [ ] 本計画に対するユーザーからのフィードバックを反映して、不要な項目/優先度を調整。

---

メモ:
- L/C/h のレンジや既定値は実際に触ってみながら微調整する前提とする（特に C の上限）。  
- 実装を始める前に、このチェックリストで削る/優先する項目があれば指示をもらってから着手する。
