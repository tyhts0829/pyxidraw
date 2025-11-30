# エフェクト角度パラメータ degree 統一計画

## 目的・ゴール

- すべての「エフェクト」の角度系引数を degree ベースに統一する。
- パラメータ名・`__param_meta__`・docstring・API スタブ・テスト・ドキュメントを一貫させる。
- 内部実装では必要に応じて radian を使いつつ、外部インターフェースは degree としてクリーンに保つ。
- 角度に関する引数名から `rad/deg` などの単位サフィックスを排除し、意味ベースの名前で統一する。

---

## 現状の角度関連パラメータ一覧（エフェクト）

### rotate エフェクト

- ファイル
  - `src/effects/rotate.py`
- 公開関数
  - `rotate(g, *, auto_center, pivot, angles_rad)`
- 現状
  - 引数 `angles_rad: Vec3` … XYZ 回転角 [rad]。
  - `PARAM_META["angles_rad"]` … `min=(-π, -π, -π)`, `max=(π, π, π)`。
  - モジュール/関数 docstring も「ラジアン」表記。
  - `LazyGeometry.rotate`（`src/engine/core/lazy_geometry.py`）から `angles_rad` でパラメータ連携。
  - API スタブ `src/api/__init__.pyi` に `rotate(..., angles_rad: Vec3, ...)` としてエクスポート。
  - 各種呼び出し箇所:
    - `src/engine/core/geometry.py` 内の使用例。
    - `docs/spec/pipeline.md` 内の使用例。
    - `architecture.md` の説明（affine の例を含む）。
    - テスト: `tests/effects/test_rotate.py`, `tests/api/test_pipeline_*`, `tests/perf/test_pipeline_perf.py`, `tests/smoke/test_fill_rotation_invariance.py` など。
    - スケッチ: `sketch/251117.py`, `sketch/251120.py`, `sketch/251129.py` など。

### affine エフェクト

- ファイル
  - `src/effects/affine.py`
- 公開関数
  - `affine(g, *, auto_center, pivot, angles_rad, scale, delta)`
- 現状
  - 引数 `angles_rad: Vec3` … Rz·Ry·Rx の合成回転角 [rad]。
  - `PARAM_META["angles_rad"]` … `min=(0,0,0)`, `max=(2π,2π,2π)`。
  - docstring の既定値記述・単位も「ラジアン」。
  - 内部で `rotate_radians = np.array(angles_rad, ...)` として NumPy ベクトルに変換。
  - エンジン側:
    - `src/engine/core/affine_ops.py::rotate` … `angles_rad` を受け取り `Geometry.rotate` へ渡す純関数。
    - `LazyGeometry.rotate` からも `angles_rad` として affine_ops に受け渡し。
  - API スタブ: `src/api/__init__.pyi` の `affine(..., angles_rad: Vec3, ...)`。
  - 呼び出し箇所:
    - `architecture.md`（affine の代表シグネチャ）。
    - テスト: `tests/effects/test_affine_*`, `tests/smoke/test_fill_rotation_invariance.py`。
    - スケッチ: `sketch/251117.py`, `sketch/251120.py`, `sketch/251129.py` など。

### repeat エフェクト

- ファイル
  - `src/effects/repeat.py`
- 公開関数
  - `repeat(g, *, ..., angles_rad_step, ...)`
- 現状
  - 引数 `angles_rad_step: Vec3` … 終点回転角 [rad]（X, Y, Z）。0→angles_rad_step を補間。
  - `PARAM_META["angles_rad_step"]` … `min=(-π,-π,-π)`, `max=(π,π,π)`。
  - 内部で `rotate_radians = np.array(angles_rad_step, ...)` として使用。
  - API スタブ: `src/api/__init__.pyi` の `repeat(..., angles_rad_step: Vec3, ...)`。
  - 呼び出し箇所:
    - テスト: `tests/effects/test_repeat_basic.py` など。

### twist エフェクト

- ファイル
  - `src/effects/twist.py`
- 公開関数
  - `twist(g, *, angle_rad, axis)`
- 現状
  - 引数 `angle_rad: float` … 最大ねじれ角 [rad]。0 で no-op。
  - `PARAM_META["angle_rad"]` … `min=0.0`, `max=2π`。
  - 内部で `max_rad = float(angle_rad)` → `twist_rad` を計算して `np.sin/np.cos` に渡す。
  - API スタブ: `src/api/__init__.pyi` の `twist(..., angle_rad: float, ...)`。
  - 呼び出し箇所:
    - テスト: `tests/effects/test_twist*`（存在する場合）、`tests/api/__init__.pyi` 経由のパイプラインテスト。

### fill エフェクト

- ファイル
  - `src/effects/fill.py`
- 公開関数
  - `fill(g, *, angle_sets, angle_rad, density, spacing_gradient, remove_boundary)`
- 現状
  - 引数 `angle_rad: float | list[float] | tuple[float, ...]` … ハッチ角 [rad]。
  - `PARAM_META["angle_rad"]` … `min=0.0`, `max=2π`。
  - docstring で「angle_rad: ... ハッチ角（ラジアン）」と明示。
  - 内部:
    - `_as_float_seq(angle_rad)` で配列化し `angle_seq` として使用。
    - `_generate_line_fill` / `_fill_single_polygon` / `_scan_span_for_angle_xy` など多くのヘルパー関数が `angle` を radian 前提で `np.sin/np.cos` に渡す。
  - API スタブ: `src/api/__init__.pyi` の `fill(..., angle_rad=..., ...)` とメタ情報。
  - 呼び出し箇所:
    - テスト: `tests/effects/test_fill_*`, `tests/smoke/test_fill_rotation_invariance.py`, `tests/test_effect_fill_*`。
    - スケッチ: `sketch/showcase/effect_grid.py`, `sketch/showcase/shape_grid.py`。
    - ドキュメント: `docs/spec/pipeline.md`。

### wobble エフェクト

- ファイル
  - `src/effects/wobble.py`
- 公開関数
  - `wobble(g, *, amplitude, frequency, phase)`
- 現状
  - 引数 `phase: float` … 位相 [rad] として docstring に記載。
  - `PARAM_META["phase"]` … `min=0.0`, `max=2π`。
  - 内部で `np.sin(2π f * x + phase)` としてそのまま trig 引数に使用。
  - API スタブ: `src/api/__init__.pyi` の `wobble(..., phase: float, ...)` ＋「位相（ラジアン）」という説明。
  - 呼び出し箇所:
    - 現時点で大半はデフォルト値使用（`phase=0.0`）。テスト/スケッチで明示指定される可能性あり。

### mirror3d エフェクト（既に degree ベース）

- ファイル
  - `src/effects/mirror3d.py`
- 公開関数
  - `mirror3d(..., phi0_deg: float, ...)`
- 現状
  - 引数 `phi0_deg: float` … くさびの開始角 [deg]。
  - `PARAM_META["phi0_deg"]` … `min=-180.0`, `max=180.0`, `step=1.0`。
  - 内部で `np.deg2rad(phi0_deg)` により radian へ変換し使用。
  - API スタブも `phi0_deg` を degree として扱う。
  - TODO: `TODO.txt` に「mirror3d phi0_deg を radians に。引数についてもう一度検証」が残っており、今回の方針（degree 統一）と競合している。

---

## 引数名の統一ルール（案）

1. **Vec3 回転角（XYZ 回転）**
   - 引数名は `rotation` に統一する。
   - 対象:
     - `effects.rotate` / `effects.affine` の XYZ 回転角。
     - `engine.core.affine_ops.rotate` / `LazyGeometry.rotate` から渡される回転角パラメータ。
   - 単位は degree（docstring と `__param_meta__` で明示）。

2. **Vec3 の終点角（repeat の回転補間）**
   - 引数名は `rotation_step` に統一する。
   - 対象:
     - `effects.repeat` の旧 `angles_rad_step`。
   - 「0→rotation_step を degree で補間する終点角」という意味を docstring で説明。

3. **スカラー角度（単一軸の最大角など）**
   - 引数名は `angle` に統一する。
   - 対象:
     - `effects.twist` の最大ねじれ角（旧 `angle_rad`）。
     - `effects.fill` のハッチ角（旧 `angle_rad`）。
   - 単位は degree。必要に応じて「0..180 度の範囲」などを RangeHint と docstring で揃える。

4. **位相（wobble）**
   - 引数名は現行どおり `phase` を維持し、単位のみ degree に変更する。
   - 対象:
     - `effects.wobble` の `phase`。
   - RangeHint は 0..360（degree）とし、内部では `np.deg2rad` で radian に変換して使用。

5. **mirror3d の開始角**
   - 専門的な意味を持つため、引数名は現行の `phi0_deg` を `phi0` にリネームし、単位は degree とする。
   - docstring と RangeHint で degree であることを明示。

6. **その他**
   - shapes や palette 側の角度・位相引数についても最終的には同じポリシー（単位無しの名称 + degree 統一）に寄せる方針だが、本計画では effects の公開引数に限定して扱う。

---

## 角度・位相に関連する周辺箇所（影響範囲の候補）

※エフェクト以外も含むが、全体設計を揃えるための参考リスト。

- エンジン／コア
  - `src/engine/core/affine_ops.py::rotate` … `angles_rad` を受け取る純関数。
  - `src/engine/core/lazy_geometry.py::LazyGeometry.rotate` … `angles_rad` でパラメータを持ち回り。
  - `src/engine/core/geometry.py` … `E.rotate(angles_rad=(...))` の使用例。
- 形状（shapes）
  - `src/shapes/line.py` … `angle` 引数（degree ベース）。
  - `src/shapes/lissajous.py` … `phase/phase_y/phase_z` が degree ベース（`docs/spec/shapes_angle_degrees_plan.md` 参照）。
  - `src/shapes/asemic_glyph.py` … `snap_angle_degrees` など degree ベースの内部処理。
- パレット／LFO 等
  - `src/palette/style.py` … 内部で `phase_L/phase_C` に radian を使用（外部 API ではなく内部実装）。
  - `src/common/lfo.py` / `docs/spec/lfo_spec.md` … ここでは位相を「周期単位」として扱い、rad/deg とは別概念。
- API スタブ
  - `src/api/__init__.pyi`
    - `line(..., angle: float, ...)`（shape）。
    - `affine(..., angles_rad: Vec3, ...)`（effect）。
    - `rotate(..., angles_rad: Vec3, ...)`（effect）。
    - `repeat(..., angles_rad_step: Vec3, ...)`（effect）。
    - `fill(..., angle_rad: float | ..., ...)`（effect）。
    - `twist(..., angle_rad: float, ...)`（effect）。
    - `wobble(..., phase: float, ...)`（effect）。
    - `mirror3d(..., phi0_deg: float, ...)`（effect）。
- テスト・ドキュメント
  - `architecture.md` … affine/rotate の例が `angles_rad` 前提。
  - `docs/spec/pipeline.md` … `angles_rad`, `angle_rad` を使った使用例。
  - `tests/**` … 上記パラメータ名を直接指定した多数のテストケース。
  - `sketch/**` … 実験用スクリプトで radian のまま指定している箇所。
  - `TODO.txt` … mirror3d の `phi0_deg` を radian にしたい旨の TODO が存在。

---

## 共通方針（案）

1. **公開パラメータ名と単位の整合**

   - 角度を表す公開引数はすべて degree ベースに統一。
   - 引数名から単位（`rad/deg`）を示すサフィックスを廃止し、「引数名の統一ルール」に沿った名前へ変更する。
     - 例: `angles_rad` → `rotation`、`angles_rad_step` → `rotation_step`、`angle_rad` → `angle`、`phi0_deg` → `phi0`。
   - 破壊的変更を許容する前提で、旧パラメータ名（`*_rad`, `*_deg`）の互換レイヤーは基本的に設けない。

2. **内部実装の単位**

   - NumPy / 三角関数への入力は引き続き radian を使用。
   - エフェクト関数の入り口で degree → radian に変換し、それ以降の処理は現状に近い形で維持。
   - 将来的に core 側も degree ベースへ寄せるかどうかは別途検討（今回はまず `effects` 層を統一）。

3. **RangeHint（`__param_meta__`）と量子化**

   - RangeHint の `min/max/step` は degree 単位に合わせる。
     - 例（案）:
       - XYZ 回転系: `rotation` / `rotation_step` → `min=(-180,-180,-180)`, `max=(180,180,180)`, `step=1.0`。
       - 単一角度: `angle` → `min=0.0`, `max=180.0` または `max=360.0`（twist/fill の用途に応じて最終決定）。
       - 位相: `phase` → `min=0.0`, `max=360.0`, `step=1.0`。
   - `params_signature` の量子化は degree に対してそのまま適用される（step は RangeHint を優先）。

4. **ドキュメント／アーキテクチャ整合**

   - `architecture.md` と `docs/spec/pipeline.md` の角度パラメータ説明を degree ベースに更新。
   - `TODO.txt` にある「mirror3d phi0_deg を radians に」を削除または「degree に統一済み」として更新。

5. **テストポリシー**
   - 角度周りのテストは「degree で指定 → 内部結果が従来と一致する」ことを確認する形に調整。
   - 既存テストで `np.deg2rad` を使っていた箇所は、基本的に degree 値をそのまま渡す形に変更。

---

## 実装タスク チェックリスト（ドラフト）

### 共通基盤

- [x] **命名・単位ポリシーの確定**
  - 角度引数名から `_rad` / `_deg` など単位サフィックスを廃止し、「引数名の統一ルール」に沿って `rotation` / `rotation_step` / `angle` / `phase` / `phi0` などへ整理する。
- [x] **RangeHint 方針の確定**
  - `rotation` 系や単一角度の `min/max` を degree ベース（±180 など）に変更し、位相も度単位に揃える。
- [x] **API スタブの更新方針**
  - `tools/gen_g_stubs.py` を用いて `src/api/__init__.pyi` を再生成し、effects の角度パラメータを degree 仕様と同期。

### 各エフェクトの変更

- [x] **rotate エフェクトを degree 対応に変更**

  - 関数シグネチャ `angles_rad` → `rotation` へリネーム。
  - `PARAM_META["angles_rad"]` → `rotation` に変更し、RangeHint を degree に合わせて再定義。
  - docstring の単位・既定値の記述を degree ベースに更新。
  - 関数内部で `rotation`（degree） → radian 変換（`np.deg2rad`）を挟んで `Geometry.rotate` に渡す。
  - `LazyGeometry.rotate`（`angles_rad` キー）との整合方法を決め、必要に応じて key 名を `rotation` に変更。

- [x] **affine エフェクトを degree 対応に変更**

  - 関数シグネチャ `angles_rad` → `rotation` へリネーム。
  - `PARAM_META["angles_rad"]` → `rotation` に変更し RangeHint を degree 基準に設定。
  - docstring の角度説明を degree ベースに更新。
  - `rotate_radians = np.deg2rad(rotation)` へ変更。
  - `architecture.md` の affine 例（`angles_rad` 記載）を degree 仕様に合わせて更新。

- [x] **repeat エフェクトを degree 対応に変更**

  - 関数シグネチャ `angles_rad_step` → `rotation_step` にリネーム。
  - `PARAM_META["angles_rad_step"]` → `rotation_step` に変更し RangeHint を degree に合わせる。
  - docstring の単位を rad→deg に更新。
  - 内部で `rotation_step`（degree）を radian に変換してから `rotate_radians` を構成。
  - 関連テスト（`tests/effects/test_repeat_basic.py` など）のパラメータ指定を degree に揃える。

- [x] **twist エフェクトを degree 対応に変更**

  - 関数シグネチャ `angle_rad` → `angle` にリネーム。
  - `PARAM_META["angle_rad"]` → `angle` に変更し RangeHint を degree 用に調整。
  - docstring を degree 表記に変更。
  - `max_rad = np.deg2rad(angle)` として内部の radian 計算を維持。
  - API スタブ内の説明「ラジアン」を「度」に変更。

- [x] **fill エフェクトを degree 対応に変更**

  - 関数シグネチャ `angle_rad` → `angle` にリネーム。
  - `PARAM_META["angle_rad"]` → `angle` に変更し RangeHint（0..180 or 0..360）を決める。
  - docstring（モジュール説明 + 関数 docstring）の単位を degree に更新。
  - `_as_float_seq(angle_rad)` → degree 前提に変更し、内部の `angle` 計算前に一括で `np.deg2rad` へ変換（API レベルの引数名は `angle`）。
  - `_generate_line_fill` / `_fill_single_polygon` / `_scan_span_for_angle_xy` など角度を受け取るヘルパーのインターフェース（引数名と単位）を整理。
  - 関連テスト（特に `tests/smoke/test_fill_rotation_invariance.py`）で `np.deg2rad` に依存している箇所を degree 指定に書き換え。

- [x] **wobble エフェクトを degree 対応に変更**
  - 引数名は現行どおり `phase` を維持する（単位のみ degree に変更）。
  - `PARAM_META["phase"]` の `max` を `360.0` に変更し、RangeHint を degree ベースに調整。
  - docstring の位相単位を degree に変更。
  - `_wobble_vertices` に渡す前に `phase_rad = np.deg2rad(phase)` へ変換。
  - API スタブの説明「位相（ラジアン）」を degree に合わせる。

- [x] **mirror3d エフェクトの開始角を degree 仕様として整理**
  - 引数名 `phi0_deg` を `phi0` にリネーム（単位は degree）。
  - `PARAM_META["phi0_deg"]` → `phi0` に変更し、RangeHint を degree ベースで維持。
  - 内部の `np.deg2rad(phi0_deg)` を `np.deg2rad(phi0)` に変更。
  - `TODO.txt` にある「phi0_deg を radians に」という記述を更新/削除し、本方針と整合させる。

### 呼び出し側・周辺コード

- [x] **API スタブ `src/api/__init__.pyi` の更新**

  - `affine/rotate/repeat/fill/twist/wobble/mirror3d` の角度・位相パラメータを degree 仕様に合わせて更新。
  - `tools/gen_g_stubs.py` を用いた自動再生成の実行と `tests/stubs/test_g_stub_sync.py` の確認。

- [x] **Parameter GUI 周辺の更新**

  - `tests/ui/parameters/test_value_resolver.py` などで `angles_rad` を前提にしている部分を degree 用に更新（descriptor ID の命名も含める）。
  - `tests/ui/parameters/test_runtime.py` のパラメータ名・期待値を degree ベースに修正。

- [x] **ドキュメント・アーキテクチャの更新**

  - `architecture.md` 内の affine/rotate のシグネチャと説明を degree 仕様に変更。
  - `docs/spec/pipeline.md` のコード例（`angles_rad`, `angle_rad` など）を degree ベースに更新。
  - 必要に応じて `docs/spec/drop_effect_spec.md` など関連 spec から radians 前提の文言を洗い替え。

- [x] **テストコード・スケッチの更新**

  - `tests/**` 内で `angles_rad/angle_rad/phase` を直接指定している箇所を degree 名＋値に書き換え。
    - 特に `tests/smoke/test_fill_rotation_invariance.py` のように degree→radian 変換を行っているテストを集中的に更新。
  - `sketch/**` 内の `angles_rad/angle_rad` の利用箇所を degree に更新（内部検証用スクリプトも含めて一括で揃える）。

- [x] **TODO/メモの整理**
  - `TODO.txt` にある「mirror3d phi0_deg を radians に。引数についてもう一度検証」を「mirror3d phi0 を degree 仕様に統一済み」とするメモに更新。

---

## 要確認事項（相談したいポイント）

- [ ] **XYZ 回転（rotate/affine/repeat）の RangeHint**
  - `rotation` / `rotation_step` の範囲を `[-180, 180]` で揃えるか、`[-360, 360]` を許容するか。
- [ ] **fill のハッチ角範囲**
  - ハッチ方向として一意なのは 0..180deg だが、現状は 0..2π [rad] を許容。
  - degree 版では「UI 上は 0..180deg に絞るか」「0..360deg を許容するか」を決めたい。
- [ ] **wobble の `phase` の命名**
  - `phase` のまま「degree として解釈する」前提で問題ないか。
- [ ] **エフェクト以外への適用範囲**
  - 今回は「effects の公開引数」に限定するが、将来的に shapes（例: `lissajous.phase`）や palette の位相も degree ベースに寄せるかどうか。

（上記チェックリストはまだすべて未着手のドラフトです。方針・粒度に問題なければ、このファイル上で項目をチェックしながら実装を進めます。）
