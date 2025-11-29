# シェイプ角度・位相パラメータ degree 統一計画

## 目的・ゴール

- すべての「シェイプ」の角度・位相系引数を degree ベースに統一する。
- パラメータ名・`__param_meta__`・docstring・API スタブ・テスト・ドキュメントを一貫させる。
- 内部実装では必要に応じて radian を使いつつ、外部インターフェースは degree としてクリーンに保つ。
- 角度・位相に関する引数名から `rad/deg` などの単位サフィックスを排除し、意味ベースの名前で統一する（`line` の `angle_deg` も含む）。

---

## 現状の角度・位相関連パラメータ一覧（shapes）

### line シェイプ

- ファイル
  - `src/shapes/line.py`
- 公開関数
  - `line(length: float = 1.0, angle_deg: float = 0.0, **params: Any) -> Geometry`
- 現状
  - 引数 `angle_deg: float` … 2D 線分の回転角 [deg]。0 で X 軸正方向。
  - docstring で「回転角度（度）」と明示。
  - `line.__param_meta__["angle_deg"]` … `min=0.0`, `max=360.0`, `step=1.0`。
  - 実装では `angle = float(angle_deg) % 360.0` → `np.deg2rad(angle)` として内部 radian へ変換して利用。
  - API スタブ `src/api/__init__.pyi::_GShapes.line` は `angle_deg` をそのまま公開。
  - テスト: `tests/shapes/test_line_shape.py::test_line_angle_rotates_segment` で `angle_deg=90.0` を指定。

### lissajous シェイプ

- ファイル
  - `src/shapes/lissajous.py`
- 公開関数
  - `lissajous(*, freq_x, freq_y, freq_z, phase, phase_y, phase_z, points, **params) -> Geometry`
- 現状
  - 引数 `phase`, `phase_y`, `phase_z` … 各軸の初期位相オフセット。[rad] 前提だが docstring には単位記述なし。
  - 実装では `t = np.linspace(0, 2 * np.pi, points)` とし、`np.sin(freq_x * t + phase)` などにそのまま渡す（radian 加算）。
  - `lissajous.__param_meta__` では `phase/phase_y/phase_z` の RangeHint を `min=0.0`, `max=6.28318`（≈2π）としている。
  - API スタブ `_GShapes.lissajous` は `phase`, `phase_y`, `phase_z` を単純な `float` としてエクスポート（Meta 情報は未付与）。
  - 現状テストは存在せず、主にスケッチ/インタラクティブ用途を想定。

### asemic_glyph シェイプ（内部設定）

- ファイル
  - `src/shapes/asemic_glyph.py`
- 関連箇所
  - 設定クラス `AsemicGlyphConfig.snap_angle_degrees: float = 60.0`。
  - `snap_stroke` 内で `snap_angle = config.snap_angle_degrees` → `_snap_point_njit(..., snap_angle)` として、線分方向を `snap_angle` 刻みの角度グリッドにスナップ。
- 現状
  - `snap_angle_degrees` は dataclass のフィールドとしてのみ存在し、shape 関数 `asemic_glyph(...)` の公開引数・`__param_meta__` には露出していない。
  - 内部コメントでは degree 前提で運用されており、数値も 360 の約数を想定（60 度刻み）。

### sphere シェイプ（内部位相のみ）

- ファイル
  - `src/shapes/sphere.py`
- 関連箇所
  - `_sphere_zigzag` 内で `phase = 2.0 * np.pi * (k / float(strand_count))` としてストランド間の位相オフセットを radian で計算。
- 現状
  - `phase` は内部変数のみで、公開パラメータではない。
  - `sphere.__param_meta__` には角度・位相を受け取る項目は存在しない（`subdivisions`, `sphere_type`, `mode` のみ）。

### その他のシェイプ

- `polygon`, `grid`, `cylinder`, `cone`, `capsule`, `torus`, `polyhedron`, `text`, `attractor` など
  - いずれも公開パラメータに角度・位相を取らず、内部でのみ radian ベースの三角関数を使用している。
  - 現計画では「公開 API の単位統一」が主目的のため、内部の radian 使用は維持しつつ、必要に応じて docstring から参照される箇所のみ確認対象とする。

---

## 角度・位相に関連する周辺箇所（影響範囲の候補）

- API スタブ
  - `src/api/__init__.pyi::_GShapes`
    - `line(..., angle_deg: float, ...)`
    - `lissajous(..., phase: float, phase_y: float, phase_z: float, ...)`
  - 今後の変更に伴い、`tools/gen_g_stubs.py` 再実行でシグネチャと Meta コメントを同期させる必要がある。
- Parameter GUI / ランタイム
  - shapes の角度パラメータは現状ほぼ GUI から直接操作されていないが、`__param_meta__` に基づき将来的に GUI で調整される想定。
  - `tests/ui/parameters/*` には shapes 固有の角度テストはまだ無く、今後追加余地あり。
- ドキュメント
  - `docs/spec/shapes.md` に `lissajous` の定義のみ簡単に記載（位相や単位には未言及）。
  - `docs/spec/pipeline.md` などで shapes を用いた例が追加された場合、角度の指定方法を degree ベースで示す必要がある。
- テスト・スケッチ
  - `tests/shapes/test_line_shape.py` が `angle_deg` を直接指定。
  - `sketch/` 配下には現時点で `angle_deg` や `lissajous.phase` を直接指定するコードは見当たらないが、今後追加される可能性を考慮。

---

## 共通方針（案）

1. **公開パラメータ名と単位の整合**

   - shapes の角度・位相を表す公開引数はすべて degree ベースに統一する。
   - 引数名から単位サフィックス（`_rad`, `_deg`）を廃止し、「意味ベースの名前 + degree 前提」のポリシーに揃える。
     - 例: `angle_deg` → `angle`、`phase`（rad）→ `phase`（deg）に意味そのままですり替え。
   - 破壊的変更を許容する前提で、旧パラメータ名の互換レイヤー（`angle_deg` など）は基本的に設けない。

2. **内部実装の単位**

   - NumPy / 三角関数への入力は引き続き radian を使用する。
   - shape 関数の入り口で degree → radian に変換し、それ以降の処理は現状に近い形で維持する（`line`/`lissajous` ともに入口で統一）。
   - `AsemicGlyphConfig.snap_angle_degrees` のような内部設定は degree のまま維持し、必要なら docstring コメントで単位を明示する。

3. **RangeHint（`__param_meta__`）と量子化**

   - RangeHint の `min/max/step` は degree 単位に合わせる。
     - `line.angle`（予定）: `min=0.0`, `max=360.0`, `step=1.0`。
     - `lissajous.phase/phase_y/phase_z`:
       - 案: `min=0.0`, `max=360.0`, `step=1.0`（周期性を持つ位相なので 0..360deg を 1 度刻み）。
   - `params_signature` の量子化は degree に対してそのまま適用される（`step` は RangeHint を優先）。
   - shapes 側は「鍵のみ量子化・実行引数は実値」のポリシーを維持し、effects 側の degree 統一と整合させる。

4. **docstring / アーキテクチャ整合**

   - `src/shapes/` 内の docstring と `__param_meta__` の内容（単位・範囲・既定値）を必ず一致させる。
   - `docs/spec/shapes.md` に degree ベースの角度・位相ポリシーを 1 セクション追加し、`effects_angle_degrees_plan.md` と設計意図を揃える。
   - `architecture.md` や `docs/spec/pipeline.md` に shapes の具体例を追記する場合は、ここで決めた命名/単位ポリシーを参照する。

5. **テストポリシー**

   - 角度周りのテストは「degree で指定 → 期待される幾何が得られる」ことを検証する。
     - `line`: 0 度で X 軸、90 度で Y 軸方向になること。
     - `lissajous`: `phase` が 90 度ずれると sin 波が 1/4 周期シフトすることなど。
   - 既存テストで radian を意識した指定（`np.deg2rad` など）が出てきた場合は、基本的に degree 指定に置き換える。

---

## 実装タスク チェックリスト（ドラフト）

### 共通基盤

- [ ] **命名・単位ポリシーの確定**
  - shapes における角度・位相引数の名前を `angle` / `phase` 系に統一し、単位を degree とすることを明文化。
- [ ] **RangeHint 方針の確定**
  - `angle`（方向）と `phase`（位相）の `min/max/step` を決める（特に `phase` の `max=360` 固定でよいかを確認）。
- [ ] **API スタブ更新フローの確認**
  - `src/api/__init__.pyi` を手修正せず、`tools/gen_g_stubs.py` ベースで再生成して同期を取る流れを再確認。

### 各シェイプの変更

- [ ] **line シェイプの angle パラメータ統一**

  - 関数シグネチャ `angle_deg` → `angle` へリネーム。
  - `line.__param_meta__["angle_deg"]` → `angle` に変更し、RangeHint を degree ベースで維持（`min=0.0`, `max=360.0`, `step=1.0`）。
  - docstring の Parameters セクションを `angle : float, default 0.0` ＋「回転角度（度）。0 で X 軸正方向。」に更新し、`__param_meta__` と揃える。
  - 実装の `angle = float(angle_deg) % 360.0` → `angle = float(angle) % 360.0` に変更し、その後の `np.deg2rad(angle)` を維持。
  - API スタブ `_GShapes.line` およびテスト `tests/shapes/test_line_shape.py` の引数名を `angle_deg` から `angle` に変更。

- [ ] **lissajous シェイプの phase を degree 対応に変更**

  - 関数シグネチャの引数名は `phase/phase_y/phase_z` のまま維持し、単位のみ rad→deg に変更。
  - docstring を拡張し、各引数に対し「初期位相（度）。許容 [0, 360]。」といった説明を追加。
  - `lissajous.__param_meta__` の `phase/phase_y/phase_z` を `min=0.0`, `max=360.0`, `step=1.0` に変更。
  - 実装で `phase` をそのまま `np.sin(freq_x * t + phase)` に渡している箇所を、`phase_rad = np.deg2rad(phase)` のように入口で radian へ変換してから使用するよう修正（`phase_y/phase_z` も同様）。
  - 将来のテスト追加（`tests/shapes/test_lissajous_shape.py` など）で、degree 指定に基づく位相シフト挙動を検証。

- [ ] **asemic_glyph の snap_angle_degrees の整理**

  - `AsemicGlyphConfig.snap_angle_degrees` が degree 前提であることを docstring か型コメントで明示。
  - 必要であれば `AsemicGlyphConfig` に対する簡単なテストを追加し、「60 度刻みのスナップ」が意図通り動作していることを確認。
  - 今回は公開パラメータ化は行わず、内部設定として維持するかどうかを最終的に判断。

- [ ] **その他 shapes の確認**

  - `sphere`, `torus`, `cylinder`, `cone`, `capsule`, `polyhedron`, `attractor`, `text` などについて、docstring に角度・位相パラメータが書かれていないことをざっと確認。
  - もし将来角度パラメータを追加する場合は、本計画のポリシー（名前は単位サフィックス無し、単位は degree）に従う旨を `docs/spec/shapes.md` へ追記。

### 呼び出し側・周辺コード

- [ ] **API スタブ `src/api/__init__.pyi` の更新**

  - `_GShapes.line` のシグネチャを `angle_deg` → `angle` に変更。
  - `_GShapes.lissajous` の `phase/phase_y/phase_z` に Meta コメント（range など）が付与されるよう、`tools/gen_g_stubs.py` の生成結果を確認。
  - スタブ再生成後、`tests/stubs/test_g_stub_sync.py` が緑になることを確認。

- [ ] **ドキュメントの更新**

  - `docs/spec/shapes.md` に「角度・位相パラメータは degree ベース」というルールと、`line` / `lissajous` の簡単な例を追記。
  - `docs/spec/pipeline.md` など、shapes の使用例で `angle_deg` に依存する記述があれば `angle` に書き換え。
  - `docs/spec/effects_angle_degrees_plan.md` から本ファイルへの参照を必要に応じて追加（effects 側から shapes 側のポリシーを辿れるようにする）。

- [ ] **テスト・スケッチの更新**

  - `tests/shapes/test_line_shape.py` の `G.line(length=1.0, angle_deg=90.0)` を `angle=90.0` に変更。
  - 将来的に追加する `tests/shapes/test_lissajous_shape.py`（仮）では degree 指定の位相シフトを直接検証し、rad ベースに戻っていないことを保証。
  - `sketch/` 配下に shapes の角度/位相を明示的に指定するスケッチが追加された場合、本ポリシーに従って degree で指定するようガイド。

---

## 要確認事項（相談したいポイント）

- [ ] **`line` の引数名**
  - `angle_deg` を完全に廃止して `angle` に統一してよいか（effects 側の `angle`/`rotation` 命名と揃える前提）。
- [ ] **`lissajous` の phase 範囲**
  - `phase/phase_y/phase_z` の RangeHint を `0..360` deg 固定とするか、`-180..180` や `0..720` などを許容するか。
- [ ] **`lissajous` の位相の意味**
  - LFO 周辺の spec では「位相=周期に対する相対位置」として扱っているが、`lissajous` の `phase` は「ラジアンオフセット」のまま degree 化してよいか（設計意図の再確認）。
- [ ] **`asemic_glyph.snap_angle_degrees` の扱い**
  - 将来的に GUI から制御したくなった場合、公開パラメータに昇格させるか。昇格させるならパラメータ名を `snap_angle` にするかどうか。
- [ ] **shapes と effects の一貫性**
  - 「形状の向き（`line.angle`）」「エフェクトによる回転（`rotate/affine.rotation`）」「ハッチ角（`fill.angle`）」がすべて degree ベースで直感的につながるように設計してよいか。

（上記チェックリストはすべて未着手のドラフトです。問題なければ、本ファイル上で項目をチェックしながら shapes 側の実装を進める想定です。）

