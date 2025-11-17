# draw() がリストを返す場合の affine/fill 無効化に見える問題 改善計画

目的: `sketch/251118.py` のように `draw(t)` が `Geometry | LazyGeometry` のリスト（レイヤー列）を返す場合に、「affine や fill が効いていないように見える」現象の原因を整理し、必要なコード/サンプル/テストの改善タスクをチェックリストとしてまとめる。

## 現象の整理（sketch/251118.py）

- 対象スケッチ: `sketch/251118.py`
- コア部分:
  ```python
  def draw(t: float):
      N = int(cc[1] * 10) + 1
      geos = []
      for i in range(N):
          g = G.polygon().scale(4 * i, 4 * i, 4 * i).translate(A5[0] / 2, A5[1] / 2, 0)
          p = E.label(uid="polygons").affine().fill()
          geos.append(p(g))
      return geos
  ```
- 典型的な実行条件:
  - `cc[1]` の既定値は 0.0 → `N = 1`
  - `i` は 0 のみとなるため、`scale(4 * i, ...) = scale(0, 0, 0)` となる。
- 観測される挙動:
  - 画面上には「中心に小さな多角形の輪郭のみ」が描画され、`affine()` のスケールや `fill()` によるハッチが一切見えない。
  - ユーザー視点では「draw がリストを返すと affine/fill エフェクトが無効になっている」ように見える。

## コードベースでの原因分析

### 1. draw が返すリストとレイヤー正規化

- `draw(t)` の戻り値は Worker 側で `_normalize_to_layers()` に渡される（`src/engine/runtime/worker.py`）。
- `result` が `Sequence`（かつ `Geometry`/`LazyGeometry` そのものではない）なら、要素ごとに `StyledLayer(geometry=..., color, thickness)` へ変換される。
  - 要素が `LazyGeometry` の場合:
    - `LazyGeometry.plan` から `style` 種のエフェクトのみを取り除き、残りのエフェクト（affine/fill など）はそのまま保持した `LazyGeometry` を新たに作成する。
    - `StyledLayer.geometry` にはこの `LazyGeometry` が入る。
- Renderer 側では、レイヤーごとに `layer.geometry` を `LazyGeometry.realize()` で実体化してから GPU へアップロードする。
- 別途検証した結果:
  - 単純な検証コード（`G.polygon()` に対して `E.label().affine().fill()` を適用し、その `LazyGeometry` のリストを `_normalize_to_layers` → `realize()`）では、**85 頂点・40 本のハッチ線**が生成され、`Sequence` 経由でも affine/fill が正しく適用されることを確認。
  - つまり「リストで返すこと自体」はエフェクト無効化の直接原因ではない。

### 2. 251118 固有のスケール指定と degenerate ジオメトリ

- 251118 の `draw` では、`i` に応じて形状のスケールを決めている:
  - `g = G.polygon().scale(4 * i, 4 * i, 4 * i).translate(...)`
  - `N = int(cc[1] * 10) + 1` のため、`cc[1] = 0.0` では `N = 1`、`i = 0` のみが生成される。
  - このとき `scale(0, 0, 0)` となり、**多角形が一点に潰れた degenerate ジオメトリ**になる。
- 実測結果（Python から直接呼び出し）:
  - `draw(0.0)` → `_normalize_to_layers()` → 最初のレイヤーの `geometry.realize()` を確認すると、
    - `coords.shape == (7, 3)`、`n_lines == 1` と、**元の多角形輪郭と同じ 1 本の線だけ**が存在する。
    - `LazyGeometry.plan` は `['_fx_scale', '_fx_translate', 'affine', 'fill']` を含んでいるが、`fill` の結果として新しいハッチ線は追加されていない。
- 一方、同じ `affine().fill()` を「スケール 0 ではない多角形」に適用した場合:
  - 例: `G.polygon().scale(40, 40, 40).translate(...)` に `E.label().affine().fill()` を適用すると、`n_vertices=85, n_lines=40` のハッチ結果が得られる。
  - また 251118 と同じロジックで `scale(4 * i, ...)` を使いつつ、`i=1,2,...` のレイヤーについては、いずれも `n_vertices=85, n_lines=40`（fill が有効）であることを確認。
- つまり:
  - **`i=0` のレイヤーだけが「スケール 0 の degenerate polygon」であり、その結果として fill が有効なハッチを生成できていない。**
  - デフォルトの `cc[1]=0.0` では `N=1` のため、この「スケール 0 のレイヤーのみ」が返り値となり、ユーザーからは「affine/fill が一切効いていない」ように見える。

### 3. 結論

- `draw()` がリストを返すこと自体は正しくサポートされており、`affine`/`fill` も Sequence 経由で期待どおり適用される。
- `sketch/251118.py` で「効いていないように見える」直接の原因は、
  - `cc[1]` の既定値によって `N=1` となり、
  - 最初のレイヤーが `scale(0, 0, 0)` で生成されてしまうため、
  - `fill` に渡るジオメトリが「面積 0 の多角形」（実質 1 点集約）になり、ハッチを生成できないこと。
- したがって、**エンジン側の Sequence 処理よりも、サンプルスケッチのスケール指定が主な原因**と考えられる。

## 改善方針

### 方針 A: サンプルスケッチのスケール指定を見直して footgun を除去する

- 目的: `sketch/251118.py` が「draw がリストを返す代表例」として、初期状態から affine/fill の効果が視覚的にわかるようにする。
- 方針:
  - `i=0` のときにも有意味なスケールになるように、スケール係数を `4 * (i + 1)` などに変更する。
  - パイプライン `p = E.label(...).affine().fill()` はループ外で 1 回だけ作成し、各レイヤーで再利用する（可読性とパフォーマンスのため）。
  - サンプルコメントに「draw は Geometry/LazyGeometry またはその Sequence を返せること」「Sequence の各要素がレイヤーとして描画されること」を簡潔に追記する。

### 方針 B: Sequence 経由のエフェクト適用をテストで明示的に保証する

- 目的: 「draw がリストを返しても affine/fill が効く」ことをユニットテストで担保し、将来のリファクタリングで壊しにくくする。
- 方針:
  - `_normalize_to_layers()` と `LazyGeometry.realize()` を組み合わせたテストを追加し、Sequence 内の各要素に付与された geometry 系エフェクト（affine/fill など）が保持されることを検証する。
  - スタイル系エフェクト（`style`）だけが plan から正しく抽出され、`StyledLayer.color`/`thickness` に反映されることも合わせて確認する。

### 方針 C: degenerate ジオメトリに対する fill の挙動をドキュメントで明示する

- 目的: 「スケール 0 や面積 0 の形状に対して fill がハッチを生成しない」ことを仕様として明示し、ユーザーが原因を推測しやすくする。
- 方針:
  - `src/effects/fill.py` の docstring に、入力ジオメトリが面積 0（全頂点が同一点など）の場合はハッチを生成せず輪郭のみを返す可能性がある旨を一行程度追記する。
  - 余裕があれば、開発者向けコメントとして「高さ 0 の場合は spacing が定義できないため、安定側（no-op）に倒す」ことを明記する。

## TODO チェックリスト

### A. `sketch/251118.py` の改善

- [ ] `scale(4 * i, ...)` を `scale(4 * (i + 1), ...)` などに変更し、`i=0` でも有意味なスケールになるようにする。
- [ ] パイプライン `p = E.label(...).affine().fill()` をループ外で 1 回だけ生成し、ループ内では `geos.append(p(g))` する形に整理する。
- [ ] サンプルの先頭コメントか docstring に、「draw が `Sequence[Geometry | LazyGeometry]` を返すと各要素がレイヤーとして描画される」旨を簡潔に記述する。
- [ ] 変更後の `sketch/251118.py` を実行し、`cc[1]=0`（N=1）でも affine/fill の効果が視覚的に確認できることを目視確認する。

### B. Sequence 経由エフェクト適用テストの追加

- [ ] `tests/runtime` もしくは新規テストファイル（例: `tests/runtime/test_layers_sequence_effects.py`）を作成し、以下を検証する:
  - [ ] `G.polygon()` に `E.label(...).affine().fill()` を適用した `LazyGeometry` のリストを `_normalize_to_layers()` に渡すと、各レイヤーの `geometry.realize()` が基底ポリゴンより多い頂点・線本数を持つ（fill が適用されている）。
  - [ ] 同じテスト内で、`style` エフェクトを含む `LazyGeometry` のリストを渡したとき、`style` は plan から除去され、`StyledLayer.color`/`thickness` に反映されることを確認する。
- [ ] テスト追加後、対象ファイルに対して `pytest -q {path}` を実行し緑になることを確認する。

### C. fill の degenerate 入力に関するドキュメント整備

- [ ] `src/effects/fill.py` の docstring に、「入力が面積 0 のジオメトリ（例: スケール 0 のポリゴン）の場合、ハッチを生成できず輪郭のみを返すことがある」旨を 1 行追記する。
- [ ] 必要に応じて、開発者向けコメントとして「高さ 0 の場合は spacing を定義せず no-op に倒す」意図を補足する。
- [ ] 変更後、`mypy`/`ruff`/`pytest` を対象ファイルに対して実行し、型/スタイル/テストが通ることを確認する。

---

上記チェックリストで問題なければ、この計画に沿って `sketch/251118.py`・テスト・docstring の修正を行い、完了した項目から順にチェックを付けていきます。*** End Patch
