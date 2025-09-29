# src/effects 実装の統一性レビュー（2025-09-29）

対象: `src/effects/` 配下（関数ベースの Geometry→Geometry エフェクト群とレジストリ）。
観点: API（関数シグネチャ/型/`__param_meta__`）、ドキュメンテーション体裁、挙動（no-op/早期リターン）、命名/単位、環境変数方針、登録・依存関係。

## 総評（結論）
- 良い点: すべて純関数で新規 `Geometry` を返す設計が徹底。`@effect` 登録・`Geometry.as_arrays(copy=False)` 利用・空入力/不正値での安全な早期リターンが広く実装され、ベースラインは良好。
- 改善余地: `__param_meta__` の型記述（特に Vec3）と関数シグネチャの型/単位、関数 docstring の体裁（`effects/AGENTS.md` 準拠）にばらつき。Numba の環境変数名も混在。表示上のみの RangeHint に対する実装内クランプの有無も不統一。

## 良い点（統一できている事項）
- 関数APIの純粋性とコピーセマンティクス
  - 早期 no-op が各所で実装（例: `dash.py:190` 近辺、`translate.py:20`、`subdivide.py:41`）。
  - 出力は常に新規 `Geometry`（`Geometry.from_lines` か `coords/offsets` 再構築）。
- 登録・モジュール構成
  - `effects/registry.py` の `@effect` と `__all__`、`effects/__init__.py` の最小一括 import で統一。
- パラメータ GUI 用メタ
  - 全エフェクトが `__param_meta__` を持つ（`registry.py`/`__init__.py` を除く）。
- 実装の安全性
  - 例外・未導入依存に配慮したフォールバック（`dash.py`/`collapse.py` の numba 経路）。

## 不一致・ばらつき（要修正候補）
- Vec3 の `__param_meta__` 記述
  - 事象: ベクトルを許容する引数で `type: "number"` かつ `min/max` にタプルを与えている。
    - `src/effects/wobble.py` `frequency`（関数は `float | Vec3`、meta は `type: number`, `min/max` がタプル）。
    - `src/effects/displace.py` `spatial_freq`（同上）。
  - 期待: Vec3 を受ける場合は `type: "vec3"` に統一（単一 float しか受けない場合は `number`）。

- 関数シグネチャ型と `__param_meta__` の不一致（int/float）
  - `src/effects/extrude.py:33` `subdivisions: float`（丸め処理あり）⇔ meta は `integer`。
  - `src/effects/collapse.py:465` `subdivisions: float` ⇔ meta は `integer`。
  - `src/effects/weave.py:17` `num_candidate_lines: float`, `relaxation_iterations: float` ⇔ meta は `integer`。
  - 期待: 実引数型は `int` にし、丸めは呼び出し側に委譲（または内部で受けるにしてもアノテーションは `int`）。

- 角度の単位表現の不統一
  - `rotate.py`/`affine.py`/`repeat.py` はラジアンで `angles_rad`。
  - `twist.py` は度で `angle`（meta は 0..360）。
  - 期待: 度なら `angle_deg`、ラジアンなら `*_rad` を名称で明示。`twist` は `angle_deg` へ改名提案（互換注記）。

- RangeHint（表示上のみ）と実装内クランプの不一致
  - 仕様: ルート `AGENTS.md` に「クランプは表示上のみ」と明記。
  - 実装: 実行時にクランプしている箇所がある（例: `extrude.py` の `distance/scale/subdivisions`、`repeat.py` の `count`）。一方、`offset.py` は docstring に丸め記述があるが実装でクランプしていない。
  - 方針: 実行時クランプは「数値安定・安全ガード」の最小限に限定し、docstring は実装に一致させる（`offset` は記述を修正、`extrude`/`repeat` は「安全ガード」として明示）。

- 関数 docstring 体裁のばらつき（`effects/AGENTS.md` 準拠違反）
  - 期待: 先頭1行要約 + Parameters のみ（Returns/Notes は原則不要）。
  - 例外: 詳説・実装メモ・Notes/Returns を含む関数がある。
    - 長文: `offset.py`, `explode.py`, `subdivide.py` など。
    - `fill.py` は関数 docstring が短く Parameters セクションが無い（要追加）。

- Numba 環境変数の命名不統一
  - `dash.py` は `PXD_USE_NUMBA_DASH`、`collapse.py` は `PYX_USE_NUMBA`。
  - 参考: `docs/numba-env-unification-proposal.md` に統一案あり（`PXD_NUMBA{,_<EFFECT>}`）。

- 表記/命名の細部
  - `offset.py` の docstring に「距離は [0, 25.0] に丸める」とあるがコード未実装。表現修正が必要。
  - `displace.py`/`wobble.py` の「周波数」命名を `spatial_freq` で統一するか、両者の意味差を docstring で明確化。

## 推奨ガイドライン（統一ポリシー案）
- 引数型とメタ
  - ベクトルは `Vec3` 固定。`__param_meta__` は `type: "vec3"` を使用。
  - `int` を受ける引数（反復回数など）は関数アノテーションも `int` にする（UI からは整数入力）。
  - 度/ラジアンは名称で明示（`*_deg` / `*_rad`）。
- ドキュメンテーション
  - 関数 docstring: 先頭1行 + Parameters（NumPy スタイル、日本語・事実のみ）。Returns/Notes は削除（必要ならモジュール docstringか docs/ へ）。
  - `__param_meta__` と docstring の既定値/単位/範囲を常に一致させる。
- 実行時クランプ
  - 本質的に危険/暴走回避のみ許容（例: 極大な複製数、ゼロ割回避など）。それ以外は UI の RangeHint のみで誘導。
  - クランプを行う場合は docstring で「安全ガード」として明記。
- Numba トグル
  - `PXD_NUMBA={auto|on|off}` + `PXD_NUMBA_<EFFECT>` を採用（旧名は互換・非推奨）。詳細は `docs/numba-env-unification-proposal.md` を参照。

## 影響範囲と優先度（粗め）
- High
  - Vec3 の `__param_meta__` 是正（`wobble.py`, `displace.py`）。
  - `extrude.py`/`collapse.py`/`weave.py` の整数引数アノテーション修正と docstring/メタ同期。
  - 関数 docstring 体裁の是正（長文整理、Parameters 追記）。
- Mid
  - `twist.py` の `angle` → `angle_deg`（互換影響の確認要）。
  - 実行時クランプ表現の整合（`offset.py` 記述修正、`extrude.py`/`repeat.py` に安全ガード注記）。
- Low
  - 周波数系パラメータの命名整理（`frequency`/`spatial_freq`）。

## ファイル別メモ（抜粋）
- `src/effects/wobble.py`
  - `frequency: float | Vec3` だが meta が `type: number`。`type: vec3` へ是正推奨。Parameters セクションの明確化（単位・意味）。
- `src/effects/displace.py`
  - `spatial_freq` 同様の問題。`NOISE_*` 定数は適切。docstring は簡潔で良いが Parameters 体裁に揃えると一貫性が増す。
- `src/effects/extrude.py`
  - `subdivisions` の型を `int` に。実行時クランプは「安全ガード」表現に修正。Parameters 同期。
- `src/effects/weave.py`
  - `num_candidate_lines`/`relaxation_iterations` を `int` 注釈へ。meta と同期。docstring は短く保ち Parameters を追加。
- `src/effects/subdivide.py`
  - 関数 docstring が Returns/Notes を含む。モジュール docstring側へ移すか docs に退避し、関数は要約+Parameters のみに。
- `src/effects/offset.py`
  - docstring の丸め記述を実装に合わせて削除/修正。`segments_per_circle` は `int` で統一済み。
- `src/effects/twist.py`
  - `angle` は度。名称に単位を付与（`angle_deg`）を検討。
- `src/effects/fill.py`
  - 関数 docstring に Parameters を追記（`mode`/`density`/`angle_rad`）。
- `src/effects/dash.py` / `src/effects/collapse.py`
  - Numba 環境変数の統一（`docs/numba-env-unification-proposal.md` に沿って変更予定）。

## 次アクション（変更提案チェックリスト：実施可否の確認待ち）
- [ ] `wobble.py`/`displace.py` の `__param_meta__` を `type: vec3` に変更（単一 float を入力した場合は `(f,f,f)` に内部正規化）。
- [ ] `extrude.py`/`collapse.py`/`weave.py` の整数引数を `int` 注釈へ修正し、丸め処理を整理（docstring/メタと同期）。
- [ ] `twist.py` の `angle` を `angle_deg` に改名（互換性の扱い方針確認）。
- [ ] 関数 docstring を全体でテンプレ準拠（先頭1行 + Parameters）に刷新。不要な Returns/Notes をモジュール docstring or `docs/` へ移動。`fill.py` には Parameters を追加。
- [ ] 実行時クランプの扱い方針を明文化し、各関数の docstring を実装と同期（`offset.py` 表現修正も含む）。
- [ ] Numba 環境変数を `PXD_NUMBA{,_<EFFECT>}` に統一（旧名は当面互換）。

承認後、上記チェックリストをベースに編集ファイル限定で lint/type/test を回しながら段階適用します（ルート AGENTS.md の Build/Test ルール準拠）。

