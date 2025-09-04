# Naming Review (2025-09-04)

本レビューは、コードベース全体（effects/, shapes/, api/, engine/, common/, util/, tests/ ほか）を走査し、関数名・変数名・クラス名・引数名の一貫性と可読性を評価したものです。目的は「破壊的変更（2025-09-03）」後の設計方針に沿って、実用性の高い命名改善を提案することです。ここでは即時に安全に適用できるものを優先し、影響が大きい変更は移行計画を併記します。

---

## サマリー（結論）
- 全体として命名規約（snake_case/UpperCamelCase/UPPER_SNAKE_CASE）は概ね順守できています。
- Geometry 周辺は translate/scale/rotate/concat で統一され、良好です。
- Pipeline/Effects は登録名の整合性は取れている一方で、
  - 変換系エフェクトが名詞系（translation/scaling/rotation）、その他が動詞/名詞混在（noise, extrude, wobble, dashify, webify 等）で、スタイルが揺れています。
  - 一部引数名が「正規化値なのか実数（ラジアン/スケール）なのか」が読み取りづらい箇所があります。
- 低リスクで効果が高い改善対象は「引数名の明確化」「モジュール定数の命名」「ユーティリティ変数名の具体化」です。

---

## リポジトリ横断の命名方針（提案）
- 関数/変数: `snake_case`、クラス: `CamelCase`、定数: `UPPER_SNAKE_CASE`（現状踏襲）。
- Geometry のメソッド系: 動詞（`translate/scale/rotate/concat`）。
- Effects の登録名: 現行を尊重しつつ、新規追加は「効果を表すシンプルな英単語」を推奨。
  - 変換系は名詞系（`translation/scaling/rotation`）で統一済みのため維持。
  - “-ify” の造語（`dashify`, `boldify`, `webify`）は読み手に意味が伝わりにくいので避ける。
- 正規化 vs 物理単位を明示:
  - 正規化パラメータは `*_norm`、ラジアンは `*_rad`、スケールは `*_scale` など接尾辞で区別。
  - 例: `rotate_norm`（0..1）/ `angles_rad`（ラジアン）／`scale`（倍率）。
- 数量パラメータは整数名を優先（`count`, `copies`, `n` など）。0..1 から整数に写像する場合は `*_norm` を併記して明示。
- ブールは状態を表す形（`is_*/has_*/enable_*`）。

---

## 具体的な改善提案（優先度つき）

### A. 引数名の明確化（低リスク・高効果）
- effects/rotation.py
  - 現状: `rotate: Vec3`（0..1 を 2π に変換）
  - 提案: `rotate_norm: Vec3` を許可する（`rotate` は当面維持）。
  - 目的: 値域の意図を引数名から読み取れるようにする。
- effects/array.py
  - 現状: `n_duplicates: float` → 整数へ丸め（0..1 正規化）。
  - 提案: `count: int` を第1候補に、`duplicates_norm: float` を後方互換として受理。
- effects/extrude.py
  - 現状: `subdivisions: float`（0..1→0..5）
  - 提案: `detail_norm: float` を受理。整数段数は将来的に `subdivisions: int`（本義）へ移行。
- shapes/text.py
  - 現状: `size: float`
  - 提案: `font_scale: float`（あるいは `scale`）へ。移行期は `size` 併存。

移行方針: 追加引数は後方互換で共存させ、`validate_spec` には新引数も許容させる（未知キー検出ロジックは現状 `**kwargs` 無し関数に対し厳密なので、関数シグネチャ側で新キーを追加する）。

### B. エフェクト登録名の語彙整理（中リスク）
- 造語の簡素化（意味を直接想起できる語へ）:
  - `dashify` → `dash`
  - `boldify` → `bold` または `embolden`
  - `webify` → `web`（“蜘蛛の巣化”の意なら）
- 備考: effects.registry は現在「エイリアス非対応」。命名変更は spec/チュートリアル/テストの一括更新が必要。段階導入したい場合は registry に限定的エイリアス機構を追加するか、移行スクリプトを用意（後述のチェックリスト参照）。

### C. モジュール内の定数・一時変数名（低リスク）
- effects/noise.py
  - 現状: モジュールレベル変数 `perm`, `grad3`（実質定数）
  - 提案: `NOISE_PERMUTATION_TABLE`, `NOISE_GRADIENTS_3D` など `UPPER_SNAKE_CASE` に。
  - 関数名 `perlin_core` は目的が曖昧 → `perlin_offset_field_3d` 等へ検討。
- effects/extrude.py
  - ループ内の累積変数 `acc` は `vertex_count` などにすると読みやすい。
- util/utils.py
  - `cfg` → `config`。`this_dir` → `project_root` など、役割を即時に連想できる名に。
- engine/render/line_mesh.py
  - `prim_restart_idx` と引数 `primitive_restart_index` の併存 → `primitive_restart_index` に統一し、フィールドも同名に。

### D. 「正規化値」命名の一貫化（中リスク）
- shapes/sphere.Sphere.generate, shapes/polygon.Polygon.generate など、`subdivisions: float` が「正規化値→整数」に写像されているケースを広く確認。
  - 提案: 正規化入力は `*_norm` を導入し、本義の整数は `subdivisions: int` に。短期は両対応、長期で非推奨化。

---

## 影響範囲と移行プラン
- 影響小（即時可）: C（定数・ローカル変数の改名）、A の「引数を追加して受け付ける」型の改善。
- 影響中: D（正規化と整数の二系統受理）。
- 影響大: B（エフェクト名自体の変更）。effects.registry がエイリアス非対応のため、以下のどちらか:
  1) 一括置換（effects名・to_spec/from_specのspec・tests・tutorials・docs すべて）
  2) registry に限定的な `alias(old, new)` を導入（短期のみ、起動時に DeprecationWarning を発する）。

---

## ファイル別・具体提案（抜粋）
- effects/rotation.py
  - `rotate` → 新規で `rotate_norm` を許可（0..1）。将来 `angles_rad` も許可（ラジアン直接指定）。
- effects/array.py
  - `n_duplicates` → 新規 `count: int`（優先）/ `duplicates_norm: float`（互換）。
- effects/extrude.py
  - `subdivisions: float` → `detail_norm: float`（互換）、将来的に `subdivisions: int` を本義に。
  - `acc` → `vertex_count`。
- effects/noise.py
  - `perm` → `NOISE_PERMUTATION_TABLE`、`grad3` → `NOISE_GRADIENTS_3D`。
  - `perlin_core` → `perlin_offset_field_3d`（検討）。
- util/utils.py
  - `cfg` → `config`、`this_dir` → `project_root`。
- engine/render/line_mesh.py
  - `prim_restart_idx` → `primitive_restart_index` に統一。
- engine/io/controller.py
  - `enable_debug` → `debug_enabled`（ブール状態名）。

---

## サーチ&置換チェックリスト（実行例）
> 参照だけでなく置換を伴う場合は `tests/` と `docs/` を含め一括更新してください。

- 正規化語尾の導入へ向けた棚卸し
  - `rg -n "subdivisions\s*:\s*float"`
  - `rg -n "n_duplicates|duplicates|count" effects/`
  - `rg -n "rotate=\(" effects/rotation.py tutorials/`（0..1 指定の用法確認）
- 造語系エフェクトの呼び出し箇所
  - `rg -n "dashify|boldify|webify"`（テスト・チュートリアル含む）
- ノイズ定数の参照
  - `rg -n "\bperm\b|\bgrad3\b" effects/noise.py`
- util の略称
  - `rg -n "\bcfg\b|\bconf\b" util/`、`rg -n "this_dir" util/`

---

## オープン課題（判断が必要）
- Effects 登録名のスタイル統一（名詞系に寄せるか、動詞系に寄せるか）。現状ドキュメントは `E.pipeline.rotation(...).filling(...)` を例示しており、変換系は名詞の設計思想が見て取れます。新規はこの方針に合わせ、既存の `dashify/boldify/webify` のみ語彙改善する案が妥当です。
- `validate_spec` のパラメータ検証は、先頭引数名を `'g'` として黙認する実装（先頭引数名に依存）。将来 `g → geom` へ改名したい場合は「先頭の POSITIONAL_ONLY/POSITIONAL_OR_KEYWORD をスキップする」ロジックへ変更する必要があります。
- `RenderPacket.timestamp` は現在クラス定義時に固定される（`time.time()` がデフォルト引数で評価）。命名ではないが生成時刻を持たせたい場合は `default_factory` を検討（機能面の補足）。

---

## 付録: 走査対象の主なディレクトリ
- api/: `E`, `G`, パイプライン、シリアライズ/検証
- effects/: 登録レジストリと各エフェクト関数
- shapes/: シェイプ生成とレジストリ
- engine/: core/geometry, transform_utils, render, pipeline, io
- util/, common/: 補助ユーティリティ
- tests/, tutorials/, docs/

---

### 今後の進め方（提案）
1) 低リスク（C, Aの追加受理）を先行反映 → テスト更新最小。
2) 正規化接尾辞（D）を段階導入 → ドキュメントとチュートリアルに注記。
3) 造語の語彙整理（B）を必要箇所に限定して実施 → 影響面を Issue 化し、合意後に一括適用。

以上。
