# Naming Spec (あるべき姿) — 2025-09-04

目的: 現行に縛られず、長期運用で迷いが生じにくい命名規範を提示します。設計の一貫性・読みやすさ・学習容易性を最優先に定めます。

---

## グローバル原則
- 関数/変数: `snake_case`、クラス: `CamelCase`、定数: `UPPER_SNAKE_CASE`。
- 関数名は「何をするか」を動詞で表す（副作用なしでも動詞）。
- 引数名は「単位・型・範囲」を名前で推測できるようにする（必要に応じ接尾辞）。
- 省略語は避ける（`arr/cfg/tmp/res/acc` → `array/config/temp/result/total_count`）。
- 同義語は統一（`pivot` を変換中心に、`origin` は (0,0,0) のみ）。

---

## 単位と接尾辞
- 角度: 規定はラジアン。明示は `*_rad`。度は `*_deg`（両指定はエラー）。
- 距離/長さ: mm 前提。必要に応じ `*_mm`。
- 正規化値(0..1): 原則廃止。どうしても使う場合は `*_norm`。
- 周波数: 空間は `spatial_freq`（cycles per unit）。時間は `freq_hz`。
- 時間: 秒は `*_sec`。

---

## API サーフェス
- Geometry メソッド: `translate/scale/rotate/concat`（現行踏襲）。
- Effects 登録名（`E.pipeline` 配下）: すべて動詞の命令形で統一。
  - 例: `E.pipeline.rotate(...).scale(...).translate(...).fill(...).trim(...).subdivide(...).repeat(...)`。
- Registry キー: 公開名は `snake_case` のみ（登録時に大文字/ケバブを正規化）。

---

## エフェクト命名（理想の一覧）
- 幾何変換: `translate`, `scale`, `rotate`, `affine`（複合変換）
- 分割/簡略: `subdivide`, `simplify`, `trim`
- 配列複製: `repeat`（旧: array）
- 太らせ/オフセット: `offset`（旧: buffer, boldify）
- 破裂: `explode`
- 押し出し: `extrude`
- 充填/ハッチング: `fill`（mode: `hatch|cross|dots`）
- ねじり: `twist`
- 波歪み: `ripple`（旧: wave）
- ふらつき: `wobble`
- 破線化: `dash`（旧: dashify）
- 網目化: `weave`（旧: webify）
- 変位場: `displace`（field: `perlin|curl|...`、旧: noise）

---

## 代表エフェクトの引数仕様（理想）
- `translate(delta: Vec3)`
  - `delta`: 加算変位ベクトル（mm）。
- `scale(scale: Vec3, *, pivot: Vec3=(0,0,0))`
  - `scale`: 各軸倍率。`1.0` で等倍。
  - `pivot`: 変換中心。
- `rotate(angles_rad: Vec3, *, pivot: Vec3=(0,0,0))`
  - `angles_rad`: ラジアン指定。`angles_deg` も受理（どちらか一方）。
- `fill(mode: Literal['hatch','cross','dots'], density: float, angle_rad: float=0.0)`
  - `density`: 密度（0..1 を採用）。
- `repeat(count: int, *, offset: Vec3=(0,0,0), angles_rad_step: Vec3=(0,0,0), scale_mul: Vec3=(1,1,1), pivot: Vec3=(0,0,0))`
  - 各複製ごとのステップ量を明示（角度はラジアンステップ）。
- `offset(distance_mm: float, *, join: Literal['miter','round','bevel']='round', segments_per_circle: int=8)`
- `displace(amplitude_mm: float, *, spatial_freq: Vec3|float=(0.5,0.5,0.5), t_sec: float=0.0, field: str='perlin')`

---

## 変数・定数の命名
- 定数: `UPPER_SNAKE_CASE`（例: `NOISE_PERMUTATION_TABLE`, `NOISE_GRADIENTS_3D`）。
- ローカル/一時: 役割が分かる名（`acc`→`vertex_count`、`cfg`→`config`）。
- 衝突回避: レンダリングの `SwapBuffer` と区別するため、エフェクト名に `buffer` は使わない（`offset` へ統一）。

---

## 現行→理想 マッピング（主要どころ）
- translation → translate
- scaling → scale
- rotation → rotate
- filling → fill
- trimming → trim
- subdivision → subdivide
- array → repeat
- buffer → offset
- boldify → offset（または `thicken` だが意味重複のため `offset` に統一）
- dashify → dash
- webify → weave
- wave → ripple
- noise → displace(field='perlin')

---

## validate_spec まわりの命名規範
- 先頭引数は常に `g: Geometry` 固定（検証から除外）。
- パラメータ名は上記「理想引数名」に合わせる。未知キーはエラー（`**kwargs` 非推奨）。
- Enum/モードは `mode` または用途名（`join`, `field`）で一貫化。

---

## 改善チェックリスト（実装順の推奨）
- [x] エフェクト登録名を上記「理想一覧」に統一（alias 不使用、完全移行）
- [x] `translate/scale/rotate` の引数を `delta/scale/angles_rad` と `pivot` に統一（旧名も受理）
- [x] `repeat`（旧 array）に `count/offset/angles_rad_step/scale_mul/pivot` を導入
- [x] `offset`（旧 buffer/boldify）に `distance_mm/join/segments_per_circle` を導入
- [x] `fill`（旧 filling）の `mode/density/angle_rad` を導入
- [x] `displace`（旧 noise）へ改名し、`amplitude_mm/spatial_freq/t_sec/field` を導入
- [x] `subdivide/simplify/trim/twist/ripple/wobble/dash/weave/extrude/explode` の引数を定義し直し
- [x] `effects/noise.py` の `perm/grad3` を `NOISE_PERMUTATION_TABLE/NOISE_GRADIENTS_3D` に改名（旧名は互換エイリアス残し）
- [x] ローカル変数の略称撲滅（例: `effects/extrude.py` の `acc`→`vertex_count`、`util/utils.py` の `cfg/this_dir`→`config/project_root`）。残項目は継続。
- [x] `engine/render/line_mesh.py` の `prim_restart_idx` を `primitive_restart_index` に統一（互換プロパティ追加）
- [x] `engine/io/controller.py` の `enable_debug` → `debug_enabled`（互換プロパティ追加）
- [x] ドキュメント・チュートリアル・テストの名称を一括更新
- [x] `validate_spec` を新パラメータ名に合わせて強化（param_meta による型/範囲/choices 検証）
- [x] 旧名→新名の自動変換スクリプト（scripts/rename_effects_in_specs.py）を追加

---

## ロールアウト方針（推奨）
- Phase 1（互換期）: registry に `alias(old, new)` を追加し、旧名使用時に DeprecationWarning。
- Phase 2（置換期）: 公式ドキュメント・サンプル・テストを新名へ切替、旧名の警告を強化。
- Phase 3（収束期）: 旧名を削除し、`from_spec/validate_spec` も新名のみ受理。

---

以上が「あるべき姿」の命名仕様と具体的な移行チェックリストです。
