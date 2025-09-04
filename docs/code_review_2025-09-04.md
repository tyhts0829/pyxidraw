# PyxiDraw4 コードレビュー（2025-09-04）

- 対象範囲: リポジトリ全体（直近の提案5/6/7/8の適用後）
- レビュア: 開発支援AI（自動レビュー）
- 目的: 設計/品質/安全性/性能/運用の観点での現状評価と、具体的な改善提案の提示

## サマリ
- 体系化: 単一 `Geometry` 型 + 関数エフェクト + 単一パイプライン + 単層キャッシュという設計は簡潔で拡張に強い。
- 安全性: polyhedron データ形式を pickle → npz に一本化し、任意コード実行リスクを回避済み。Pipeline の spec 検証も導入済み。
- 一貫性: 角度は `angles_rad/angles_deg`（旧 `rotate` も受理）で明示、0..1→レンジ写像の方針、`Vec3` 型別名の横断適用で API はほぼ統一。
- 可観測性: engine 層の logging への統一・renderer の DEBUG ログでデバッグしやすさ向上。
- 品質: テスト 231 件が通過（回帰なし）。README/AGENTS/ADR 等の文書整備も進捗良好。

全体評価: 健全で読みやすい構成。残りは「追加テスト」「設定/データ生成系の微整理」「ドキュメントの最終調整」を継続すれば安定フェーズに入れる。

## 設計・アーキテクチャ
- 良い点
  - `engine/core/geometry.py` のミニマル API（translate/scale/rotate/concat/from_lines/as_arrays）が明快で、後段の最適化余地も残す構造。
  - `effects/registry.py` + 関数デコレータでの登録は依存関係が薄く、テスタブル。
  - `api/pipeline.py` の単層キャッシュ（geometry_hash × pipeline_key）は予測可能性が高い。spec の往復（to_spec/from_spec）と `validate_spec()` で流通も扱いやすい。
  - `common/param_utils.py` による 0..1 正規化の集約は、仕様の一貫性維持に効く。
- 気になる点/提案
  - キャッシュキーの計算コスト: 毎回 `coords/offsets` 全体を blake2b するため、超大規模ジオメトリでは CPU 帯域を食う可能性。将来の最適化候補として、Geometry に「スナップショット用のダイジェスト」を持たせ、変換時に再計算する方式（準イミュータブル運用）を検討余地。
  - `Pipeline` のパラメータ検証はキー/JSON様式/関数シグネチャで充分に堅牢。ただし数値域の妥当性（例: `density ∈ [0,1]`）までは見ていない。将来的にエフェクト側で「param メタデータ（型/範囲/既定値）」を宣言・集約できると UX が上がる。

## API/一貫性
- `Vec3` の横断適用（effects/affine/rotate/scale/repeat/extrude/ripple/wobble 等、shapes/base、engine/core/geometry）でシグネチャの読みやすさが向上。
- `translate` は `delta: Vec3`（推奨）と `offset/offset_x/y/z` の両受理で後方互換性に配慮（良）。
- `displace.spatial_freq: float|Vec3` を明示化し、`ripple/wobble` も周波数の取り扱いを Docstring で明記（良）。

## データ/I/O/安全性
- polyhedron の .npz 一本化は妥当（`shapes/polyhedron.py` の pickle フォールバック撤去済）。
- `data/regular_polyhedron/regular_polyhedron.py` / `data/sphere/sphere.py` にデータ生成スクリプト（icecream 呼び出し）があるが、実行時依存ではないため放置でも害は少。将来の CI 環境では `icecream` を使わないログ出力への置換を検討してもよい。

## ロギング/可観測性
- engine 層は logging に統一。`engine/render/renderer.py` のアップロードサイズ/描画スキップの DEBUG ログで、描画パスの確認が容易。
- CLI/ベンチは仕様上の print を維持（適切）。

## パフォーマンス
- Numba を用いた `effects/noise.py` や `effects/filling.py` は高速化の方向性として適切。
- `Geometry.from_lines` の連結（concatenate）は大半のケースで妥当。極端に大量の細分化や配列生成では、繰り返し `np.vstack` を避けるために「最初にサイズ見積もり→一括確保」最適化の余地はある（現状問題化していない）。
- ベンチの最小フローを README に追記済。`compare` での敷居値/指標（例: 幾何点数 vs 速度）テンプレ化は今後の改善点。

## テスト
- 総数 231 のテストが通過。`validate_spec` まわりの正常/異常系テストが追加済で良い。
- 追加の観点（任意）
  - `effects/*` の境界値（density=0/1、subdivisions=0/1等）と極端値（大規模 points）での動作確認。
  - `api/pipeline.Pipeline.cache` のキャッシュヒット/ミスの明示的検証。
  - `shapes/polyhedron` の .npz ローダのキー種別（`arrays` vs `arr_0..N`）の網羅。

## ドキュメント/開発体験
- README の「角度/スケール取り扱い」「ベンチ基準化」「シリアライズ/検証」「キャッシュ制御」など、実務導入に必要な事項が揃っている。
- ADR（npz 採用理由）で意思決定が残っており良い。将来的に Pipeline spec ルール（validate_spec の範囲/方針）についても ADR 化すると、拡張時の議論がやりやすい。

## リスク/改善提案（優先度つき）
1) 中（機能安全）: `validate_spec` の「数値域検証」を段階導入
   - 方式案: 各エフェクト関数に `__param_meta__ = {name: {type, min, max, choices}}` を任意付与 → `validate_spec` があれば尊重。
2) 中（パフォーマンス）: キャッシュダイジェストの二度計算回避
   - 方式案: `Geometry` に `digest` フィールド（オプション）を持たせ、変換 API 内で更新。測定とトレードオフ評価が必要。
3) 低（DX）: ベンチ比較テンプレの同梱
   - `benchmarks/config/compare_template.yaml` を追加し、敷居値や対象ターゲットの共通定義を提供。
4) 低（整備）: データ生成スクリプトの `icecream` 依存整理
   - `print` or `logging` に置換するか、`if __name__ == '__main__'` 節のみで import する形に変更。

## 具体的アクション（短期ロードマップ）
- [ ] エフェクト 2〜3個で `__param_meta__` の PoC（最小: `fill.density`, `extrude.subdivisions`, `offset.distance`）
- [ ] `benchmarks/config/compare_template.yaml` の追加と README からの参照
- [x] `data/*` 生成スクリプトの `icecream` 参照を `logging` に切り替え（不要: 現状 `icecream` 参照なし）
- [x] ADR: 「Pipeline Spec 検証ポリシー」を起票（済: docs/adr/0002-pipeline-spec-validation.md）

以上。
