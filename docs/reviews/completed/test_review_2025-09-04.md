# テスト設計レビュー（2025-09-04）

対象: プロジェクト全体（pytest ベースの `tests/`）、補足として `benchmarks/` の役割分離を考慮。

## 現状スナップショット
- ランナー: pytest（`pytest.ini` で `testpaths=tests`、`addopts=-v --tb=short`）。
- 構成: 効果（effects）、変換（affine/translate/rotate/scale）、幾何（unified geometry）、
  パイプライン（builder/serialization/spec-validation）、機能系（repeat/offset/extrude/fill/subdivide）などの単体テストが中心。
- 追加: ベンチ系は `benchmarks/` に分離（実測と可視化、比較は CLI で実施）。
- 結果: 現状 231 テストが安定通過（回帰なし）。

## 良い点
- 粒度: 1 つの機能に 1～数ケースの単体テストを配置。失敗時の切り分けが容易。
- 仕様カバー: `Pipeline` の `to_spec/from_spec/validate_spec` までカバー済で、外部表現の破壊的変更を検知しやすい。
- シンプル: ハードウェア依存（MIDI/GL）を避け、計算結果（座標/オフセット）の検証に集中。
- 実用: `benchmarks` を別系統にして性能/可視化を担わせる思想が明確。

## 改善余地（観点別）
- 仕様境界の網羅性
  - 効果パラメータの端/外れ値（density=0/1、subdivisions=0/1、distance=0 など）の系統網羅。
  - ベクトル引数（Vec3）でのスカラ/1要素/3要素の受理と正規化の確認（いくつかは実施済）。
- ランダム/性質テスト
  - 幾何変換の基本性質（結合性、単位元、逆元）を性質ベース（property-based）でサンプリング検証。
  - `displace` や `wobble/ripple` は近傍値の連続性・出力レンジなどの不変条件を軽く確認。
- 併走/キャッシュの可観測性
  - `Pipeline` のキャッシュヒット/ミスの明示的テスト（既存 API テストに加え、keyの違いの確認）。
  - `WorkerPool` の並走での落ちない/リークしない性質（最小限の統合テスト）。
- ロギング/エラー
  - 例外ポリシー（`InvalidPortError` など）が runner で適切に変換されるかの最小テスト。
- 機能拡張の回帰防止
  - `extrude(center_mode="auto")` の新挙動が従来既定と両立するか（互換ケースのスナップショット）。

## 推奨アクション（チェックリスト）
- [x] 端/外れ値の系統網羅（優先: 中）
  - [x] effects/fill: density=0/1 と angle の代表値でライン数・ドット数の境界確認
  - [x] effects/subdivide: 0/1/最大の分割数で頂点数増加の期待値確認
  - [x] effects/offset: distance=0 で恒等、join の代表 3 つで例外が出ないこと
- [x] ベクトル正規化と受理形の確認（優先: 中）
  - [x] affine/repeat: rotate=(s,) 相当と (x,y,z) の 0..1→2π 変換の一致（rotate は Vec3 受理）。新APIの `angles_rad_step` でも一致
- [x] translate: 旧名 `offset(_x/_y/_z)` は廃止。`delta` のみを受理することを確認
- [x] キャッシュ挙動の観測（優先: 中）
  - [x] Pipeline: 同一 `Geometry` + 同一 `Pipeline` でキャッシュヒット、関数コード変更/params変更でミス。
- [x] 並走/安定性の最小統合（優先: 低）
  - [x] WorkerPool: 2～3 フレームだけ回して `RenderPacket` が順序を保ち、close でハングしない
- [x] ロギング/エラー経路の確認（優先: 低）
  - [x] `InvalidPortError` → runner が `SystemExit(2)` に変換（mido 入力名をモック）
- [x] 新オプション回帰抑止（優先: 低）
  - [x] extrude: center_mode="origin" と "auto" の差分検証（頂点数は同一で座標差異）
- [x] property-based tests の導入（優先: 低）
  - [x] hypothesis（任意依存）がある場合に transform/concat の基本性質を検証（tests/test_properties_hypothesis.py）。
        無い場合はテストをスキップするため CI への影響はない。

## 実装メモ（ガイド）
- 命名/場所: 既存の `tests/test_*.py` に沿って機能別に追加。境界値テストは既存ファイルへ追記。
- 乱択: hypothesis の `@given` を使う際はシード固定や小さな探索空間で短時間に抑える。
- ベンチ分離: 性能は `benchmarks/` に任せ、pytest は「正しさ」「契約の維持」を軸にする。

以上。小さな PR を積み重ねる方針で、上のチェックボックスを潰していくのがおすすめです。
