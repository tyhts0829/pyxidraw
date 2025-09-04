# 破壊的変更 提案ダイジェスト（2025-09-04 更新）

この文書は「現状把握と残課題の一覧」に特化した短縮版です。詳細な背景/メリデメはコミットメッセージと各モジュールの docstring を参照してください。

## 現状サマリ（達成状況）
- [x] 提案1: Geometry 統一（`engine/core/geometry.py`）
- [x] 提案2: エフェクト関数化（`effects/*` は `@effect` 関数のみ。`effects/registry.py` は関数以外を拒否）
- [x] 提案3: パイプライン一本化（`api/pipeline.py` の `E.pipeline...build()(g)`）
- [x] 提案4: 単層キャッシュ（`Pipeline` の `(geometry_hash, pipeline_key)` のみ）
- [x] 提案5: 命名/型/0–1 写像の共通化（Vec3 適用・0..1→レンジ写像/角度正規化の統一、Doc/README 反映完了）
- [x] 提案6: シリアライズ/検証簡素化（`Pipeline.to_spec/from_spec`。polyhedron データは npz のみ）
- [x] 提案7: ログ/例外の標準化（エンジン層は logging に統一。CLI/ベンチは仕様上の print を維持）
- [x] 提案8: ディレクトリ再編（__init__.py 追加で import を安定化、不要ディレクトリ掃除完了）

## 直近で完了した主な変更
- 互換層とレガシー清掃
  - `old/` ディレクトリ削除。`E.add(...).result()` 系の痕跡除去。
  - `effects/registry.py` を関数専用に簡素化（クラス登録不可）。
  - `tutorials` の誤 API（`.result()` 等）修正、ログ文の `E.pipeline` 表記統一。
- ベンチマークの新 API 対応
  - `benchmarks/plugins/serializable_targets.py` を `E.pipeline` ベースに統一。
  - `subdivision/extrude/buffer` を新パラメータにマッピングし実行可能化。
  - 全体実行成功（52 targets, 100% success）。
- ドキュメント刷新
  - README を PyxiDraw4 表記・最新 API に更新。チュートリアル README も更新。

## 未消化/保留（やることリスト）
- 提案5（命名/スケール/型の統一）は完了（Vec3 適用と 0..1 正規化の統一を含む）
- 提案7（logging 統一）は完了（engine 層は logging、CLI/ベンチは print 維持）
- 提案8（ディレクトリ再編）は完了（`__init__.py` 追加・不要物削除）
- polyhedron データの完全移行は完了（pickle フォールバック撤去）

### 決定記録: API パラメータの完全切替（2025-09-04）

- 旧名パラメータは廃止。新名のみ受理。
  - rotate(0..1) → angles_rad（ラジアン）/ pivot（中心）
  - translate: offset/offset_x/y/z → delta
  - fill: pattern/angle → mode/angle_rad
  - repeat: n_duplicates/rotate/center → count/angles_rad_step/pivot
  - displace: intensity/frequency/time → amplitude_mm/spatial_freq/t_sec
  - offset: join_style/resolution → join/segments_per_circle

理由: 一貫性・可読性・検証強化（validate_spec と param_meta の単純化）。

- 安全性: pickle はロード時に任意コード実行のリスクがある一方、npz は `allow_pickle=False` 前提で純データのみを扱える。
- 互換性: pickle は Python 実装/バージョン/クラス定義に依存が強い。npz は環境非依存で将来の移行コストが低い。
- 再現性: dtype(`float32`)/shape/配列順を明示でき、`arr_0, arr_1, ...` 規約で決定的ロードが可能。
- 単純化: 読み込みが「配列を開く」だけになり、互換クラスやラッパ不要。`shapes/polyhedron.py` のローダが簡潔。
- 配布効率: zip 圧縮でサイズ削減・配布容易（CI アーティファクト/キャッシュとの相性が良い）。
- 運用整合: 本プロジェクトの設計方針（データはデータ形式、ロジックはコード）および `Pipeline.to_spec/from_spec` の方針と一貫。

補足:
- 互換期間は終了。`.npz` のみを使用（`shapes/polyhedron.py` の pickle フォールバックを撤去）。
  一括移行用に `scripts/convert_polyhedron_pickle_to_npz.py` を提供。
 - 関連 ADR: `docs/adr/0001-npz-over-pickle.md`

## マイグレーション・チートシート（主要置換）
- 変換 API: `size→scale`, `at→translate`, `spin→rotate(z: 0..1→2π)`, `move→translate`, `grow→scale`。
- エフェクト: クラス継承は禁止。`effects.registry.effect` で登録する関数のみ（`Geometry -> Geometry`）。
- パイプライン: `E.pipeline ... .build()(g)` に統一（単層キャッシュ）。

## 参照ポイント（実装の所在）
- Geometry: `engine/core/geometry.py`（純関数 `translate/scale/rotate/concat`）
- パイプライン: `api/pipeline.py`（`Pipeline/PipelineBuilder` と to_spec/from_spec）
- エフェクト: `effects/*`（`@effect` 関数。rotate/translate/scale/affine/displace/...）
- レジストリ: `effects/registry.py`, `shapes/registry.py`
- 形状 API: `api/shape_factory.py`（`G.*` は `Geometry` を返却）
- ベンチ: `benchmarks/plugins/*`, `benchmarks/core/*`（CLI: `python -m benchmarks`）

---

この文書は、提案の「進捗と残課題」を常に最新へ短く保つことを目的としています。詳細な議論・根拠は各 PR/コミットに付随する記述を参照してください。

## 採用優先度（コスト/効果の目安）
1) Geometry 統合（高効果・中コスト）
2) パイプライン一本化＋キャッシュ単層化（高効果・中コスト）
3) エフェクト関数化（中効果・中コスト）
4) 命名/型/写像の一貫化（中効果・低コスト）
5) シリアライズ/検証簡素化（中効果・低コスト）
6) ログ統一（低効果・低コスト）
7) ディレクトリ再編（中効果・中コスト、最後に）

---

## 参考コードスケッチ（抜粋）

（あくまで方向性の共有用。実装時は PR を小さく分割して進める）

```
# engine/core/geometry.py
@dataclass(slots=True)
class Geometry:
    coords: np.ndarray
    offsets: np.ndarray
    def translate(self, dx, dy, dz=0): ...
    def scale(self, sx, sy=None, sz=None): ...
    def rotate(self, x=0, y=0, z=0, center=(0,0,0)): ...

# effects/noise.py（概念）
@effect
def noise(g: Geometry, *, amplitude_mm=0.5, spatial_freq=(0.5,0.5,0.5), t_sec=0.0) -> Geometry:
    ...

# api/pipeline.py（概念）
@dataclass(frozen=True)
class Step:
    name: str
    params: dict
    fn: Callable[[Geometry], Geometry]

class Pipeline:
    def __call__(self, g: Geometry) -> Geometry: ...
```

---

## 実装ログ（進捗・検証結果）

本提案のうち、互換性を考慮せず「美しく・シンプル・読みやすく」を優先した範囲で以下を実装・確認済み。

- 実施サマリ:
  - Geometry 統合: `engine/core/geometry.py` を単一データクラス化。最小 API は `translate/scale/rotate/concat/from_lines/as_arrays`。純粋・新インスタンス返却。
  - GeometryData 統合: `engine/core/geometry_data.py` は暫定的に `GeometryData = Geometry` として公開（GUID 等の複雑さを撤廃）。将来的に `GeometryData` 参照は完全撤去予定。
  - 変換ユーティリティ: `engine/core/transform_utils.py` は実質 `Geometry` 前提に簡素化（入力型と同型で返却）。
  - エフェクト関数化: `effects/noise.py`, `effects/filling.py`, `effects/array.py`, `effects/translation.py`, `effects/rotation.py`, `effects/scaling.py` を関数ベースへ（`@effect` 登録、`Geometry -> Geometry`）。
  - レジストリ: `effects/registry.py` を関数レジストリ化（明示名のみ、エイリアス非対応）。
  - パイプライン: `api/pipeline.py` で `E.pipeline` を提供。単層キャッシュ（`(geometry_hash, pipeline_hash)`）。
    - geometry_hash: `coords/offsets` を blake2b でハッシュ。
    - pipeline_hash: ステップ名 + 正規化パラメータ + 関数バージョン（`__code__.co_code` のダイジェスト）。
  - API エクスポート: `api/__init__.py` は `E.pipeline` と `Geometry` を公開（`E` はシングルトン）。

- テスト整備（現行）:
  - テストは `tests/` に集約（v3 スイート）。
  - pytest 非依存の最小ランナーは `scripts/run_unified_geometry_check.py` に移設。

- サニティ実行（最小ランナー）:
  - `scripts/run_unified_geometry_check.py` を用意（pytest不要、コア動作の最低限の確認に利用）。

- 既知の制約/今後の作業:
  - 旧 `EffectChain`/`GeometryAPI` 系は破壊的変更により未対応（使用停止前提）。
  - 残りの主要エフェクトの関数化（`subdivision/extrude/buffer/...`）、旧 API/クラスの完全撤去、サンプル・ドキュメント更新。
  - `shapes/` の戻り値の明示的な `Geometry` 型統一（実体は統一済みだが命名の整備）。

### 追補（GUI ランタイムの移行・旧 API 撤去）

- ランタイム/レンダラを Geometry 化:
  - `api/runner.py`: `user_draw: (t, cc) -> Geometry` に統一。
  - `engine/pipeline/{buffer, worker, packet}`: バッファ/ワーカー/パケットを `Geometry` 型に移行。
  - `engine/monitor/sampler.py`: 頂点数計測を `Geometry` ベースに変更。
  - `engine/render/renderer.py`: VBO/IBO 変換を `Geometry.coords/offsets` から直接生成。

- デモの移行（動作確認済み）:
  - `main.py`: `E.pipeline` + 関数エフェクト（noise/filling/rotation）に全面移行（GUI起動確認済み）。
  - `shapes_grid_cc.py`: `.size/.at` → `.scale/.translate`、回転は `E.pipeline.rotate(...).build()(g)` に変更。
  - `simple.py`: 最小サンプルを `Geometry` + `E.pipeline` 化（統計のみ出力）。

- 形状ファクトリの統一:
  - `api/shape_factory.py`: 返り値を `Geometry` に統一。`G.from_lines/G.empty` も `Geometry` を返却。

- ベンチマークの暫定対応:
  - `benchmarks/plugins/serializable_targets.py`: `api.effect_chain` → `api.pipeline`。`noise/filling/array` は新パイプラインで適用。
  - `subdivision/extrude/buffer` は未関数化のため `NotImplementedError` を明示（後続で対応）。

- チュートリアルの更新（部分）:
  - `tutorials/01_basic_shapes.py`: `.scale/.translate` に更新、Geometry 出力へ。
  - `tutorials/README.md`: `E.pipeline` と Geometry 変換 API（`.scale/.translate`）に沿って記述を更新。
  - `tutorials/02_multiple_shapes.py`: `.size/.at` と `.add()` を `.scale/.translate` と `+` に置換。
  - `tutorials/03_basic_effects.py`: `E.add()` のチェーンを `E.pipeline` に置換。未実装の細分化は削除。
  - `tutorials/04_custom_shapes.py`: `GeometryAPI` 依存を排し、`GeometryData.from_lines` でカスタム形状を構築。
  - `tutorials/05_custom_effects.py`: `@E.register()` を `effects.registry.effect` に置換。関数エフェクト化（wave/explode/twist/gradient）。
  - `tutorials/06_advanced_pipeline.py`: 諸所の `E.add()` を `E.pipeline` に置換。未実装エフェクトは代替で構成。

- 旧 API の撤去（破壊的変更）:
  - 削除: `api/geometry_api.py`, `api/effect_chain.py`, `api/effect_pipeline.py`。

- 既知の影響/残タスク:
  - 旧 API を参照するテストやチュートリアルの残存箇所はエラーとなる（段階的に新 API へ移行予定）。
  - ベンチ対象の `subdivision/extrude/buffer` の関数化と登録。
  - `AGENTS.md` などドキュメントの旧 API 記載の更新・整理。

---

この設計は「型 1・関数・単一パイプライン・単一キャッシュ」という最小構成を核に、読みやすさ・保守性・拡張性のバランスを最適化することを狙います。採用可否や優先順位のフィードバックをいただければ、段階実装の計画に落とし込みます。

---

## ここまでのまとめ（完了事項）

- Geometry 統一
  - `engine/core/geometry.py`: 統合 Geometry（translate/scale/rotate/concat/from_lines/as_arrays）。
  - `engine/core/geometry_data.py`: 削除（`GeometryData` 互換は撤廃済み）。
  - `shapes/*`: すべて `Geometry` を返却するよう統一。

- エフェクトの関数化（登録は effects/__init__ の副作用で実行）
  - 標準: translation / rotation / scaling / noise / filling / array / subdivision / transform / extrude / buffer / dashify / wobble / boldify / collapse / trimming / webify（すべて関数）。
  - `effects/registry.py`: 関数レジストリ（@effect / @effect() / @effect("name") 対応、エイリアス非対応）。
  - パイプライン: `api/pipeline.py` に単層キャッシュ Pipeline（`E.pipeline...build()(g)`）。
  - 削除: `effects/base.py`（BaseEffect）, `effects/pipeline.py`。

- 形状 API の単純化
  - `api/shape_factory.py`: `G.*` が `Geometry` を直接返却。`G.from_lines/G.empty` も `Geometry`。

- 旧 API の撤去（破壊的変更）
  - 削除: `api/geometry_api.py`, `api/effect_chain.py`, `api/effect_pipeline.py`, `effects/base.py`, `effects/pipeline.py`。

- サンプル/チュートリアル/README の更新
  - `main.py`: 新パイプラインに全面移行（GUI起動確認済み）。
  - `shapes_grid_cc.py`: `.scale/.translate` と `E.pipeline.rotation` に置換。
  - tutorials: 01–06 を新 API・非MIDI化・`SQUARE_300` に統一。
  - `README.md`: 例を `Geometry + E.pipeline` に刷新。

- ベンチの対応
  - `benchmarks/plugins/serializable_targets.py`: `api.pipeline` を使用。主要エフェクトの関数化に追随済み。

- 新規テストと手動確認（現行）
  - テストは `tests/` に集約済み（v3 スイート）。
  - 最小ランナー `scripts/run_unified_geometry_check.py` を提供（pytest 非依存で一部確認）。

---

## 再開時の TODO（次にやること）

- 旧表記/旧APIの残滓の整理（ドキュメント）
  - [x] `AGENTS.md` の GeometryAPI/EffectChain 記述を更新（最新仕様に同期済み）。
  - [x] README/チュートリアルの細部（語彙・角度0..1→2πの指針）を明記・統一。

- テストスイートの刷新
  - [x] 旧API依存のテキスト/コメントの掃除（`benchmarks/tests/test_plugins.py` から旧表記のモック記述を除去）。
  - [x] 暫定の `new_tests/` は整理済み（内容は `tests/` に統合、残置物を削除）。

- 型/設計の仕上げ
  - [x] `shapes/` 戻り値注釈の再点検（`Geometry` に統一済みを確認）。
  - [x] `common/cacheable_base.py` の依存は温存。環境変数での無効化/サイズ制御追加により実運用負荷を低減（段階縮小は不要と判断）。
    - [x] 環境変数で LRU キャッシュの無効化/サイズ指定を可能化（`PXD_CACHE_DISABLED`, `PXD_CACHE_MAXSIZE`）。

- 命名/型/0–1 写像の共通化（提案5の仕上げ）
  - [x] `common/param_utils.py` を主要エフェクトへ適用（rotation/transform/noise/array/buffer/extrude/trimming/subdivision/collapse/webify）。
    - [x] `buffer/explode/transform/trimming/collapse` に適用。ドキュメントを更新（README/Docstring）。

- シリアライズ/検証（提案6）
  - [x] `[{name, params}]` 形式のシリアライザ/バリデータ追加（`api.pipeline.validate_spec`）。未知名・不正型（非JSON様式/未定義パラメータ※kwargs未対応エフェクトのみ）を早期失敗。
  - [x] 旧 pickle 資産 → npz 変換スクリプトを `scripts/convert_polyhedron_pickle_to_npz.py` として同梱（削除オプション付き）。

- ログ/例外（提案7）
  - [x] `print` を `logging` に統一（engine/io, render/renderer を整備）。

- ディレクトリ再編（提案8）
  - [x] 最終フェーズで import 安定化（`engine/*`, `util/` に `__init__.py`）。

- ベンチ/最適化
  - [x] ベンチマークは `E.pipeline` 前提に統一済み（plugins を確認）。最小フローを README に記載し、回帰検知手順を標準化。
  - [~] `geometry_hash` の最適化は現状不要（性能上の懸念が顕在化した時点で再検討）。

---

## 参考メモ

- 方針の原則
  - 型は `Geometry` の 1 本化。
  - エフェクトは「純関数（Geometry -> Geometry）」に統一。
  - パイプラインは 1 本・単層キャッシュのみ（予測可能性重視）。
  - レジストリはエイリアス非対応（明示名のみ）。

- 運用ノート
  - チュートリアルはすべて MIDI/arc なしで動作（runner がダミー CC を供給）。
  - GUI 経路（pyglet/moderngl）は Geometry 直処理。例外時は worker -> receiver 経由でメインへ再送。
