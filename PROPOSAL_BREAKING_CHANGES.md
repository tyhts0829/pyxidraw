# 破壊的変更前提の簡素化提案（2025-09-01）

本ドキュメントは「既存 API 互換を気にせず、コードベースを美しく・シンプルに・読みやすく」するための破壊的変更案をまとめたものです。各提案ごとにメリット・デメリット・影響範囲/移行メモを併記します。

---

## 進捗ステータス（2025-09-03 更新）
- [x] 提案1 Geometry 統一（GeometryData/GeometryAPI 撤廃）
- [x] 提案2 エフェクト関数化（BaseEffect 撤廃）
- [x] 提案3 パイプライン一本化（`E.pipeline` のみ）
- [x] 提案4 単層キャッシュ（Pipeline のみ）
- [~] 提案5 命名/型/0–1 写像の共通化（主要エフェクトへ適用拡大中：rotation/noise/array/wobble 反映）
- [x] 提案6 シリアライズ/検証の簡素化（Pipeline to_spec/from_spec 実装、pickle→npz 変換スクリプト同梱）
- [~] 提案7 ログ/例外の logging 統一（engine/pipeline/worker と engine/io/controller に適用、残りは段階適用）
- [ ] 提案8 ディレクトリ再編（未）

## 背景と目標
- 目的: 学習コストの削減、責務の明確化、抽象の重複解消、単純で予測可能なキャッシュ戦略、命名/単位/型の一貫性。
- 現状の課題:
  - Geometry の重複（GeometryData / Geometry / GeometryAPI）。
  - EffectChain と EffectPipeline の二重系統・キャッシュ多層化。
  - エフェクトがクラス＋キャッシュ抽象に依存し理解コストが高い。
  - 命名・パラメータ・型表現が箇所により揺れる。

---

## 提案 1: Geometry を 1 型に統合

- 提案概要:
  - `GeometryData`/`Geometry`/`GeometryAPI` を廃し、`Geometry`（データクラス）に統一。
  - 変換 API は `translate/scale/rotate/concat` の最小セットだけ提供し、純粋に新インスタンスを返す。

- メリット:
  - レイヤ/概念の重複が解消され、読みやすさ向上。
  - 変換の副作用がなく、推論・テストが容易。
  - API 表面積が縮小し、ドキュメント/チュートリアルが簡潔に。

- デメリット:
  - 既存の `GeometryAPI` 連鎖（`size/at/spin` 等）を置換する必要。
  - 既存のサンプル・テスト多数の更新が必要。

- 影響範囲/移行:
  - `api/geometry_api.py` と `engine/core/geometry_data.py` を廃止し、`engine/core/geometry.py` に集約。（実装済み）
  - `shapes/*` の戻り値型を `Geometry` に統一（実装済み）。
  - 置換ガイド: `size→scale`, `at→translate`, `spin→rotate(z=deg2rad)`, `move→translate`, `grow→scale`。

---

## 提案 2: エフェクトを関数ベースに（クラス/キャッシュ抽象を撤廃）

- 提案概要:
  - `effects/*.py` の `BaseEffect` 継承＋`apply` をやめ、`@effect` 登録の純粋関数に統一（`Geometry -> Geometry`）。
  - 各エフェクト内で入力コピー・出力返却を徹底（副作用なし）。

- メリット:
  - 概念が「関数」に一本化され、理解しやすい。
  - `LRUCacheable` 等の抽象を削減し、実装/依存の見通しが良くなる。
  - テストがシンプル（直接 `effects.noise.noise(geom, ...)`）。

- デメリット:
  - 既存のクラス API を使用するコード/テストの全面置換が必要。
  - 将来エフェクト固有の状態管理が必要になった場合、関数型では表現工夫が必要。

- 影響範囲/移行:
  - `effects/base.py` を削除し、関数エフェクトに一本化（実装済み）。
  - エフェクトは `@effect()` 登録の `Geometry -> Geometry` 関数のみ（実装済み）。

---

## 提案 3: パイプラインは 1 本に統合（EffectChain 撤廃）

- 提案概要:
  - `EffectChain` を廃し、`E ... .build() -> Pipeline` だけに統一。
  - `Pipeline.__call__(geometry)` が順次適用するだけの単純モデルに。

- メリット:
  - 表現が一貫し、ドキュメント/学習コストを削減。
  - キャッシュ/最適化の適用位置が明確（Pipeline のみ）。

- デメリット:
  - `E.add(g).xxx().result()` の使用感が変わる（互換層で一時吸収可）。
  - Chain ベースの表現に依存した既存デモ/チュートリアルの更新。

- 影響範囲/移行:
  - `E.add(...).result()` は撤廃し、`E.pipeline...build()(g)` のみを採用（実装済み）。

---

## 提案 4: キャッシュはパイプライン単層のみ

- 提案概要:
  - ステップキャッシュ/チェーンキャッシュ/エフェクト内 LRU をすべて撤廃し、`Pipeline` の `(geometry_hash, pipeline_hash)` のみでキャッシュ。
  - `geometry_hash` は `coords/offsets` のバイト列から安定ハッシュ（GUID 依存をやめる）。

- メリット:
  - キャッシュの挙動が予測可能、デバッグ容易。
  - 設計/実装が簡素化し、バグ発生点が減る。

- デメリット:
  - ステップ単位の再利用最適化は失われる（ただし単純で十分速いケースが多い）。
  - ハッシュ計算コストが入力サイズに依存（大規模データでは注意）。

- 影響範囲/移行:
  - `EffectChain` のキャッシュ削除（実装済み）。`common/cacheable_base.py` の利用は段階的縮小予定。

---

## 提案 5: 命名・型・スケールの一貫化

- 提案概要:
  - 変換 API 名を `translate/scale/rotate` に統一。
  - `common/types.py` に `Vec2/Vec3` 型別名を定義し、API シグネチャで採用。
  - 0–1 正規化パラメータ→実数/整数の写像を `common/param_utils.py` に集約（線形/指数など選択可能）。
  - `noise` の `intensity *= 10` を廃止し、仕様としてレンジ/単位を明記（例: `strength: 0..1` を素直に反映）。

- メリット:
  - 横断的な読みやすさ・推測可能性が向上。
  - UI（MIDI）とコアのスケール変換が共有化され、重複とバグが減る。

- デメリット:
  - 一時的にレンジが変わることで既存アートの見え方が変わる可能性。

- 影響範囲/移行:
  - サンプル/チュートリアルのパラメータ調整、`tests/` の期待値更新。

---

## 提案 6: シリアライズ/検証の簡素化

- 提案概要:
  - `SerializablePipeline` は `[{name: str, params: dict}]` の配列に簡素化。ロード時に未登録名/不正型は即例外（早期失敗）。
  - `shapes/polyhedron.py` の事前計算データを `npz/json` に移行（`pickle` 廃止）。

- メリット:
  - フォーマットが人間可読・差分フレンドリー・安全。
  - エラーの早期検出で運用事故を減らす。

- デメリット:
  - 旧ファイル資産（pickle）の再生成/移行が必要。

- 影響範囲/移行:
  - 変換スクリプトを一時同梱（pickle→npz/json）。ロードは新形式に一本化。

---

## 提案 7: ログ/例外の一貫化（標準 logging）

- 提案概要:
  - ランタイムメッセージ/例外通知の `print` を `logging` に統一。子プロセス例外は親側で整形し、ポリシー（継続/停止）を明示。

- メリット:
  - 本番/開発の切替・粒度制御が標準手段で可能。
  - エラー追跡性が向上。

- デメリット:
  - 初期はログ設定の標準化コスト（フォーマッタ/ハンドラ定義）。

- 影響範囲/移行:
  - `engine/pipeline/worker.py`, `engine/io/controller.py`, `effects/buffer.py` などの出力箇所置換。

---

## 提案 8: ディレクトリ構成のシンプル化（案）

- 提案概要:
  - `api/`: `pipeline.py`（E ビルダー＋Pipeline）, `__init__.py`（E, G, Geometry 再エクスポート）
  - `engine/core/`: `geometry.py`（統合）, `transform_utils.py`, `render_window.py`, ...
  - `effects/`: 関数実装＋`registry.py`（クラス撤廃）
  - `shapes/`: 生成器（関数/最小限クラス）＋`registry.py`
  - `common/`: `types.py`, `param_utils.py`, `base_registry.py`

- メリット:
  - レイヤの責務が構造に反映され、探索性が高い。

- デメリット:
  - 既存ファイル移動に伴うインポートパス修正が広範囲。

- 影響範囲/移行:
  - リネーム/移動は最後に実行し、段階移行の最終フェーズで適用。

---

## 段階的移行ロードマップ（推奨順）
1) Geometry 統合（`GeometryAPI`/`GeometryData` の置換）
2) エフェクトを関数化（旧クラス→新関数ブリッジ、一時互換）
3) Pipeline 導入と EffectChain 撤廃（`E.add(...).result()` は暫定互換で中継）
4) キャッシュ単層化（Pipeline のみ、GUID 依存排除）
5) 命名/型/0–1 写像の共通化（`types.py`/`param_utils.py`）
6) シリアライズ/検証の簡素化（pickle 廃止、未知名は即例外）
7) ログ/例外の logging 統一
8) ディレクトリ再編（最終段階）

---

## リスクと緩和策
- 大規模な破壊的変更により短期的にテストが赤くなる → フェーズごとに PR を分割し、互換ブリッジで段差を小さくする。
- パラメータレンジの変更で描画結果が変わる → サンプル/スクリーンショットを更新し、CHANGELOG に「見た目が変わる」旨を明記。
- キャッシュ戦略の変更で性能特性が変わる → ベンチマークを併走し、必要ならパイプライン鍵生成を調整（例: 近似ハッシュ/サマリーハッシュ）。

---

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
def noise(g: Geometry, *, intensity=0.5, frequency=(0.5,0.5,0.5), time=0.0) -> Geometry:
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

- new_tests に追加/更新:
  - `new_tests/test_unified_geometry.py`: Geometry 基本 API とノイズ関数の統合テスト（関数版に更新）。
  - `new_tests/test_transform_utils_geometry.py`: transform_utils が Geometry を受け取り Geometry を返すことを確認。
  - `new_tests/test_translation_geometry_path.py`: `translation(g, ...) -> Geometry` の確認。
  - `new_tests/test_effects_functions_fill_array.py`: `filling/array` が Geometry を返し、線数/重複数が期待通り変化することを確認。
  - `new_tests/run_unified_geometry_check.py`: pytest 不要の最小ランナー（依存が乏しい環境向け）。

- サニティ実行（`new_main.py`）:
  - 追加: `new_main.py`（`noise -> filling -> array` の関数パイプライン確認用）。
  - 実行例（ユーザー環境の実測）:
    - `[base] points=5, lines=1, bbox=(-30.0, -30.0, 0.0)->(30.0, 30.0, 0.0)`
    - `[out1] points=220, lines=104, bbox≈(-30.38, -30.03, -0.27)->(330.02, 30.20, 0.35)`
    - `[out2] points=220, lines=104, bboxは out1 と同一（キャッシュヒット）`

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
  - `shapes_grid_cc.py`: `.size/.at` → `.scale/.translate`、回転は `E.pipeline.rotation(...).build()(g)` に変更。
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

- 新規テストと手動確認
  - `new_tests/` に Geometry と関数エフェクトの検証を追加（サンドボックスでは実行せず、ローカルで pytest/簡易ランナー可）。
  - `new_main.py` によるパイプライン sanity（キャッシュヒット確認済み）。

---

## 再開時の TODO（次にやること）

- 旧表記/旧APIの残滓の整理（ドキュメント）
  - [ ] `AGENTS.md` の GeometryAPI/EffectChain 記述を更新。
  - [ ] README/チュートリアルの細部（用語・コメント）を新API前提で統一。

- テストスイートの刷新
  - [ ] 旧API依存のテキスト/コメントの掃除。必要に応じて追加ケースを拡充。

- 型/設計の仕上げ
  - [ ] `shapes/` 戻り値注釈の再点検（`Geometry` で統一済みだが表記揺れを解消）。
  - [ ] 可能なら `common/cacheable_base.py` 依存の段階的縮小（過剰なキャッシュ層の廃止）。

- 命名/型/0–1 写像の共通化（提案5の仕上げ）
  - [ ] `common/param_utils.py` の横断適用（全エフェクトで一貫仕様へ）。

- シリアライズ/検証（提案6）
  - [ ] `[{name, params}]` 形式のシリアライザ/バリデータ追加。未知名・不正型の早期失敗。
  - [ ] 旧 pickle 資産 → json/npz 変換スクリプトの同梱。

- ログ/例外（提案7）
  - [ ] `print` を `logging` に統一（特に engine/io, pipeline/worker）。

- ディレクトリ再編（提案8）
  - [ ] 最終フェーズで物理移動・インポート整理。

- ベンチ/最適化
  - [ ] ベンチマークを `E.pipeline` 前提に整理し、性能特性を再計測。
  - [ ] `geometry_hash` の最適化（必要なら近似/要約ハッシュを追加検討）。

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
