# 破壊的変更前提の簡素化提案（2025-09-01）

本ドキュメントは「既存 API 互換を気にせず、コードベースを美しく・シンプルに・読みやすく」するための破壊的変更案をまとめたものです。各提案ごとにメリット・デメリット・影響範囲/移行メモを併記します。

---

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
  - `api/geometry_api.py` と `engine/core/geometry_data.py` を統合し、`engine/core/geometry.py` に集約。
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
  - 暫定ブリッジ: 旧クラスが新関数を内部呼び出し（段階移行時）、最終的にクラス削除。

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
  - 暫定互換: `E.add(g)...result()` は内部で `E...build()(g)` を呼ぶ薄い互換実装に一時対応。

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
  - `EffectChain` のキャッシュ削除、`common/cacheable_base.py` の利用縮小/撤廃。

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

この設計は「型 1・関数・単一パイプライン・単一キャッシュ」という最小構成を核に、読みやすさ・保守性・拡張性のバランスを最適化することを狙います。採用可否や優先順位のフィードバックをいただければ、段階実装の計画に落とし込みます。

