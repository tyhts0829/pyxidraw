# クラス設計レビュー（美しさ） — 2025-09-04

本ドキュメントは、2025-09-03 の破壊的変更方針（Geometry 統一 / 関数エフェクト / E.pipeline / 仕様のシリアライズと検証）に準拠した現行コードを対象に、クラス設計の「美しさ」を観点別に評価し、改善余地を提案します。

## 総評

- 変換を純関数化した単一 `Geometry` と、関数ベースのエフェクト + 最小限のクラスで支える実行系（`Pipeline`・`Tickable` 群）の組み合わせが、たいへん端正です。
- クラスは「状態と責務の境界」を明確化する箇所にのみ配置（ダブルバッファ、ワーカ、レンダラ等）。一方で、拡張性が要る領域（エフェクト）は関数化し、クラス階層の重さを避けています。
- `@dataclass(slots=True)` や `Protocol` の活用、`frozen=True` なメッセージ型など、意図がコードに直裁に表れ、読み手に負担をかけません。

## 美しい点（賛辞）

1) Geometry（engine/core/geometry.py）
- 単一型の統一: 旧型を統合し、`coords (N,3)` と `offsets (M+1,)` の不変条件が明瞭。
- 純関数変換: `translate/scale/rotate/concat` がすべて新インスタンスを返し、副作用がありません。
- ダイジェスト設計: キャッシュキーとしての `digest` を環境変数で有効/無効制御でき、性能と再現性のトレードオフが素直。
- 工場メソッド: `from_lines()` が多様な入力形状を正規化し、表現の自由度と一貫性を両立。

2) パイプライン最小核（api/pipeline.py）
- 明瞭な構成: `Step(dataclass)` + `Pipeline` + `PipelineBuilder`。責務が小さく可読性が高い。
- 単層キャッシュ: 入力ジオメトリのハッシュ × パイプライン定義のハッシュという単純で十分強力な鍵設計。
- 仕様の往復変換: `to_spec/from_spec/validate_spec` により、エディタ/外部UI/テストの往来が快適。
- 署名照合: エフェクト関数のシグネチャに基づく早期検証は、実行前に「美しく失敗」させる工夫として秀逸。

3) Tickable と小さなクラス群
- `Tickable(Protocol)` によるインターフェイス表明が軽量で読みやすい。
- `RenderPacket` / `RenderTask` は `@dataclass(frozen=True, slots=True)` でメッセージ性を明確化。
- `FrameClock`・`SwapBuffer`・`StreamReceiver`・`WorkerPool`・`LineRenderer` の分割は単一責務原則に忠実で、依存の向きも素直。

4) レジストリとファクトリ
- `BaseRegistry` のキー正規化（Camel→snake、小文字化）が Shapes/Effects で一貫。
- `@shape` デコレータ + `ShapeFactory(G)` の動的ディスパッチは実用上の快適さと見通しを両立（`G.sphere(...)` など）。
- 形状生成は `BaseShape.generate()` による明快な抽象化で、生成ロジックが局所化。

5) パフォーマンス配慮の明示
- `slots=True` の徹底、Numba 化（例: `effects/noise.py`）、必要十分なバッファ設計などが設計としても美しい。

## 改善余地（美しさをさらに磨く）

- Dataclass の時刻既定値: `engine/pipeline/packet.py` の `RenderPacket.timestamp: float = time.time()` はインポート時に固定されます。`field(default_factory=time.time)` への変更が安全で意図に合致。
- レジストリの統一感: `effects/registry.py` はモジュール内 dict、`shapes/registry.py` は `BaseRegistry`。実装差は最小ですが、後者へ統一すると概念的負荷がさらに下がります（メソッド群・無効化・検査を流用可）。
- 未使用メタクラスの整理: `common/meta_factory.py` の `UnifiedFactoryMeta` 系は現状参照が見当たりません。削除または「外部向け拡張ポイント」として明記すると設計の輪郭が締まります。
- 変換 API の統一的語彙: `Geometry.scale(..., center=...)` と `effects.rotate(pivot=...)` で `center/pivot` が混在。どちらかに寄せる（例: `center` に統一し alias を短期提供）と認知負荷が低減。
- キャッシュ層の整合: 形状生成は `G._cached_shape(lru_cache)` で十分に見えます。`BaseShape` 側のキャッシュ（既定は無効）については、利用局面を README で明確化 or 実運用で不要なら段階的縮退も検討余地。
- 旧ベクトル/行列型の見直し: `data/regular_polyhedron/regular_polyhedron.py` の `Vector3/Matrix33` が `numbers.Integral` を継承しており抽象契約と不整合です。`@dataclass` + `numpy` ベースか Protocol 化に改めると設計面でも整います。
- Pipeline の厳格モード: `validate_spec` は外部仕様に強いですが、ビルド時オプション（例: `E.pipeline.strict(True)`）で未知パラメータを早期拒否できると、実行経路がより自己記述的に。
- API 可視性の補助: `G.__getattr__` の動的ディスパッチは実用的ですが、`G.__dir__` で登録済みシェイプを返す／型スタブを用意するなど、IDE 体験の美しさもさらに向上します。

## 快さを支える小さな決定（Good Patterns）

- `__repr__` が有用（`Pipeline(...)`）。デバッグ・チューニング時に構造が一目で把握可能。
- ハッシュの考慮が局所化（`_geometry_hash` / `_params_digest` / `_fn_version`）。衝突確率に対するバランス感覚がよい。
- 例外の文脈化（`WorkerTaskError(frame_id=...)`）が原因追跡を助ける。
- 環境変数で振る舞いを切り替え（ダイジェスト・キャッシュ）。探索的開発に優しい。

## まとめ

本プロジェクトのクラス設計は「必要最小限のクラスに責務を集約し、その他は関数で軽やかに拡張する」という美しく現代的なバランスにあります。上記の軽微な磨き込み（特に dataclass 既定値、レジストリ統一、用語一貫性）を加えると、読みやすさ・可搬性・IDE 体験がさらに向上し、設計としての端正さが一段と際立つでしょう。

## 改善チェックリスト（抜け漏れ防止）

- [ ] Dataclass 既定値の是正（RenderPacket.timestamp）
  - [ ] `engine/pipeline/packet.py` の `timestamp: float = time.time()` を `field(default_factory=time.time)` に変更
  - [ ] `from dataclasses import field` を追記し、既存 import と整合
  - [ ] 生成タイミング差が反映されることを確認するテストを追加（例: `tests/test_pipeline_serialization.py` 近傍に新規）

- [ ] レジストリ実装の統一（effects を BaseRegistry 化）
  - [ ] `effects/registry.py` の内部 dict ベースを `common.base_registry.BaseRegistry` に置換
  - [ ] `@effect` デコレータを `BaseRegistry.register` に委譲（キー正規化ポリシーを共有）
  - [ ] `get_effect/list_effects/clear_registry` をレジストリへ委譲し API を維持
  - [ ] `api/pipeline.validate_spec` と相互作用を確認（未知名・署名検証が従来通り機能）
  - [ ] ドキュメント更新（`docs/guides/effects_authoring.md`、`AGENTS.md`）

- [ ] 未使用メタクラス群の整理（`common/meta_factory.py`）
  - [ ] `rg "UnifiedFactoryMeta|ShapeFactoryMeta|EffectFactoryMeta"` で実使用を確認
  - [ ] 未使用なら削除、もしくは「拡張ポイント」として用途/サンプルを `docs/` に明記
  - [ ] `common/__init__.py` の `__all__` を更新し、影響テストを実行

- [ ] 変換 API 語彙の統一（center/pivot）
  - [ ] 公式語彙を「center」に統一（要合意）
  - [ ] `effects/rotation.rotate` の `pivot` を `center` に改名、`pivot` は非推奨エイリアスとして受理＋警告
  - [ ] `rotate.__param_meta__` を更新（`center`）
  - [ ] 使用箇所一括置換（`rg "pivot=|\bpivot\b"`）とチュートリアル/README/AGENTS の修正

- [ ] キャッシュ層の整合（G と BaseShape）
  - [ ] `BaseShape` のキャッシュ利用実態を調査（`enable_cache=True` の使用有無）
  - [ ] 方針決定：`G` の LRU を一次キャッシュとし、`BaseShape` のキャッシュは原則無効 or 段階的廃止
  - [ ] README に運用方針と環境変数（`PXD_CACHE_DISABLED/PXD_CACHE_MAXSIZE`）の位置づけを明記
  - [ ] ベンチマークでキャッシュ構成差の影響を確認（`python -m benchmarks run`）

- [ ] 旧 Vector/Matrix 実装の見直し
  - [ ] `data/regular_polyhedron/regular_polyhedron.py` の `Vector3/Matrix33` が `numbers.Integral` 継承なのを是正
  - [ ] 選択肢A: `@dataclass` + `numpy` ベースへ置換（演算は関数/メソッドで提供）
  - [ ] 選択肢B: ランタイム依存を断ち「データ供給専用」に限定、演算は共通ユーティリティへ移管
  - [ ] 影響テスト追加（正多面体生成があれば回帰確認）

- [ ] Pipeline の厳格モード導入
  - [ ] `PipelineBuilder.strict(True)` または `build(strict=True)` を追加し、ビルド時にパラメータ名を事前検証
  - [ ] `validate_spec` と重複せず補完する責務に限定（未知キー/型を実行前に検出）
  - [ ] 単体テスト追加（未知キーの早期失敗、`**kwargs` 許容関数の挙動差）
  - [ ] ドキュメント更新（使用例と失敗例）

- [ ] API 可視性の向上（IDE 補助）
  - [ ] `G.__dir__` を実装して登録済みシェイプを列挙
  - [ ] `api/__init__.py` に `list_effects` の再エクスポートを追加（`from effects.registry import list_effects`）
  - [ ] 型スタブ（`api/__init__.pyi` など）で主要ファクトリの補完性を改善
  - [ ] チュートリアルに補完の活用法を追記

### 却下した項目（2025-09-04 判定）

- 却下: レジストリ実装の統一（effects を BaseRegistry 化）
  - 理由: 現状でも `BaseRegistry._normalize_key` を共有しており実害が小。移行コスト（テスト更新・回帰リスク）の割に概念的利得が限定的。

- 却下: 変換 API 語彙の統一（center/pivot）
  - 理由: `validate_spec`／既存 spec／チュートリアル／ユーザコードへの影響が大きく、エイリアス＋警告管理で API 表面が肥大化。命名一貫性の利得は相対的に小。

- 却下: Pipeline の厳格モード導入
  - 理由: `validate_spec` が既に事前検証を担っており重複。例外の発生点が増え学習コストと実装複雑性が上がる。

- 却下: 旧 Vector/Matrix 実装の全面見直し
  - 理由: 影響範囲と工数が大。現状は主にデータ供給用途で致命的課題は見当たらない。必要性が顕在化した段階で計画的に実施するのが妥当。

- 部分却下: API 可視性の向上のうち「型スタブ（.pyi）追加」
  - 理由: 維持コストが相対的に高い。代替として `__dir__` 実装と `list_effects` 再エクスポートで実用的な補完を確保。
