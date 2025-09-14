# Registry / Cache 改善チェックリスト（提案）

本チェックリストは、レジストリ（`common/base_registry.py`・`shapes/registry.py`・`effects/registry.py`）と
キャッシュ（`api/shape_factory.py`・`api/pipeline.py`）周りの「必要十分な堅牢さ」を満たしつつ、
複雑化を避けて読みやすさ/美しさを高めるための具体アクションを短く分解したものです。
変更前にあなたの承認を得て、合意項目から順に小さく適用します。進行中は本ファイルに進捗を逐次追記します。

## 現状サマリ（観測）

- Registry 共通基盤: `BaseRegistry`
  - キー正規化（Camel/Snake/大小・ハイフン → アンダースコア）。重複登録は `ValueError`。
  - `list_all()` は未ソート返却。`registry` プロパティはコピーを返し外部不変を担保。
- Shapes/Eeffects レジストリ
  - `shapes.registry`: デコレータ `@shape`（名前省略可）。型検査は未実施（実質 Any 登録）。
  - `effects.registry`: デコレータ `@effect` は関数のみ許可（`isfunction` で型検査）。`list_effects()` はソート。
- 形状キャッシュ: `api/shape_factory.ShapeFactory`
  - `functools.lru_cache(maxsize=128)` 固定。キー正規化は堅牢（dtype/shape/Blake2b 指紋、集合/バイト列対応）。
  - 同一入力で同一インスタンスを返す（テストも `is` チェック）。
- パイプラインキャッシュ: `api/pipeline.Pipeline`
  - 単層 LRU 風（`OrderedDict`）。キーは `(Geometry.digest, pipeline_key)`。
  - `pipeline_key` は「名前」「関数バイトコード（近似）」「パラメータの `repr()` 正規化 8B ハッシュ」。
  - 既定キャパは `None`（無制限）。`PXD_PIPELINE_CACHE_MAXSIZE` で上書き可。`maxsize=0` で無効。
- 重要な相互作用
  - Geometry の `digest` は有効時に保持（遅延計算）。外部から配列を書き換えると「digest が古いまま」になる懸念。
    パイプラインキャッシュのキー整合性に直結（詳細は下の A1 関連）。

## 合意が必要な確認事項（要回答）

- [YES] A1: `Geometry.as_arrays(copy=False)` を“読み取り専用ビュー”に変更して良いか
  （キャッシュ安全性向上のため。関連: `docs/review/geometry_improvement_checklist.md` の A2）。
- [YES] A2: `shapes.registry` の登録対象を `BaseShape` 派生クラスに型制約して良いか（関数等の誤登録防止）。
- [YES] A3: `list_*()` の並び統一（両レジストリとも「ソートして返す」方針に統一）で良いか。
- [YES] A4: `Pipeline` のパラメータダイジェストを `ShapeFactory` と同等の堅牢正規化（dtype/shape/Blake2b）へ置換して良いか。
- [YES] A5: `Pipeline` 内部 LRU に簡易ロック（`threading.RLock`）を入れて良いか（低コストのスレッド安全性）。
- [YES] A6: `PXD_PIPELINE_CACHE_MAXSIZE` に負値が来た場合の扱いを「0 と同義（無効）」に正規化して良いか。
- [YES] A7: `api/shape_registry.unregister_shape()` の冗長 `try/except` を削除して良いか（`BaseRegistry.unregister` は無例外）。
- [YES] A8: `get_registry()` の返却を「読み取り専用ビュー（`Mapping`）」に狭めて良いか（外部からの破壊防止）。

## タスク一覧（DoD 付き）

### T1: 破壊的変更リスクの事前低減

- [x] T1-1: A1 の可否確定（Geometry 読み取り専用方針）。
  - DoD: 方針合意（YES）→ 実装済み（`Geometry.as_arrays(copy=False)` 読み取り専用ビュー）。

### T2: Registry の型安全性/一貫性

- [x] T2-1: `shapes.registry.shape` で `issubclass(obj, BaseShape)` を検証。
  - DoD: 間違った登録で `TypeError`。既存テスト緑（`tests/shapes/test_registry_and_sphere.py`）。
- [x] T2-2: `list_shapes()` を `sorted(_registry.list_all())` に変更（`effects` と対称）。
  - DoD: `dir(G)` 出力の順序が安定。既存テストへ影響なし（`__dir__` 側で `sorted` 済）。
- [x] T2-3: `api/shape_registry.unregister_shape` の冗長 `try/except` を削除。
  - DoD: 機能等価（未登録時に無例外）。

### T3: Pipeline キャッシュ鍵の堅牢化/安定性

- [x] T3-1: `_params_digest` を `ShapeFactory._params_to_tuple` と同等の変換則へ置換（小型ユーティリティに抽出）。
  - DoD: `np.ndarray` の dtype/shape 差異で衝突しない。集合/bytes/bytearray の扱いが対称。
- [x] T3-2: `PipelineBuilder.cache(maxsize=...)` の入力正規化（負値 →0）。環境変数も同様にクリップ。
  - DoD: `maxsize<0` で「無効」。
- [x] T3-3: LRU 操作部に `RLock` を導入（読み書き周りの最小ガード）。
  - DoD: シングルスレッド性能劣化が測定上無視できる（コメント根拠で可）。

### T4: 公開 API/ドキュメント整流

- [ ] T4-1: `effects.registry`/`shapes.registry` の docstring を NumPy スタイルの日本語で統一（API 対称性・例外条件を明記）。
  - DoD: 行長 ≤100、用語統一、公開 API 完備。
- [ ] T4-2: `architecture.md` の「キャッシュ方針（G=LRU128 / Pipeline=単層 LRU）」を現状実装と完全同期。
  - DoD: 当該セクションに実コード参照（関数/行）を明記。

### T5: 最小限の観測/運用補助（任意）

- [x] T5-1: `Pipeline` に `cache_info()` を追加（サイズ/ヒット回数程度）。
  - DoD: 依存/複雑化を避け、必要最小の情報のみ。

## 追加の気づき（参考・今回スコープ外でも共有）

- `Geometry` の可変配列と LRU の相性
  - 現状は外部書き換えで `digest` が古くなる可能性があり、`Pipeline` キャッシュキーの誤同一化を招き得る。
  - A1（読み取り専用ビュー）を採ると根本的に緩和可能。互換性への影響はテストで吸収可能と判断。
- `BaseRegistry` のキー正規化（ハイフン →`__` 合成）はテスト仕様に沿う（変更不要）。

## 検証（編集ファイル優先の高速ループ）

- Lint/Format/Type（変更ファイルのみ）
  - `ruff check --fix {changed}` / `black {changed} && isort {changed}` / `mypy {changed}`
- Targeted tests（代表）
  - レジストリ: `pytest -q tests/common/test_base_registry.py tests/shapes/test_registry_and_sphere.py`
  - 形状 LRU: `pytest -q tests/api/test_shape_factory.py::test_shape_factory_lru_returns_same_instance`
  - パイプライン LRU: `pytest -q tests/api/test_pipeline_cache.py`
  - 厳格検証: `pytest -q tests/api/test_pipeline_spec_and_strict.py`

---

更新履歴:

- 2025-09-14: 初版（レビュー待ち）
- 2025-09-14: 実装第一弾完了（A1/A2/A3/A4/A5/A6/T5-1）。
