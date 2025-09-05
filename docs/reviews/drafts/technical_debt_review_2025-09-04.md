# 技術的負債レビュー（2025-09-04）

対象: PyxiDraw4（現行ブランチ、破壊的変更適用後の構成）

## 概要（結論）
- 破壊的変更ガイド（Geometry 統一／関数エフェクト／E.pipeline／spec 検証）は概ね順守。`api/pipeline.py` と `effects/registry.py` は簡潔でテスタブル。
- 一方で、依存関係の明示不足、パイプラインキャッシュの無制限成長、`**kwargs` に起因する検証の緩さ、未テストのエフェクト群など、拡張時に効いてくる負債が点在。
- まずは「ドキュメント整備」「LRU 導入」「テストの穴埋め」「例外の粒度改善」を小さく速く進めるのが効果的。

---

## 高優先度（直近のリスク/影響が大）

### 1) 依存関係ドリフト（README 未記載）
- 観測: SciPy（`shapes/asemic_glyph.py`）、Shapely（`effects/buffer.py`）、`arc`（`README.md`/`engine/io/controller.py`）が使用されるが、README の最小インストール例に未記載。
- 影響: 初回セットアップ失敗／環境差による再現性低下。
- 提案: README を「必須 / 機能別オプション / 開発補助」に三分割。`shapes`（SciPy）, `buffer`（Shapely）, `midi`（mido/arc）などの“機能別”で明示。

### 2) パイプライン単層キャッシュの無制限成長 + ハッシュ計算コスト
- 観測: `api/pipeline.Pipeline` の `_cache: Dict[Tuple[bytes, bytes], Geometry]` は LRU なし。`_geometry_hash` は毎回ジオメトリ全体を blake2b。
- 影響: 長時間セッションでメモリ膨張／CPU 帯域消費。
- 提案: `Pipeline(..., cache_maxsize: int|None=None)` を追加し LRU 化（`functools.lru_cache` 相当の薄ラッパ）。将来案として `Geometry` に任意 `digest` を保持し、変換時に更新（再ハッシュ局所化）。

### 3) 仕様検証（validate_spec）の緩さ
- 観測: 安定エフェクトでも `**_params` を受ける実装があり（例: `effects/trimming.py`）、未知キー検出が効かない場面がある。
- 影響: スペック誤記が実行時まで潜伏。
- 提案: 安定エフェクトから順次 `**kwargs` を撤廃し、`__param_meta__` を補強。ガイド（`docs/guides/effects_authoring.md`）に「`**kwargs` 非推奨」を明記。

### 4) 広域 `except Exception` の多用
- 観測: `engine/pipeline/worker.py` ほかで包括捕捉→親へ再送。ベンチ系も同様の箇所あり。
- 影響: 失敗の分類・復旧方針（スキップ/再試行）の分岐が困難。
- 提案: 主要経路は例外型を絞るか、分類キー（effect/shape/stage）をログへ付与。HUD/CLI でのユーザ通知も検討。

---

## 中優先度（保守性/可読性の改善）

### 5) 未テストのエフェクト群
- 観測: `boldify/collapse/dashify/explode/trimming/twist/wave/webify/wobble` が tests から直接参照されていない。
- 提案: 各 1–2 本のスモーク＋境界テスト（恒等近傍、空/極端入力、`to_spec/from_spec` 往復整合）。

### 6) 型安全性のばらつき（`type: ignore` の点在）
- 観測: `moderngl/mido` の型未解決に起因する `type: ignore` が複数。
- 提案: 侵入口をインタフェース層に局所化（`Protocol` を最小定義）。`mypy.ini` で外部ライブラリはモジュール単位許容。

### 7) 変換 API の重複
- 観測: `Geometry` メソッドと `engine/core/transform_utils.py` が並存。
- 提案: 現方針（`transform_combined` 主用途、個別は補助）をドキュメントで明示し選択指針を強化。

### 8) 大型モジュールの段落化
- 観測: `effects/webify.py`, `shapes/asemic_glyph.py` が長大。
- 提案: 「前処理／コア計算／後処理」に分ける私的関数抽出＋先頭にアルゴリズム概要コメント（部分対応済み）。

---

## 低優先度（小リスクの改善・一貫性）

### 9) 未実装 API の残置
- 観測: `api/shape_registry.unregister_shape()` が `pass`。`common/base_registry.py` には `unregister` 実装済み。
- 提案: `shapes.registry.get_registry().unregister(name)` を呼ぶ実装に置換。

### 10) 命名/用語の揺れ
- 観測: `pivot/center` の併存表記が残存。角度は `angles_rad` のみが正。
- 提案: README など公開ドキュメントは `pivot`/`angles_rad` のみを使用し、旧名（`center`/`angles_deg`/`rotate(0..1)`）は移行ガイド内の注記に限定する。

### 11) スクリプトの `print` ログ
- 観測: `scripts/convert_*` 系などに `print` ベースのログが残存。
- 提案: `logging` へ統一（INFO/ERROR、`--verbose`）。

---

## 改善アクション（チェックリスト）

- [x] README の「インストール」章を再編（必須/オプション/開発）。SciPy/Shapely/mido/arc を機能別に明示する。
- [x] `docs/guides/effects_authoring.md` に「`**kwargs` 非推奨」「`__param_meta__` 推奨」を追記する。
- [x] `api/pipeline.Pipeline` に `cache_maxsize` を追加し LRU 化する（`clear_cache()` も提供）。
- [x] 将来最適化: `Geometry` に任意 `digest` を持たせる（遅延計算のプロパティ）。`api.pipeline` は `g.digest` を優先使用。
- [x] `translate/scale/rotate/concat/from_lines` 実行時に `digest` を再計算して保持（変換時更新）。
- [x] 未テストの 9 エフェクト（boldify/collapse/dashify/explode/trimming/twist/wave/webify/wobble）へスモーク/境界テストを追加する。
- [x] `effects/*` の安定関数で `**kwargs` を撤廃し、`validate_spec()` が未知キーを検知できる状態にする。（対応: trim/filling/dashify/wobble/collapse/extrude/buffer/webify/transform/noise/subdivision）
- [x] `engine/pipeline/worker.py` 等の包括捕捉を見直し、例外に分類情報（effect/shape/stage）を付与してログ出力する。
- [x] `api/shape_registry.unregister_shape()` を実装（`shapes.registry.get_registry().unregister(name)` を呼ぶ）。
- [x] `scripts/convert_*` 系の `print` を `logging` に置換。`--verbose` でレベル切り替えを可能にする。（convert_polyhedron_* 対応済み。convert_pickle_assets は既に logging ベース）
- [x] boldify の numba 依存を排し、2 パス事前確保で安定化（スモークテスト通過）。
- [x] Geometry digest の計測ガードとベンチスクリプトを追加（`PXD_DISABLE_GEOMETRY_DIGEST=1` 環境変数、`scripts/bench_geometry_digest.py`）。
- [x] README の命名指針を更新し、推奨表記（`pivot`, `angles_rad`）を強調する。

追加タスク（未対応）

- [ ] 型安全性の強化: `mido`/`moderngl` 等の外部依存を `Protocol` で局所化し、`type: ignore` を削減。`mypy.ini`（または pyproject）での設定追加を含む。
- [ ] 変換 API の選択指針を明文化: README か guides に「Geometry メソッド vs `transform_utils.transform_combined` の使い分け」を短章で追記。
- [ ] 大型モジュールの段落化: `effects/webify.py` と `shapes/asemic_glyph.py` を「前処理／コア計算／後処理」に分ける私的関数抽出、先頭にアルゴリズム概要コメントを追加。
- [ ] `effects/*` の `__param_meta__` を拡充: 境界/範囲/choices をなるべく網羅（例: `repeat.scale` の範囲、`offset.join` の choices はOKだが他も明示）。
- [ ] README にヘッドレス実行の最小例を追加し、`arc` なしでも動くことを明示（`run(..., use_midi=False)` のサンプルコード）。

---

## 参考（主な観測箇所）
- パイプライン/検証: `api/pipeline.py`
- エフェクト登録: `effects/registry.py`
- 依存の重い実装: `effects/buffer.py`（Shapely）, `shapes/asemic_glyph.py`（SciPy）
- GPU 経路: `engine/render/renderer.py`, `engine/render/line_mesh.py`
- MIDI/Runner: `api/runner.py`, `engine/io/*`
- 変換: `engine/core/geometry.py`, `engine/core/transform_utils.py`

---

## 補足
小さく速い変更（README/未実装 API/スモークテスト）から着手することで、開発者体験と品質の双方を早期に底上げできます。LRU 化は影響が読みやすいため第二フェーズでの導入が適切です。
