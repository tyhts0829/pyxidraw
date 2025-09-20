"""
どこで: リポジトリ全体のコードレビュー（全体観・運用観点）
何を: 設計適合/責務分離/API/ランタイム/テスト/依存/ドキュメントを俯瞰し、良い点と改善提案を整理
なぜ: 現状の完成度を把握し、次の小さな改善サイクルの着地点を明確にするため
"""

# コードレビュー（全体）

## 要約
- 設計一貫性が高く、公開 API は薄く、層の責務分離が明確。統一幾何 `Geometry` と純関数エフェクト、厳格なパイプライン検証/キャッシュは堅牢。
- UI パラメータ（0..1 正規化→実レンジ）/レジストリ/スタブ自動生成がよく統合され、テスト/CI も実用的。
- 改善は主に実行依存の明示（moderngl/pyglet）、パッケージ名整合（v6 表記と一致）、任意依存の遅延 import の徹底、doc の数行補強。

## 良い点（ハイライト）
- 層と依存方向の明示と検証（アーキテクチャテスト）。tests/test_architecture.py:1
- 公開 API の薄い入口と遅延 import（sketch ランナー）。src/api/__init__.py:1, src/api/sketch.py:232
- 統一幾何と純関数変換。digest による同一性・キャッシュ連携が明快。src/engine/core/geometry.py:1
- エフェクト直列の `Pipeline` と厳格検証・単層 LRU 風キャッシュ。src/api/effects.py:160
- 形状/エフェクトのレジストリ正規化・拡張性。src/common/base_registry.py:1
- スタブ自動生成と同期テスト（公開面の契約維持）。tools/gen_g_stubs.py:1, tests/stubs/test_g_stub_sync.py:1
- 並行実行（WorkerPool）とダブルバッファの結線がシンプル。src/engine/runtime/worker.py:1, src/engine/runtime/buffer.py:1

## 主要所見と提案

### パッケージング/依存
- 実行依存（moderngl/pyglet）の明示が不足。`python main.py` 実行には必要なため、`[optional]` または新設 `[runtime]` へ追加推奨。README の導入手順にも明記。README.md:23
- プロジェクト名の整合: `pyproject.toml` の `name = "pyxidraw5"` と API/リポの v6 系が乖離。`pyxidraw6` 等へ改名を検討（破壊的変更可の方針）。pyproject.toml:8

### API/パイプライン
- `E.pipeline` の `.cache(maxsize=...)` は丁寧に設計。メモリ運用の注意（大規模ジオメトリ時は上限設定を推奨）を README/API doc へ一言追補すると親切。src/api/effects.py:240
- 「UI 経由は 0..1 正規化、関数直呼は実レンジ」の説明を `api/shapes.py`/`api/effects.py` の docstring 冒頭へ明示（既に言及ありだが導入部で強調）。src/api/shapes.py:1, src/api/effects.py:1

### Geometry/レンダリング
- 1 ドロー（LINE_STRIP + Primitive Restart）の方針は適切。シェーダは最小で可読。環境互換性重視の将来案として、GS なし版を ADR にメモしておくと選択肢提示として有益。src/engine/render/renderer.py:84, src/engine/render/shader.py:1

### ランタイム/IO/MIDI
- `api.sketch` の MIDI 初期化は厳格/非厳格で妥当。src/api/sketch.py:154
- 一貫性のため `engine/io/controller.py` も `mido` を遅延 import（現在はトップレベル import）。`manager.py` と同様に使用箇所で import ガード化を推奨。src/engine/io/controller.py:20, src/engine/io/manager.py:1

### UI/Parameters
- 0..1 正規化→RangeHint→実レンジの流れが `ValueResolver` に集約されており良好。導入の「最短ルート」を `engine/ui/parameters/AGENTS.md` に既に追記済みでわかりやすい。src/engine/ui/parameters/value_resolver.py:74, src/engine/ui/parameters/AGENTS.md:1

### テスト/CI
- スタブ同期とアーキテクチャ検査が CI に組み込まれ、破壊的変更の検知が可能。`.github/workflows/verify-stubs.yml`:1
- optional マーカーの分離とダミー依存注入（tools/dummy_deps.py）で CI 耐性が高い。tools/dummy_deps.py:1

## リスク/優先度（上→中→下）
1) 実行依存の明示不足（moderngl/pyglet）
   - 影響: 初回実行体験の躓き、README と実挙動の乖離
   - 対処: optional へ追加 or runtime セット新設、README 追記
2) パッケージ名の不整合（v5 名称のまま）
   - 影響: インストール/配布時の混乱
   - 対処: `pyxidraw6` 等へ改名
3) 依存遅延 import の一貫性（mido）
   - 影響: サブモジュール単体 import 時の例外（軽微）
   - 対処: `controller.py` の遅延化
4) ドキュメント補強（数行）
   - 影響: 規約理解の速度（小）
   - 対処: API doc/README に一言追記

## クイックウィン（軽作業で効果がある項目）
- optional 依存へ `moderngl`, `pyglet` を追加し README に明記（手順 2–3 行）。pyproject.toml:1, README.md:23
- `pyproject.toml` の `name` を v6 系へ整合。
- `engine/io/controller.py` の `mido` を関数内遅延 import + 分かりやすい例外へ差し替え。
- `api/shapes.py`/`api/effects.py` docstring に「正規化/実レンジ」の注意を冒頭へ再掲。

## 参考（主要ファイル）
- API 入口とバージョン: `src/api/__init__.py`:40
- sketch（遅延 import・MIDI 初期化）: `src/api/sketch.py`:232, `src/api/sketch.py`:154
- Geometry 中核: `src/engine/core/geometry.py`:1
- Pipeline/Builder/strict: `src/api/effects.py`:160
- レジストリ基盤: `src/common/base_registry.py`:1
- ランタイム/バッファ/ワーカ: `src/engine/runtime/buffer.py`:1, `src/engine/runtime/worker.py`:1
- レンダリング（1 ドロー方針）: `src/engine/render/renderer.py`:84
- アーキテクチャテスト: `tests/test_architecture.py`:1
- CI（スタブ同期など）: `.github/workflows/verify-stubs.yml`:1
- パッケージ名: `pyproject.toml`:8

## 次アクション（任意・要確認）
- 上記クイックウィンの反映をご希望なら、個別 PR（小粒）として順に対応可能です。
  - 例: 「optional に moderngl/pyglet を追加＋README 追記」→ 「package 名整合」→ 「mido 遅延 import 化」→ 「docstring 追記」。

