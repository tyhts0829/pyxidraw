# AGENTS.md と実装の整合性監査レポート（2025-10-07）

本レポートは、リポジトリ直下の AGENTS.md（以下「AGENTS」）とコード実装の整合性を、リポジトリ全体を対象に確認した結果の要約である。配下ディレクトリにも複数の AGENTS.md が存在するが（src/engine/ 等）、本監査では「ルート AGENTS を基準」とし、下位 AGENTS は補足/詳細として扱った。

## 結論（概要）
- 主要な設計方針・ビルド/テスト手順・公開 API 位置・GUI/パラメータ解決・量子化仕様・スタブ生成/CI は AGENTS と実装が概ね一致。
- 重要な不一致は 1 点のみ検出（`quantize_params` の関数内 docstring の既定ステップ記述）。実装自体は AGENTS と一致している。
- 実装に存在し AGENTS に未記載の拡張が一部あり（情報共有の観点で AGENTS 追記候補）。

---

## 一致確認（主な項目）

- Build/Test 環境/手順
  - Python 3.10 指定: `pyproject.toml:13` の `requires-python = ">=3.10"` を確認。
  - 開発 extras（ruff/black/isort/mypy/pytest 等）: `pyproject.toml:22` 以降に `dev` 定義。
  - 実行エントリ: `main.py:1` が存在し、`src` をパス追加して `from api import ...` を解決。
- スタブ生成/同期検証
  - 生成スクリプト: `tools/gen_g_stubs.py:1`。日本語 docstring と堅牢な生成ロジックを確認。
  - 生成物: `src/api/__init__.pyi:1`。`G/E/cc/lfo` 型面を Protocol で定義。
  - 同期テスト: `tests/stubs/test_g_stub_sync.py:1` と `tests/stubs/test_pipeline_stub_sync.py:1` が一致を検証。
  - CI ジョブで差分検出: `.github/workflows/verify-stubs.yml:23`（`git diff --exit-code`）。
- プロジェクトマップ（配置）
  - 公開 API: `src/api/__init__.py:1`, `src/api/__init__.pyi:1`。
  - 形状/エフェクト登録: `src/shapes/registry.py:1`, `src/effects/registry.py:1`。
  - 中核ジオメトリ: `src/engine/core/geometry.py:1`（日本語ヘッダあり）。
  - レンダリング/ランタイム/共通ユーティリティ: `src/engine/render/`, `src/engine/runtime/`, `src/common/`, `src/util/` を確認。
  - 設定: ルート `config.yaml:1` と `configs/default.yaml:1` を確認。
  - CI: `.github/workflows/verify-stubs.yml:1` を確認。
- Parameter GUI / cc 仕様
  - `cc` は `api.cc` のグローバル辞書: `src/api/cc.py:1`（未定義は 0.0, 0..1 float）。
  - GUI 表示は「既定値採用の引数のみ」: `src/engine/ui/parameters/value_resolver.py:71` 以降（`source == "default"` のみ Descriptor 登録）。
  - RangeHint は `__param_meta__` がある場合のみ使用／無い場合は 0–1 既定レンジ: `src/engine/ui/parameters/value_resolver.py:127`（meta→RangeHint）、`src/engine/ui/parameters/dpg_window.py:154`（fallback に `ParameterLayoutConfig.derive_range()` を使用）。ベクトルも同様の既定あり。
  - 優先順位「明示引数 > GUI > 既定値」: `value_resolver` が provided 値は登録せず原値を返す実装（`_resolve_scalar/_resolve_vector`）。
  - Runtime は `set_inputs(t)` のみを扱い cc は関知しない: `src/engine/ui/parameters/runtime.py:44`（コメントと実装）。
  - macOS における DPG 駆動: `src/engine/ui/parameters/dpg_window.py:74`（`pyglet.clock.schedule_interval` を優先、未導入時はバックグラウンドスレッド）。
- 量子化/署名（キャッシュ鍵）
  - 実装の要点（AGENTS と一致）
    - 「float のみ量子化、int/bool はそのまま」: `src/common/param_utils.py:134`（`_quantize_scalar`）。
    - 既定ステップ 1e-6（環境変数 `PXD_PIPELINE_QUANT_STEP` で上書き）: `src/common/param_utils.py:124`（`_env_quant_step`）。
    - ベクトルは成分ごと。`step` が不足する場合は末尾値で補完: `src/common/param_utils.py:165`（`steps[idx if idx < len(steps) else -1]`）。
    - Effects は量子化後の値が実行引数にも渡る: `src/api/effects.py:256`→`_params_signature` を実行→CompiledPipeline へ渡す。
    - Shapes は鍵のみ量子化（実行はランタイム解決値）: `src/api/shapes.py:188`（鍵は `_params_signature`、実実行は `_generate_shape_resolved`）。
- テスト/CI ルール
  - マーカー群: `pytest.ini:13` に `smoke/integration/e2e/perf/optional` を定義。
  - CI DoD 相当: ruff/black/isort/mypy/pytest 実行を `.github/workflows/verify-stubs.yml:31` 以降で確認。
- ドキュメンテーション/型
  - 主要公開 API/中核モジュールに日本語ヘッダと NumPy スタイル節（Parameters/Returns）を確認（例: `src/api/effects.py:1`, `src/engine/core/geometry.py:1`）。

---

## 不一致（要修正）

1) 量子化の既定ステップ値（docstring のみ不一致）
- 期待（AGENTS）: 未指定時の既定は 1e-6（`PXD_PIPELINE_QUANT_STEP` で上書き可）。
- 実装: コードは 1e-6 を使用（`src/common/param_utils.py:124`）。
- 不一致箇所: `src/common/param_utils.py:149` の docstring に「無ければ 1e-3」と記載。
- 影響: 実装の挙動は正しいが、読者に誤解を与えるためドキュメント修正推奨。
- 提案対応: 上記 docstring を 1e-6 に修正（テスト/挙動影響なし）。

---

## 仕様に未記載（実装に存在：追記候補）

- Compiled Pipeline のグローバルキャッシュ上限（環境変数）
  - 実装: `src/api/effects.py:63` 付近で `PXD_COMPILED_CACHE_MAXSIZE` を受け付け、ビルダー/パイプライン共有キャッシュの上限を制御。
  - AGENTS: 明記なし。運用上のチューニング・トラブルシュートの助けになるため、AGENTS の「キャッシュ設計」節に追記候補。
- Parameter GUI の override 永続化
  - 実装: `src/engine/ui/parameters/persistence.py:1`（保存/復元、保存先 `data/gui/*.json`、float の保存時量子化に 1e-6 既定）。
  - AGENTS: GUI の表示仕様・優先順位は記載ありだが、永続化についての明確な記述は無し。運用ルールとして記載候補。

---

## 注記（網羅困難な事項と見解）

- 「すべての公開 API に NumPy スタイル docstring + 型ヒント」は広範囲のため、本監査では代表的な公開面（api.*, engine/core, registry, 主要 effects/shapes）を抽出確認。大勢は一致だが、全関数レベルの網羅は未実施。
- 「architecture.md は実装と同期」については高粒度の方針が一致していることを確認（例: 量子化仕様は 1e-6）。ただし全文差分検証は範囲外。

---

## まとめ/次アクション（提案）

- 速攻修正（ドキュメントのみ）
  - `src/common/param_utils.py:149` の docstring を 1e-6 に修正。
- AGENTS 追記候補（任意）
  - `PXD_COMPILED_CACHE_MAXSIZE`（compiled pipeline キャッシュ上限）
  - Parameter GUI の override 永続化仕様（保存先/量子化/復元条件）

以上。

