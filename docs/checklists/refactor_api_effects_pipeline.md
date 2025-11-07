# API Effects/Pipeline リファクタ — 実施チェックリスト（提案）

目的
- `src/api/effects.py` の堅牢性・整合性・ドキュメント同期を改善し、利用者 API を分かりやすく保つ。

スコープ
- 対象: `src/api/effects.py`
- 関連: `architecture.md`, `tools/gen_g_stubs.py`, `src/api/__init__.pyi`

前提観測（現状）
- dataclass フィールドの誤用
  - `Pipeline._cache` が `dataclass(init=False, repr=False)` で定義されており、`field()` の誤用。
  - ただし `__post_init__` で `OrderedDict()` を代入しているため、即時の不具合は出にくい。
- `Pipeline.realize()` の挙動非対称
  - `PipelineBuilder.realize()` は `_cache_maxsize` を引き継ぐが、`Pipeline.realize()` は引き継がない。また `_compiled_steps` も失われる。
- ドキュメント/API スタブと実装の不整合
  - `architecture.md` と `src/api/__init__.pyi` に `PipelineBuilder.intermediate_cache(maxsize)` / `label(uid)` が記載。
  - 実装側は `cache(maxsize)` のみ。`intermediate_cache` と `label` が無く、`__getattr__` で誤って「エフェクト名」として解決される恐れ（実行時例外）。
- HUD メトリクスの不足
  - `global_cache_counters()` は `step_hits/step_misses` を 0 で返す。Prefix キャッシュ（`engine.core.lazy_geometry`）の統計と同期していない。

やること（タスク分解）
1) フィールド初期化の是正（安全・最小）
   - [ ] `Pipeline._cache` を `field(default_factory=OrderedDict, init=False, repr=False)` に修正。
   - [ ] 簡潔な docstring 追記（LRU 風の終端キャッシュ）。

2) `Pipeline.realize()` の一貫性確保
   - 方針A（推奨）: 既存設定を引き継ぐ。
     - [ ] `_cache_maxsize` と `_compiled_steps` を引き継いで新しい `Pipeline` を返す。
     - [ ] docstring に「戻り値は実行時に `LazyGeometry` を実体化するパイプライン（副作用なし）」を明記。
   - 方針B（代替）: 現状の非対称仕様を仕様として明記。
     - [ ] docstring/architecture.md を更新し、`Pipeline.realize()` の非対称動作を明文化。
   - [ ] A/B どちらにするか要確認（下段「質問」参照）。

3) `PipelineBuilder` の API 整合
   - `intermediate_cache(maxsize)`（選択肢あり）
     - 選択肢A（実装）: パイプライン単位の prefix-cache 制御を導入。
       - [ ] Builder/Pipeline に `_step_cache_maxsize` を追加し、`LazyGeometry.realize()` の Prefix 保存時キーにパイプライン固有タグを混ぜる／または保存可否を委譲。
       - [ ] 簡易版として「0: 無効、それ以外: 有効（上限はグローバル上限を尊重）」の読み書きフラグに留める実装でも可。
       - [ ] HUD 集計に pipeline 単位の `step_hits/step_misses` を反映（可能なら、まずはグローバル合計のみ）。
     - 選択肢B（仕様同期のみ）: 現実装に合わせて記述削除。
       - [ ] `architecture.md` の該当節（intermediate_cache の説明）を削除/将来計画へ移動。
       - [ ] `tools/gen_g_stubs.py` から当該メソッド生成を削除し、`PYTHONPATH=src python -m tools.gen_g_stubs` で `src/api/__init__.pyi` を再生成。
     - [ ] A/B どちらにするか要確認（下段「質問」参照）。
   - `label(uid: str)`（実装容易）
     - [ ] `PipelineBuilder.label(uid)` を追加し、`self._uid` を明示設定（runtime.before_effect_call に渡す UID の固定化）。
     - [ ] docstring を短く追加（HUD/GUI でのグルーピング・表示に使用）。

4) HUD メトリクスの拡充
   - [ ] `global_cache_counters()` に prefix-cache の統計を反映（`engine.core.lazy_geometry` の `_PREFIX_*` を読み取り、なければ 0）。
   - [ ] `api.sketch_runner.utils.hud_metrics_snapshot()` 側は変更多し（キーは既存 `step_hits/step_misses` を流用）。

5) 付随の軽微改善（任意・簡潔）
   - [ ] `E: Final` の型注釈（IDE 補完の意図明示）。
   - [ ] `_GLOBAL_PIPELINES`/`_GLOBAL_COMPILED` の用途コメント（WeakSet は観測、OrderedDict は再利用）。
   - [ ] `Pipeline`/`PipelineBuilder` のクラス先頭に短い docstring。

6) ドキュメント/スタブの同期
   - [ ] `architecture.md` のパイプライン節を現実装に同期（2/B・3/B を選ぶ場合は記述修正）。
   - [ ] `PYTHONPATH=src python -m tools.gen_g_stubs && git add src/api/__init__.pyi`（必要時のみ）。

影響/互換性
- 破壊的変更
  - 方針B を選ぶ場合は公開 API（`intermediate_cache` の撤回）を伴うため、`__init__.pyi` の更新が生じる。
- 後方互換
  - `label(uid)` の追加は非破壊。
  - `Pipeline.realize()` の引継ぎ強化（方針A）は互換向上に寄与。

検証（編集ファイル限定の高速チェック）
- Lint/Format/Type: `ruff check --fix {changed}`, `black {changed} && isort {changed}`, `mypy {changed}`
- テスト（重点）
  - `pytest -q tests/api/test_pipeline_cache.py`
  - `pytest -q tests/api/test_pipeline_more.py`
  - `pytest -q tests/stubs/test_pipeline_stub_sync.py`（スタブ変更時）

完了条件
- 変更ファイルに対する ruff/black/isort/mypy/pytest が緑。
- 必要に応じてスタブ再生成済み、ドキュメント同期済み。

質問（ご確認ください）
1) `Pipeline.realize()` は「設定引継ぎ（方針A）」で進めて良いですか？ それとも現状仕様を文書化（方針B）しますか？
2) `intermediate_cache(maxsize)` はどちらで進めますか？
   - A: 実装（最小実装は“有効/無効”の読み書きフラグ）。
   - B: いったん仕様撤回（ドキュメント/スタブから削除）。
3) `global_cache_counters()` に prefix-cache の統計を追加して良いですか？（`_PREFIX_DEBUG` 無効時は 0 のまま）
4) 上記で合意後、段階的に着手しますか？（1→2→3→4→5→6 の順を想定）

タイムライン（目安）
- 1) フィールド修正 + 2) realize 引継ぎ + 5) 微修正: 0.5 日
- 3) intermediate_cache（A: 1–2 日 / B: 0.5 日）
- 4) HUD メトリクス拡充: 0.5 日
- 6) ドキュメント/スタブ同期: 0.5 日

