# テスト全面再設計ドラフト（承認依頼）

目的: 現行コード（2025-09-10 時点）の公開 API と中核ロジックを、最小コストで堅牢に検証できるテスト群へ刷新する。原則「短く・具体的・反復可」。本ドラフトはチェックリストを含む計画書であり、実装前に合意を得る。

---

## 方針（要点）

- レイヤ別に責務を明確化してテスト種別を分離（core / api / effects / pipeline / render / io）。
- 最小スモーク → 単体 → 連携 → E2E（ヘッドレス限定）の順に拡張。毎回“編集ファイル限定実行”で高速に回す。
- 決定的（純関数）な箇所を優先（Geometry, registries, pipeline spec/strict, builder cache）。
- 重依存（OpenGL/pyglet/shapely/numba/mido 等）は optional/integration マーカーで隔離。Ask-first 運用。
- 乱数は固定し再現性確保。スナップショット系は別途 opt-in（Ask-first; PXD_UPDATE_SNAPSHOTS）。

---

## マーカー運用

- smoke: 最小動作確認（秒で終わる）。
- integration: サブシステム間の連携（プロセス/スレッド/キュー）。
- e2e: `api.run_sketch(init_only=True)` など UI 初期化手前までの通し。
- optional: 重依存（例: shapely/numba/mido/GL）。CI 本線は除外。
- perf/property/snapshot: 将来拡張（Ask-first）。

pytest.ini のマーカー定義と整合済み。

---

## テスト対象と狙い

1. Core（engine/core）

- Geometry
  - from_lines 正規化（2D→Z=0、1D→reshape、空/単頂点、offsets 整合）。
  - translate/scale/rotate/concat の純粋性（新インスタンス・配列コピー方針・is_empty）と数値妥当性。
  - digest 有効/無効（環境変数）と例外仕様、as_arrays(copy) の copy/view 期待。
- transform_utils
  - transform_combined の順序性（Scale→Rotate→Translate）。
- frame_clock
  - tick の dt 非指定分岐（軽い単体）。

2. Common / Registry / Types

- BaseRegistry
  - キー正規化（Camel→snake, -, 大文字）、重複登録防止、unregister、list_all/is_registered。

3. API レイヤ（api/\*）

- shape_factory (G)
  - 動的ディスパッチ（存在/不存在）、LRU キャッシュ（\_params_to_tuple の正規化: dict/ndarray/set/bytes）。
  - G.empty/from_lines の型整合。
- pipeline (E.pipeline)
  - PipelineBuilder.strict: 未知パラメータで TypeError（allowed 名を含むメッセージ）。
  - to_spec/from_spec/validate_spec: JSON-like 判定、choices/min/max/type 反映。
  - Pipeline キャッシュ: 同一入力でヒット、maxsize=0 で無効、maxsize=1 の追い出し。
- runner
  - run_sketch(init_only=True) が重依存を読み込まず早期 return（smoke）。

4. Shapes（shapes/\*）

- registry デコレータで登録されること、get/list が機能すること。
- 代表: Sphere.generate の出力が Geometry（座標/offsets 整合）であること。パラメータ域のクリップ。

5. Effects（effects/\*）

- rotate: Geometry.rotate 委譲の動作（pivot/angles_rad）。
- displace: amplitude=0 で恒等、spatial_freq のスカラー/ベクトル正規化、numba 未使用経路のフォールバック（optional）。
- offset: shapely 依存のため optional。距離 0 で恒等、join/segments の境界（軽い入力）。

6. Pipeline サブシステム（engine/pipeline/\*）

- SwapBuffer.push/try_swap/version/is_data_ready の原子性（単体; lock の挙動は状態遷移で間接検証）。
- StreamReceiver: 最新 frame_id 優先、Exception バケツリレー、max_packets_per_tick 制限。
- WorkerPool（integration）: ダミー draw を使い、結果キューから RenderPacket を受け取れる。WorkerTaskError 伝播。

7. Render ユーティリティ（engine/render/\*）

- \_geometry_to_vertices_indices（private util）: 複数ライン →primitive restart index の配置が正しい。
  - GPU 実体（moderngl）は触らない。LineRenderer.tick/draw は smoke（index_count=0 経路のみ）。

8. Stubs/Docs 連携（scripts/gen_g_stubs.py, api/**init**.pyi）

- test_g_stub_sync: `generate_stubs_str()` の結果とディスク上の `src/api/__init__.pyi` が一致。
- test_pipeline_stub_sync: エフェクト登録の変更が Stub に反映されていること（Protocol メソッドの存在）。

---

## 目標レベル（段階）

- Phase 1（当日着手）: smoke/単体を中心に 20–30 ケース、core/api/pipeline を緑化。
- Phase 2: integration（Worker/Receiver）と render/util、shapes 代表。
- Phase 3: optional（shapely/numba/mido/GL）を opt-in で追加。snapshot/property/perf は合意後。

---

## テストファイル構成案（作成順）

- tests/core/test_geometry.py [smoke]
- tests/core/test_transform_utils.py [smoke]
- tests/common/test_base_registry.py [smoke]
- tests/api/test_shape_factory.py [smoke]
- tests/api/test_pipeline_spec_and_strict.py [smoke]
- tests/api/test_pipeline_cache.py [smoke]
- tests/api/test_runner_init_only.py [smoke]
- tests/shapes/test_registry_and_sphere.py [smoke]
- tests/effects/test_rotate.py [smoke]
- tests/effects/test_displace_minimal.py [optional]
- tests/effects/test_offset_minimal.py [optional]
- tests/pipeline/test_swap_buffer_and_receiver.py [smoke]
- tests/pipeline/test_worker_pool_smoke.py [integration]
- tests/render/test_renderer_utils.py [smoke]
- tests/stubs/test_g_stub_sync.py [smoke]
- tests/stubs/test_pipeline_stub_sync.py [smoke]

---

## 代表シナリオ（抜粋の期待仕様）

- Geometry.from_lines
  - 2D 入力: Z=0 補完、offsets=[0, N]。空入力: coords.shape==(0,3), offsets==[0]。
  - 1D 長さ ≠3k で ValueError。copy=False で view、copy=True で deep copy。
- Geometry.digest
  - 既定有効: bytes(16) を返す。一度アクセス後は同一オブジェクトを返す。
  - 環境 `PXD_DISABLE_GEOMETRY_DIGEST=1` で RuntimeError。
- PipelineBuilder.strict
  - rotate に未知キーで TypeError: メッセージに allowed 名（pivot, angles_rad）。
- Pipeline.cache
  - 同一 g+pipeline で 2 回目はヒット（計測は内部状態で間接確認: out is cached obj）。
  - maxsize=0 なら常にミス。maxsize=1 で 2 キー目で追い出し。
- validate_spec
  - 非 JSON 値（set(), object()）は TypeError。choices/min/max/type を各 1 ケース。
- shape_factory
  - 未登録名は AttributeError。np.ndarray を含むパラメータでも \_params_to_tuple が安定キー化。
- runner.init_only
  - 例外を出さずに None を返す（重依存未導入環境で動作）。
- SwapBuffer/StreamReceiver
  - push→try_swap→get_front の状態遷移、最新 frame_id のみ受理、Exception はそのまま raise。
- \_geometry_to_vertices_indices
  - 2 本のラインで indices の末尾に primitive restart が 2 つ入る（個数/位置一致）。
- Stub 同期
  - 生成文字列とファイルが完全一致。E.pipeline の登録関数が Protocol に出力される。

---

## フィクスチャ設計（conftest.py）

- `np_seed`（session）: numpy RNG 固定。
- `geom_empty` / `geom_line2` / `geom_two_lines`: 小さな Geometry 試料。
- `small_sphere`: `G.sphere(subdivisions=0.0)` 代表形状。
- `make_pipeline`: `E.pipeline.strict(True)` を返すヘルパ。
- `env_no_digest`: `PXD_DISABLE_GEOMETRY_DIGEST=1` 一時設定（monkeypatch）。

---

## 実行方法（開発ループ）

- 初期化（Ask-first 前提; ネットワーク有）
  - `python3.10 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e .[dev]`
- 単体/変更ファイル優先
  - Lint: `ruff check --fix {path}`
  - Format: `black {path} && isort {path}`
  - Type: `mypy {path}`
  - Test: `pytest -q {test_path}` または `pytest -q -k {expr}`
- スモーク一括
  - `pytest -q -m smoke`
- integration のみ
  - `pytest -q -m integration`
- スタブ同期チェック
  - 生成差分を見る: `PYTHONPATH=src python -c "from scripts.gen_g_stubs import generate_stubs_str;print(generate_stubs_str()[:500])"`
  - 更新（Ask-first）: `PYTHONPATH=src python -m scripts.gen_g_stubs && git add src/api/__init__.pyi`

---

## リスクと回避策

- multiprocessing 周りのフレーク：タイムアウトと最小データで短時間に限定。CI では integration を段階導入。
- numba/shapely/GL 未導入環境：`-m optional` を本線から除外。必要時のみ依存を導入（Ask-first）。
- スナップショット肥大：`tests/_snapshots/` に限定し、更新は環境変数ゲート（Ask-first）。

---

## 実装チェックリスト（合意後に着手）

- [x] tests/core/test_geometry.py（from_lines/ops/digest/as_arrays）
- [x] tests/core/test_transform_utils.py（順序検証）
- [x] tests/common/test_base_registry.py（登録/重複/正規化）
- [x] tests/api/test_shape_factory.py（動的呼び出し/キャッシュ/例外）
- [x] tests/api/test_pipeline_spec_and_strict.py（strict/validate_spec）
- [x] tests/api/test_pipeline_cache.py（maxsize/clear_cache）
- [x] tests/api/test_runner_init_only.py（init_only 経路）
- [x] tests/shapes/test_registry_and_sphere.py（登録/Geometry 整合）
- [x] tests/effects/test_rotate.py（pivot/angles_rad）
- [x] tests/pipeline/test_swap_buffer_and_receiver.py（最新優先/例外伝播）
- [x] tests/pipeline/test_worker_pool_smoke.py（integration; RenderPacket/TaskError）
- [x] tests/render/test_renderer_utils.py（indices 生成）
- [x] tests/stubs/test_g_stub_sync.py（generate_stubs_str 一致）
- [x] tests/stubs/test_pipeline_stub_sync.py（Protocol メソッド）
- [x] tests/effects/test_displace_minimal.py（amplitude=0 恒等）
- [x] tests/effects/test_offset_minimal.py（距離 0 恒等; shapely）
- [ ] optional: snapshot/property/perf の設計追記

---

## 合意が必要な点（確認事項）

1. まずは Phase 1（約 1–2 日相当）を最小構成で実装して良いか。→ 実装済み
2. optional（shapely/numba/mido/GL）はテストから初期除外で合意できるか。→ 除外せず実装（環境依存は安全化）
3. スナップショットテストの導入タイミング（Phase 2/3 に後ろ倒し可）。→ テスト実装済（更新は env ゲート）
4. カバレッジ目安（core/api/pipeline 80% 目標、全体 60% 以上）を採用するか。→ 採用（次段で計測）

承認後、各テストファイルを段階的に追加し、編集ファイル優先の高速ループで緑化します。

---

## 追加カバレッジ計画（Phase 2: Coverage Uplift）

目標: core/api/pipeline ≥ 80%、全体 ≥ 60%（現状 36%）。影響小・効果大の単体テストを追加して段階的に引き上げる。

完了条件
- 下記チェックリストの「必須」項目をすべて完了し、局所カバレッジ目標を満たす。
- `pytest --cov=src --cov-report=term-missing` が緑で完走し、しきい値報告を記録。

チェックリスト（必須）
- api/pipeline.py（+3〜5%）
  - [x] to_spec/from_spec の往復で Pipeline 等価性を検証（順序・パラメータ保持）。
  - [x] validate_spec: `min/max/choices/type` メタの検証（rotate/displace/offset 代表で1ケースずつ）。
  - [x] PipelineBuilder.cache(maxsize=None) 挙動と `clear_cache()` の動作確認。
  - [x] `_is_json_like` の境界（tuple/list/dict 混在の許容と set/object の拒否）。

- common/base_registry.py（+2〜4%）
  - [x] `unregister` の no-op 経路（未登録名で例外にならない）。
  - [x] `list_all` の出力に登録キーが含まれる（順序前提なし）。
  - [x] キー正規化のハイフン混在（"My-Effect" → 正規化キー）。

- api/shape_registry.py（+15〜25%）
  - [x] `unregister_shape` が存在しないキーで例外を出さない（内部握りつぶし）。
  - [x] `get_shape_generator` の未登録名で `ValueError`。

- api/runner.py（+10〜20% / GL 非依存で分岐網羅）
  - [x] `use_midi=True, midi_strict=True` かつ mido 不在相当で `SystemExit(2)`（monkeypatch で擬似）。
  - [x] `use_midi=True, midi_strict=False` で mido 不在相当でも `init_only=True` により早期 return（例外なし）。
  - [x] 環境変数 `PYXIDRAW_MIDI_STRICT` による既定値補完の検証（true/false 両値）。
  - [x] `init_only=True` が moderngl/pyglet の import を要求しないこと（ImportError を誘発するモックで確認）。

チェックリスト（任意・効果中）
- effects/offset.py（shapely 経路）
  - [x] join='round'/'mitre'/'bevel' の最小入力での成功を smoke で確認。
- util/utils.py
  - [x] `_find_project_root` のフォールバック経路（実ファイル操作なしのパッチで）。

実行順とコマンド
- Phase 2-1（pipeline/base_registry/shape_registry）: `pytest -q tests/api/test_pipeline_* tests/common/test_base_registry.py tests/api/test_shape_registry_extra.py`
- Phase 2-2（runner 分岐）: `pytest -q tests/api/test_runner_*`
- 最終確認: `pytest --cov=src --cov-report=term-missing -q`

想定インパクト（到達見込み）
- api/pipeline.py: 77% → 82–85%
- common/base_registry.py: 78% → 85%+
- api/shape_registry.py: 76% → 95%+
- api/runner.py: 31% → 45–55%
