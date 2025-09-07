# テスト作成計画（TEST PLAN）

> 目的: アーキテクチャ文書（docs/architecture.md）で定義したコア境界を守るテストを優先し、幾何→レジストリ→パイプライン→API→統合の順で堅牢性と回帰検知を高める。

## 1. 方針と優先順位（Architecture-Driven）
- 優先1: Geometry データモデルの不変条件（正規化/純関数性/ダイジェスト）。
- 優先2: レジストリ（effects/shapes）のキー正規化と登録取得の一貫性。
- 優先3: パイプライン（strict 検証、spec シリアライズ、LRU キャッシュ）。
- 優先4: API 公開面（`api.__all__`、`G/E/run/Geometry` の基本連携）。
- 優先5: 統合/スモーク（`main.draw` 経由で最小系が動く）。
※ Rendering/IO 層はユニット対象外。`moderngl`/`pyglet`/MIDI 系（例: `mido`）は `pytest.importorskip(...)` で回避し、描画はユニット外とする。

## 2. カバレッジ目標
- 対象: `engine/`, `effects/`, `shapes/`, `api/`（テスト補助や CLI は除外）。
- 軸: line 覆率を主指標、可能なら branch も参考値として収集。
- 初期目標: line 50%（各層に最低1本のユニット）
- 拡張目標: line 70%（主要エフェクト/シリアライズ経路まで）

## 3. ディレクトリとファイル（アーキテクチャ対応）
- `tests/`
  - `conftest.py`（共通フィクスチャ: `rng`, `tiny_geom`, `grid_geom`, `env_digest_on/off`, `digest_hex`）
  - `_utils/snapshot.py`（digest 比較/更新ユーティリティ）
  - `test_geometry_model.py`（正規化/コピー性/純関数/消化無効時の挙動）
  - `test_registry_normalization.py`（Camel→snake、重複登録エラー、未登録アクセス）
  - `test_shapes_factory.py`（`G.list_shapes`、`G.<name>(...)` の解決、LRU キャッシュ同一性）
  - `test_shapes_output.py`（`G.grid`/`G.polygon`/`G.sphere` の出力検証）
  - `test_pipeline_strict_and_spec.py`（`strict(True)` 未知キー検出、`to_spec/from_spec/validate_spec`）
  - `test_pipeline_cache.py`（digest 有効/無効の両モードでキャッシュヒットとLRU追い出し）
  - `test_effects_invariants.py`（translate/rotate/fill の代表不変と効果確認）
  - `test_api_surface.py`（`from api import G,E,run,Geometry` が最低限動作）
  - `test_smoke.py`（`main.draw` が Geometry を返す、マーカー `smoke`）
  - `test_g_stub_sync.py`, `test_pipeline_stub_sync.py`（スタブ同期: 維持）

## 4. 具体テスト項目（優先順）
- Geometry（`engine/core/geometry.py`）
  - 2D→3D 正規化: `from_lines([[(0,0),(1,0)]])` で `coords.shape==(2,3)`、Z=0 埋め。
  - offsets 整合: 例の本数 M に対し `offsets.shape==(M+1,)` かつ `offsets[-1]==len(coords)`。
  - `as_arrays(copy=...)` のコピー性: `copy=True` で別参照、`False` で同一参照。
  - 純関数性: `g2=g.translate(...); g is not g2` かつ元配列不変。
  - digest: 生成可能、内容変化で変わる、同内容で等しい。`PXD_DISABLE_GEOMETRY_DIGEST=1` で `g.digest` は `RuntimeError`。

- Registry（`common/base_registry.py`, `effects/shapes/registry.py`）
  - 正規化: `MyEffect` → `my_effect`、`foo-bar` → `foo_bar`。
  - 既存キーへの重複登録で `ValueError`、未登録取得は `KeyError`。

- Pipeline（`api/pipeline.py`）
  - strict: 既存エフェクトに未知パラメータで `TypeError`（例: `E.pipeline.strict(True).rotate(bogus=1).build()`）。
  - spec 検証: `validate_spec` が未知キー/型不正で `TypeError`、既存エフェクト名以外は `KeyError`。
  - シリアライズ往復: `to_spec(build(...))` → `from_spec(...)` で等価処理。
  - キャッシュ: `cache(maxsize=1)` で同入力2回は同一オブジェクト参照（`is`）、異なるキー追加で追い出し。
  - digest無効時のフォールバック: 環境変数で無効化してもキャッシュがヒットする（配列ハッシュ計算にフォールバック）。

  注記（strict と validate_spec の厳格度差）:
  - `validate_spec`: 未知パラメータは禁止（`**kwargs` 例外なし）。
  - `PipelineBuilder.strict`: 署名が `**kwargs` を受ける関数は未知キー検査の対象外。

- Effects（代表）
  - translate: 頂点数不変、重心移動量が `delta` に一致、`delta=(0,0,0)` では内容等価だが別インスタンス。
  - rotate: 距離保存（原点回り/任意 pivot）。閉路の長さが不変（数値許容差内）。
  - fill: `G.polygon(n_sides=...)` に対し `density=0` はコピー、`density>0` で線本数（`len(offsets)-1`）が増加。
           `mode` 間の関係は「cross は lines 以上」を目安（退化形に配慮し、閾値は +1 本以上、失敗時は半径/密度を上げて再試行）。

- Shapes（代表）
  - grid: `subdivisions=(sx,sy)` に対し、線本数は `Grid.MAX_DIVISIONS` に依存（現行=50）。検証は実装定数（`shapes.grid.Grid.MAX_DIVISIONS`）を参照して期待値を計算し、Z=0 を確認。
  - polygon: `n_sides=k` で閉路（最終点==先頭点）を含む1本のライン、半径≒0.5。
  - sphere: `sphere_type` 切替で非空を保証し、全頂点のノルムが≲0.5（±ε）。

- API/Smoke
  - `from api import G,E,run,Geometry` が import 可能で `G.sphere(...).translate(...)` など最小動作。
  - `main.draw(t, cc)` が `Geometry` を返す（最小 CC: dict で必要キーのみ 0/1）。

## 5. フィクスチャ/マーカーと設定
- `conftest.py`
  - `rng`: 固定シード `np.random.default_rng(0)`。
  - `tiny_geom`: 1 本/2D の最小 Geometry。
  - `grid_geom`: 規則格子（`G.grid(subdivisions=(sx, sy))`）。
  - `env_digest_on/off`: 環境変数 `PXD_DISABLE_GEOMETRY_DIGEST` の一時切替（`monkeypatch`）。
  - `digest_hex(g)`: digest の16進取得（無効時はフォールバック実装）。
  - `cc_min`: `main.draw` 用の最小 CC（必須キー: 1..9）。既定値は `{1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.0, 7:0.0, 8:1.0, 9:1.0}`。
  - `shapes_exist`: 最低限の実体確認（`{"grid","polygon","sphere"} ⊆ set(G.list_shapes())`）。
- `pytest.ini`
  - マーカー `smoke`, `snapshot`, `slow`, `property` を追記（既存の `-q --tb=short` は維持）。

## 6. スナップショット運用
- Geometry の大規模出力はダイジェストで比較。更新は `PXD_UPDATE_SNAPSHOTS=1` を前提にユーティリティで再生成。
- 保存先と命名: `tests/_snapshots/<module>__<testname>.snap`（1行=1ダイジェスト）。複数ケースは複数行で順序固定。
- PR 運用: 差分は .snap の行単位でレビュー。期待更新が含まれる場合は、コミットメッセージに「snapshot: update」等を含める。
- 画像スナップショットは対象外（レンダラ依存のため）。

## 7. 実装上の注意
- モジュール副作用（effects の自動登録）に依存するテストは、必要に応じて個別に `import effects` を明示して安定化。
- GPU/ウィンドウ依存箇所は `pytest.importorskip("moderngl")` / `pytest.importorskip("pyglet")` を使用し、ユニット層では直接触れない。MIDI 系（例: `mido`）も同様。
- 浮動小数比較は `np.allclose(..., rtol=1e-6, atol=1e-6)` を既定（大きさ依存の誤差に対応）。
- 失敗時メッセージは部分一致で確認（`in` を用いたサブストリング）。

## 8. 実行コマンド（例）
- 既定: `pytest`（pytest.ini の `--tb=short -v` を正とする）
- スモークのみ: `pytest -m smoke`
- スナップショット更新: `PXD_UPDATE_SNAPSHOTS=1 pytest -m snapshot`

## 9. 前提となる小改修（別変更で実施）
- `scripts/gen_g_stubs.py` に純粋関数 `generate_stubs_str()` を追加（I/O分離して stub 同期テストを安定化）。
- `pytest.ini` にマーカー宣言を追記。

関数仕様（テストからの呼び出しを想定）
```python
# scripts/gen_g_stubs.py
def generate_stubs_str() -> str:
    """api/__init__.pyi の期待内容を文字列として生成し返す（ファイルI/Oはしない）。"""
    ...
```
テストは `from scripts.gen_g_stubs import generate_stubs_str` として呼び、`Path('api/__init__.pyi').read_text()` と比較する。

## 10. 作業計画（テスト専用・4人並列）
目的: テスト整備のみを4名で並列に進め、進捗を可視化する（期間やGit運用の前提なし）。

- 共通ルール
  - ファイルオーナーシップ: すべて `tests/` 配下。必要に応じて `pytest.ini`・`scripts/gen_g_stubs.py` の最小変更のみ可。
  - マーカー未宣言時は `pytest --disable-warnings -q` で一時運用（後で `pytest.ini` に追記）。

- 担当A: テスト基盤・フィクスチャ・スナップショット
  - [x] ひな形作成: `tests/conftest.py`, `tests/_utils/snapshot.py`
 - [x] フィクスチャ: `rng`, `tiny_geom`, `grid_geom`, `env_digest_on/off`, `digest_hex`, `cc_min`, `shapes_exist`
  - [x] マーカー方針: `smoke`, `snapshot`, `slow`, `property`（必要なら `pytest.ini` 追記）
  - [x] スナップショット格納: `tests/_snapshots/` を設け、`PXD_UPDATE_SNAPSHOTS=1` で更新
  - [x] optional依存: `pytest.importorskip('moderngl')`/`'pyglet'`/`'numba'` のラッパ整備

- 担当B: Geometry・Registry のユニットテスト
  - [x] `test_geometry_model.py`: from_lines 正規化、`as_arrays(copy)`、`is_empty`、純関数性、digest(ON/OFF)
  - [x] `test_geometry_invariants.py`: translate/scale/rotate/concat の不変条件（行数・距離・同一頂点集合）
  - [x] `test_registry_normalization.py`: Camel→snake、`-`→`_`、重複登録 `ValueError`、未登録 `KeyError`
  - [x] エラー系: `from_lines` の無効入力で `ValueError`、`validate_spec` の型不正で `TypeError`

- 担当C: Shapes・Effects のユニットテスト
  - [x] `test_shapes_factory.py`: `G.list_shapes`、`G.<name>(...)` 解決、LRU同一性、numpyスカラ受容
  - [x] `test_shapes_output.py`: `grid`（線本数とZ=0・定数参照）、`polygon`（閉路と半径）、`sphere`（半径上限+非空）
  - [x] `test_effects_invariants.py`: translate/rotate の不変、入力 `Geometry` の非破壊性（同一参照でない）
  - [x] `test_effects_fill.py`: density=0はコピー、>0で本数増、`lines/cross/dots` の単調関係、角度指定の変化

- 担当D: Pipeline・API・スモーク
  - [x] `test_pipeline_strict_and_spec.py`: `strict(True)` 未知キー、`validate_spec`（型/範囲/choices/JSON性）、`KeyError`（未登録）
  - [x] `test_pipeline_cache.py`: `cache(maxsize=1)` のヒット/追い出し、digest無効時のフォールバック
  - [x] `test_pipeline_equivalence.py`: 連続適用（効果関数直接適用）とパイプライン結果の等価性
  - [x] `test_api_surface.py`: `from api import G,E,run,Geometry` の import と最小連携
  - [x] `test_smoke.py`: `main.draw(t=0, cc=cc_min)` が `Geometry` を返す（`-m smoke`）
  - [x] スタブ同期: `scripts.gen_g_stubs` に `generate_stubs_str()` を追加して同期待ち合わせ（必要最小）

- 完了定義（Test-Only）
  - `pytest -q` 緑、`-m smoke`/`-m snapshot` が機能。
  - 主要層（Geometry/Registry/Shapes/Effects/Pipeline/API）が少なくとも1ファイルずつカバー。
  - digest(ON/OFF) 両モードでキャッシュ動作が検証済み。

補足（ランダム性/プロパティテスト）
- 乱数を用いるテスト（例: ランダム回転/並進）は `rng` フィクスチャを必須とし、`@pytest.mark.property` で明示。
- Hypothesis を用いる場合は事前に種を固定し、事後に失敗例をスナップショットへ保存可。

等価性の判定基準（to_spec/from_spec）
- 優先1: `to_spec(p1) == to_spec(p2)`（順序不問の辞書は並べ替え比較）。
- 優先2: 同一入力 Geometry に適用した結果の `digest_hex` が一致。
