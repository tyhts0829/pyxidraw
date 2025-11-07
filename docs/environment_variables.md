# 環境変数インベントリ（pyxidraw）

目的: リポジトリ内で参照されている環境変数を網羅し、整理（不要の削減と集中管理）のベース資料とする。

調査方法: `rg` により `os.environ/getenv` と `PXD_/PYX_` を検索し、実装・テスト・ドキュメントを横断して確認。

## 実装で参照される（ランタイム有効）

- `PXD_PIPELINE_QUANT_STEP`
  - 種別/既定: float（既定 1e-6）
  - 目的: 署名生成/GUI 保存時の float 量子化刻み（Effects は量子化後を実行引数に使用）
  - 参照: src/common/param_utils.py:126, src/engine/ui/parameters/persistence.py:28

- `PXD_IBO_FREEZE_ENABLED`
  - 種別/既定: bool（文字列 0/1, 既定 1）
  - 目的: 連続フレームで offsets が不変な場合に IBO を再利用（VBO のみ更新）
  - 参照: src/engine/render/renderer.py:83

- `PXD_IBO_DEBUG`（未使用気味）
  - 種別/既定: bool（0/1, 既定 0）
  - 目的: IBO 固定化のデバッグフラグ（現状コード内で未使用）
  - 参照: src/engine/render/renderer.py:84

- `PXD_INDICES_CACHE_ENABLED`
  - 種別/既定: bool（0/1, 既定 1）
  - 目的: オフセット署名ベースの indices LRU の有効/無効
  - 参照: src/engine/render/renderer.py:308

- `PXD_INDICES_CACHE_MAXSIZE`
  - 種別/既定: int（既定 64, 負値は 0 に丸め＝無効）
  - 目的: indices LRU の上限件数
  - 参照: src/engine/render/renderer.py:309

- `PXD_INDICES_DEBUG`（未使用気味）
  - 種別/既定: bool（0/1, 既定 0）
  - 目的: indices LRU のデバッグフラグ（現状コード内で未使用）
  - 参照: src/engine/render/renderer.py:315

- `PXD_SHAPE_CACHE_MAXSIZE`
  - 種別/既定: int | None（既定 128, 0 で無効, None で無制限想定）
  - 目的: 形状生成結果 LRU の上限件数
  - 参照: src/engine/core/lazy_geometry.py:204

- `PXD_PREFIX_CACHE_ENABLED`
  - 種別/既定: bool（0/1, 既定 1）
  - 目的: エフェクトの静的プレフィックス（中間結果）LRU の有効/無効
  - 参照: src/engine/core/lazy_geometry.py:209

- `PXD_PREFIX_CACHE_MAXSIZE`
  - 種別/既定: int | None（既定 128, 0 で無効）
  - 目的: プレフィックス LRU の上限件数
  - 参照: src/engine/core/lazy_geometry.py:210

- `PXD_PREFIX_CACHE_MAX_VERTS`
  - 種別/既定: int（既定 10_000_000, 0 で保存しない）
  - 目的: プレフィックス LRU に保存する Geometry の頂点数上限
  - 参照: src/engine/core/lazy_geometry.py:211

- `PXD_DEBUG_PREFIX_CACHE`
  - 種別/既定: bool（0/1, 既定 0）
  - 目的: プレフィックス LRU のカウンタ更新/ログの有効化
  - 参照: src/engine/core/lazy_geometry.py:212

- `PXD_COMPILED_CACHE_MAXSIZE`
  - 種別/既定: int | None（既定 128, 0 で無効）
  - 目的: `api.effects` のステップ列（コンパイル済み `Pipeline`）キャッシュの上限件数
  - 参照: src/api/effects.py:29

- `PXD_DEBUG_FONTS`
  - 種別/既定: bool（存在で有効）
  - 目的: フォント探索ユーティリティでの軽量デバッグ出力
  - 参照: src/util/fonts.py:99

- `PYX_USE_NUMBA`
  - 種別/既定: bool（文字列 0/1, 既定 1）
  - 目的: `effects.collapse` で numba 実装を使うかの切替
  - 参照: src/effects/collapse.py:390

- `WINDIR`（OS 標準）
  - 種別/既定: str（既定 `C:\\Windows`）
  - 目的: Windows のフォントディレクトリ解決
  - 参照: src/shapes/text.py:125, src/util/fonts.py:62

## テスト/開発ユースのみ

- `PXD_UPDATE_SNAPSHOTS`
  - 目的: digest スナップショットの更新トグル（存在で更新・skip）
  - 参照: tests/snapshot/test_geometry_digest_snapshot.py:29, pytest.ini:12

- `PXD_DISABLE_GEOMETRY_DIGEST`（現状未使用・撤廃方向）
  - 目的: Geometry.digest の無効化トグルの名残（実装側では digest を廃止済み）
  - 参照: tests/perf/test_pipeline_perf.py:61, tests/core/test_geometry.py:70

## ドキュメントにのみ言及（未実装）

- `PXD_PIPELINE_CACHE_MAXSIZE`
  - 目的: `Pipeline.cache(maxsize=...)` の既定を環境変数で上書き（仕様案）
  - 出典: docs/spec/pipeline.md:40, architecture.md:361

## 所感（重複・未使用・危険箇所）

- 未使用の疑い: `PXD_IBO_DEBUG`, `PXD_INDICES_DEBUG` はフラグを読むが実際の分岐/ログが無い。
- 撤廃候補: `PXD_DISABLE_GEOMETRY_DIGEST` は実装が digest 非依存に移行しており、テストのみ参照。整理対象。
- 文書/実装の差分: `PXD_PIPELINE_CACHE_MAXSIZE` はドキュメントにのみ存在。導入するか、記述を削除して整合を取るべき。

## 集中管理への提案（最小方針）

- 単一の定義モジュール: `src/common/settings.py`（または `common/env_vars.py`）に以下を集約。
  - 定義: 変数名、型、既定、簡潔な説明。
  - 読み出し: 起動時に 1 回だけ `common.env.env_bool/env_int` などで読み取り、属性として公開。
- 呼び出し側の置換: 各所の `os.getenv` 直参照を当該モジュール経由に変更。
- 未使用の整理: `PXD_IBO_DEBUG`, `PXD_INDICES_DEBUG` は用途が無ければ削除（または実装側で使用を追加）。
- ドキュメント同期: 上記モジュールの内容を人間可読の一覧へ自動出力するスクリプトを用意（将来）。

補足: `fonts.search_dirs` は `~` と環境変数展開を許可しているため、任意の OS 環境変数を含められる（プロジェクト固有ではないため本一覧からは除外）。

## 集中管理モジュール: 実装計画（要確認）

目的: 環境変数の定義・既定・型・説明を単一モジュールに集約し、参照側はそのモジュール経由に統一する。

方針
- 追加: `src/common/settings.py` を新設し、型付きで読み取り・公開。
- 実装: 既存の `common.env.env_int/env_bool` を利用して起動時一括ロード。`reload_from_env()` を提供（テストや再読み込み用途）。
- 置換: 実装箇所の `os.getenv/os.environ/env_*` を段階的に `common.settings` 参照へ置換。
- 非対象: OS 既定の `WINDIR` はそのまま OS 依存扱い（集中管理の対象外）。

置換対象（優先順）
- 最小セット（安全）:
  - `PXD_PIPELINE_QUANT_STEP`: `src/common/param_utils.py`, `src/engine/ui/parameters/persistence.py`
  - `PXD_COMPILED_CACHE_MAXSIZE`: `src/api/effects.py`
- キャッシュ層:
  - `PXD_SHAPE_CACHE_MAXSIZE`, `PXD_PREFIX_CACHE_ENABLED`, `PXD_PREFIX_CACHE_MAXSIZE`, `PXD_PREFIX_CACHE_MAX_VERTS`, `PXD_DEBUG_PREFIX_CACHE`: `src/engine/core/lazy_geometry.py`
- レンダラ関連:
  - `PXD_IBO_FREEZE_ENABLED`, `PXD_IBO_DEBUG`, `PXD_INDICES_CACHE_ENABLED`, `PXD_INDICES_CACHE_MAXSIZE`, `PXD_INDICES_DEBUG`: `src/engine/render/renderer.py`
- ユーティリティ/エフェクト:
  - `PXD_DEBUG_FONTS`: `src/util/fonts.py`
  - `PYX_USE_NUMBA`: `src/effects/collapse.py`

ステップ計画（段階導入）
1) settings 追加（読み出し・docstring・`reload_from_env()`）。
2) 最小セット置換（`param_utils`, `persistence`）。
3) lazy_geometry のキャッシュ設定を置換（LRU の挙動不変）。
4) renderer の IBO/indices 設定を置換（ログ/デバッグの分岐も温存）。
5) util/effects の置換（`PXD_DEBUG_FONTS`, `PYX_USE_NUMBA`）。
6) 未使用/文書のみの変数の扱い方針を決定（削除 or 実装反映）。

検証とDoD（変更単位）
- 変更ファイルに限定して `ruff/black/isort/mypy/pytest` を実施。
- 既存挙動の維持（既定値・境界値の互換）。
- テスト中に環境変数を書き換えるケースに備え、`common.settings.reload_from_env()` を用意。

詳細なタスク分解とチェックリストは docs/plans/env_centralization.md に記載。承認後に着手します。
