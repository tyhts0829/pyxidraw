# README と実装の整合性チェックレポート

本レポートは、リポジトリ直下の `README.md` に記載された内容と、実装（コード/設定/ドキュメント）との整合性を確認した結果をまとめたもの。

- 対象: リポジトリ全体（`src/`, `configs/`, `docs/`, `tools/`, `tests/`, ルート設定類）
- 目的: README の主張・リンク・使用例が現在の実装と矛盾しないかの点検
- 生成物: 本ファイル（機械作成）、コード変更は行っていない

## 不一致（要修正）

- 名称/バージョン表記の不一致
  - README のタイトルは「PyXidraw6」だが、パッケージ名は `pyxidraw5` のまま。
    - 該当: `README.md:1`「# PyXidraw6」 / `pyproject.toml:6`「name = "pyxidraw5"」
  - 併せて `pyproject.toml:11` の作者表記も「PyXidraw5 contributors」。

- README 内リンクの不整合・リンク切れ
  - 「目的/全体像: `docs/architecture.md`」→ 実ファイルはルート直下 `architecture.md`。
    - 該当: `README.md:10`（リンク先存在せず）/ 実体: `architecture.md`
  - 「エフェクト一覧: `docs/effects.md`」→ 該当ファイルが存在しない。
    - 該当: `README.md:12`（リンク先なし）
  - 「開発環境セットアップ: `docs/dev-setup.md`」→ 該当ファイルが存在しない。
    - 該当: `README.md:15`（リンク先なし）

- サンプル設定ファイルの記述
  - 「サンプル: `configs/example.yaml`」→ 該当ファイルが存在しない。
    - 該当: `README.md`「設定ファイル」節 / 実体なし（`configs/default.yaml` は存在）

## 整合確認済みの主事項（OK）

- 公開 API エントリ
  - `from api import G, E, cc, lfo, run/run_sketch, shape, effect` が提供されている。
    - 該当: `src/api/__init__.py`（`__all__` に含まれる）, `src/api/shapes.py`（`G` 実体）, `src/api/effects.py`（`E.pipeline`）
- 形状は関数ベースで `@shape` 登録（BaseShape 継承なし）
  - 該当: `src/shapes/registry.py`（`@shape` デコレータ）, 各 `src/shapes/*.py`
- パイプライン `E.pipeline.<effect>(...).build()` のビルダー/実行モデル
  - 該当: `src/api/effects.py`（`PipelineBuilder`/`Pipeline` 実装, キャッシュ仕様）
- LFO/CC の利用例
  - `lfo(...)` は `__call__(t)` で 0..1 を返却、`cc[i]` は未定義 0.0。
    - 該当: `src/common/lfo.py`, `src/api/cc.py`
- HUD 設定の例
  - `HUDConfig` と `run_sketch(..., hud_config=cfg)` が存在し、例のフラグも有効。
    - 該当: `src/engine/ui/hud/config.py`, `src/api/sketch.py`
- スタブ生成/スモークテスト
  - `tools.gen_g_stubs` により `src/api/__init__.pyi` を生成、`pytest -q -m smoke` マーカーも存在。
    - 該当: `tools/gen_g_stubs.py`, `tests/...` 各 smoke テスト
- 設定読み込みの仕様
  - `configs/default.yaml` と ルート `config.yaml` を浅いマージで返却。
    - 該当: `src/util/utils.py` の `load_config()`

## 補足的観測（README 由来ではない軽微な表現差）

- ドキュメント/コメントの表現差
  - `common.param_utils.quantize_params` の docstring に「既定 1e-3」とあるが、実装と README は 1e-6。
    - 該当: `src/common/param_utils.py`（実装は 1e-6／`PXD_PIPELINE_QUANT_STEP` で上書き）
  - `tools/gen_g_stubs.py` に旧「クラス+generate()」スタイルに言及する補助関数が残存するが、最終的な出力は関数ベースの shape から生成されており挙動は一致。

## 推奨アクション（順不同）

- パッケージ名の更新
  - `pyproject.toml` の `[project].name` を "pyxidraw6" 相当に更新（必要なら `authors` も）。
- README のリンク修正
  - `docs/architecture.md` → `architecture.md`（ルート）に修正。
  - `docs/effects.md` と `docs/dev-setup.md` のいずれか: ドキュメントを新規追加するか、README の該当行を削除/差し替え。
- 設定サンプルの整合
  - `configs/example.yaml` を追加するか、README からサンプル記述を削除。
- 参考: 実装コメントの整合
  - `src/common/param_utils.py` の docstring を実装/README に合わせて 1e-6 表記へ統一（任意）。

以上。README のリンク切れとパッケージ名の表記が主要な不一致であり、API/使用例・実装は概ね一致していることを確認した。
