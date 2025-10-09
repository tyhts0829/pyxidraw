# dpg_window.py リファクタリング計画（提案）

目的: Dear PyGui ベースのパラメータ GUI 実装を、明確な責務分割・例外ガードの縮小・テスト容易性向上の観点で簡潔化する。現行の公開 API（`ParameterWindow` の挙動とインポートパス）とパラメータ GUI の仕様は維持する。

関連ファイル（参照）
- `src/engine/ui/parameters/dpg_window.py:37`（DPG 未導入時のダミー実装）
- `src/engine/ui/parameters/dpg_window.py:64`（本実装のクラス定義）
- `src/engine/ui/parameters/dpg_window.py:115-124`（pyglet 統合/スレッド駆動の分岐）
- `src/engine/ui/parameters/dpg_window.py:126-155`（可視化/クローズの広域例外ガード）
- `src/engine/ui/parameters/dpg_window.py:157-223`（ウィジェット構築・マウント）
- `src/engine/ui/parameters/dpg_window.py:224-333`（各 ValueType のウィジェット生成）
- `src/engine/ui/parameters/dpg_window.py:444-457`（駆動ループ/フレーム呼び出し）
- `src/engine/ui/parameters/dpg_window.py:459-520`（テーマ適用。例外ガードが多く分岐が複雑）

指摘事項（現状の課題）
1) 多責務・肥大化
   - DPG ライフサイクル、ウィジェット構築、イベント同期、テーマ、駆動（pyglet/Thread）を単一クラスに内包（約560行）。
2) 例外ガードの過多と広域捕捉
   - `except Exception: pass` が多数存在し、失敗要因の可視性/保守性が低い（例: `:133, :139, :144, :150, :154, :199, :306, :450, :456, :498` など）。
3) 実装分岐の重複/密結合
   - DPG 未導入スタブと本実装が同一ファイル・同名クラスで二系統。pyglet 統合と Thread 駆動も同クラスに混在。
4) リテラル/タグの散在
   - 固定タグ（`"__pxd_param_root__"`, `"__pxd_param_scroll__"`, `"__pxd_param_stage__"`）や列幅ロジックが点在し、再利用しづらい。
5) テスト容易性が低い
   - GUI 起動に依存するためユニットテストが困難。ウィジェット構築や値変換の純粋関数化が不足。

非機能要件（維持すべき挙動）
- 既存の公開 API と import 互換（`from engine.ui.parameters.dpg_window import ParameterWindow`）。
- パラメータ GUI の仕様（AGENTS.md 記載）を順守：
  - GUI は「draw 内で未指定（既定値採用）の引数のみ」を対象（ValueResolver/Store の既存挙動を尊重）。
  - RangeHint/VectorRangeHint/choices を UI に反映。
  - ヘッドレス/未導入環境でも import 失敗しない。
  - macOS では可能なら pyglet 駆動（メインスレッド）。

進め方（段階的チェックリスト）
1) 土台と責務の分割（内部 API 整理）
   - [ ] `ParameterWindowBase`（最小インタフェース: `mount`, `set_visible`, `close`）。
   - [ ] `NullParameterWindow`（DPG 未導入/ヘッドレス用スタブを別クラス化）。
   - [ ] `DpgParameterWindow`（DPG 実装本体）。外部公開名 `ParameterWindow` は既存互換のエイリアスにする。

2) 駆動レイヤの分離
   - [ ] `IDpgDriver` 的薄いプロトコルに切り出し（`pyglet` 版とスレッド版）。
   - [ ] `DpgParameterWindow` からスケジューリング分岐を排除（ドライバ委譲）。
   - [ ] 例外はドライバ側で限定捕捉し、ログへ出力（丸めない）。

3) UI 構築の関数化
   - [ ] `build_root(viewport_cfg, layout_cfg)` と `build_grouped_table(descriptors, layout)` を純粋関数寄りに分離。
   - [ ] 各 `ValueType` ごとのウィジェット生成を小関数に分割（`create_bool`, `create_enum`, `create_string`, `create_vector`, `create_int`, `create_float`）。
   - [ ] タグ/定数（root/scroll/stage/format など）をモジュール定数に集約。

4) 例外ガード/ログの整理
   - [ ] 広域 `except Exception: pass` を撤廃し、失敗を局所で把握（DPG 古いバージョン対応箇所などは明示コメント）。
   - [ ] 失敗時のフォールバックは最小限に限定し、`logging` で発生箇所を記録（テストしづらい箇所は `pragma: no cover` 付与可）。

5) テーマ適用の単純化
   - [ ] スタイル/カラー適用マップ生成を小関数化し、`Sequence` 判定や多値処理を集約。
   - [ ] 既定の最小テーマ生成を `theme_utils.py` 的ヘルパへ移動（任意）。

6) 型/Docstring/ヘッダ整備
   - [ ] 主要関数に日本語の NumPy スタイル docstring を追加。
   - [ ] Value/Descriptor 受け渡しは `Any` を避け、可能な限り型を明示。

7) 検証・互換
   - [ ] 変更ファイルに限定した高速チェック: `ruff/black/isort/mypy`。
   - [ ] パラメータ GUI の最小テスト群（存在する場合）: `pytest -q tests/ui/parameters`。
   - [ ] `api.sketch` から `use_parameter_gui=True` で起動し、基本操作が維持されることを手動確認（ヘッドレス環境ではスキップ）。

提案するディレクトリ/構造（最小）
- `src/engine/ui/parameters/dpg_window.py`
  - `ParameterWindowBase`（抽象/最小）
  - `NullParameterWindow`
  - `DpgParameterWindow`（実装本体）
  - 公開名: `ParameterWindow = DpgParameterWindow or NullParameterWindow`（環境に応じて）
- 任意: `src/engine/ui/parameters/_dpg_driver.py`（pyglet/Thread 駆動）
- 任意: `src/engine/ui/parameters/_dpg_widgets.py`（ValueType 別ウィジェット生成）

受入条件（DoD）
- 既存 API 互換を保ちつつ、`dpg_window.py` の行数と複雑度が有意に低減。
- GUI が無い環境でもインポート/実行が落ちない（現状互換）。
- 変更ファイルの `ruff/mypy/pytest` が成功（テストがある場合）。
- 例外ガードの広域 `except Exception: pass` の大半を排除（不可避な箇所のみ縮退）。

要確認/オープンな論点（事前合意したい点）
- pyglet 駆動の既定レート（現状 60Hz: `src/engine/ui/parameters/dpg_window.py:117-120`）を維持でよいか。
- `string` 入力での既定高さ/改行（`__param_meta__` 優先。未指定時は 1 行で良いか）。
- enum の閾値（5 以下はラジオボタン、それ以上はコンボ）の閾値を変更するか。
- ベクタ入力（3/4 成分）のデフォルト範囲を 0..1 維持で良いか（RangeHint 無し時）。

実施順（小さな PR 単位の目安）
1. Base/Null の導入と公開名の差し替え（無挙動変更）。
2. ドライバ分離（pyglet/thread）とクラスからの除去。
3. ウィジェット生成の小関数化とタグ/定数集約。
4. 例外ガードの局所化＋ログ化、テーマ適用の整理。
5. 仕上げ（型/Docstring）＋最小テスト実行。

備考
- 既存の `ParameterValueResolver`/`ParameterStore` の契約には手を入れず、GUI 層のみの純粋分割を徹底する（仕様整合のため）。
- macOS のメインスレッド制約は現状通り `pyglet.clock.schedule_interval` を優先し、利用不可時のみ Thread 駆動へフォールバックする。

