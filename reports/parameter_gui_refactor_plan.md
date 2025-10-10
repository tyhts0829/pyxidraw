# DPG Parameter GUI リファクタ計画（dpg_window 分割）

目的
- 保守性・可読性・テスト容易性の向上（単一大ファイルから関心分離）。
- 既存仕様/挙動は変更せず、内部構造のみの整理を目指す（非破壊的）。

スコープ
- 対象: `src/engine/ui/parameters/dpg_window.py` の分割と再配置。
- 非対象: UI 機能追加、パラメータランタイム仕様変更、外部 API 変更。

提案ディレクトリ構成（新規）
- `src/engine/ui/parameters/dpg/`
  - `__init__.py`（最小再エクスポート）
  - `window.py`（ParameterWindow オーケストレータ）
  - `drivers.py`（DPG 駆動: `_PygletDriver` / `_ThreadDriver`）
  - `layout.py`（見出し/2 カラムテーブル/比率計算/行生成）
  - `widgets.py`（値型別ウィジェット生成: bool/enum/string/vector/int/float）
  - `theme.py`（既定スタイル/設定スタイル/色正規化 `to_dpg_color`）
  - `tags.py`（`ROOT_TAG`/`SCROLL_TAG`/`STAGE_TAG` 定数）
  - `headless.py`（DPG 未導入時の `ParameterWindow` スタブ）
  
注: 互換シム（`dpg_window.py` の再エクスポート）は作成しない。参照元は新パスへ更新する。

実施チェックリスト（順序）
- [ ] `dpg/` サブパッケージ作成（空ファイル群 + __init__ 雛形）
- [ ] `headless.py` にスタブ `ParameterWindow` を切り出し（import ガードはここに集約）
- [ ] `drivers.py` へ `_PygletDriver` / `_ThreadDriver` を移動（公開は最小）
- [ ] `tags.py` にタグ/定数集約（`ROOT_TAG`/`SCROLL_TAG`/`STAGE_TAG`）
- [ ] `theme.py` へ `_apply_default_styles`/`_apply_styles_from_config`/`_apply_colors_from_config`/`_to_dpg_color` を移動
- [ ] `layout.py` へ `_build_root_window`/`_label_value_ratio`/`_add_two_columns`/`_build_grouped_table`/`_create_row` の UI 構築ロジックを移動
- [ ] `widgets.py` へ `_create_bool/_enum/_string/_vector/_int/_float` を移動（必要データを引数化）
- [ ] `window.py` を再構成（購読/マウント/ドライバ選択・起動/可視切替/クリーンアップのみ保持）
- [ ] 参照元を新モジュールパスへ更新（下記「参照更新リスト」）
- [ ] `architecture.md` の記述を `dpg.window` 実体に同期
- [ ] 変更ファイルのみの Lint/Format/Type/Test を実行
  - [ ] `ruff check --fix {changed}`
  - [ ] `black {changed} && isort {changed}`
  - [ ] `mypy {changed}`
  - [ ] `pytest -q tests/ui/parameters/test_dpg_mount_smoke.py -m smoke -q`

非機能要件・成功条件
- [ ] 参照更新後の全テスト緑（ヘッドレス環境ではスタブ経路で失敗しない）
- [ ] `manager/controller` から見た寿命管理/表示切替/購読が従来通り
- [ ] `architecture.md` 同期済み

リスクと対策
- DPG/pyglet の環境差異: try-import ガードを `headless.py`/`drivers.py` に集約し、例外はログのみで継続。
- UI API 差異（古い DPG で列ストレッチ不可）: 現行のフォールバック分岐を `layout.py` に維持。
- 循環参照: DPG 以外の依存は `state` のみとし、他層（runtime 等）へは依存しない。

確認事項（要回答）
- [ ] `widgets.vector` のテーブル内セル padding テーマは維持で良いか（互換性のため既定 ON）
- [ ] `theme.colors` のキー集合（text/window_bg/frame_bg/...）の拡張要望有無

実行・検証コマンド（編集ファイル優先）
- 初期化: `python3.10 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e .[dev]`
- 単体チェック（編集ファイルのみ）:
  - `ruff check --fix {path}`
  - `black {path} && isort {path}`
  - `mypy {path}`
  - `pytest -q tests/ui/parameters/test_dpg_mount_smoke.py::test_mount_headless_ok -q`

メモ/今後の拡張（提案）
- `ParameterWindowBase` を `dpg/` に移し、将来の他 GUI 実装（例: Qt/PySide）追加に備える。
- `widgets` のコールバックを関数型に寄せ、`window` から DI できるようにしてテスト容易化。
- `theme` にスキーマ（許容キー/型）を追加し、設定の型エラーを開発時に早期検出。

現状ステータス
- 作業前（未着手）。本チェックリストで合意後、段階的に実施します。

参照更新リスト（互換シムなし）
- `src/engine/ui/parameters/controller.py:11`
  - 変更前: `from .dpg_window import ParameterWindow`
  - 変更後: `from .dpg.window import ParameterWindow`
- `tests/ui/parameters/test_dpg_mount_smoke.py:3`
  - 変更前: `from engine.ui.parameters.dpg_window import ParameterWindow`
  - 変更後: `from engine.ui.parameters.dpg.window import ParameterWindow`
- `architecture.md:61`
  - 変更前: 実体パスが `engine.ui.parameters.dpg_window`
  - 変更後: 実体パスを `engine.ui.parameters.dpg.window` に更新
- `docs/user_color_inputs.md:20`
  - 変更前: 実装参照に `src/engine/ui/parameters/dpg_window.py:662-689` を含む
  - 変更後: `src/engine/ui/parameters/dpg/theme.py` 等の新構成へ書き換え（行番号参照は削除/更新）
- `docs/user_color_inputs.md:75`
  - 変更前: `src/engine/ui/parameters/dpg_window.py:620-760`
  - 変更後: 新構成の `theme.py`/`layout.py` などへ分割参照（行番号は削除/更新）
- `docs/user_color_inputs.md:85`
  - 変更前: `src/engine/ui/parameters/dpg_window.py:700` 近傍
  - 変更後: 新構成の `theme.py:_to_dpg_color` へ移行（行番号は削除/更新）

備考（ドキュメント参照の方針）
- dpg_window.py の行番号参照は分割後に無効化されるため、概念参照（モジュール/関数名）へ置換する。
- 実装行番号を維持する場合は分割後に再採番するが、今後の保守性を考慮し極力行番号参照は避ける。
