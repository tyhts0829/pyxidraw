# DPG ParameterWindow 復元計画（詳細コントロール × シンプル実装）

- 目的: 旧来の詳細 UI（Display/HUD/カテゴリ別テーブル/各型ウィジェット/Store 双方向同期）を、現在の「シンプルでクリーンな」実装・依存方針（ローカル import、スタブ撤去）に合わせて復元する。
- 非目標: 新機能の追加（レイアウト大幅刷新/高度テーマ）。まずは実装の復元と安定化に限定。

## 方針（依存/構造）
- Dear PyGui は実行時必須。import 失敗時は GUI 機能を使わない限り影響なし（Controller 側を遅延 import）。
- pyglet はローカル import に統一（あれば `clock.schedule_interval`、無ければスレッド駆動）。
- `auto_show` オプションを維持（テストで窓を開かないため）。
- コードは関数粒度で小分割（Display/HUD/テーブル生成/型ごとウィジェット/購読ハンドラ）。

## タスク（チェックリスト）

1) ルート/スクロール/プライマリ設定の復元（最小）
- [x] `ROOT_TAG`/`SCROLL_TAG` を用意し、`_build_root_window()` でルート+スクロール領域を生成
- [x] `dpg.set_primary_window(root, True)` を適用

2) Display/HUD セクションの復元
- [x] `build_display_controls(parent, store)` を復元（背景/線色 → ColorEdit、HUD: text/meter/meter_bg）
- [x] `util.color.normalize_color` を用いた 0..1 / hex 正規化 → 0..255 表示
- [x] `store_rgb01(pid, app_data)` で Store override を保存（RGBA 0..1）
- [x] 起動直後の `sync_display_from_store()`（Store→GUI 初期反映）

3) カテゴリ別テーブル/行生成の復元
- [x] `_build_grouped_table(parent, descriptors)`：runner.* を除外、カテゴリ毎に Header→2カラムテーブル
- [x] `_label_value_ratio()` と `_add_two_columns(left, right)` の比率/フォールバック
- [x] `_create_row(table, desc)`：左ラベル/右ウィジェット

4) 各型ウィジェットの復元（シンプル化込み）
- [x] bool/int/float/enum/string/vector の各 `_create_*`
- [x] vector は `id::{axis}` の子スライダ、VectorRangeHint 未設定時はレイアウトから既定レンジ
- [x] コールバック `_on_widget_change` で Store 側に反映（vector は部分更新対応）

5) Store→GUI の双方向同期
- [x] `_on_store_change(ids)` で差分更新（scalar: `set_value(id, …)`、vector: `id::{axis}`）
- [x] `ParameterManager.initialize()` 後の購読を継続（実装済み）

6) テーマ/スタイル（最小）
- [x] `_setup_theme()` を復元（config→style/colors、失敗は既定スタイル）
- [x] 既定スタイルは padding 程度に留める（簡素）

7) ドライバと可視制御
- [x] `auto_show=True` 時に `show_viewport()` と driver 起動（pyglet 優先→thread fallback）
- [x] `set_visible(True/False)` で driver start/stop 連動、`close()` で確実に停止

8) テスト/動作確認（対象限定）
- [x] ruff/black/isort/mypy（`src/engine/ui/parameters/dpg_window.py`）
- [ ] `pytest -q tests/ui/parameters`（Dear PyGui 導入時、`auto_show=False` スモークも確認）
- [x] `pytest -q -m smoke`（全体の回帰確認）
- [ ] `python main.py`（Dear PyGui 導入環境で Display/HUD が見える）

## 受け入れ基準（DoD）
- [ ] ウィンドウに Display/HUD セクションとカテゴリ別パラメータが表示され、操作で Store に反映される
- [ ] Store 側の変更が GUI に同期される（Display/HUD/ベクトルも含む）
- [ ] pyglet 有無に関わらずフレームが進む（pyglet or thread）
- [ ] 変更ファイル限定の Lint/Type/Test 緑、smoke 緑

## 備考
- 旧実装のロジック構造は維持しつつ、import 方針（ローカル化）と冗長な防御を軽量化する。
- UI 詳細（ツールチップ/ハイライト/外観の細かい差）は段階的復帰とし、まずは機能の復元を優先。
