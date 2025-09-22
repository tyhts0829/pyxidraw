# Parameters GUI: Dear PyGui 導入計画（詳細）

目的

- 既存の `pyglet` ベース UI を置き換え/併存し、見た目と操作性を向上しつつコード量を削減する。
- 既存の `ParameterStore / ParameterDescriptor / RangeHint` をそのまま利用し、UI 実装のみを差し替える。

採用ライブラリ / 前提

- Dear PyGui（即時モード GUI, GPU アクセラレーション, 豊富なウィジェット, テーマ/ドッキング）
- 前提: dearpygui は導入済み（ネットワーク操作は不要）

設計方針

- UI バックエンドを差し替え可能にする薄い抽象（インタフェース）を導入。
- `ParameterStore` の購読/更新を唯一のデータフローとし、UI は stateless に近い設計（即時モードに適合）。
- 既存 `pyglet` 実装は当面温存し、環境変数/設定で DPG と切替（段階移行）。

ファイル構成（予定）

- `src/engine/ui/parameters/dpg_backend.py`（新規）
  - DpgParameterWindow（DPG 実装のウィンドウ）
  - Descriptor→ ウィジェット生成マッピング、イベントハンドラ、テーマ適用
- `src/engine/ui/parameters/window.py`（更新）
  - バックエンド選択ロジック（env `PXD_UI_BACKEND=dpg|pyglet|stub`）
- `src/engine/ui/parameters/__init__.py`（必要ならエクスポート追加）
- `tests/ui/parameters/test_dpg_backend.py`（最小の生成/購読テスト、headless で import/構築のみ）

機能要件（マッピング）

- float/int
  - DPG: `add_slider_float` / `add_slider_int`
  - `RangeHint.min/max/step` と `ParameterDescriptor.default_value` を反映
  - 値変更 →`store.set_override(id, value)`
- bool
  - DPG: `add_checkbox`（テーマでトグル風にも）
- enum
  - 選択肢<=5: `add_radio_button(items=choices)`
  - 選択肢>5: `add_combo(items=choices)`
  - 値変更 →`store.set_override`
- vector（x/y/z/w）
  - 既存はコンポーネント分割済み（`vector_group` 付き Descriptor）
  - DPG: 横並び配置（`add_slider_float` を 3/4 個）または `add_input_float3/4`
  - ラベルはグループ見出しのみ、各コンポは短ラベル（x/y/z/w）
- グルーピング
  - `collapsing_header` で `scope.name#index` 見出し（例: `affine#0`）
  - 見出し下に各 param 行（上: ラベル, 下: ウィジェット）
- 変更マーカー/Reset
  - 値が `default_value` と異なる場合にラベル色/● ドット付与
  - 右クリックメニュー: Reset（`store.clear_override` or `set_override(default)`）
- ツールチップ
  - `help_text`（Descriptor）を hover で表示
- ショートカット/操作性
  - `←/→` で enum の移動、`1..9` で快速選択（Key handler）
  - フィルタ入力（上部に検索ボックス）で param を絞り込み（後続）
- テーマ/見た目
  - 暗色テーマ（基本色/アクセント、角丸、パディング、フォント）
  - Enum の radio/コンボのコントラスト調整

イベント/データフロー

- 描画: 起動時に Descriptor リストを走査してウィジェット生成
- 変更通知: `ParameterStore.subscribe` で UI へ反映（`set_value`）
- 入力: DPG コールバック →`store.set_override` へ集約
- 更新間引き: UI 側は値差分のみ `set_value` して無限ループを回避

切替戦略

- `src/engine/ui/parameters/window.py`
  - 現状: `pyglet` or headless stub
  - 追加: `if os.getenv("PXD_UI_BACKEND") == "dpg": from .dpg_backend import DpgParameterWindow as ParameterWindow`
  - 既定は現状維持（pyglet）→ 移行期のリスク低減

実装チェックリスト（段階導入）

1) バックエンド切替の土台
- [ ] `src/engine/ui/parameters/dpg_backend.py` を新規作成（空の DpgParameterWindow 骨格: `__init__/set_visible/close`）
- [ ] `src/engine/ui/parameters/window.py` にバックエンド選択を追加
  - [ ] `PXD_UI_BACKEND` を参照し `dpg` の場合のみ DPG を import
  - [ ] ImportError 時は安全に既存（pyglet or stub）へフォールバック
- [ ] 既存コードの動作確認（未設定時は従来 UI 起動）

2) 最小ウィジェット生成（float/int/bool/enum）
- [ ] `DpgParameterWindow` に `mount(descriptors: list[ParameterDescriptor])` を実装
- [ ] グルーピング: `collapsing_header` で `scope.name#index` を見出しに
- [ ] float → `add_slider_float`（min/max/step を RangeHint から反映）
- [ ] int → `add_slider_int`（step=1）
- [ ] bool → `add_checkbox`（見た目は後でテーマ調整）
- [ ] enum（choices あり）
  - [ ] 候補数 <= 5 → `add_radio_button`
  - [ ] 候補数 > 5 → `add_combo`
- [ ] 生成と同時に `store.set_override(..., value=default)` は行わない（現在値は Store から取得）

3) Store 連携（単方向/双方向）
- [ ] UI → Store: すべてのウィジェット `callback` で `store.set_override(id, value)` を呼ぶ
- [ ] Store → UI: `store.subscribe` で `set_value(widget_id, value)` を反映（差分のみ）
- [ ] `param_id -> dpg_item_id` のマップを持つ（辞書）

4) 変更マーカー/Reset/ツールチップ
- [ ] `default_value` と比較し、差分があればラベルに●（または色）を付与
- [ ] 右クリックで Reset（`set_override(default)`）を提供（DPG の context menu）
- [ ] `help_text` があれば `set_tooltip` を設定

5) レイアウト/テーマ（最小）
- [ ] ダークテーマとフォント/角丸/パディングの適用
- [ ] enum の radio/コンボのコントラスト調整（選択時の視認性）
- [ ] vector は当面“分割表示のまま”（横並びは後続）

6) キー操作（最小）
- [ ] enum に ←/→ で前後移動（radio でのインデックス操作 / combo の選択変更）
- [ ] 数字キー 1..9 で上から n 番目を選択（範囲外は無視）

7) ベクトル/複合（拡張）
- [ ] `vector_group` を使い x/y/z/w を 1 行横並びで表現（`add_input_float3/4` or 横並びスライダ）
- [ ] 軸ラベル（x/y/z/w）を小さく表示

8) 仕上げ/安定化
- [ ] 大量パラメータ時の更新間引き（同値更新回避）
- [ ] ラベルの長文折返し/ツールチップ化
- [ ] 例外ガード（DPG 例外時も落ちない）

9) テスト/検証
- [ ] headless でも import/生成→破棄が行える簡易スモーク（DPG 未導入環境では skip）
- [ ] 代表パラメータ（float/int/bool/enum）の round-trip（UI→Store→UI）が成立する
- [ ] `PXD_UI_BACKEND=dpg` と未設定（既存 UI）の両方で起動確認

10) ドキュメント/整備
- [ ] README/AGENTS に切替方法を 2-3 行追記（任意）
- [ ] 変更ファイル限定の `ruff/black/isort` を実行

変更対象一覧

- 追加: `src/engine/ui/parameters/dpg_backend.py`
- 更新: `src/engine/ui/parameters/window.py`（バックエンド切替）
- 追加（任意）: `tests/ui/parameters/test_dpg_backend.py`
- 変更なし: `ParameterStore/ValueResolver/Introspector`（既存 API 利用）

リスクと緩和

- 依存追加（dearpygui）が重い: 段階導入と環境変数での opt-in。CI では import-guard で回避
- 二重実装の保守: バックエンド抽象を薄くして重複ロジックを最小化
- パフォーマンス: 大量パラメータ時の create/update をバッチ化（DPG は高速だが更新間引きは実装）

受け入れ条件（DoD）

- 既存機能（float/int/bool/enum/vector/グループ/Reset/ツールチップ）が DPG で再現
- `PXD_UI_BACKEND=dpg` で新 UI が起動し、未設定では既存 UI のまま
- 変更ファイルに対する `ruff/black/isort` 緑
- headless/CI でも import 失敗しない（guard 済み）

運用/導入手順（ローカル）

- 起動: `PXD_UI_BACKEND=dpg python main.py`
- 既存 UI に戻す: `PXD_UI_BACKEND=pyglet python main.py`（既定）

メモ（後続拡張）

- Docking で「プロパティパネル」+「ライブプレビュー」2 ペイン構成
- プリセット保存/読込、フィルタ検索、キーバインドカスタム
- テーマスイッチ（ライト/ダーク）

---

この計画で実装に進めてよいか確認してください。依存追加（dearpygui）の承認後、ステップ 1 から着手します。
