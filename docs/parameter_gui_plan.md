# パラメータ GUI 実装計画

## ゴール
- `draw` 関数内で使用される Shape/Effekt 呼び出しの各パラメータを自動検出し UI に列挙する。
- 描画ウィンドウとは独立したパラメータ編集ウィンドウを提供し、水平スライダーを縦方向に並べてマウスで調整できるようにする。
- 既存の描画パイプラインに過剰な結合を生まず、フレームレートへの影響を最小に抑える。

## 前提・制約
- 既存の `api.G` / `api.E` の利用方法（ユーザコード側の `draw` 実装）は変更不要にする。
- GUI は既存依存の `pyglet` を用いて実装し、新規依存追加は避ける。
- MIDI (`cc`) 入力は現状通り生かしつつ、GUI からの入力を優先またはブレンドできる仕組みを用意する。
- 将来別 GUI 実装へ差し替えやすいよう、パラメータ検出ロジックと描画実装を分離する。

## アーキテクチャ概要
1. **パラメータ検出層 (`parameter_scan`)**
   - `ParameterTracer` コンテキストで `ShapesAPI` / `PipelineBuilder` の呼び出しをフックし、
     `draw` 実行時に現れた Shape / Effect 名と引数辞書を収集。
   - 各引数は「識別子（例: `G.sphere#0.subdivisions`）」「型・既定値」「元の評価値」「呼出し回数」を保持。
2. **メタデータ & 状態管理 (`parameter_state`)**
   - `ParameterDescriptor` / `ParameterValue` データクラスで GUI 表示用情報を正規化。
   - 値のソースを `original`（コード）と `override`（UI/MIDI）に分け、優先順位制御を提供。
   - GUI から更新された値は `ParameterStore` に反映し、`draw` 呼び出し時に override を適用。
3. **パラメータ GUI (`parameter_panel`)**
   - `pyglet.window.Window` を継承した `ParameterWindow` を新設。
   - スライダー描画は軽量な自前描画（矩形 + ドラッグ）か `pyglet.gui` を評価し、操作レスポンスを最優先。
   - 各スライダーは `[min, max]` 範囲・補助ラベルを表示。範囲はヒューリスティック（デフォルト ± スケール、型別既定など）で決定し、ユーザ調整も許容。
4. **実行フロー統合 (`api.sketch` 拡張)**
   - `run` 時に `draw` をラップする `ParameterizedDraw` を注入。
   - 初回実行で `ParameterTracer` によりスキャンし、GUI スレッドを起動。
   - フレーム毎に `ParameterStore.snapshot()` を `cc` とマージし、`draw` へ渡す値を決定。
   - GUI ウィンドウの Tick をメインループに統合しつつ、レンダリングループへの影響を避けるためイベント処理を独立させる。

## パラメータウィンドウ詳細設計

- **主要クラス構成**
  - `ParameterWindow(pyglet.window.Window)`
    - レイアウト管理と描画、入力（マウス・スクロール）イベント処理を担当。
    - 内部に `ParameterPanel` を保持し、描画時に委譲する。
    - `attach_store(store: ParameterStore)` で監視対象を差し替え可能にする。
    - `on_draw`/`on_resize`/`on_mouse_press`/`on_mouse_drag`/`on_mouse_release`/`on_mouse_scroll` をオーバーライドし、`ParameterPanel` へイベントを渡す。
  - `ParameterPanel`
    - スライダー群の並べ替えと範囲計算を担当する薄いコンテナ。
    - `update_descriptors(list[ParameterDescriptor])` で項目一覧を更新し、`draw()` で各ウィジェットを描画。
    - `hit_test(x, y)` / `on_drag(dx)` を提供し、ウィンドウからのイベントを処理。
    - `widgets_by_id` / `ordered_widgets` を保持し、差分更新時に再生成コストを最小化。
  - `SliderWidget`
    - 単一パラメータの UI 表現。内部ステートとして `value`, `min_value`, `max_value`, `step` を保持。
    - 描画では背景バー・現在値ラベル・ホバー表示を担当。
    - `from_descriptor(ParameterDescriptor, ParameterValue)` で生成し、値更新を `ParameterStore` に反映。
  - `ParameterStore`
    - GUI 側 override と MIDI/コード入力をマージする権威。既存セクションの実装タスクと一致。
    - `subscribe(callback)` を介し変更通知を `ParameterWindow` へ送る（push 型で再描画最小化）。

- **描画・イベントループ設計**
  - `ParameterWindow` は `pyglet.clock.schedule_interval` で 30Hz 程度の再描画を行い、変更通知時は即再描画を要求。
  - ウィンドウサイズ変更に追従し、`ParameterPanel.layout(width, height)` を再計算。
  - マウスドラッグ（左クリック）でスライダー値を更新し、`ParameterStore.set_override(param_id, value)` を呼ぶ。
  - マウスホイールで微調整、ダブルクリックで既定値へリセットするショートカットを提供。

- **ライフサイクルと疎結合の確保**
  - `run_sketch` からは `ParameterWindowController`（軽量 Facade）を経由して GUI を管理。
    - `ParameterWindowController.start()` がウィンドウ生成とスケジューリングを行い、メインループ終了時に `close()`。
    - `ParameterWindowController.apply_overrides(cc_snapshot)` がフレーム毎に呼ばれ、GUI 値と MIDI 値のマージ済み辞書を返す。
  - GUI 側は `ParameterDescriptor` の変更（新しい shape/effect の登場）を検出すると `ParameterPanel` を再構築。
  - GUI スレッドは `pyglet` メインスレッドと共有（`pyglet` のシングルスレッド制約）。パラメータストア更新はスレッド安全なキュー経由で受け取る。

- **パフォーマンス考慮**
  - スライダーの描画は `pyglet.graphics.Batch` を活用し、静的パーツ（背景・ラベル）はキャッシュして更新時だけ再構築。
  - 値更新は浮動小数点→文字列のフォーマットを事前に関数化し、GC 負荷を抑制。
  - `ParameterStore` 側は差分更新（値が変わったときのみ通知）を保証し、無変更フレームの再描画を防ぐ。

- **拡張ポイント**
  - `SliderWidget` をインターフェイス化し、将来的にトグル/カラー等の別ウィジェットを追加可能にする。
  - `ParameterWindowController` にプリセット保存・読込 API を追加する余地を確保（現段階では非実装）。

## コンポーネント仕様詳細

### parameter_scan
- `ParameterTracer`
  - `__enter__` で `G`/`E.pipeline` にパッチを当て、`__exit__` で原状復帰。
  - `trace_shapes(fn)` / `trace_effects(pipeline_builder)` で個別に呼び出しを包み、`CallSnapshot` を生成。
  - `wrap_shape(name, fn)` / `wrap_effect(name, fn)` でラップ関数を生成し、呼び出し毎に `CallRecord` をバッファへ push。
  - `enable_lazy(lazy=True)` オプションで初回フレームのみトレースと常時トレースを切り替え。
  - 具体的には `ShapesAPI._build_shape_method` と `PipelineBuilder.__getattr__` を monkey patch し、戻り値をトレース付きクロージャへ差し替える。
- `CallSnapshot`
  - `id: str`（`{kind}.{name}#{sequence}.{param}`）
  - `source: Literal["shape", "effect"]`
  - `name: str`
  - `params: dict[str, Any]`
  - `defaults: dict[str, Any]`（シグネチャから抽出）
  - `frame_index: int`（出現順）
  - `occurrence: int`（同フレーム内出現回数）
- `ParameterRegistryBuffer`
  - スレッドセーフな `SimpleQueue` を用い、トレース結果をメインスレッドへ渡す。
  - `drain()` で `CallSnapshot` のバッチを返し、`ParameterDescriptorFactory` が処理。
- 既定値解析
  - `inspect.signature` を用いて呼び出し前にデフォルト・型注釈を取得。
  - `Enum` や `Sequence` の判定は型ヒントと値からヒューリスティックに行い、GUI の部品選択に活用。
  - 呼び出し中に再帰する `G`/`E` 呼び出しはスタック管理し、`context_id` を付与して無限ループを回避。
- キャッシュ/制限
  - 同一フレームで同名シグネチャが連続した場合は `occurrence` のみ増加し、Descriptor 生成処理は 1 度に限定。
  - `ParameterTracer` は `max_records`（既定 128）を超えた場合に古い記録を破棄し、ログで通知。
  - `lazy=True` の際は初回フレームでトレース停止し、以後はレジストリキャッシュのみ使用。

### parameter_state
- `ParameterDescriptor`
  - `id`, `label`, `source`, `category`（`shape`/`effect`/`global`）
  - `value_type: Literal["float", "int", "bool", "enum", "vector"]`
  - `range_hint: RangeHint | None`（`min`, `max`, `step`, `scale`）
  - `default_value`
  - `help_text: str | None`（docstring やヒューリスティックから抽出）。
- `ParameterDescriptorFactory`
  - `from_snapshot(call_snapshot)` で Descriptor 群を生成。
  - Docstring 取得には `inspect.getdoc(fn)` を用い、1 行目を `help_text` として利用。
- `ParameterValue`
  - `original: Any`
  - `override: Any | None`
  - `timestamp: float`
- `ParameterStore`
  - `set_override(id, value)` → 値検証後に `override` を更新し通知。
  - `clear_override(id)` → `override=None`。
  - `snapshot(cc_snapshot)` → GUI override を適用した辞書を返却。
  - `subscribe(cb)` / `unsubscribe(cb)` → push 通知。
  - `register(descriptor, value)` で初期値を保存し、GUI への通知を発火。
- `OverrideResult`
  - `value: Any`, `clamped: bool`, `source: Literal["gui", "midi"]` を保持。
  - GUI へ返却し、ラベル色や警告表示を制御。
- `ParameterLayoutConfig`
  - `row_height`, `padding`, `font_size`, `value_precision`, `default_range_multiplier` を保持。
  - `derive_range(descriptor, observed_value)` が範囲推定（float: `±|value|*multiplier + epsilon`, int: `±max(1, value*multiplier)`）。
- 優先順位
  1. GUI override（`override is not None`）
  2. MIDI `cc` 入力（`cc_override_map`）
  3. `draw` 元コードの `original`
- ベクトル引数（例: `angles_rad`）は `ParameterDescriptor` を配列要素ごとに分割した派生 id (`angles_rad.x`) として扱う。
  - 各派生 id の `category` は親と共有し、`vector_group` 属性でグルーピング。

### parameter_panel
- レイアウト
  - 1 行は `SliderWidget` + ラベル + 現在値ラベル。
  - パディングや行高は `WINDOW_PADDING=8`, `ROW_HEIGHT=28` を定数化。
  - ウィンドウ高より項目数が多い場合、スクロールバーを描画し、ホイール操作で `scroll_offset` を更新。
- ウィジェット
  - `SliderWidget` は `value` を 0-1 の正規化値に変換してドラッグ処理。
  - `EnumWidget`（拡張余地）を別クラスとして定義し、`ParameterDescriptor.value_type == "enum"` の場合に使用。
  - 100 件以上のパラメータを想定し、`visible_rows` のみ描画する仮想スクロール方式を実装。
  - 仮想スクロールでは `first_visible_index` と `scroll_offset_px` を保持し、`layout()` 実行時に更新。
- 文字描画
  - `pyglet.text.Label` を利用し、Batch への登録と `document` 再利用で GC を抑制。
  - 値ラベルは `value_precision` 設定で丸め、`enum` は `DropdownWidget`（必要時）を利用。
  - ヘルプテキストは `ParameterDescriptor.help_text` が存在する場合、ホバー時にツールチップ（半透明パネル）として表示。

### parameter_window_controller
- `ParameterWindowController`
  - `__init__(store: ParameterStore, layout: ParameterLayoutConfig)`
  - `start()` で `ParameterWindow` を生成し、`pyglet.app` のイベントループにウィンドウを追加。
  - `tick(dt)` を `FrameClock` から呼び出し、GUI 側の pending イベントを処理。
  - `apply_overrides(cc_snapshot)` が `store.snapshot(cc_snapshot)` を呼び、辞書を返す。
  - `shutdown()` でウィンドウクローズとフック解除。
  - `set_visibility(visible: bool)` を提供し、パラメータなし時にウィンドウを隠す。
- マルチウィンドウ管理
  - 既に GUI が生成済みの場合はフォーカスを与える。
  - メインウィンドウ終了時に `shutdown()` を必ず呼ぶため、`atexit` ハンドラと `pyglet.app.event_loop.exit_funcs` を利用。
  - バックグラウンドワーカーからのイベントは `queue.SimpleQueue` で受け取り、`pyglet.app.platform_event_loop.post_event` でメインスレッドへ再配送。

### MIDI ブレンド仕様
- `cc` 値は 0.0-1.0 の正規化値とし、`ParameterDescriptor` の `range_hint` を用いて実値へ変換。
- GUI override が存在しない場合のみ `cc` を適用。
- ベクトルパラメータに対しては、`cc` のキー範囲を `[base_index, base_index + n)` としてマッピングし、ルールをプランに明記。
- 将来的なブレンド（加算/乗算）に備えて `ParameterStore` 内部で `blend_strategy` を設定可能にするが初期実装では `replace` 固定。

### エラーハンドリング / フォールバック
- `ParameterTracer` のパッチ適用に失敗した場合は警告ログを残し、GUI 機能を無効化して既存挙動を維持。
- GUI 初期化で `pyglet` が例外を投げた場合は、`--no-parameter-gui` と等価な動作に切り替え、`ParameterStore` の状態のみ保持。
- GUI 未サポートパラメータ（関数や巨大配列など）は `ParameterDescriptor.supported=False` とし、ウィンドウには表示せずログに詳細を出力。
- `ParameterStore.set_override` に範囲外値が指定された場合はクリップし、`OverrideResult` に `clamped=True` を含めて caller（GUI）に返す。
- ハンドリング不能な例外は `ParameterWindowController` がキャッチし、ユーザへ HUD 通知（右上メッセージ）を出して GUI を閉じる。

### データフロー / イベントシーケンス
1. `run_sketch` が `use_parameter_gui=True` なら `ParameterWindowController` を生成。
2. 初回 `user_draw` 呼び出し前に `ParameterTracer` を起動し、`G`/`E` 呼び出しをラップ。
3. `user_draw` 実行で得た `CallSnapshot` 群を `ParameterDescriptorFactory` が解析し、`ParameterStore.register(descriptor)` を行う。
4. `ParameterWindowController.start()` が `ParameterWindow` を生成し、`ParameterPanel.update_descriptors()` に一覧を渡す。
5. メインループ毎に `ParameterStore.snapshot(cc_values)` を呼び、GUI/MIDI をマージしたマップを `user_draw` に渡す。
6. GUI 操作（ドラッグ/ホイール）で `SliderWidget` が `ParameterStore.set_override()` を呼び、差分通知でウィンドウを再描画。
7. メインウィンドウ終了時、`shutdown()` が呼ばれ GUI ウィンドウも連動して閉じる。

簡易シーケンス図:
```
run_sketch ─┐
            │ frame loop
            ▼
  ParameterWindowController ─▶ ParameterWindow ─▶ SliderWidget
            │                        ▲                 │
            │ snapshot()             │ value change     │ set_override()
            ▼                        │                 │
        ParameterStore ◀─────────────┘                 │
            │ merged values                               │
            └────────────▶ user_draw ▶ G/E ▶ ParameterTracer
```

### モジュール構成（提案）
- `src/engine/ui/parameters/__init__.py`
- `src/engine/ui/parameters/tracer.py`
- `src/engine/ui/parameters/state.py`
- `src/engine/ui/parameters/panel.py`
- `src/engine/ui/parameters/window.py`
- `src/engine/ui/parameters/controller.py`
- テスト
  - `tests/ui/parameters/test_tracer.py`
  - `tests/ui/parameters/test_state.py`
  - `tests/ui/parameters/test_panel.py`
  - `tests/ui/parameters/test_integration.py`

## 実装タスク
- parameter_runtime
  - [x] ShapesAPI/Pipeline フックでの実パラメータ捕捉と override 適用。
  - [x] `ParameterRuntime` によるベクタ分解とドキュメントキャッシュ実装。
  - [ ] `lazy`/`max_records` 設定を CLI 引数・設定ファイルから受け取る仕組みを追加。
- parameter_state
  - [x] `ParameterDescriptor`/`ParameterValue` データクラス定義とバリデーション関数実装。
  - [x] `ParameterStore` の CRUD/通知 API 実装。
  - [x] ベクトル分解ロジックの単体テスト追加（`tests/ui/parameters/test_runtime.py`）。
  - [x] `ParameterLayoutConfig` の範囲推定アルゴリズムを実装し、代表ケースのテストを追加。
- parameter_panel
  - [x] `ParameterPanel` のレイアウト計算・スクロール処理実装。
  - [x] `SliderWidget` の入力処理・描画を実装。
  - [x] `ParameterWindow` イベントハンドラ（resize/draw/mouse/scroll）実装。
  - [ ] ツールチップ表示や仮想スクロールの描画最適化テストを追加。
- controller 統合
  - [x] `ParameterWindowController` の起動/停止 API 実装。
  - [x] `run_sketch` 内での初期化フロー拡張、`use_parameter_gui` フラグ追加。
  - [ ] GUI 生成失敗時のフォールバックを確認するテストケースを追加。
- ドキュメント/テスト
  - [x] `architecture.md` に新コンポーネントの記述を追記。
  - [ ] 必要に応じた ADR の検討・記録（今回は未着手）。
  - [ ] `tests/ui/test_parameter_window.py` で描画コンポーネントの smoke テスト作成。
  - [x] `tests/ui/parameters/test_parameter_store.py` / `test_runtime.py` を追加。
  - [x] `AGENTS.md` に高速チェック手順（`pytest -q tests/ui/parameters`）を追記。

## 意思決定サマリ
- スライダー範囲: `ParameterLayoutConfig.derive_range` の方針を採用。float は `default ± max(|default|, 1.0)`、int は `default ± max(|default|, 1)` を基準にし、名称に `scale`/`subdiv`/`freq` を含む場合は下限 0。デフォルトが 0 の float は `[0, 1]` を初期レンジとする。bool はトグル、enum はプルダウンを用いる。
- トレースオーバーヘッド: 既定で `lazy=True`（初回フレームのみトレース）とし、必要に応じて CLI/設定で常時トレースへ切り替える。
- GUI ライフサイクル: `ParameterWindowController` がメインウィンドウ終了時に自動終了し、パラメータが検出できなかった場合はウィンドウを生成せずログ通知のみ行う。
- 大量パラメータ対応: 仮想スクロールとツールチップにより 100 件超でも負荷を抑制。ズーム操作はホイール+Shift で 0.1 倍、ホイール+Ctrl/Cmd で 10 倍変化量に切り替える。
- 並列実行: パラメータ GUI 有効時は `InlineWorker` モード（単一プロセス）で実行し、monkey patch の伝播を明示的に制御。通常モードでは従来通りマルチプロセス。
- override 優先順位: GUI → MIDI → コードの順で固定し、`blend_strategy` は内部的に `replace` のみを実装。設定オプションは公開しない。
- プリセット機能: 初期実装では非対応。将来的な拡張余白のみ `ParameterWindowController` に確保。
- 追加ショートカット: ダブルクリックでデフォルト値へリセット、Shift+ドラッグで 0.1 倍、Ctrl/Cmd+ドラッグで 10 倍の操作感度を導入。
- スレッド/プロセス安全性: `ParameterTracer` は GUI 有効時のみ適用し、InlineWorker 内で同期するため追加のスレッドローカルは不要。
- パラメータ数ゼロ時の挙動: GUI を非表示にし、HUD ログと標準出力に通知して操作対象が無いことを示す。
