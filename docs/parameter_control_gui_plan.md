# Parameter GUI 実装計画

目的: `draw` 内の全 shape/effect 引数を自動検出し、横向きスライダー群で操作できる独立ウィンドウを追加する。

スコープ: `main.py` が呼ぶ `run()` パスにパラメータ GUI を統合し、ランタイムで値を調整→再描画できるようにする。
非スコープ: MIDI デバイス固有の UI、外部設定ファイル保存、既存描画ロジックの大幅改変。

## タスクチェックリスト
- [ ] パラメータ呼び出しキャプチャ層を実装する（`ShapesAPI`/`PipelineBuilder` をラップし、呼び出し回数・引数値・メタ情報を収集）。
- [ ] 収集データを保持・更新する `ParameterModel`（ID 付与、型/範囲推定、現在値管理、GUI からの更新伝播）を設計・実装。
- [ ] `pyglet` ベースのコントロールウィンドウを新設し、横向きスライダーを縦並びでレンダリング＋マウスドラッグで値更新できるようにする。
- [ ] `run_sketch` のワーカー/レンダリングサイクルに `ParameterModel` を接続し、GUI 更新時に描画フレームへ新しいパラメータを反映させる再評価経路を構築。
- [ ] MIDI CC 入力と GUI パラメータが競合しないよう調停（優先順位・ブレンド戦略・無効化オプション）を決めて実装。
- [ ] キャプチャ結果の再利用やデバウンスでリレンダリング頻度を抑え、パフォーマンス劣化を防ぐ仕組みを追加。
- [ ] 主要コンポーネントの単体テスト（キャプチャ層・ParameterModel）と最小統合テスト（GUI なし環境向けスタブ）を追加。
- [ ] `architecture.md` と利用ドキュメントに新 GUI の概要・制約・操作手順を追記。

## 検討が必要な論点
- `tkinter` ではなく `pyglet` を使い同一イベントループで2窓を管理する案の実装詳細。
- 形状パラメータに `__param_meta__` が未整備な場合のスライダー範囲ヒューリスティック。
- 動的に変化する呼び出し（ループ内で可変ステップ数）の扱いをどう安定化するか。

## アーキテクチャ指針
- **イベントループ統合**: 既存の `pyglet` メインループにコントロールウィンドウを追加し、描画ウィンドウと同一スレッドでメッセージポンプを共有する。`pyglet.app.run()` の呼び出し箇所は既存フローを流用し、追加ウィンドウは `pyglet.window.Window(visible=True)` で生成。
- **パラメータ評価経路**: `run_sketch` のワーカーが `user_draw` を呼ぶ前に `ParameterModel` から最新値を注入。GUI 変更→モデル更新→`SwapBuffer.request_invalidate()` → 次フレームで `user_draw` 再評価、のパイプラインを構築。
- **競合解決**: MIDI からの CC 値と GUI スライダーの値は `ParameterBlender` で合成。既定は GUI 優先、オプションで「加算」「乗算」「無効化」を選択できる拡張ポイントを残す。
- **拡張性**: Shape/Effect 呼び出しキャプチャは `ParamIntrospector` に切り出し。将来の外部エディタ連携を想定し、モデルはシリアライズ可能な dict 表現を返せるようにする。

## クラス設計（案）
- `ParamIntrospector`
  - 役割: `ShapesAPI` / `EffectsAPI` 呼び出しをプロキシ化し、関数名・引数・メタ情報を収集。
  - 主API: `wrap_shapes_api(G) -> ShapesAPI`, `wrap_pipeline_builder(E.pipeline) -> PipelineBuilder`。
  - 内部: `inspect.signature()` と `__param_meta__` から型/範囲を推定。呼び出しごとに一意の `ParameterKey` を発行。

- `ParameterModel`
  - 役割: パラメータ群の単一ソース。現在値・既定値・レンジ・ステップを保持。
  - 主API: `get_value(key)`, `set_override(key, value)`, `iter_controls()`。
  - 内部: `ParameterEntry(dataclass)` に `key`, `label`, `value`, `default`, `min`, `max`, `step`, `source` を保持。スレッド安全性のため `RLock` で保護。

- `ParameterBlender`
  - 役割: MIDI 値と GUI 値のマージポリシーを提供。
  - 主API: `blend(key, midi_value, gui_value) -> float`。
  - ポリシー: `PriorityPolicy(Enum)`（`GUI_DOMINANT`, `MIDI_DOMINANT`, `ADD`, `MULTIPLY`）。設定は `ParameterModel` から参照。

- `ParameterWindow`
  - 役割: コントロール専用 `pyglet.window.Window`。縦方向にスクロール可能なスライダーリストを描画。
  - 主API: `bind_model(ParameterModel)`, イベントハンドラ `on_draw`, `on_mouse_drag`, `on_mouse_scroll`。
  - 描画: Pyglet の `Batch` と `VertexList` を使い、各コントロールを横長のバー + ハンドルで表現。ラベルは `pyglet.text.Label`。
  - 入力: マウスドラッグでバーハンドルを移動し、`ParameterModel.set_override()` を呼ぶ。

- `ParameterController`
  - 役割: 全体の司令塔。モデル・ウィンドウ・ブレンダーを束ね、`run_sketch` とワーカー側のフックを提供。
  - 主API: `inject_into(draw_fn) -> WrappedDraw`, `tick(dt)`（ウィンドウ更新用）、`teardown()`。
  - 内部: `Queue` で GUI→ワーカー通知、`threading.Event` で無駄な再評価を抑制。

- `DrawContext`
  - 役割: `draw` 実行時の文脈をカプセル化し、パラメータ参照を一元化。
  - 主API: `resolve_param(key, fallback)`, `as_mapping()`（`cc` 相当を再構築）。`user_draw` に渡す `cc` マップを GUI 値に置き換える仕組みを提供。

## コントロールレイアウト
- ウィンドウ幅: 480px 固定（設定可）。
- 各スライダー: 高さ 32px、ラベル部 160px、バー部 280px。
- スクロール: 10 個以上のコントロールで縦スクロールバーを表示。
- 色テーマ: 背景 #1E1E1E、バー #333、ハイライト #5AA9FF、文字 #EEE（可視性重視）。

## データフロー
1. `run()` 起動時、`ParameterController` が `E`/`G` を差し替えて `draw` を初回評価→キャプチャ。
2. 収集済みパラメータで `ParameterModel` を初期化し、GUI にコントロールを生成。
3. GUI で値変更が発生 → `ParameterModel.set_override()` → イベントキューに `ParameterUpdate` を投入。
4. ワーカーはキューを監視し、更新があるフレームで `user_draw` を再実行して新しい `Geometry` を生成。
5. MIDI 入力がある場合は `ParameterBlender.blend` を通じて最終値を決定し、`DrawContext` が `cc` マップとして `user_draw` に渡す。
6. 描画ウィンドウは `SwapBuffer` から結果を取得し、通常通りレンダリング。

## スレッド・同期戦略
- GUI・レンダリング: メインスレッドで `pyglet` イベントループを共有。
- ワーカー: 既存の `WorkerPool`（マルチプロセス/スレッド）を維持。`ParameterController` はワーカー共有メモリに触れない。
- 同期プリミティブ: `ParameterModel` 内部で `RLock`、GUI からの更新通知に `queue.Queue`、ワーカー側デバウンスに `threading.Event` を使用。
- レイテンシ最小化: 大量更新時は最後の値のみワーカーへ渡すフェンス処理を行う。

## 例外処理・フェイルセーフ
- パラメータ推定に失敗（範囲未決定）の場合は `min=0`, `max=1`, `step=0.01` のデフォルトを適用し、GUI 上に警告アイコンを表示。
- GUI ウィンドウ生成失敗時はログ警告を出し、既存の MIDI 制御フローにフォールバック。
- ブレンダーでの合成中にエラーが発生した場合は安全側（GUI 値採用）で処理し、ログに詳細を落とす。

## 主要コンポーネント詳細仕様（案）
- `ParamIntrospector`
  - `wrap_shapes_api(original: ShapesAPI) -> ShapesAPI`
    - 返すプロキシは `__getattr__` をオーバーライドし、取得した callable に対して動的ラッパーを生成。
    - ラッパーは呼び出し時に `ParameterKey(shape_name, position)` を発行し、`ParameterModel.register()` へ初回登録を要求。
  - `wrap_pipeline_builder(factory: Callable[[], PipelineBuilder]) -> Callable[[], PipelineBuilder]`
    - `E.pipeline` のアクセサを置き換え、`PipelineBuilder.__getattr__` で取得されるエフェクト adder をラップ。
    - ステップ生成時に `ParameterKey(effect_name, step_index)` を発行し、`ParameterModel` にメタ情報を提供。
  - 追加関数: `_resolve_meta(fn) -> ParameterMeta`（signature と `__param_meta__` の統合結果）
  - 例外方針: シグネチャ解決に失敗した場合は `ParameterMeta` を `None` で返し、呼び出し自体は継続。

- `ParameterModel`
  - `register(key: ParameterKey, meta: ParameterMeta, default: float | tuple[float, ...]) -> None`
    - 同じ `key` で二重登録された際は既存レコードを返す（エイリアス対応）。
  - `get_value(key: ParameterKey) -> float | tuple[float, ...]`
    - GUI override があればそれを、なければ既定値または MIDI 直値を返す。
  - `set_override(key: ParameterKey, value: float | tuple[float, ...], source="gui") -> None`
    - 範囲外の値は `clamp` し、変更があれば `ParameterUpdate` をイベントキューに積む。
  - `iter_controls() -> Iterable[ParameterEntry]`
    - コントロール生成に必要な順序付きイテレータ。呼び出し時にスナップショットを返し、ロックを速やかに解放。
  - 内部補助: `_infer_bounds(meta) -> Bounds`（min/max/step 推定）、`_format_label(key, meta) -> str`。
  - 例外方針: 未登録キーは `KeyError`。値型不正は `TypeError` を送出。

- `ParameterBlender`
  - `set_policy(key: ParameterKey, policy: PriorityPolicy) -> None`
  - `blend(key: ParameterKey, midi_value: float | None, gui_value: float | None, default: float) -> float`
    - ポリシーごとに合成。`None` の入力は無視。
  - `reset_overrides(source: Literal["midi", "gui"]) -> None`
    - 特定ソースからの上書きを一括解除し、`ParameterModel` に反映。
  - 例外方針: 未設定ポリシーは `PriorityPolicy.GUI_DOMINANT` を既定とする。

- `ParameterWindow`
  - `__init__(width: int = 480, height: int = 720, theme: ThemeConfig | None = None)`
  - `bind_model(model: ParameterModel, *, poll_interval: float = 0.1) -> None`
    - モデル参照とともに、`pyglet.clock.schedule_interval` で UI 更新を登録。
  - イベントハンドラ: `on_draw()`, `on_mouse_press(x, y, button, modifiers)`, `on_mouse_drag(x, y, dx, dy, buttons, modifiers)`, `on_mouse_scroll(x, y, scroll_x, scroll_y)`。
    - 各ハンドラは押下中のコントロールと同期して `set_override` を呼ぶ。
  - `update_layout(force: bool = False) -> None`
    - モデル側の変更を検知し、必要に応じて UI 要素を再構築。
  - 例外方針: GL リソース確保失敗は `RuntimeError` を送出し、上位がフォールバックを実施。

- `ParameterController`
  - `__init__(model: ParameterModel, window_factory: Callable[..., ParameterWindow])`
  - `inject_into(draw_fn: DrawFn) -> DrawFn`
    - `DrawFn` は `Callable[[float, Mapping[int, float]], Geometry]`。ラップ後は `DrawContext` を介してパラメータを差し替え。
  - `notify_update(update: ParameterUpdate) -> None`
    - GUI からの更新イベントを受信し、`queue.Queue` に投入。
  - `tick(dt: float) -> None`
    - ウィンドウの `dispatch_events()` と `update_layout()` を呼び出し。
  - `teardown() -> None`
    - スケジューラ解除、ウィンドウ破棄、キュークリアを実施。
  - 例外方針: `inject_into` はドキュメント化されたインタフェース外の引数で呼ばれた場合 `TypeError` を送出。

- `DrawContext`
  - `__init__(model: ParameterModel, blender: ParameterBlender, midi_snapshot: Mapping[int, float])`
  - `resolve_param(key: ParameterKey, fallback: float) -> float`
    - `ParameterModel` の値を取り出し、`ParameterBlender` で最終値を確定。
  - `as_mapping() -> Mapping[int, float]`
    - `draw` に渡す CC マップを構築。GUI 未介入の CC は `midi_snapshot` を透過。
  - `update_midi(snapshot: Mapping[int, float]) -> None`
    - MIDI 側の最新値を反映し、ブレンダーの結果を更新。
  - 例外方針: 未知キーは `fallback` を返す。型不一致は `TypeError`。
