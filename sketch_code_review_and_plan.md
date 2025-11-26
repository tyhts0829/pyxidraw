# sketch.py 実装コードレビュー & 改善計画

対象: `src/api/sketch.py`（スケッチ実行ランナー）

## レビュー結果

### 良い点

1. **役割とフローが docstring で丁寧に説明されている**
   - 冒頭のモジュール docstring で、責務・実行フロー・引数の意味・注意点が具体的に整理されており、初見でも「何をしているモジュールか」が把握しやすい。
   - `run_sketch` 内も ①〜⑨ のセクションコメントで大まかな処理順が示されていて、上から読むと実行フローが追える構成になっている。

2. **純粋関数・ユーティリティの分離が進んでいる**
   - `resolve_fps` / `build_projection` / `hud_metrics_snapshot` を `api.sketch_runner.utils` に切り出しており、トップレベルに純粋関数を集約する方針が明確。
   - MIDI・パラメータ GUI・エクスポート・録画なども `sketch_runner.*` サブモジュールに委譲していて、ドメイン別に依存が分かれている。

3. **ヘッドレス/重依存対策としての遅延 import が機能している**
   - `pyglet` や `ModernGL` など重い依存は `run_sketch` の内部で遅延 import しており、`init_only=True` で早期 return するフローも用意されている。
   - MIDI 未導入/未接続・HUD 無効・GUI 無効といった条件に応じて、必要なコンポーネントだけを初期化する分岐が整理されている。

4. **ランタイム構成要素の結線が素直で追いやすい**
   - `SwapBuffer`・`WorkerPool`・`StreamReceiver`・`FrameClock`・`OverlayHUD`・`MetricSampler` などのオブジェクトは、生成→接続→スケジューリングまでが 1 箇所にまとまっており、データフローの把握がしやすい。
   - 品質最優先モード（録画中）と通常モードの切り替えも `sketch_runner.recording` に委譲する形で分離されている。

### 改善したい点

1. **`run_sketch` が長大かつ多責務で、読み書きの負荷が高い**
   - 1 関数内で「設定解決」「MIDI/GUI 初期化」「ワーカー/バッファ構成」「HUD/メトリクス」「エクスポート（PNG/G-code/Video）」「品質モード」「イベントハンドラ登録」「終了処理」「シグナル/atexit/KeyboardInterrupt 対応」までを扱っており、500 行超のボリュームになっている。
   - 内部のネスト関数（`_handle_save_png` / `_enter_quality_mode` / `_leave_quality_mode` / `on_key_press` / `on_close` など）が多数あり、それぞれが多くの外側変数に依存しているため、変更時に影響範囲を追うコストが高い。
   - `nonlocal` を使う箇所もあり、状態のライフサイクル（特に品質モード移行まわり）が一読で把握しづらい。

2. **`engine.ui/*` への依存がモジュールトップにあり、AGENTS の「遅延 import 方針」と少しずれている**
   - `HUDConfig` (`engine.ui.hud.config`) と `ParameterManager` (`engine.ui.parameters.manager`) をモジュールトップで import しており、ヘッドレス環境や GUI 未導入環境での import コスト・失敗可能性を増やしている。
   - `src/api/AGENTS.md` では `api/sketch.py` が `engine.render/*`・`engine.runtime/*`・`engine.ui/*`・`engine.io/*` に依存する際は「関数内の遅延 import」とする方針が記載されているため、この 2 つも `TYPE_CHECKING` + 関数内 import へ寄せると整合が取れる。

3. **例外処理が広範に「握りつぶし」になっており、問題検知が難しい**
   - 多くの処理が `try: ... except Exception: pass` もしくは HUD に軽いメッセージを出すだけになっており、ログにすら出ないケースが散見される。
     - 例: パイプラインキャッシュ初期化、pyglet ログレベル設定、HUD メトリクス取得、Parameter GUI の store 操作、`pyglet.clock.unschedule`、`worker_pool.close()`、`video_recorder.stop()` など。
   - 初期化やクリーンアップの一部は「失敗しても致命的ではない」が、開発時には失敗に気付きたいものが多く、最低限 `logging.debug`/`logging.warning` で記録しておくとデバッグしやすくなる。
   - どこまでクラッシュ回避を優先して握りつぶすか、どこからは例外をそのまま上げる／ログに残すべきかのポリシーが明文化されていない。

4. **ライフサイクル関連の処理が複数箇所に散らばっている**
   - Parameter GUI の終了処理が `_shutdown_parameter_gui` / `on_key_press`（ESC） / `on_close` / `_sig_handler` / `_at_exit` の複数経路から呼ばれており、挙動がやや追いにくい。
   - `on_close` とシグナル/atexit ハンドラがどちらも `rendering_window.close()` を呼び、さらに `on_close` 自体も冪等化している構造になっていて、制御フローが複雑。
   - WorkerPool/録画/HUD/メトリクスのクリーンアップも主に `on_close` に寄っているが、品質モードの `_leave_quality_mode` との関係がやや分かりづらい。

5. **型ヒントまわりで `# noqa` / `type: ignore` が残っている**
   - `_on_metrics` は `# type: ignore[no-untyped-def]`、`on_key_press` / `on_close` / `_silent_excepthook` などは `# noqa: ANN001` でシグネチャ型注釈が省略されている。
   - いずれも `int` や `Mapping[str, str]` など単純な型で表現可能なため、最小限の注釈を付ければ mypy/ruff との整合が取りやすくなる。

6. **コメント・ドキュメントの軽微な不整合**
   - `run_sketch` の docstring の `workers` 説明が「default 4」となっている一方、実際のデフォルト値は `workers: int = 6` になっている。
   - `line_color` の docstring の型表記と実際の型ヒント（`str | tuple[float, float, float] | tuple[float, float, float, float] | None`）が少しずれている。
   - セクション番号コメントが「⑤ Window」「⑤.5 Video Recorder」「⑦ FrameCoordinator」「⑨ Parameter GUI」「⑧ pyglet イベント」と並んでおり、読み手がフローを追う際に少し混乱しやすい。

7. **小さなスタイル・名前付けの改善余地**
   - `RGBA` 型エイリアスが現状このモジュール内で実質使われておらず、不要であれば削除してシンプルにできる可能性がある。
   - `rendering_window, mgl_ctx, line_renderer, _bg_rgba, _line_rgba = ...` のうち `_bg_rgba` / `_line_rgba` が未使用であり、将来の用途が無ければ戻り値の形を整理するか、明示的に無視する意図をコメントで補足してもよい。

## 実装改善計画（チェックリスト）

目的: `run_sketch` の読みやすさ・保守性・方針整合（AGENTS の規約・例外処理ポリシー）を高めつつ、挙動はできる限り現状維持とする。

### 設計・分割

- [ ] `run_sketch` の内部を 3〜5 個程度の「セットアップ段階」に分解し、トップレベルのプライベート関数（例: `_setup_runtime_components`, `_setup_hud_and_overlay`, `_setup_export_and_recording`, `_setup_event_handlers`）として切り出す。
- [ ] 品質モード（Video 録画用）の状態と操作を小さなコンテキスト（`@dataclass` など）にまとめ、`nonlocal` に依存しない形で `_enter_quality_mode` / `_leave_quality_mode` を整理する。
- [ ] Parameter GUI / HUD / WorkerPool / StreamReceiver / FrameClock / VideoRecorder など「ランタイム構成要素」の束を簡潔に表すコンテキスト構造体を導入し、イベントハンドラやクリーンアップ処理へ渡しやすくする。

### 依存・import 方針の整理

- [ ] `HUDConfig` と `ParameterManager` の import を `TYPE_CHECKING` ブロックと関数内遅延 import に分離し、`api/AGENTS.md` の「engine.ui/* への依存は遅延 import」の方針と揃える。
- [ ] `_prepare_parameter_gui` / `_build_hud_config` 内で必要なクラスを import する形に変更し、モジュール import 時には軽量な型情報以外を読み込まないようにする。

### 例外処理・ログ方針

- [ ] `try/except Exception: pass` となっている箇所を洗い出し、「ユーザー体験を守るために沈黙させたい箇所」と「開発中に異常を検知したい箇所」に分類する。
- [ ] 前者は `logging.debug` 程度のログに、後者は少なくとも `logging.warning` を出すか例外を伝播させるように変更し、最低限の観測可能性を確保する。
- [ ] `on_close` / シグナルハンドラ / `atexit` ハンドラまわりの例外処理を見直し、二重終了や二重解放は黙殺しつつも、本質的なリソースリークや保存失敗（MIDI/動画など）はログを残す。

### ライフサイクル・クリーンアップ

- [ ] Parameter GUI の終了処理を 1 箇所に集約し、`on_key_press`（ESC）/`on_close`/シグナル/atexit からはその共通ヘルパーを呼ぶだけにする。
- [ ] `rendering_window.close()` をトリガとして終了処理が 1 パスで走るように整理し、`on_close` と `_sig_handler`/`_at_exit` の役割分担を明確にする（例: シグナル/atexit は「閉じる要求」だけを出し、実際の終了処理は `on_close` に委譲）。
- [ ] 品質モード中の録画停止と通常モードへの復帰の経路を 1 箇所にまとめ、`on_key_press`（V キー）と `on_close` の双方から共通ヘルパーを使うようにする。

### 型ヒント・コメント整備

- [ ] `_on_metrics` や `on_key_press` / `on_close` / `_silent_excepthook` などのネスト関数に対して、`int` / `Mapping[str, str]` 等の簡潔な型注釈を付け、`# noqa: ANN001` / `# type: ignore[no-untyped-def]` を削減する。
- [ ] `run_sketch` の docstring を実際のシグネチャと一致させる（`workers` のデフォルト値、`line_color`/`background` の型表現、`use_parameter_gui` の default 記述など）。
- [ ] セクションコメント番号（①〜⑨）を実際の処理順に揃え、「Window & GL」「HUD」「FrameClock」「Parameter GUI 監視」「イベント/録画」といった論理まとまりが一目で分かるようにする。
- [ ] 未使用の型エイリアスやローカル変数（`RGBA`, `_bg_rgba`, `_line_rgba` など）を確認し、不要であれば削除するか用途をコメントで明示する。

### 動作確認・テスト

- [ ] `run_sketch` のリファクタ後に、手元での Smoke テスト（`python main.py`、簡単な `user_draw`）と、関連するテスト（存在する場合は `tests/ui/parameters` や HUD 関連テスト）を実行して挙動が変わっていないことを確認する。
- [ ] 変更したファイルに対して `ruff` / `mypy` を実行し、型とスタイルのチェックを通す。

## 追加確認・相談したい点

- 例外処理ポリシー:
  - 「ユーザーのスケッチ実行を止めない」ことをどこまで優先すべきか。たとえば HUD の更新失敗や録画失敗は HUD メッセージ + ログにする一方で、初期化フェーズ（`WorkerPool` 生成や `create_window_and_renderer`）の失敗は例外をそのまま上げる、という線引きでよいか。
- 構造の分割レベル:
  - `run_sketch` をあくまで単一関数のまま、小さなセットアップ関数に分解する「軽量な分割」に留めるか、それとも内部に小さなコンテキストクラス（例: `_SketchRuntime`）を導入してイベントハンドラをメソッド化するレベルまで踏み込むか。
- 互換性に関する許容度:
  - `workers` デフォルト値のドキュメント修正や例外処理の見直しで、挙動がやや変化する（今まで黙殺されていた例外がログに出る等）ことは許容してよいか。破壊的変更が許容される前提であれば、よりシンプルな実装へ寄せる方向で進めたい。

