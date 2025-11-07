# コードレビュー: api.sketch と関連モジュール

- 対象: `src/api/sketch.py`（実行ランナーの中核）
- 関連: `src/api/sketch_runner/{utils,render,params,midi,export,recording}.py`、`src/api/cc.py`
- 参照: アーキテクチャ規約（architecture.md, src/api/AGENTS.md）に整合

## 総評（Summary）
- 責務分離が明確で、`api.sketch` 本体はオーケストレーションに徹し、具体実装は `sketch_runner/*` に分割されている。遅延 import（pyglet/ModernGL 等）も適切に配置され、ヘッドレス環境への配慮が行き届いている。
- ワーカ（生成）と描画（UI）の分離、HUD/メトリクス、MIDI フォールバック、動画/PNG/G-code エクスポートなど、実行環境まわりのUXが一通り揃っている。
- 一方で小さな不整合がいくつか存在（軽微なバグ1件、import/型スタイルの細部、一部の重複）。設計の大枠に影響する問題は見当たらない。

## 良い点（Strengths）
- 遅延 import と安全なフォールバック
  - pyglet/ModernGL/各種サービスは関数内遅延 import。ヘッドレス/未導入環境対策（`src/api/sketch.py:186`、`src/api/sketch_runner/render.py:26`）。
  - MIDI は Null 実装へ安全にフォールバックし、ログ通知（`src/api/sketch_runner/midi.py:29`）。
- 責務分割の明確さ
  - 初期化と純粋関数は `sketch_runner/utils.py`（FPS/キャンバス/投影/HUDメトリクス）。
  - GL/Window 初期化は `sketch_runner/render.py` に集約。
  - Parameter GUI の適用/購読は `sketch_runner/params.py` に分離。
  - 録画モード切替は `sketch_runner/recording.py` に分離。
  - PNG/G-code エクスポートの薄いヘルパは `sketch_runner/export.py` に分離。
- 並行・スレッド/プロセス安全の配慮
  - WorkerPool への注入関数は spawn 安全なトップレベル関数を使用（`apply_param_snapshot` 等は import 参照、`hud_metrics_snapshot` はトップレベル定義）。
  - 終了処理（`on_close`）は冪等に配慮（`src/api/sketch.py:516` 以降）。
- API/ドキュメント
  - `run_sketch` の docstring は日本語のNumPyスタイルで、引数の意味が明確（`src/api/sketch.py:122`）。
  - アーキテクチャ記載の投影・座標系方針と一致（architecture.md, `api.sketch_runner.utils.build_projection`）。

## 気づいた問題（Issues）
- バグ: 品質モード中の HUD 描画判定が常に無効化される可能性
  - `src/api/sketch.py:559` にて `if "quality_recording" in locals():` を用いているが、`_capture_frame` 内では `quality_recording` はローカルではなく外側スコープの変数。`locals()` に現れないため、常に分岐に入らない恐れがある。
  - 期待挙動: 品質モード中（FBO→screen ブリット後）に HUD を重ねたい（コメント `src/api/sketch.py:558`）。
  - 提案: 単に `if quality_recording and overlay is not None:` とすべき（`locals()` チェックは削除）。
- import の一貫性（軽微）
  - `engine.export.gcode.GCodeWriter` の import がモジュールトップにある（`src/api/sketch.py:87`）。`src/api/AGENTS.md` の「sketch.py は重い依存は遅延 import」に照らすと、ここも関数内へ移動するとより一貫する。実害は薄いが設計上の統一性のために移動推奨。
- import 重複（軽微）
  - `apply_initial_colors` を 2 回 import している（`src/api/sketch.py:205`, `src/api/sketch.py:328`）。後者は不要（削除可能）。
- 型表現スタイル（微調整）
  - ガイドラインではビルトインジェネリクス＋`|`演算子を推奨（`AGENTS.md`）。`Optional[...]` は `| None` への揃えを推奨（例: `src/api/sketch.py:247` の `Optional[Callable[...]]`）。
- ロギングの粒度
  - HUD キャッシュ更新の try/except は debug ログを出しており方針は良い（`src/api/sketch.py:259`）。初期化の失敗/フォールバックのログは十分だが、重要な例外（ウィンドウ生成失敗など）はユーザー向けメッセージに昇格させる余地あり。

## 設計適合性（Architecture Fit）
- 依存方向はアーキテクチャ規約の L3(API)→L2(Engine) 準拠。`engine/*` から `api/*` への参照は無し。
- `api/sketch.py` の `engine.render/runtime/ui/io` への依存は関数内遅延 import でほぼ統一（例外: `engine.export.gcode`）。
- HUD 構成の解決は `HUDConfig` を活用（`src/api/sketch.py:211-217`）。優先順位ルールも明記されておりシンプル。
- WorkerPool とのやり取りは「API 層が cc/param の snapshot を用意→worker へ渡す」契約で一貫（`src/engine/runtime/worker.py`）。`param_snapshot` は親プロセス内実行のため picklable 要件を回避できているのも良い設計。

## 詳細観点ごとの所感
- 実行制御
  - FPS 解決/キャンバス解決/投影行列が純粋関数で切り出されテスト容易（`src/api/sketch_runner/utils.py`）。
  - フレームドライバは `FrameClock` で固定順序（`src/engine/core/frame_clock.py`）。
- レンダリング
  - 背景/線色の自動決定（輝度ベース）は合理的（`src/api/sketch_runner/render.py:37` 以降）。
  - LineRenderer は indices LRU を持ち HUD に統計提供（`src/engine/render/renderer.py`）。
- パラメータ GUI 連携
  - 初期色の適用・変更購読が UI スレッドへ正しくスケジューリング（`src/api/sketch_runner/params.py`）。
- エクスポート
  - PNG 保存は画面/高品質をモード分岐（`src/api/sketch_runner/export.py`）。
  - G-code は Service にジョブ投入し、進捗/HUD 連携（`src/api/sketch_runner/export.py` → `src/engine/export/service.py`）。
- MIDI
  - デバイス未接続時の UX が明示（警告ログ＋Null）（`src/api/sketch_runner/midi.py`）。

## 改善提案（Actionable）
- バグ修正
  - `src/api/sketch.py:559` を `if quality_recording and overlay is not None:` に修正。
- import 整理
  - `src/api/sketch.py:87` の `from engine.export.gcode import GCodeWriter` を関数内へ移動、または `sketch_runner/export.py` に統合して `ExportService` 生成を一箇所に寄せる。
  - `src/api/sketch.py:328` の重複 import を削除。
- 型スタイル統一
  - `Optional[...]` を `| None` に置換（例: `src/api/sketch.py:247` など）。
- ログ/メッセージ
  - `create_window_and_renderer` 失敗時のユーザー向けメッセージ強化（HUD か stderr）。

## 簡易チェック（Spot Checks）
- `resolve_fps`（`src/api/sketch_runner/utils.py:12`）: `None` → 設定→既定の順で 1 以上にクランプ。OK。
- `resolve_canvas_size`（`src/api/sketch_runner/utils.py:28`）: 小文字キー/不正タプル検証/許容値表示。OK。
- `build_projection`（`src/api/sketch_runner/utils.py:47`）: mm ベースの正射影（転置済）。OK。
- `make_gcode_export_handlers`（`src/api/sketch_runner/export.py`）: 進捗ポーリング/HUD 連携/キャンセル。OK。
- `enter_quality_mode` / `leave_quality_mode`（`src/api/sketch_runner/recording.py`）: インライン worker＋固定刻み tick／通常モード復帰。OK。

## リスク/注意点（Risks）
- 環境依存: pyglet/ModernGL は環境により初期化失敗あり。現在も try/except は点在するが、起動初期の致命的失敗はユーザー向けに分かりやすく案内すると良い。
- 生成と描画の責務境界: LazyGeometry 実体化は worker 側で原則実施しているが、将来の拡張で boundary が崩れないように留意。

## 参考（コード参照）
- `src/api/sketch.py:106` `run_sketch` 定義
- `src/api/sketch.py:211` HUDConfig 解決
- `src/api/sketch.py:246` HUD cache ステータス更新コールバックの定義
- `src/api/sketch.py:270` GL/Window 初期化呼び出し
- `src/api/sketch.py:323` ExportService + GCodeWriter 接続
- `src/api/sketch.py:354` Parameter GUI の色変更購読
- `src/api/sketch.py:456` pyglet キーイベント
- `src/api/sketch.py:516` on_close（冪等化）
- `src/api/sketch.py:559` 品質モード中 HUD 描画判定（要修正）

---

このレビューは実装の挙動を変えない観点での指摘に留めています。修正の実施をご希望であれば、上記「改善提案」をチェックリスト化して順に対応します。
