# 描画ウィンドウの保存機能 仕様案（PNG / G-code）

本書は、実行中の描画ウィンドウ内容を保存する機能（PNG 画像 / G-code）の最小で明快な仕様を定義する。
初回は依存追加なしで完結し（pyglet / ModernGL 既存機能を活用）、将来の拡張余地を残す。

---

## 目的 / スコープ
- 目的: 実行中のスケッチをワンアクションで保存できるようにする。
- 出力: PNG（ラスター）、G-code（2D ツールパス）。
- スコープ: ランタイムのホットキー操作と、簡易なプログラマブル API。
- 非スコープ: ファイルダイアログ、PDF/EPS/SVG 出力、複雑なスタイル指定（将来拡張）。

## ユーザー体験（UX）
- ホットキー（`src/api/sketch.py` の pyglet イベントに追加。公開 API 追加は行わない）
  - `P`: 画面解像度で PNG 保存（既定: オーバーレイ含む）。
  - `Shift+P`: 高解像度 PNG 保存（`scale=2.0`）。
  - `G`: G-code 保存（ジオメトリから生成、オーバーレイ非含有、非ブロッキング）。
  - `Shift+G`: 実行中の G-code エクスポートをキャンセル。
- 保存先/ファイル名
  - PNG: ルート直下 `screenshots/`（存在しなければ作成）。
    - 既定ファイル名: `YYYYmmdd_HHMMSS_{w}x{h}.png`。
  - G-code: `data/gcode/`（存在しなければ作成）。
    - 既定ファイル名: `YYYYmmdd_HHMMSS_{W}x{H}_mm.gcode`。
- 成功時のフィードバック
  - HUD に一時メッセージ（例: `Saved PNG: screenshots/20250920_223012_1200x1200.png` / `Saved G-code: data/gcode/...`）。
  - 失敗時は HUD に理由込みで表示（例外の要約）。

## 出力フォーマット仕様
- PNG（ラスター）
  - 色: 8-bit RGBA、既定はウィンドウの見た目をそのまま取得。
  - 解像度: 既定はウィンドウのピクセル。`scale` によりオフスクリーンで拡大描画（1.0–4.0 目安）。
  - オーバーレイ: 既定は含む（window バッファキャプチャ）。`include_overlay=False` 指定時はオフスクリーン描画で除外。
  - 透過背景: `transparent=True` 指定時のみ（オフスクリーン時にクリア色の α=0 で描画）。
- G-code（2D ツールパス）
  - ソース: 直近フレームの `Geometry`（`SwapBuffer.get_front()` のスナップショット）。
  - 単位/座標: mm、絶対座標（`G21`/`G90`）。内部ジオメトリは Y 上向きのため、そのまま出力（`y_down=False`）。
  - パス表現:
    - 行間移動（ペンアップ）: `G0 X.. Y..`（必要に応じて `Z=z_up`）。
    - 描画（ペンダウン）: `G1 X.. Y.. F{draw_feed}`（最初の描画時のみ `Z=z_down`）。
    - 線分が 1 点のみの場合はスキップ（出力なし）。
  - 既定ヘッダ/フッタ:
    - ヘッダ例: `; Pyxidraw G-code v1` `G90` `G21` `G92 X0 Y0` `G0 Z{z_up}`。
    - フッタ例: `G0 Z{z_up}` `G0 X0 Y0`。
  - フィードレート: `travel_feed`（移動）, `draw_feed`（描画）を mm/min で指定。
  - 角丸/補間: なし（直線補間のみ）。
  - 互換性: 既定は GRBL/Marlin を想定した最小コマンド集合（スピンドル/PWM/カスタム M コードは非対応、将来拡張）。

## 操作（ホットキー中心）
- ユーザーはインポート不要。ホットキーのみで保存機能を利用可能。
- プログラマブル API は現段階では「内部専用」。`api/` 配下には追加しない。
- 内部層（`engine/export`）に関数を実装し、`api/sketch.py` からのみ呼び出す。

## 内部設計（最小実装）
- 参照箇所
  - ウィンドウ: `src/engine/core/render_window.py`
  - レンダラ: `src/engine/render/renderer.py`
  - シェーダ: `src/engine/render/shader.py`
  - バッファ: `src/engine/runtime/buffer.py`
  - ランナー: `src/api/sketch.py`
- PNG 実装
  - 低コストパス（既定）: `pyglet.image.get_buffer_manager().get_color_buffer().save(path)` でウィンドウをそのまま PNG へ（オーバーレイ含む）。
  - クリーンパス（overlay なし・高解像度・透過対応）: ModernGL のオフスクリーン FBO を作成し、`LineRenderer.draw()` だけを描画 → `fbo.read(components="RGBA")` → `pyglet.image.ImageData.save(path)`。
  - 依存追加なし（Pillow 不要）。
- G-code 実装（非ブロッキング）
  - スナップショット: `SwapBuffer.get_front()` の `coords/offsets` を C 連続配列に即時コピー（UI ブロックを最小化）。
  - ExportService: 専用スレッド＋ジョブキュー（同時 1 件）。API:
    - `submit_gcode_job(snapshot, params) -> job_id`
    - `cancel(job_id)`（`Shift+G` で発火）
    - `progress(job_id): {done_vertices, total_vertices, state}`
  - 変換/書き出し: `offsets` で行分割し、`G0`（ペンアップ）→`G1`（ペンダウン）の列へ変換。
    - I/O はバッファ付きでチャンク書き出し（数千〜数万行単位で `"\n".join`）。
    - 小数は小数点 3 桁固定で丸め（例: `12.345`）。
  - 進捗/HUD: HUD に `NN% (processed/total)` と状態を表示。完了/失敗メッセージを短文で表示。
  - 再入: 実行中に `G` を押した場合は「実行中」メッセージのみ（新規ジョブは拒否）。
  - キャンセル: `Shift+G` でキャンセルフラグを立て、チャンク境界で早期終了。
- スレッド/タイミング
  - ホットキー押下はメインスレッド。PNG は同期で保存（軽量）。G-code は ExportService でバックグラウンド実行。
  - スナップショットコピーのみメインスレッドで実行し、その後は非ブロッキング。
- エラー処理
  - 出力先ディレクトリ作成失敗・書込失敗は例外捕捉し HUD に短文表示。ファイル名衝突時はサフィックス `-1`, `-2`。
  - G-code は `.gcode.part` に逐次書き出し、成功時に最終ファイル名へリネーム。キャンセル/失敗時は `.part` を削除。

## 設定値と既定
- 既定ディレクトリ: PNG は `screenshots/`、G-code は `data/gcode/`。
- PNG 既定: `scale=1.0`、`include_overlay=True`、`transparent=False`。
- G-code 既定: `travel_feed=3000.0`、`draw_feed=1500.0`、`z_up=5.0`、`z_down=0.2`、`y_down=False`、`origin=(0.0,0.0)`。
- 上限/警告: `scale>4.0` は GPU メモリ次第で失敗し得るため HUD で警告。

## 互換性 / 影響範囲
- 公開 API の変更なし。`api/` 配下に新規ファイルを追加しない（方針: `api` は G/E/薄い再エクスポートのみ）。
- 実装は `engine/export/*` と `api/sketch.py` のイベント配線のみで完結。
- 依存の追加なし。pyglet/ModernGL のみで完結。

## テスト観点
- 単体
  - G-code: ヘッダ/フッタ、`G0/G1` 行、座標丸め、行間移動の有無が想定どおり。
  - PNG: オフスクリーン FBO で所定サイズの RGBA が得られること（ピクセル数のみ検証）。
- E2E（任意）
  - ホットキー押下で `screenshots/` と `data/gcode/` にファイルが生成されること。
  - `include_overlay=False` で HUD 文言が PNG に含まれないこと（差分判定の簡易チェック）。

## 実装タスクリスト（最小）
1) `engine/export/service.py` 追加（スレッド＋ジョブキュー＋進捗/キャンセル）。
2) `engine/export/gcode.py` 追加（Geometry→G-code 変換とチャンク書き出し）。
3) `engine/export/image.py` 追加（PNG: FBO/pyglet キャプチャのラッパ）。
4) `api/sketch.py` にホットキー `P`/`Shift+P`/`G`/`Shift+G` を配線（内部関数呼び出しのみ）。
5) HUD 簡易メッセージ/進捗 API を `OverlayHUD` に追加（短文表示・%表示）。
6) 既定保存先ユーティリティ `util/paths.py` に `ensure_screenshots_dir()` と `ensure_gcode_dir()` を追加。
7) smoke テスト追加（`tests/ui/test_export_minimal.py`）。
8) README/architecture.md に仕様要点を追記（導線のみ）。

## 将来拡張の余地
- PDF/EPS エクスポート、SVG スタイル（色/パターン/レイヤ）、パス最適化（RDP 簡略化）
- バッチ保存（N フレーム保存）、自動保存（秒間隔/キーフレーム）
- HDR/カラーマネジメント、EXR/TIFF 等の高ビット深度

---

### 参考（関連コード）
- ランナー: `src/api/sketch.py`
- ウィンドウ: `src/engine/core/render_window.py`
- レンダラ: `src/engine/render/renderer.py`
- シェーダ: `src/engine/render/shader.py`
- バッファ: `src/engine/runtime/buffer.py`

---

## 実装計画（チェックリスト + 合格基準）

目的: 非ブロッキングな PNG/G-code 保存を段階的に実装し、UI の滑らかさを維持する。

前提/制約:
- `api/` 配下に新規ファイルは追加しない（公開 API は変更しない）。
- 依存追加なし。
- 変更ファイルに対して `ruff/mypy/pytest` を実行して緑化する（AGENTS.md 方針）。

Stage 0 — 設計スケルトンの用意（コードは薄く、空実装あり）
- [x] `engine/export/` パッケージを新設（空の `__init__.py`）。
- [x] `engine/export/gcode.py` に空の実装を追加:
  - [x] `@dataclass class GCodeParams:`（`travel_feed`, `draw_feed`, `z_up`, `z_down`, `y_down`, `origin`, `decimals=3`）。
  - [x] `class GCodeWriter:`（空の本体）。
    - [x] `def write(self, coords: np.ndarray, offsets: np.ndarray, params: GCodeParams, fp: IO[str]) -> None:` を定義のみ（`pass`）。
  - [x] 型ヒントは Python 3.10 範囲で記述（`typing` は最小限）。
- 合格基準: mypy が型未解決を出さない（空実装でOK）、ruff/black/isort 緑。

Stage 1 — 保存先ユーティリティ
- [x] `util/paths.py` を追加し、保存先を提供:
  - [x] `ensure_screenshots_dir() -> Path`
  - [x] `ensure_gcode_dir() -> Path`
  - [x] 既存 `data/` が無ければ作成、並行呼び出しでも安全。
- 合格基準: 同関数を呼び出すだけで存在ディレクトリが返る。存在/非存在の両ケースで例外なし。

Stage 2 — ExportService（非ブロッキング基盤）
- [x] `engine/export/service.py` を追加:
  - [x] 単一ワーカースレッド＋ジョブキュー（最大同時実行 1）。
  - [x] API: `submit_gcode_job(snapshot, params) -> str(job_id)`、`cancel(job_id) -> None`、`progress(job_id) -> Progress`。
  - [x] `Progress(state: Literal["pending","running","cancelling","completed","failed","cancelled"], done_vertices:int, total_vertices:int, path:Path|None, error:str|None)`。
  - [x] `.part` に逐次書き出し、完了時に `rename`、失敗/キャンセルで削除。
- 合格基準: 疑似ジョブ（モック writer）で進捗が 0→100% へ単調増加。キャンセル時に `cancelled` で終了。

Stage 3 — G-code 変換（空のクラスを保持）
- [x] `GCodeWriter` は引き続き空（本体の実装は後日）。
- [x] ExportService 側は `GCodeWriter.write(...)` を呼ぶだけにして、ここでは try/except と進捗更新の骨組みのみ実装。
- 合格基準: 実ジョブ投入で `NotImplementedError` 等を適切に扱い、HUD に失敗が表示される（UI は止まらない）。

Stage 4 — PNG エクスポートラッパ
- [x] `engine/export/image.py` を追加:
  - [x] `save_png(window, path=None, *, scale=1.0, include_overlay=True, transparent=False) -> Path`。
  - [ ] `include_overlay=True` はウィンドウバッファ直保存、`False` は FBO 経由で `LineRenderer.draw()` のみ描画。
- 合格基準: 生成ファイルのピクセル数が期待通り（`scale` 反映）、エラー時に説明的例外。

Stage 5 — HUD 拡張（メッセージ/進捗）
- [x] `engine/ui/overlay.py` に簡易 API を追加:
  - [x] `show_message(text: str, level: Literal["info","warn","error"] = "info", timeout_sec=3)`
  - [x] `set_progress(key: str, done: int, total: int)` / `clear_progress(key: str)`
- 合格基準: 実行時に一時メッセージが表示され、進捗が % で更新/消去できる。

Stage 6 — ランナー配線（ホットキー）
- [ ] `src/api/sketch.py` キーイベント:
  - [ ] `P` / `Shift+P` → `save_png(...)` を呼び出し、完了メッセージを HUD 表示。
  - [x] `G` → `SwapBuffer.get_front()` の `coords/offsets` を C 連続コピーでスナップショットし、ExportService へ投入。
  - [x] 実行中に再度 `G` → 「エクスポート実行中」の HUD メッセージ。
  - [x] `Shift+G` → 現行ジョブをキャンセル、HUD にキャンセル通知。
- 合格基準: 押下直後のフレーム落ちが最小（体感でカクつかない）。PNG は即時保存、G-code は進捗が更新される。

Stage 7 — エラー処理/ロギング
- [x] 例外は捕捉して HUD に短文表示（詳細は `logging`）。
- [x] ファイル名衝突時は `-1`, `-2` … を付与。
- 合格基準: 想定外エラーでも UI は継続、`.part` 残存なし。

Stage 8 — テスト（smoke/最小）
- [x] `tests/ui/test_export_minimal.py`（スキップ許容の smoke）
  - [x] `ensure_*_dir` がディレクトリを返す。
  - [x] ExportService 疑似ジョブで完了/キャンセル/失敗の3経路が検証できる。
- 合格基準: CI でスモークが緑。公開 API への影響なし。

備考:
- G-code 変換ロジックは後日 `GCodeWriter` に実装（今回の PR では空）。
- 大規模ジオメトリへの最適化（間引き/優先度制御）は非対象（将来拡張）。
