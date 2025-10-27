# 描画ウィンドウの動画書き出し 仕様案（MP4）

本書は、実行中の描画ウィンドウを MP4 動画として保存する最小かつ明快な仕様を定義する。
初回は「録画トグル（V キー）」のみを実装対象とし、依存追加は任意（存在時に活用、非存在時は明確に失敗を通知）。

---

## 目的 / スコープ
- 目的: 実行中のスケッチの画面をそのまま MP4 に保存できるようにする。
- 出力: MP4（H.264/AVC を想定、拡張子は `.mp4` 固定）。
- スコープ: ランタイムのホットキー操作と、内部用の極小 API（公開 API の追加は行わない）。
- 非スコープ: ファイルダイアログ、可変ビットレート/画質プリセット、音声合成、字幕。

## ユーザー体験（UX）
- ホットキー（`src/api/sketch.py` の pyglet イベントに追加）
  - `V`: 録画の開始/停止をトグル。
    - 開始時: HUD に「REC 開始」を短時間表示、以降は軽い常時表示（REC インジケータは任意、Stage 2 で検討）。
    - 停止時: 即座に MP4 をクローズして保存し、HUD に保存先を表示（例: `Saved MP4: data/video/...mp4`）。
- 速度/フレームの扱い
  - 録画中は UI の fps が低下してもよい（GPU からのピクセル読み出し・エンコードが同期で発生）。
  - 「録画された映像上ではフレームが飛ばない」ことを優先し、描画されたフレームごとに必ず 1 枚ずつ書き出す（キャプチャの取りこぼし無し）。
  - 再生 fps はランナーの `fps` 設定値を採用（例: 60）。UI が追いつかない場合でも、動画内の時間は一定フレーム間隔で進む。
    - Stage 2 で「固定タイムステップ記録（録画中は `t` を 1/fps 刻みで進める）」をオプション提供し、見た目の速度安定性を高める。
- 保存先/ファイル名
  - 既定: `data/video/`（無ければ作成）。
  - 例: `{scriptName}_{W}x{H}_{fps}fps_{yymmdd_hhmmss}.mp4`（`scriptName` は実行スクリプト名）。
  - H.264 の制約に合わせ、幅・高さが奇数のときは 1px 切り下げて偶数に調整（上下左右 1px 以内）。

## 内部設計（最小実装）
- 参照箇所
  - ランナー: `src/api/sketch.py`（キーイベント、HUD、FrameClock/Tickable 構成）
  - ウィンドウ: `src/engine/core/render_window.py`（`add_draw_callback`/`on_draw`）
  - レンダラ: `src/engine/render/renderer.py`（`draw()`）
  - PNG 保存: `src/engine/export/image.py`（既存のバッファ保存/FBO 参照）
  - パス解決: `src/util/paths.py`（`ensure_*_dir`）

- コンポーネント
  - `engine/export/video.py` に極小 `VideoRecorder` を追加（内部用）。
    - 依存: 可能なら `imageio-ffmpeg`（推奨）。未導入時は分かりやすく例外を投げ、HUD に通知。
    - API（想定）:
      - `start(window, width, height, fps, *, include_overlay=True, name_prefix=None) -> None`
      - `stop() -> Path`（保存先を返す）
      - `is_recording: bool`
      - `attach_draw_callback(window)`（on_draw 後段にキャプチャ関数を登録）
    - 実装要点:
      - キャプチャは `pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()` を利用。
      - `get_data("RGBA", pitch=width*4)` → `numpy.frombuffer` → `[H, W, 4]` に整形 → RGB へ変換。
      - 上下反転を補正（`np.flipud`）し、ライタへ同期書き込み（フレーム未欠落を優先）。
      - ライタ作成時に `fps` を設定。幅・高さが偶数でない場合は切り下げてクロップ。
      - 例外時は `stop()` 内でリソースを確実に解放し、上位へ伝搬。

- ランナーへの配線（`api/sketch.py`）
  - 生成: Window/MGL/Renderer 初期化後に `recorder = VideoRecorder(...)` を準備（遅延 import）。
  - on_draw: 既存の `line_renderer.draw() → overlay.draw()` の後に、録画中のみ `recorder.capture_current_frame()` を呼ぶコールバックを最後に登録。
  - キーイベント: `V` 押下で `recorder.start(...)` or `recorder.stop()` をトグル。
    - 成功時は HUD に一時メッセージ。失敗時は理由を含むメッセージ（依存未導入/権限/保存失敗など）。

## 品質・制約
- フレーム未欠落: キャプチャ/エンコードは同期で行い、バックプレッシャを受けたら UI 側が遅くなる設計（録画映像は連番フレームを維持）。
- 画質/ビットレート: 初回はデフォルト（`imageio-ffmpeg` の既定）とし、将来 `quality`/`bitrate` をオプション化。
- オーバーレイ: 初回は「ウィンドウ見た目そのまま」を保存（HUD 含む）。将来 `Shift+V` で HUD なし（FBO 経由）を検討。
- 依存: 外部バイナリ `ffmpeg` への直接依存は避け、Python パッケージ（`imageio-ffmpeg`）を推奨。未導入時は明示的に失敗。

## エッジケース
- 解像度が奇数: 幅/高さのいずれかが奇数の場合、最後列/行をクロップして偶数に調整（1px 以内）。
- ヘッドレス/pyglet 未初期化: 録画操作は無効化し、HUD に警告。
- 長時間録画: 同期実装のため UI が重くなる。Stage 2 で非同期キュー + ノンドロップ戦略（ブロック/バックプレッシャ）を検討。

---

## 実装計画（チェックリスト + 合格基準）

前提/制約:
- 公開 API（`api/`）は変更しない。実装は `engine/export/` と `api/sketch.py` のみ。
- 依存追加は任意（Ask-first）。導入しない場合は録画を明確に無効化。
- 変更ファイルに対して `ruff/mypy/pytest` を実施して緑化する（編集ファイル優先）。

Stage 0 — 保存先ユーティリティの拡張
- [x] `util/paths.py` に `ensure_video_dir() -> Path` を追加（`data/video/`）。
- 合格基準: 既存/新規の両ケースで例外なし、Path を返す。

Stage 1 — VideoRecorder（最小実装）
- [x] `engine/export/video.py` を新設し、`VideoRecorder` を実装。
  - [x] `start(...)`/`stop()`/`is_recording` を提供（draw コールバックはランナー側で登録）。
  - [x] `imageio`/`imageio-ffmpeg` 未導入時は `RuntimeError` を投げる。
  - [x] バッファ読み出し（RGBA→RGB、上下反転、偶数解像度クロップ）。
- 合格基準: 単体で 10〜60 フレーム程度を生成可能（目視と簡易スモーク）。

Stage 2 — ランナー配線/キーイベント
- [x] `api/sketch.py` に `V` キーのトグルを追加。
- [x] Window 初期化後に録画フックを最後に登録（overlay の後段）。
- [x] HUD で開始/停止/失敗メッセージを表示。
- 合格基準: 実行時に `V` で録画開始、再度 `V` で停止し、`.mp4` が保存される。

Stage 3 — UX 改善（任意）
- [x] 常時表示の REC インジケータ（HUD へ小さな赤丸 + ラベル）。
- [x] `Shift+V`: HUD を含まない録画（FBO 経由、`LineRenderer.draw()` のみ）。
- 合格基準: 視認性が良く、誤操作が起こりにくい。

Stage 4 — テスト/CI（最小）
- [ ] `tests/ui/test_video_minimal.py`（スモーク、optional マーカーで許容）。
  - [ ] `ensure_video_dir()` がディレクトリを返す。
  - [ ] `VideoRecorder` が依存未導入時に明快な例外を投げる。
- 合格基準: CI の smoke が緑（optional は別ジョブ）。

---

## 実装メモ（抜粋コード方針）
- ピクセル取得（overlay 含む）
  - `buf = pyglet.image.get_buffer_manager().get_color_buffer()`
  - `img = buf.get_image_data()`
  - `raw = img.get_data('RGBA', img.width * 4)`
  - `arr = np.frombuffer(raw, dtype=np.uint8).reshape(img.height, img.width, 4)`
  - `rgb = np.flipud(arr)[..., :3]` で上下反転かつ RGB 抽出
- ライタ（`imageio-ffmpeg`）
  - `iio.get_writer(path, fps=fps, codec='libx264', quality=8)`（初回は既定で十分）
  - フレームごとに `writer.append_data(rgb)`、停止時に `writer.close()`

---

## リスクと回避策
- 依存の有無差: `imageio-ffmpeg` 未導入環境では録画が無効。HUD に説明つきエラーを表示し、操作は継続。
- パフォーマンス: 同期実装のため負荷が高い。要件上「fps 低下許容」なので優先度は低い。将来、非同期キュー/バックプレッシャで改善。
- 解像度制約: 奇数ピクセルは H.264 と相性が悪い。1px クロップで回避（画面への影響は極小）。

---

## 変更点サマリ（予定）
- 追加: `src/engine/export/video.py`（新規）
- 追加: `src/util/paths.py` に `ensure_video_dir()`
- 変更: `src/api/sketch.py`（V キーの追加、録画の開始/停止、HUD 通知、draw コールバック登録）
- 変更（任意）: `src/engine/ui/hud/overlay.py`（REC インジケータ）

---

以上。ご確認ください。OK であればチェックリストに沿って実装を進めます。
