# Shift+V（HUDなし録画）の画質をV（HUDあり）に合わせる 改善チェックリスト

目的: Shift+V（FBO 経由, HUD なし）の映像が V（画面バッファ, HUD あり）と比べてジャギー/解像度差/色味差が出る問題を改善し、見た目を揃える。

想定原因（現状差分）
- 経路差: V=カラーバッファ直接, Shift+V=FBO再描画。
- MSAA差: V=WindowでMSAA(4x)有効, Shift+V=FBOが非MSAA。
- 解像度差: V=実フレームバッファ（HiDPI）ピクセル, Shift+V=論理解像度ベース。
- 色空間差: V=sRGB補正後の見た目に近い, Shift+V=線形空間出力。

---

## 実装タスク（最小で効果が大きい順）

- [x] 1) FBO の解像度を「実フレームバッファのピクセル数」に合わせる
  - 録画開始時（Shift+V）に `pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()` の `width/height` を取得し、FBO サイズに採用（Retina 環境で 2x を反映）。
  - 取得不能時は従来の `window.width/height` にフォールバック。

- [x] 2) FBO を MSAA（例: 4x）にする → 単一サンプルへ解決
  - ModernGL で `renderbuffer(samples=window.config.samples or 4)` を利用して MSAA FBO を構築。
  - 描画後、単一サンプルの FBO/テクスチャへ blit/resolve → RGBA 読み出し。
  - サンプル数が取得できない/非対応なら非MSAAでフォールバック。

- [ ] 3) 偶数ピクセル制約の維持（現行踏襲）
  - H.264 の都合で偶数へクロップする処理を維持（±1px 以内）。

- [ ] 4) 色空間（任意・効果小）
  - `FRAMEBUFFER_SRGB` が使える場合のみ有効化を検討。未対応なら現状維持（差は軽微）。

- [ ] 5) 失敗時のフォールバックとHUD通知
  - MSAA/FBO 生成に失敗した場合は非MSAA・従来解像度へフォールバックし、HUD に簡潔な警告を表示。

- [ ] 6) 確認観点（手動）
  - 同一シーンで V / Shift+V の1フレームを PNG で比較し、輪郭のジャギーと解像度が一致すること。
  - HiDPI 環境でのピクセル数一致（例: 1200x1200 → 2400x2400 相当）

---

## 変更予定ファイル
- `src/engine/export/video.py`（FBO サイズ取得, MSAA FBO→解像, 読み出し周り）
- （必要なら）`src/api/sketch.py`（サンプル数の取得/受け渡し。基本は `window.config.samples` を `video_recorder.start(..., …)` へ渡さずとも内部で参照可能）

## 受け入れ基準（DoD）
- Shift+V と V で、同一フレームのジャギー感・解像度が視覚的に同等。
- HiDPI 環境でも実フレームバッファ解像度へ追随。
- 依存追加なし。非対応環境では安全にフォールバックし、録画は継続可能。

---

OK であれば、このチェックリストに沿って実装を進め、完了項目にチェックを入れていきます。
