どこで: 背景変更時の点滅修正の“不要差分”整理（クリーンアップ計画）
何を: 最小修正（Renderer のレイヤー未到達フレームで前回レイヤーを再描画）に絞り、今回のデバッグ過程で混入した無関係/冗長変更を元に戻す計画。
なぜ: 問題の本質は on_draw が tick より先行してレイヤー未到達になるフレームが挟まることだった。根本対策は「レイヤー未到達フレームで前回レイヤーを再描画」だけで十分であり、GUI 配線や基準色 API の変更は不要だから。

保持する変更（根本対策）
- Renderer のフォールバック再描画（前回レイヤーのスナップショットを再描画）
  - ファイル: `src/engine/render/renderer.py`
  - 影響範囲: レイヤー未到達フレームのみ。見た目を前フレームと等価に維持し、点滅（片レイヤー消失・単色化）を抑止。

巻き戻す（または縮小する）変更（不要/副次）
1) draw 冒頭の `try_swap`（早取り）
   - 理由: フォールバック再描画で十分。二重で swap する必要は無く、責務の分離（tick が供給、draw は消費）を保つ。
   - 変更箇所: `src/engine/render/renderer.py: draw()` 冒頭の早取りロジックを削除。

2) `set_base_line_color` API と GUI 側の呼び出し切替
   - 理由: 今回の点滅とは無関係。既存の `set_line_color` で十分。基準色の一貫性は本件の症状に寄与していない（ログ上、各レイヤーは常に style.color 指定）。
   - 変更箇所: `src/engine/render/renderer.py` の新規 API を削除し、
               `src/api/sketch_runner/params.py` の `apply_initial_colors`/`subscribe_color_changes` を `set_line_color` に戻す。

3) 背景→自動線色の単一コールバック化/再試行/ヒステリシス
   - 理由: 点滅原因は「追加 on_draw と tick の非同期」によるレイヤー未到達であり、runner.line_color の挙動は無関係だった。
   - 変更箇所: `src/api/sketch_runner/params.py`
     - 背景と自動線色のまとめ適用（_apply_bg_then_maybe_auto）を削除し、元の個別スケジュールへ戻す。
     - レイヤー活動時の再試行（最大3回）を削除。
     - 自動線色のヒステリシスを削除（将来案として残すなら別PRで）。

4) 受信/描画のデバッグ出力の削除（PXD_DEBUG_GLOBAL）
   - 理由: 解析は完了。運用ノイズを減らし、最小差分へ。
   - 変更箇所: `src/engine/runtime/receiver.py`、`src/engine/render/renderer.py` の追加 print を削除。

実施手順（チェックリスト）
- [x] 1) renderer.py: draw 冒頭の早取り `try_swap` を削除
- [x] 2) params.py: `apply_initial_colors`/`subscribe_color_changes` を `set_line_color` に戻す
- [x] 3) renderer.py: `set_base_line_color` を削除（参照も除去）
- [x] 4) params.py: 背景・自動線色の単一コールバック/再試行/ヒステリシスを削除
- [x] 5) receiver.py/renderer.py: デバッグ print を削除（必要なら logger.debug に差し替え）
- [x] 6) sketch.py: 未使用の `subscribe_style_changes` 呼び出しを削除
- [x] 7) params.py: 未使用の `subscribe_style_changes` 定義/エクスポートを削除
- [x] 8) 動作確認（GUI 背景スライダ操作中の点滅再現が無いこと）
       - 背景操作中でも「レイヤー未到達フレームは前回レイヤー再描画」で崩れないこと
       - PXD_DEBUG_GLOBAL 無効で不要ログが出ないこと
- [ ] 9) docs: 既存プラン（fix_background_linecolor_flicker.md）を“最終案”へ反映（保持/削除の要約）

追加クリーンアップ（反映済み）
- [x] params.py: `_apply_line_color` の layers-active ガードを撤去（更新取りこぼし防止のため簡素化）
- [x] receiver.py: layers/geometry push の不要な try/except + フォールバックを削除（単純化）

補足（追跡調査の結果）
- 残存コードの確認結果:
  - `set_base_line_color` の参照・実装は完全削除済み。
  - `PXD_DEBUG_GLOBAL` を用いた print は全削除済み（src/ 配下に該当なし）。
  - Renderer の早取り `try_swap` は削除済み（受信は tick のみ）。
  - 残すのは Renderer のレイヤースナップショット再描画のみ（根本対策）。

検証観点（最終）
- 背景スライダを高速に往復しても、レイヤー描画が片方/単色にならない（目視）。
- on_draw と tick の順序（内部）は変えずとも、描画破綻が起きない（フォールバックで吸収）。
- 生成/受信/描画の各責務境界は維持（早取り削除後も問題なし）。

備考（将来の改善アイデアとして分離）
- 背景スケジュールのデバウンス（例 16ms）: GUI連打時の追加 on_draw をコアレッシング。
- レイヤーの GPU サイド保持/再描画: VBO/IBO をフレーム間保持して CPU 再アップロードを削減。
- 自動線色のヒステリシス: 仕様強化は別PRで扱う（今回の症状とは独立）。
