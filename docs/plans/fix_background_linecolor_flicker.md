どこで: runner 背景色変更時の描画点滅対策（api/sketch_runner + engine/render）
何を: 背景変更（runner.background）操作中に線が点滅・消える現象の対策計画（原因と対策、段階的チェックリスト）
なぜ: GUI操作時のみ発生する一過性の見づらさ/点滅を抑え、レイヤー描画と即時適用の整合性を高めるため。

現象（参考ログの抜粋）
- 背景適用/スケジュールが高頻度で発火し、各フレームでレイヤー2本が描画される。
  - 例: `[GLOBAL] apply background -> (...)
          [GLOBAL] schedule background
          [GLOBAL] on_draw frame N bg=(...)
          [GLOBAL] frame N layers-drawn
          [GLOBAL] renderer.set_line_color -> (...layer1...)
          [GLOBAL] draw: apply layer.color=(...layer1...)
          [GLOBAL] renderer.set_line_color -> (...layer2...)
          [GLOBAL] draw: apply layer.color=(...layer2...)` 
- 提示区間では点滅は再現していないが、実運用では「背景スライダ操作中」に線が見えづらくなる/消える瞬間がある。

原因仮説（コード根拠）
- 背景変更と「自動線色（黒/白）」の同一tickスケジュールが、レイヤー描画中の線色即時更新ブロック（layers active）と競合し、線色の適用/巻き戻りがフレーム跨ぎで揺れる。
  - 背景→自動線色の順序が未決定（schedule_once(0.0)を複数投げ）、かつブロック時の再試行がない。
  - レイヤーで style.color=None の場合、renderer.draw は「基準色（_base_line_color）」へ戻すが、GUIからの set_line_color では基準色が更新されないため、フレーム間で統一が崩れやすい。
- 自動線色の 0.5 閾値にヒステリシスがなく、境界付近で黒/白が反転しやすい（点滅感の助長）。

対応方針（段階導入・安全重視）
1) スケジューリングの直列化（順序決定）
   - 変更箇所: `src/api/sketch_runner/params.py`
   - 対応: 「背景適用」と「自動線色（未設定時のみ）」を“単一の schedule_once(0.0)” 内で直列実行（背景→自動線色）。
   - 目的: 同一フレーム内の適用順序を決定し、線色と背景の同期ズレを減らす。

2) ブロック時の再スケジュール
   - 変更箇所: `src/api/sketch_runner/params.py::_apply_line_color`
   - 対応: layers active でスキップした場合は、次tickに再度 schedule_once する（最大再試行回数を設け無限再試行を回避）。
   - 目的: レイヤー活動が続く状況でも runner.line_color の適用漏れを減らす。

3) 基準色の一貫性確保
   - 変更箇所: `src/engine/render/renderer.py`
   - 対応案A: `set_base_line_color(rgba)` を新設し、_base_line_color と uniform を同時更新。
   - 対応案B: 既存 `set_line_color` に `update_base: bool=False` を追加（デフォルト互換）。GUI経由は True を渡す。
   - GUI側: `apply_initial_colors` / `subscribe_color_changes` は“基準も更新する”経路へ切替。
   - 目的: layer.color=None のレイヤーが混在しても、フレーム間で色基準がブレないようにする。

4) 自動線色のヒステリシス
   - 変更箇所: `src/api/sketch_runner/params.py`（背景変更に伴う“動的”な自動線色適用に導入）
   - 対応: しきい値を 2値化（例: >0.55=黒, <0.45=白, 中間は前回値保持）。
   - 目的: 背景輝度が境界付近を往復しても黒白の反転頻度を抑制。

5) デバッグと回帰試験
   - PXD_DEBUG_GLOBAL=1 で、以下のログの順序と安定性を確認。
     - schedule background（統合後は“単一コールバック”内で背景→線色）
     - skip runner.line_color (layers active) 時の再スケジュール有無
     - draw: apply base_line_color=... の出現頻度（基準更新後に減ること）
   - 必要に応じ `-m smoke` の最小テスト、manual操作の再現シナリオを記録。

実装チェックリスト（要同意後着手）
- [x] 1) params.py: 背景/自動線色スケジュールを単一コールバック化（背景→自動線色）
- [x] 2) params.py: _apply_line_color のブロック時に再スケジュール（最大N回、例:3）
- [x] 3-A) renderer.py: set_base_line_color(rgba) を追加（uniform と _base 同期）
- [x] 3-B) params.py: GUI経路を set_base_line_color に切替（初期色/変更時）
- [x] 4) params.py: 自動線色のヒステリシス導入（0.45/0.55）
- [x] 5) ログ/デバッグ: 競合が収束したことを PXD_DEBUG_GLOBAL=1 で確認（提供ログと同粒度）
- [ ] 6) ドキュメント: README/architecture に「背景/線色適用の順序と設計意図（ヒステリシス含む）」追記

検証シナリオ（手動）
- 背景スライダを 1.0→0.4→1.0 と往復しながら、
  - 線が消える/点滅がなくなること（コントラスト維持）
  - ログ上、背景→（必要時）線色の順に同一コールバックで適用されること
  - layers active 時に line_color がスキップされた場合でも、翌フレームで適用されること
  - style.color=None を含むレイヤー混在でも、基準色の巻き戻りが出ないこと

リスクと緩和
- renderer API 追加に伴う既存呼び出しの影響
  - 既存 set_line_color のデフォルトは維持。GUI経路のみ新APIへ切替。
- 自動線色の揺れ止めにより、既存の見た目がわずかに変化
  - しきい値は環境変数/設定で調整可能にしておく（将来改修）。

ロールバック
- API 追加（3-A）が問題なら、GUI側で「set_line_color + _base に直接代入」の暫定回避可（ただしカプセル化低下）。

承認依頼
- 上記チェックリストで進めて良いかご確認ください。段階的にPRを分割し、各段でログ/挙動を貼って検証報告します。
