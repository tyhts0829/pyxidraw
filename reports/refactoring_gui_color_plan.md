# GUI 色指定まわりのリファクタリング計画（提案）

目的

- 初期復帰・実行時反映・保存/復元の経路を単純化し、保守性と可観測性を高める。
- Dear PyGui の表示仕様（0–255 RGB）と内部（0–1 RGBA）を明確に分離・変換し、再発を防止。

背景（現状と改善済みの要点）

- 共通パーサ `util.color` に集約済み（Hex / 0–1 / 0–255 対応） [完了]
- ColorEdit は 0–255 RGB（no_alpha=True）に統一し、初期/同期は整数で強制反映 [完了]
- 描画側の初期適用（store→Renderer/Window）を `run_sketch` 起動直後に明示 [完了]
- DPG/GL のコンテキスト競合は pyglet.schedule 経由で解消 [完了]
- 二重登録（runner.\*）を排除し、登録は ParameterManager.initialize() に一元化 [完了]

スコープ

- UI（ParameterWindow/dpg_window.py）・API ランナー（api/sketch.py）・ドキュメントの最小整理。
- 仕様変更なし（振る舞いは現状維持）。

チェックリスト（TODO）

- [x] dpg_window の局所関数の分割
  - [x] `build_display_controls(parent, store)` を導入（色ピッカー構築/初期反映を包含）
  - [x] `force_set_rgb_u8(tag, rgb_u8)` と `store_rgb01(store, pid, app_data)` のヘルパ分離
- [x] store→GUI 同期の一本化
  - [x] `sync_display_from_store(store)` を追加し、初回と購読の両方から呼ぶ
  - [x] ランナー色（runner.\*）のみ整数 RGB で強制反映（他の型は既存経路）
- [x] sketch.py の初期適用の関数化
  - [x] `apply_initial_colors(parameter_manager, window, renderer)` を導入（BG/LINE + 自動線色）
- [x] ドキュメント差分
  - [x] docs/user_color_inputs.md に「GUI は 0–255 RGB 表示・内部は 0–1 RGBA」を明記
  - [x] architecture.md に初期適用（store→renderer/window/overlay）の記述を追記

完了扱い（実施済み）

- [x] 共通パーサ導入（util/color.py）
- [x] ColorEdit の 0–255 表示固定（no_alpha=True / DisplayRGB / DisplayInt / InputRGB）
- [x] 初期同期で整数 RGB の強制反映（store→GUI）
- [x] 描画側の初期適用（store→renderer/window）
- [x] DPG→ 描画の反映は pyglet.schedule 経由
- [x] runner.\* の登録を ParameterManager.initialize() に一元化

受け入れ条件（DoD）

- GUI/描画とも「保存 → 終了 → 再起動」で同じ色に復帰（run の色引数を外した状態）
- GUI の表示値は 0–255 の整数、内部保存は 0–1 RGBA（JSON で確認可能）
- 主要テスト（変更ファイル優先）がグリーンで、追加の簡易スモークを手元で確認

リスク/留意点

- DPG バージョン差による引数名の差異（現状は getattr で安全側に回避）
- 今後 alpha を再導入する場合は、Display 側と Store 側での型/成分数整合が再度必要

オープン質問（要確認）

- GUI の色表示を 0–255 で固定のままで良いか（0–1 表示に切替たい場面があるか）；はい
- 起動直後の描画色は「常に保存値を優先」で良いか（config 既定を優先するモードは不要か）；引数指定 → 保存色 →config の順でいいよ。それ以外のモードは不要
- 他にも保存/復帰したい描画系のランナー項目（線幅 等）があれば追加対象にするか；とりあえずなしで

実施順序（提案）

1. dpg_window の関数分割（Display 構築と同期の切り出し）
2. sketch.py の初期適用を関数化
3. ログ制御（環境変数）
4. ドキュメント差分の反映

ロールバック

- 各ステップはファイル局所の関数抽出のみで、振る舞いの変更は無し。問題発生時は該当差分のみ戻す。
