どこで: `src/api/sketch.py` の `run_sketch(..., show_hud=...)` と Parameter GUI（`src/engine/ui/parameters/dpg_window.py`）による HUD トグル連携  
何を: `show_hud=False` で起動したときに、Parameter GUI から `runner.show_hud` を True に変更しても HUD が表示されない問題の原因整理と実装改善計画  
なぜ: ランタイムでの HUD 可視性制御の仕様と UI の挙動がずれており、ユーザにとって直感的でないため

# 背景 / 現状

- 再現手順（想定）:
  - `run_sketch(user_draw, use_parameter_gui=True, show_hud=False)` で起動。
  - Parameter GUI の「HUD」セクションに表示される `Show HUD` チェックボックス（`runner.show_hud`）をオンにする。
  - 期待: HUD が表示状態になる。
  - 実際: HUD は表示されない（常に非表示のまま）。
- 関連実装:
  - HUD 有効フラグの解決:
    - `src/api/sketch.py:190` 付近  
      `hud_conf = HUDConfig()` / `HUDConfig(enabled=bool(show_hud))` / `replace(hud_config, enabled=bool(show_hud))`
  - HUD オーバレイ生成と初期可視性:
    - `src/api/sketch.py:230` 付近  
      `if (use_parameter_gui and not init_only) or hud_conf.enabled:` で `OverlayHUD` を生成。  
      直後に `overlay.set_enabled(bool(hud_conf.enabled))` を呼び、HUD の初期可視性を決定。
  - Parameter GUI からの HUD トグル購読:
    - `src/api/sketch_runner/params.py:215` 以降 `subscribe_hud_visibility_changes()`  
      `lock=True` または `parameter_manager/overlay` が None の場合は何もしない。  
      そうでない場合に限り、`runner.show_hud` の変更を購読し、`overlay.set_enabled(...)` をスケジュール。
    - `src/api/sketch.py:383` 付近での呼び出し:  
      `_subscribe_hud_visibility_changes(parameter_manager, overlay, pyglet, lock=(show_hud is not None))`
  - Parameter GUI 側の `Show HUD` ウィジェット:
    - `src/engine/ui/parameters/dpg_window.py:548` 付近  
      `dpg.add_checkbox(tag="runner.show_hud", ...)` を常に追加し、`store.set_override("runner.show_hud", bool(a))` をコールバックに持つ。
- ドキュメント上の仕様:
  - `architecture.md:193` 付近:
    - 「`show_hud=None` かつ Parameter GUI 有効時は、GUI の `Show HUD（runner.show_hud）` で表示/非表示を動的に切替できる（明示引数がある場合は引数優先でロック）。」
  - `README.md:54` 付近:
    - 「`use_parameter_gui=True` かつ `show_hud=None` のとき、GUI に `Show HUD` が現れ、トグルで表示/非表示を切替可能。  
       `show_hud` を明示した場合は引数が優先され、GUI トグルは無効（ロック）。」
- 現実の挙動:
  - Parameter GUI の `Show HUD` チェックボックスは `show_hud` の有無に関わらず常に表示される。
  - しかし `run_sketch()` 側では `lock=(show_hud is not None)` を渡しているため、`show_hud` を明示した場合は
    `subscribe_hud_visibility_changes()` が早期リターンし、`runner.show_hud` の変更が `OverlayHUD` に伝播しない。
  - 結果として、「トグルが表示されているが、明示引数を渡したケースでは動作しない」という UI／仕様のズレが生じている。

# 原因（まとめ）

- `run_sketch(..., show_hud=False, use_parameter_gui=True)` の場合:
  - `hud_conf.enabled` は False に解決される。
  - `(use_parameter_gui and not init_only)` が True のため `OverlayHUD` 自体は生成されるが、初期状態で `overlay.set_enabled(False)` が呼ばれ HUD は非表示。
  - `subscribe_hud_visibility_changes(..., lock=(show_hud is not None))` により `lock=True` となり、購読処理がスキップされる。
  - Parameter GUI 側の `Show HUD` チェックボックスは `store.set_override("runner.show_hud", bool(a))` を更新するが、
    その変更を受けて `overlay.set_enabled(...)` を呼ぶリスナーが存在しない。
- ドキュメント上は「明示引数がある場合はロック」とある一方で、GUI 上は常にトグルが表示されるため、
  ユーザ視点では「トグルが効かないバグ」と見える状態になっている（UI と仕様の一貫性が不足）。

# 目標（Goal）

- `show_hud` 引数と Parameter GUI の `Show HUD` トグルの関係を明確化し、実装・UI・ドキュメントを揃える。
- `show_hud=False` で起動した際にも、「トグルが表示されているなら最低限その挙動は直感的」である状態にする。
- 既存の HUD 機能（メトリクス表示、キャッシュステータス、録画時のメッセージ等）の挙動は変えない。

# 非目標（Non‑Goals）

- HUD の表示内容やレイアウトの全面的な見直し。
- Parameter GUI 全体の構成変更（HUD セクション以外の改修）。
- MIDI／WorkerPool／レンダリングパイプラインの仕様変更。

# 変更方針（Design）

## 採用方針: `show_hud` を「初期値」とみなし、Parameter GUI を常に優先する（B 案）

- 意味づけ:
  - `show_hud` は「起動時の HUD 初期値」を決めるだけとし、`use_parameter_gui=True` の場合は
    その後の HUD 可視性は常に Parameter GUI (`runner.show_hud`) が決定する。
  - `use_parameter_gui=False` のときのみ、`show_hud` がランタイム中も唯一の制御手段となる。
- 実装方針:
  - `run_sketch()` からの購読登録時に、Parameter GUI 有効時は常に購読を有効化する。  
    具体的には `_subscribe_hud_visibility_changes(...)` 呼び出しの `lock` を  
    `lock=not use_parameter_gui`（= GUI 無効時のみロック扱い）とする方向で実装する。
  - 起動時の `parameter_manager.store.set_override("runner.show_hud", bool(hud_conf.enabled))` により、
    Parameter GUI の初期チェック状態は引数/設定に同期される（現状の処理を維持）。
  - 以降は `runner.show_hud` の変更が常に `overlay.set_enabled(...)` へ反映される。
- ドキュメント修正:
  - `architecture.md` / `README.md` を「Parameter GUI 有効時は GUI トグルが最終的なソースになる」ように書き換える。
  - 既存の「明示引数でロック」の記述は削除または補足（`use_parameter_gui=False` のケースに限定）。

メリット:
- ユーザ視点で「トグルが見えている限り必ず効く」挙動になり、直感的。
- 実装変更は `run_sketch()` 側の `lock` 条件と、必要なら若干の doc 修正で済む。

デメリット:
- `show_hud` の意味が若干変わる（「固定」から「初期値」へ）。ただし現時点ではユーザが居ない前提のため影響は限定的。

# 作業手順（チェックリスト）

- [x] 現状の実装とドキュメント（`architecture.md` / `README.md`）を確認して差分を整理
- [x] バグの直接原因（`lock=(show_hud is not None)` により購読が無効化されている点）を特定
- [x] 最終仕様の選定（B 案: `show_hud` 初期値 + Parameter GUI 優先）について、オーナーと合意を取る
- [ ] B 案に基づき、`run_sketch()` 内の HUD 初期化ロジックと `subscribe_hud_visibility_changes()` 呼び出しを更新
- [ ] 必要に応じて Parameter GUI 側（`dpg_window.py`）の `Show HUD` ウィジェットの表示/有効化条件を調整
- [ ] `architecture.md` / `README.md` の該当箇所（HUD/Parameter GUI 周り）を更新し、実装と揃える
- [ ] 変更ファイルに対して Lint/Format/Type チェックを実行  
      （例: `ruff check --fix src/api/sketch.py src/api/sketch_runner/params.py src/engine/ui/parameters/dpg_window.py` 等）
- [ ] 手動確認:  
      - `show_hud=None` / `True` / `False` × `use_parameter_gui=True/False` の組み合わせで HUD の挙動を確認  
      - 特に `show_hud=False, use_parameter_gui=True` で GUI トグルが期待通りに動作すること
- [ ] 必要であれば HUD トグル挙動に関する pytest を追加（最小限の統合テスト）

# 検証（Acceptance Criteria）

- [ ] `run_sketch(..., show_hud=False, use_parameter_gui=True)` で起動し、Parameter GUI の `Show HUD` をオンにしたときの挙動が「仕様として合意したもの」と一致する。
- [ ] `show_hud=None` のときは README/architecture.md に記載された通りに GUI から HUD をオン/オフできる。
- [ ] `use_parameter_gui=False` のときは `show_hud`（および `hud_config.enabled`）だけで HUD の有効/無効が一意に決まる。
- [ ] Lint/Format/Type チェックがすべて通る。

# 影響範囲・リスク

- 影響範囲:
  - `src/api/sketch.py`（HUD 初期化と購読登録）
  - `src/api/sketch_runner/params.py`（HUD 可視性購読ロジック）
  - `src/engine/ui/parameters/dpg_window.py`（Show HUD トグルの表示条件・有効化）
  - `architecture.md` / `README.md`（仕様の記述）
- リスク:
  - HUD の初期可視性や GUI トグルの挙動が変わるため、今後の利用コードとの互換性に影響し得る（ただし現時点ではユーザ不在のため許容しやすい）。
  - Parameter GUI と HUD の間の依存関係が増える場合、今後のリファクタリング時に考慮事項が増える可能性。

# 事前確認事項（Open Questions）

- 現時点では追加の事前確認事項なし（B 案前提で実装を進める）。

# ロールバック

- 変更対象は主に API 層（`run_sketch`）と UI 層（Parameter GUI）の局所的なロジックに限定されるため、
  問題が発生した場合は当該ファイルの差分を元に戻すだけで復旧可能。
