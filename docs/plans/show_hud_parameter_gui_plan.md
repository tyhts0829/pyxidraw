# show_hud を Parameter GUI から制御する改善計画

目的: `run_sketch(..., show_hud=...)` の HUD 表示可否を Parameter GUI からトグル可能にする。最小実装で動的切替を実現し、既存の API 互換とシンプルさを優先する。

## 要件（最小）

- GUI 上で HUD の表示/非表示をトグルできる。
- 優先順位は「明示引数 > GUI > 既定」。
  - `show_hud` を明示（True/False）した場合は GUI 側の変更は適用しない（ロック扱い）。
  - `show_hud=None` の場合に GUI トグルが有効。
- `use_parameter_gui=True` のときに動的切替をサポート（GUI なし実行は従来挙動を維持）。
- 破壊的でないが、必要十分な堅牢さで実装。複雑化を避ける。

## 影響範囲と方針

- 新規 GUI パラメータを追加:
  - id: `runner.show_hud`（bool, category="HUD", label="Show HUD"）。
  - 既定値は後述の解決結果（`hud_conf.enabled`）を初期同期で反映。
- HUD 可視性の切替は Overlay 側で吸収:
  - `engine/ui/hud/overlay.py` に `set_enabled(on: bool)` を追加し、`draw()` で `_enabled` を確認して早期 return。
  - Tick は維持（描画のみ停止）。最小変更でオーバーヘッド小。必要なら将来 tick もガードに拡張。
- 生成タイミング:
  - `use_parameter_gui=True` の場合は、`hud_conf.enabled` に関わらず `MetricSampler`/`OverlayHUD` を生成しておく（後からトグル可にするため）。
  - `use_parameter_gui=False` の場合は従来どおり `hud_conf.enabled` が False なら生成しない。
- 初期状態同期:
  - `hud_conf` を解決後、`parameter_manager.store.set_override("runner.show_hud", hud_conf.enabled)` を適用し、GUI 表示と実状態を揃える。
- 購読（UI スレッド反映）:
  - `src/api/sketch_runner/params.py` の購読系を拡張し、`runner.show_hud` 変更時に `overlay.set_enabled(...)` を `pyglet.clock.schedule_once` 経由で反映。
  - 明示引数ロック時（`show_hud is not None`）は購読処理を無効化。
- メトリクス連携の単純化:
  - Worker への `metrics_snapshot` は `hud_conf.show_cache_status` のみに基づいて有効化し、`enabled` とは独立させる（HUD 非表示でも計測は継続）。
    - 理由: HUD の可視性と計測を分離することで、トグル後の表示に即応させる。オーバーヘッドは軽微。

## 変更ファイル（予定）

- `src/engine/ui/hud/overlay.py`: `set_enabled()` と `_enabled` フラグ、`draw()` ガード追加。
- `src/engine/ui/parameters/manager.py`: `runner.show_hud` の Descriptor 登録（bool, category="HUD"）。
- `src/api/sketch.py`:
  - `hud_conf` 解決後の初期同期（Store へ `runner.show_hud` を反映）。
  - `use_parameter_gui=True` 時は HUD を常に生成し、`overlay.set_enabled(hud_conf.enabled)` を初期適用。
  - `metrics_snapshot_fn` の有効化条件を `hud_conf.show_cache_status` のみに変更。
- `src/api/sketch_runner/params.py`: `subscribe_color_changes(...)` を拡張、または `subscribe_hud_visibility_changes(...)` を新設して `runner.show_hud` を監視し `overlay.set_enabled()` を呼ぶ。
- `README.md`（任意最小）: パラメータ GUI で HUD を切替可能である旨を一行追記。
- `architecture.md`（必要なら）: API レイヤ（`api.sketch`）と GUI の責務分担更新。

## 実装タスクリスト（チェックリスト）

- [x] overlay: `set_enabled(on: bool)` を追加し、`draw()` 冒頭で `_enabled` を確認（既定 True）。
- [x] parameters: `ParameterManager.initialize()` で `runner.show_hud` を登録（bool, default=True, category="HUD"）。
- [x] sketch: `hud_conf` 解決（`src/api/sketch.py:216` 付近）直後に `initial_hud_on = bool(hud_conf.enabled)` を決定。
- [x] sketch: `use_parameter_gui=True` のとき HUD を常に生成し、`overlay.set_enabled(initial_hud_on)` を適用。
- [x] sketch: Parameter GUI が有効かつ `parameter_manager` があれば、`parameter_manager.store.set_override("runner.show_hud", initial_hud_on)` で初期同期。
- [x] sketch_runner: `params.py` に HUD 可視性購読を追加。`show_hud is None` のときのみ有効化。
- [x] sketch: `metrics_snapshot_fn` の有効化条件を `hud_conf.show_cache_status` のみに変更（`src/api/sketch.py:231-236` 付近）。
- [x] README/architecture（最小更新）: HUD トグルの追記。
- [x] 変更ファイルの `ruff`/`black`/`isort`/`mypy` を通す（変更ファイル限定）。

## 受け入れ条件（DoD）

- 変更ファイルに対する `ruff`/`black`/`isort`/`mypy` が成功。
- サンプル実行で以下が確認できる:
  - `use_parameter_gui=True, show_hud=None` で GUI の「Show HUD」チェックにより HUD が即時表示/非表示に切り替わる。
  - `use_parameter_gui=True, show_hud=True/False` では GUI トグルが無効（ロック）で、初期状態が維持される。
  - 録画/スクリーンショット等の既存操作（`V`, `P`）は回帰しない。

## 動作確認手順（手動）

- 実行: `python main.py`（適当な `sketch/*.py` を読む既定動作）。
- GUI 有効: `sketch/xxx.py` 内で `use_parameter_gui=True` を渡すか、`main.py` の呼び出し側で指定。
- GUI の「HUD」セクションに `Show HUD` が表示され、チェックで HUD の表示が変わることを確認。
- `show_hud=True/False` を明示した場合はトグルを変更しても状態が変わらないことを確認。

## リスクと対応

- Overlay を常に生成（GUI 有効時）→ オーバーヘッド: 小。描画は `set_enabled(False)` で抑止。
- 明示引数と GUI の衝突: ルール（明示 > GUI）で解決。必要なら UI 上でロック状態の注記を検討（今回は非対応）。
- 既存コードとの整合: `metrics_snapshot` の条件変更により HUD 非表示でも計測が走る。オーバーヘッドは軽微でメリット（即時反映）が勝ると判断。

## 確認事項（要回答）

- 明示引数ロック時に GUI の `Show HUD` を無効表示（グレイアウト）にする対応は必要か（今回は非対応の予定）。→ 不要
- GUI 無効時（`use_parameter_gui=False`）でも後からホットキー等で HUD を出す要件はあるか（無しと想定）。→ なし

---

承認いただければ、上記チェックリストに沿って着手します。必要な調整があればコメントください。
