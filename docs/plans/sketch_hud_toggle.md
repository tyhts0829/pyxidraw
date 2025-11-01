# api.run で HUD 表示を制御できるようにする — 改善計画（提案）

目的
- 利用者が `api.run`/`api.run_sketch` の引数で HUD の表示/非表示を直接トグルできるようにする。
- 既存の `HUDConfig` による詳細設定との両立（後方互換）を維持する。

背景
- 現状は `src/api/sketch.py:205` で `hud_conf: HUDConfig = hud_config or HUDConfig()` としており、
  既定で HUD は有効。引数からの直接トグル導線は無い（`architecture.md:67` にも「HUD は引数指定の導線なし」とある）。

仕様案（最小で明快）
- `run_sketch` にオプション引数 `show_hud: bool | None = None` を追加（`hud_config` の直前に配置）。
  - 既定 `None` は「上書きしない」を意味し、従来挙動（HUD 有効）を保つ。
  - `True/False` 指定時は HUD の有効フラグだけを明示的に上書きする。
- 優先順位（明示 > 既存設定 > 既定）
  1) `show_hud is not None` の場合、その値を優先（`hud_config.enabled` より強い）。
  2) `show_hud is None` かつ `hud_config` が与えられていれば、その `enabled` を採用。
  3) いずれも無ければ従来通り HUD 有効（`True`）。
- 実装方針
  - `hud_conf` の決定は `dataclasses.replace(hud_config, enabled=...)` で「enabled のみ差し替え」。
  - `hud_config is None` の場合は `HUDConfig(enabled=<resolved>)` を新規生成。
  - HUD 無効時は従来通り `overlay=None` として分岐（既存コードは `if overlay is not None:` で安全）。

API 変更（提案シグネチャ）
- 変更箇所: `src/api/sketch.py:105`
  ```py
  def run_sketch(
      user_draw: Callable[[float], Geometry],
      *,
      canvas_size: str | tuple[int, int] = "A5",
      render_scale: float = 4.0,
      line_thickness: float = 0.0006,
      line_color: str | tuple[float, float, float] | tuple[float, float, float, float] | None = None,
      fps: int | None = None,
      background: str | tuple[float, float, float, float] | None = None,
      workers: int = 4,
      use_midi: bool = True,
      init_only: bool = False,
      use_parameter_gui: bool = False,
      show_hud: bool | None = None,           # ← 追加（hud_config より前）
      hud_config: HUDConfig | None = None,
  ) -> None:
  ```

実装タスク（チェックリスト）
- [x] `run_sketch` に `show_hud: bool | None = None` を追加（`src/api/sketch.py:105`）。
- [x] Docstring（Parameters 節）に `show_hud` の説明と優先順位を追記（`src/api/sketch.py:120` 付近）。
- [x] `hud_conf` 解決ロジックを置換（`src/api/sketch.py:205` 付近）。
  - [x] `from dataclasses import replace` を追記し、`hud_config` が与えられた際は `replace(hud_config, enabled=...)` を使用。
  - [x] `metrics_snapshot_fn` の有効/無効判定（`hud_conf.enabled and hud_conf.show_cache_status`）は現行維持。
  - [x] 参照サイトの None 安全性を再確認（`overlay` 使用箇所は全て `if overlay is not None` でガード済み）。
  - [x] `api.run` エイリアスは変更不要だが、使用例に `show_hud` を追加（`architecture.md` に注記を追加／`main.py` は現状維持）。

ドキュメント更新
- [x] `architecture.md` の該当箇所を更新し、`run(..., show_hud=False)` に触れる注記を追加。
- [ ] 実行例への `show_hud` の明示追加は任意（現状は注記のみ）。
- [ ] 必要なら `README.md` の簡易例にも 1 行追記（任意）。

スタブ/テスト/チェック
- [ ] 公開 API 変更に伴いスタブ再生成：
  - 実行: `PYTHONPATH=src python -m tools.gen_g_stubs && git add src/api/__init__.pyi`
- [ ] 変更ファイルに限定したチェックを実行：
  - `ruff check --fix src/api/sketch.py`
  - `black src/api/sketch.py && isort src/api/sketch.py`
  - `mypy src/api/sketch.py`
- [ ] 既存の軽量テストは維持（GL なし）。追加の最小確認：
  - [ ] `tests/api/test_sketch_init_only.py` に `show_hud=False` を渡したケースを 1 ケース追加（早期 return の挙動に影響しないことを確認）。
  - （任意）HUD 解決ロジックを関数化する場合はその関数を単体テスト。

非目標（今回やらないこと）
- HUD の詳細項目（配置/色/フォントなど）の API 扱いは現状維持（`HUDConfig` で指定）。
- Parameter GUI からの HUD トグルは追加しない。

影響範囲と互換性
- 追加パラメータのみの後方互換変更。既存コードは無変更で動作。
- `hud_config` と併用時は `show_hud` が `enabled` を上書き（他の HUD 設定値は保持）。

確認事項（要回答）
- [ ] パラメータ名は `show_hud` で良いか（候補: `hud`, `hud_enabled`）。
- [ ] 既定値は `None`（非上書き）で良いか。`True` 既定にして単純化も可能だが、`hud_config.enabled=False` を渡すケースの意図を尊重するため `None` を提案。
- [ ] `hud_config` 併用時の優先順位（`show_hud` が強い）で問題ないか。

実装後の最小動作例
```python
from api import run

def draw(t):
    ...

run(draw, canvas_size=(400, 400), render_scale=4, show_hud=False)
```

メモ（関連ソースの参考）
- `src/api/sketch.py:270-297` — HUD/Sampler/Overlay と描画統合部。
- `src/engine/ui/hud/config.py:59` — `HUDConfig` 定義（`enabled` 既定 True）。
- `architecture.md:67` — 現状の記述（引数導線なし）。

スケジュール（目安）
- 実装 0.5h、ドキュメント修正 0.5h、スタブ/最小テスト/整形 0.5h。
