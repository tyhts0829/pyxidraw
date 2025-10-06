# HUD 横棒メータ実装計画（提案）

目的
- CPU 使用率、メモリ使用量、ライン本数、頂点数、FPS をスマートな横棒メータで可視化し、数値だけでなく直感的な負荷把握を可能にする。
- 既存 HUD（テキストラベル）を壊さず、同一の更新サイクル内で低オーバーヘッドに描画する。

対象メトリクス（5 種）
- `CPU`：プロセスの CPU 使用率 [%]
- `MEM`：プロセス RSS [bytes]
- `LINE`：ポリライン本数 `Geometry.n_lines`
- `VERTEX`：頂点数 `len(coords)`
- `FPS`：実効フレームレート [Hz]

仕様概要（バー表示）
- レイアウト：既存テキストの右側に横棒メータをインライン表示（`inline_right` 固定）。下段配置は当面非対応（将来拡張）。
- バー幅/高さ：既定 `width=160px`, `height=6px`。行間 `gap=6px`。
- 色：背景（薄灰, alpha 120）/ 前景（アクセント単色, alpha 220）。段階色は使用しない。
- 数値表記：従来の `key : value` テキストは維持。バーは補助視覚要素。
- スムージング：EMA（指数移動平均）で値揺れを軽減（`alpha=0.5` 既定）。

正規化（0..1 へのマッピング）
- `CPU`：`value / 100`。
- `MEM`：`rss / mem_max_bytes`。`mem_max_bytes` は `psutil.virtual_memory().total` を既定とし、設定で `process_peak`/`custom_bytes` も可（`configs/default.yaml` で上書き可）。
- `FPS`：`fps / 60.0`（固定 60 を目標）。超過時は 1.0 で頭打ち。
- `VERTEX`：`min(vertices / vertex_max, 1.0)`。`vertex_max` は既定 10,000,000（1000万点）。
- `LINE`：`min(lines / line_max, 1.0)`。`line_max` は既定 5,000,000（500万線）。

設定拡張（`HUDConfig` 案）
- `show_meters: bool = True`（メータ全体の有効/無効）
- `meter_width_px: int = 160`, `meter_height_px: int = 6`, `meter_gap_px: int = 6`
- `meter_alpha_fg: int = 220`, `meter_alpha_bg: int = 120`
- `meter_color_fg: tuple[int,int,int] = (0, 120, 220)`（単色）
- `smoothing_alpha: float = 0.5`（EMA）
- `target_fps: float = 60.0`（固定 60）
- `mem_scale: Literal['system_total','process_peak','custom'] = 'system_total'`（`configs/default.yaml` で上書き可）
- `mem_custom_bytes: int | None = None`
- `vertex_max: int = 10_000_000`（既定）
- `line_max: int = 5_000_000`（既定）

実装変更点（小さく安全に）
1) Sampler を拡張（数値の併走保管）
   - `MetricSampler` に `values: dict[str, float]` を追加（CPU[%], MEM[bytes], FPS[Hz], VERTEX[int], LINE[int]）。
   - EMA 用に `ema: dict[str, float]`、ピーク用に `peaks: dict[str, float]` を保持。
   - `target_fps` を `HUDConfig` から参照。未設定は 60。
   - `mem_max_bytes` 解決（`psutil.virtual_memory().total`/peak/custom）。

2) Overlay を拡張（横棒描画）
   - `OverlayHUD` にメータ描画を追加。`pyglet.shapes.Rectangle` もしくは `pyglet.graphics.Batch` で矩形を再利用し、毎フレームの新規生成を避ける。
   - テキストの計算済み座標に対し、右横（または下）にバーを配置。`config.meter_gap_px`で間隔調整。
   - 正規化は `values` と `peaks/target` に基づき 0..1 を算出。色は単色（段階色なし）。

3) 設定/API
   - `HUDConfig` に上記項目を追加。既定でメータ ON、互換性維持（設定未指定でも従来表示は維持）。
   - `configs/default.yaml` に `hud.meters.mem_scale`、`hud.meters.vertex_max`、`hud.meters.line_max` 等のキーを追加し、`util.utils.load_config()` 経由で解決。
   - 既存のテキスト HUD と順序ロジックを流用（`resolved_order()`）。

4) ドキュメント
   - `architecture.md` の HUD 節に「横棒メータ」追記（簡潔に）。
   - 簡易スクリーンショットを `screenshots/` に追加（任意）。

5) テスト/検証
   - ユニットテスト（ロジック中心）
     - 正規化ヘルパ（MEM/FPS/PEAK）と EMA の単体テスト。
     - `MetricSampler` が `values` を正しく更新すること。
   - 手動目視（smoke）
     - 小/中/大の幾何でメータ変化を確認。`Shift+P` の PNG 保存時も HUD が崩れないことを確認。

パフォーマンス/安全性
- 描画は矩形再利用でアロケーションを抑制。更新は座標/幅のみ。
- EMA/ピーク更新はサンプル周期（既定 0.5s）でのみ計算。
- 依存追加なし（`psutil` は既存利用）。

確定事項（反映済み）
- 配置：右側（inline_right 固定）。
- FPS 目標：固定 60。
- MEM スケール：システム総メモリに対する比（`configs/default.yaml` で上書き可）。
- 色：単色（段階色なし）。
- バー寸法：160x6（既定）。
- VERTEX の 100%：1000万点で 100%（固定上限、超過はクリップ）。
- LINE の 100%：500万線で 100%（固定上限、超過はクリップ）。

タスクチェックリスト（進行管理）
- [ ] HUDConfig にメータ設定を追加（既定 ON、inline_right 固定）
- [ ] MetricSampler に `values/ema/peaks` を追加し、各値更新
- [ ] MEM/FPS/正規化ヘルパ（固定上限スケーリング）を実装（単体テスト付き）
- [ ] OverlayHUD にバー描画（矩形再利用）を実装（単色）
- [ ] `configs/default.yaml` に `hud.meters.mem_scale` / `hud.meters.vertex_max` / `hud.meters.line_max` を追加し、読込実装
- [ ] ドキュメント更新（architecture.md）
- [ ] 簡易スクショ作成（任意）
- [ ] 変更ファイルの ruff/black/isort/mypy/最小 pytest を実施

影響ファイル（予定）
- `src/engine/ui/hud/config.py`
- `src/engine/ui/hud/sampler.py`
- `src/engine/ui/hud/overlay.py`
- `src/engine/ui/hud/fields.py`（変更なし予定）
- `architecture.md`（短い追記）
- `configs/default.yaml`（設定キー追加）

DoD（完了条件）
- 設定未指定で従来 HUD の文言が維持され、追加で横棒メータが表示される。
- CPU/MEM/LINE/VERTEX/FPS の 5 本が適切に 0..1 スケーリングされ、EMA により過度のちらつきがない。
- ruff/black/isort/mypy が緑。任意の小テストが緑。

補足（実装方針の簡潔な根拠）
- 既存の HUD は Label ベースでシンプル。矩形も最小限で保ち、描画負荷を増やさずに可視性を上げる設計とする。
- メモリとジオメトリ規模は絶対値の意味付けが環境依存のため、相対バー + 数値の併記で認知負荷を下げる。
