# HUD リファクタ残タスク（未実施チェックリスト）

目的
- これまでの実装を踏まえ、未実施項目を明確化し追跡する。

未実施（必須）
- [x] `engine/runtime/cache_snapshot.py` の依存を `api.*` から下位層へ置換
  - [x] effects 側: `api.effects.global_cache_counters` を `effects.metrics.global_cache_counters` へ移動/再エクスポート
  - [x] shapes 側: `ShapesAPI.cache_info()` 代替の下位 API を `shapes.metrics.shape_cache_counters()` として提供
  - [x] `snapshot_counters()` の import を上記へ切替
  - [x] `architecture.md` を更新（依存方向図を反映）

未実施（任意/提案）
- [x] HUD モジュール再配置（提案A）
  - [x] `src/engine/ui/monitor.py` → `src/engine/ui/hud/sampler.py`
  - [x] `src/engine/ui/overlay.py` → `src/engine/ui/hud/overlay.py`
  - [x] import と `architecture.md` の参照更新

ドキュメント整備
- [x] README に `HUDConfig` の使用例を追記（`from engine.ui.hud import HUDConfig`）
- [x] `docs/hud_metrics_refactor_plan.md` の参照（`docs/overlay_cache_status_plan.md`）を整理（ファイル非存在のため修正/削除）
- [x] `docs/hud_directory_reorg_proposal.md` に Plan A 採用済みの注記を追記（現状説明の整合確認）

スモーク/計測
- [ ] HUD ON/OFF と CACHE ON/OFF で FPS/CPU 平均（60秒）を比較・記録
- [ ] `workers=0/複数` で `CACHE/SHAPE`/`CACHE/EFFECT` が更新されることを確認

追加オプション（任意）
- [ ] `main.py` に `hud_config` を渡す CLI/設定フック（例: `--hud-cache on/off`）
- [ ] psutil 未導入時の挙動を README に注記（`show_cpu_mem=False` で回避可能）
