# Changelog

## v4.0.0 — 2025-09-06

破壊的変更（クリーンAPIへの統一）

- effects/noise.displace: 旧引数 `intensity/frequency/time` を削除し、`amplitude_mm/spatial_freq/t_sec` のみを受理。
- PipelineBuilder: 既定で `.strict(True)`（未知パラメータはビルド時に TypeError）。
- validate_spec: 各ステップのパラメータ名を関数シグネチャで照合し、未知キーを常に TypeError。
- Text 形状: `size` → `font_size` に改名（エイリアス無し）。
- api/shape_registry: `CustomShape/ValidatedCustomShape` を削除し、`shapes.base.BaseShape` + `@shapes.registry.shape` に一本化。

新規/改善

- docs/migration.md を追加（v3 → v4 の最小置換表）。
- README に厳格化と `PXD_PIPELINE_CACHE_MAXSIZE` の説明を追記。

