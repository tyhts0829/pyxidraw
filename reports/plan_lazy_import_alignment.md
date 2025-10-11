# Lazy import alignment（撤回メモ）

本ドキュメントで予定していた E/F（numba_compat と OptionalDependencyError）の検討は、
`reports/plan_lazy_import_simplification.md` の適用により不要となったため撤回する。

- E: numba_compat の追加整備 → 撤回（`njit` は no-op デコレータで吸収）。
- F: OptionalDependencyError の導入 → 撤回（ImportError をそのまま上げ、レイヤ境界で扱う）。

以後、依存は「性能目的の遅延のみ」かつ「実利用時 import」で統一する。

