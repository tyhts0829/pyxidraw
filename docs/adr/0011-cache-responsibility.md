# ADR 0011: Cache Responsibility (Shapes vs. Factory vs. Pipeline)

Date: 2025-09-04

## Status
Accepted

## Context
- Shapes historically implemented caching via `BaseShape` (LRU-based) and the high-level `ShapeFactory` also cached shape instances/outputs. This led to potential double-caching and unclear ownership.
- The new unified `Geometry` model and function-based effects favor simple, predictable cache boundaries.

## Decision
- Cache ownership for shape generation is centralized in `api/shape_factory.ShapeFactory`.
- `shapes.base.BaseShape` keeps a cache facility but is disabled by default (`enable_cache=False`). Authors may opt-in per-shape if they have strong locality requirements.
- Effect-level caching is not introduced; `api.pipeline.Pipeline` keeps a single-layer LRU-like cache keyed by `(geometry_digest, pipeline_key)` with optional env override `PXD_PIPELINE_CACHE_MAXSIZE`.

## Consequences
- Removes ambiguity and reduces risk of cache incoherence.
- Keeps an escape hatch for shape-specific micro-caching without changing the public API.

## Migration
- Existing shapes continue to work; no code change is required for users.
- Contributors adding new shapes should prefer relying on `ShapeFactory` and leave BaseShape caching disabled unless justified.

## Related
- ADR 0005: Pipeline and cache
- docs/simplicity_readability_audit_2025-09-04.md
