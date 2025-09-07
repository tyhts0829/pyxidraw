# Optional Dependency Tests

These tests exercise integrations that require optional, heavier dependencies. They are all marked with the `optional` pytest marker and use `pytest.importorskip` to skip cleanly when the dependency is not available.

## Targeted libraries
- `shapely` — geometry buffering used by `effects.offset`
- `fontTools` + `fontPens` + `numba` — font outline rendering used by `shapes.text`
- `numba` — JIT-accelerated math paths (e.g., `shapes.torus`)
- `mido` — MIDI message handling in `engine.io`

## Setup
Install the libraries you want to exercise. For all tests:

```bash
pip install shapely fonttools fontpens numba mido
```

Notes:
- `fontPens` is a separate package required by `shapes.text`.
- On Linux, ensure at least one system font (e.g. DejaVuSans) is present under `/usr/share/fonts`.

## Run
Execute only the optional test group:

```bash
pytest -m optional -q
```

## Expectations
- When the corresponding dependency is present, each file runs at least one “happy-path” assertion.
- When a dependency is missing, the tests are reported as `skipped` (not failed).

