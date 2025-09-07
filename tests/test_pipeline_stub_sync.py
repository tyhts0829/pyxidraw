from __future__ import annotations

from pathlib import Path

from scripts.gen_g_stubs import generate_stubs_str

# What this tests (TEST_PLAN.md §Stub Sync)
# - Generated stub contains Pipeline builder protocol and known effects names.
# - On-disk api/__init__.pyi exactly equals generator output.


def test_pipeline_stub_contains_effects_and_matches_file():
    content = generate_stubs_str()
    # sanity: builder class and at least one known effect should appear
    assert "class _PipelineBuilder(Protocol):" in content
    assert " def rotate(" in content

    path = Path(__file__).resolve().parents[1] / "src" / "api" / "__init__.pyi"
    on_disk = path.read_text(encoding="utf-8")
    assert on_disk == content
