from __future__ import annotations

from pathlib import Path

from scripts.gen_g_stubs import generate_stubs_str

# What this tests (TEST_PLAN.md §Stub Sync)
# - The generated stub string (scripts.gen_g_stubs.generate_stubs_str) matches api/__init__.pyi on disk.




def test_g_stub_file_matches_generator_output():
    expected = generate_stubs_str()
    path = Path(__file__).resolve().parents[1] / "src" / "api" / "__init__.pyi"
    on_disk = path.read_text(encoding="utf-8")
    assert on_disk == expected
