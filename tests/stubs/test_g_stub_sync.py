from __future__ import annotations

from pathlib import Path

from scripts.gen_g_stubs import generate_stubs_str


def test_api_stub_file_is_in_sync_with_generator() -> None:
    expected = generate_stubs_str()
    pyi_path = Path(__file__).resolve().parents[2] / "src" / "api" / "__init__.pyi"
    actual = pyi_path.read_text(encoding="utf-8")
    assert actual == expected
