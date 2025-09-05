import ast
from pathlib import Path


def parse_builder_methods(text: str) -> set[str]:
    tree = ast.parse(text)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_PipelineBuilder":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    names.add(item.name)
            break
    return names


def test_pipeline_stub_sync():
    from effects.registry import list_effects

    pyi_path = Path(__file__).resolve().parents[1] / "api" / "__init__.pyi"
    assert pyi_path.exists(), "api/__init__.pyi not found; run `python -m scripts.gen_g_stubs`"
    stub_text = pyi_path.read_text(encoding="utf-8")

    stub_methods = parse_builder_methods(stub_text)
    reg_effects = set(list_effects())

    missing_in_stub = sorted(reg_effects - stub_methods)
    extra_in_stub = sorted(stub_methods - reg_effects - {"build", "strict", "cache", "__call__"})

    assert not missing_in_stub and not extra_in_stub, (
        "Effect names mismatch between registry and stub.\n"
        f"Missing in stub: {missing_in_stub}\n"
        f"Extra in stub: {extra_in_stub}\n"
        "Regenerate stubs: python -m scripts.gen_g_stubs"
    )
