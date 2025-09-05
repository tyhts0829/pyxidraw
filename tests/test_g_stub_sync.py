import ast
import re
from pathlib import Path


def parse_stub_shape_names(text: str) -> set[str]:
    """Parse `_GShapes` Protocol in a .pyi and collect method names via AST.

    Supports both one-line stubs ("def f(...): ...") and block-style with docstrings.
    """
    tree = ast.parse(text)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_GShapes":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    names.add(item.name)
            break
    return names


def is_valid_identifier(name: str) -> bool:
    return re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name) is not None


def test_g_stub_sync():
    import shapes  # ensure registration
    from api.shape_registry import list_registered_shapes

    pyi_path = Path(__file__).resolve().parents[1] / "api" / "__init__.pyi"
    assert pyi_path.exists(), "api/__init__.pyi not found; run `python -m scripts.gen_g_stubs`"
    stub_text = pyi_path.read_text(encoding="utf-8")

    stub_names = parse_stub_shape_names(stub_text)
    reg_names = {n for n in list_registered_shapes() if is_valid_identifier(n)}

    missing_in_stub = sorted(reg_names - stub_names)
    extra_in_stub = sorted(stub_names - reg_names)

    assert not missing_in_stub and not extra_in_stub, (
        "Shape names mismatch between registry and stub.\n"
        f"Missing in stub: {missing_in_stub}\n"
        f"Extra in stub: {extra_in_stub}\n"
        "Regenerate stubs: python -m scripts.gen_g_stubs"
    )
