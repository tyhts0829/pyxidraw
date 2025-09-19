from engine.ui.parameters.introspection import FunctionIntrospector


def sample_function(a: float = 1.0, *, b: int = 2) -> None:
    """Sample doc first line.

    Additional detail.
    """


sample_function.__param_meta__ = {
    "b": {"min": 0, "max": 10},
}


def test_function_introspector_resolves_doc_signature_and_meta():
    introspector = FunctionIntrospector()

    info = introspector.resolve(kind="shape", name="sample", fn=sample_function)

    assert info.kind == "shape"
    assert info.name == "sample"
    assert info.doc == "Sample doc first line."
    assert info.signature is not None
    assert info.signature.parameters["b"].default == 2
    assert info.param_meta["b"]["max"] == 10

    # キャッシュ動作の確認: 再度解決しても値が変わらない
    info_cached = introspector.resolve(kind="shape", name="sample", fn=sample_function)
    assert info_cached.doc == info.doc
    assert info_cached.param_meta == info.param_meta
