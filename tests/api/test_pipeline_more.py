from __future__ import annotations


from api import E
from api.effects import Pipeline, _is_json_like
from engine.core.geometry import Geometry


def test_noop_placeholder_to_keep_file_active() -> None:
    # 仕様縮減により to_spec/from_spec/validate_spec を削除。
    # 本ファイルでは _is_json_like のテストと他のケースを維持する。
    assert isinstance(Pipeline.__name__, str)


def test_pipeline_cache_none_and_clear_cache_behavior() -> None:
    # maxsize=None（無制限） + clear_cache の挙動
    g = Geometry.from_lines([[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]])
    p = E.pipeline.cache(maxsize=None).rotate(angles_rad=(0.0, 0.0, 0.0)).build()
    a = p(g)
    b = p(g)
    assert a is b  # ヒット
    p.clear_cache()
    c = p(g)
    assert c is not b


def test__is_json_like_boundaries() -> None:
    assert _is_json_like({"a": [1, 2.0, "x", True, None]})
    assert _is_json_like(((1, 2), [3, 4]))
    assert not _is_json_like(set([1, 2]))

    class X:  # noqa: D401 - テスト用
        pass

    assert not _is_json_like(X())
