import numpy as np
import pytest


@pytest.mark.smoke
def test_api_shape_import_and_registration():
    from api import G, shape
    from engine.core.geometry import Geometry
    from shapes import registry as reg
    from shapes.base import BaseShape

    NAME = "tmp_shape_smoke"

    # 事前に同名があれば掃除
    reg.unregister(NAME)

    @shape(NAME)
    class _TmpShape(BaseShape):
        def generate(self, *, n: int = 3, r: float = 1.0, **_):
            th = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=np.float32)
            xy = np.c_[r * np.cos(th), r * np.sin(th)]
            return Geometry.from_lines([xy])

    # レジストリに登録されていること
    assert reg.is_shape_registered(NAME)

    # G ファクトリ経由で Geometry を取得できること
    g = G.tmp_shape_smoke(n=5, r=2.0)
    assert isinstance(g, Geometry)
    assert not g.is_empty

    # 後片付け
    reg.unregister(NAME)


def test_old_import_path_is_removed():
    with pytest.raises(ImportError):
        # 破壊的変更: 旧経路は不可
        from api.shape_registry import shape  # noqa: F401
