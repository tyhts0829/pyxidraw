import numpy as np

from api import E
from engine.core.geometry import Geometry


def _geom():
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    return Geometry.from_lines([pts])


def test_pipeline_cache_hit_and_miss():
    g = _geom()
    p = (E.pipeline.rotation(rotate=(0.0, 0.0, 0.25)).build())

    out1 = p(g)
    out2 = p(g)
    # 単層キャッシュ: 同一 Geometry + Pipeline で同一オブジェクト参照
    assert out1 is out2

    # パラメータが違えばミス
    p2 = (E.pipeline.rotation(rotate=(0.0, 0.0, 0.26)).build())
    out3 = p2(g)
    assert out3 is not out1

    # Geometry が変わればミス
    g2 = _geom().translate(1, 0, 0)
    out4 = p(g2)
    assert out4 is not out1
