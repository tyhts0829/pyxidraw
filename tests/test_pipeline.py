"""
api.E パイプライン（新API）に合わせたパイプラインテスト
"""
import numpy as np
import pytest

from api import E, Geometry
from engine.core.geometry import Geometry as GGeom


@pytest.fixture
def simple_geometry():
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    return GGeom.from_lines(lines)


class TestPipelineBuilder:
    def test_empty_pipeline_passthrough(self, simple_geometry):
        out = E.pipeline.build()(simple_geometry)
        assert isinstance(out, Geometry)
        c0, _ = simple_geometry.as_arrays()
        c1, _ = out.as_arrays()
        np.testing.assert_allclose(c0, c1, rtol=1e-6)

    def test_order_matters(self, simple_geometry):
        p1 = E.pipeline.scale(scale=(2.0, 2.0, 2.0)).translate(delta=(5.0, 0.0, 0.0)).build()
        p2 = E.pipeline.translate(delta=(5.0, 0.0, 0.0)).scale(scale=(2.0, 2.0, 2.0)).build()
        r1 = p1(simple_geometry)
        r2 = p2(simple_geometry)
        assert isinstance(r1, Geometry)
        assert isinstance(r2, Geometry)
        c1, _ = r1.as_arrays()
        c2, _ = r2.as_arrays()
        assert not np.allclose(c1, c2)

    def test_progressive_vs_pipeline(self, simple_geometry):
        # 段階適用
        from effects.rotate import rotate
        from effects.scale import scale
        from effects.translate import translate

        step1 = rotate(simple_geometry, angles_rad=(0.0, 0.0, 1.5707963267948966))
        step2 = scale(step1, scale=(2.0, 2.0, 2.0))
        step3 = translate(step2, delta=(5.0, 3.0, 0.0))

        # パイプライン適用
        p = (
            E.pipeline
            .rotate(angles_rad=(0.0, 0.0, 1.5707963267948966))
            .scale(scale=(2.0, 2.0, 2.0))
            .translate(delta=(5.0, 3.0, 0.0))
            .build()
        )
        pr = p(simple_geometry)

        c3, _ = step3.as_arrays()
        cp, _ = pr.as_arrays()
        np.testing.assert_allclose(c3, cp, rtol=1e-6)
