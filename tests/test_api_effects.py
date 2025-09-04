"""
api.E パイプライン（新API）のテスト

提案3/4に基づく単一パイプライン＋単層キャッシュの基本動作を検証。
"""
import math

import pytest

from api import E, G, Geometry


@pytest.fixture
def simple_geometry():
    return G.polygon(n_sides=3)


@pytest.fixture
def complex_geometry():
    return G.sphere(subdivisions=0.3)


class TestPipeline:
    def test_basic_pipeline(self, simple_geometry):
        pipeline = (
            E.pipeline
            .rotate(angles_rad=(0, 0, 0.2 * 3.141592653589793))
            .scale(scale=(1.5, 1.5, 1.5))
            .build()
        )
        out = pipeline(simple_geometry)
        assert isinstance(out, Geometry)

    def test_translation_rotation_scaling(self, simple_geometry):
        out = (
            E.pipeline
            .translate(delta=(10, 20, 0))
            .rotate(pivot=(0, 0, 0), angles_rad=(0, 0, 1.5707963267948966))
            .scale(pivot=(0, 0, 0), scale=(2, 2, 2))
            (simple_geometry)
        )
        assert isinstance(out, Geometry)

    def test_noise_and_filling(self, simple_geometry):
        out = (
            E.pipeline
            .displace(amplitude_mm=0.3, spatial_freq=(0.2, 0.2, 0.2), t_sec=0.0)
            .fill(mode="lines", density=0.5, angle_rad=0.0)
            (simple_geometry)
        )
        assert isinstance(out, Geometry)

    def test_complex_pipeline(self, complex_geometry):
        out = (
            E.pipeline
            .rotate(angles_rad=(0.2 * 3.141592653589793, 0.2 * 3.141592653589793, 0.2 * 3.141592653589793))
            .scale(scale=(1.2, 1.2, 1.2))
            .subdivide(subdivisions=0.6)
            .displace(amplitude_mm=0.2, t_sec=0.5, spatial_freq=0.3)
            (complex_geometry)
        )
        assert isinstance(out, Geometry)

    def test_pipeline_cache(self, simple_geometry):
        builder = (
            E.pipeline
            .rotate(angles_rad=(0, 0, 0.2 * 3.141592653589793))
            .scale(scale=(1.5, 1.5, 1.5))
        )
        p = builder.build()
        r1 = p(simple_geometry)
        r2 = p(simple_geometry)
        assert isinstance(r1, Geometry)
        assert isinstance(r2, Geometry)
