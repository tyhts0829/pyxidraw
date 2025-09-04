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
            .rotate(rotate=(0, 0, 0.1))
            .scale(scale=(1.5, 1.5, 1.5))
            .build()
        )
        out = pipeline(simple_geometry)
        assert isinstance(out, Geometry)

    def test_translation_rotation_scaling(self, simple_geometry):
        out = (
            E.pipeline
            .translate(offset_x=10, offset_y=20, offset_z=0)
            .rotate(center=(0, 0, 0), rotate=(0, 0, 0.25))
            .scale(center=(0, 0, 0), scale=(2, 2, 2))
            (simple_geometry)
        )
        assert isinstance(out, Geometry)

    def test_noise_and_filling(self, simple_geometry):
        out = (
            E.pipeline
            .displace(intensity=0.3, frequency=(0.2, 0.2, 0.2), time=0.0)
            .fill(pattern="lines", density=0.5, angle=0.0)
            (simple_geometry)
        )
        assert isinstance(out, Geometry)

    def test_complex_pipeline(self, complex_geometry):
        out = (
            E.pipeline
            .rotate(rotate=(0.1, 0.1, 0.1))
            .scale(scale=(1.2, 1.2, 1.2))
            .subdivide(subdivisions=0.6)
            .displace(intensity=0.2, time=0.5, frequency=0.3)
            (complex_geometry)
        )
        assert isinstance(out, Geometry)

    def test_pipeline_cache(self, simple_geometry):
        builder = (
            E.pipeline
            .rotate(rotate=(0, 0, 0.1))
            .scale(scale=(1.5, 1.5, 1.5))
        )
        p = builder.build()
        r1 = p(simple_geometry)
        r2 = p(simple_geometry)
        assert isinstance(r1, Geometry)
        assert isinstance(r2, Geometry)
