"""
effects.transform モジュールのテスト
"""
import numpy as np
import pytest

from effects.transform import transform
from engine.core.geometry import Geometry


@pytest.fixture
def simple_geometry():
    """テスト用の簡単なGeometry"""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32),
    ]
    return Geometry.from_lines(lines)


class TestTransform:
    def test_identity_transform(self, simple_geometry):
        """恒等変換テスト"""
        result = transform(
            simple_geometry,
            center=(0, 0, 0),
            scale=(1, 1, 1),
            rotate=(0, 0, 0)
        )
        assert isinstance(result, Geometry)
        # 恒等変換なので元と同じ
        np.testing.assert_allclose(result.coords, simple_geometry.coords, rtol=1e-6)

    def test_scale_only_transform(self, simple_geometry):
        """スケールのみ変換テスト"""
        result = transform(
            simple_geometry,
            center=(0, 0, 0),
            scale=(2, 2, 2),
            rotate=(0, 0, 0)
        )
        assert isinstance(result, Geometry)
        # 2倍スケールされることを確認
        expected = simple_geometry.coords * 2.0
        np.testing.assert_allclose(result.coords, expected, rtol=1e-6)

    def test_rotate_only_transform(self, simple_geometry):
        """回転のみ変換テスト"""
        result = transform(
            simple_geometry,
            center=(0, 0, 0),
            scale=(1, 1, 1),
            rotate=(0, 0, 0.25)  # Z軸90度回転
        )
        assert isinstance(result, Geometry)

    def test_combined_transform(self, simple_geometry):
        """複合変換テスト"""
        result = transform(
            simple_geometry,
            center=(0.5, 0.5, 0),
            scale=(2, 1.5, 1),
            rotate=(0, 0, 0.25)
        )
        assert isinstance(result, Geometry)

    def test_center_transform(self, simple_geometry):
        """中心点指定変換テスト"""
        center = (1.0, 1.0, 0.5)
        result = transform(
            simple_geometry,
            center=center,
            scale=(2, 2, 2),
            rotate=(0, 0, 0)
        )
        assert isinstance(result, Geometry)

    def test_non_uniform_scale(self, simple_geometry):
        """非均一スケール変換テスト"""
        result = transform(
            simple_geometry,
            center=(0, 0, 0),
            scale=(3, 0.5, 2),
            rotate=(0, 0, 0)
        )
        assert isinstance(result, Geometry)

    def test_multi_axis_rotation(self, simple_geometry):
        """多軸回転変換テスト"""
        result = transform(
            simple_geometry,
            center=(0, 0, 0),
            scale=(1, 1, 1),
            rotate=(0.1, 0.2, 0.3)
        )
        assert isinstance(result, Geometry)

    def test_negative_scale(self, simple_geometry):
        """負のスケール変換テスト（反転）"""
        result = transform(
            simple_geometry,
            center=(0, 0, 0),
            scale=(-1, 1, 1),
            rotate=(0, 0, 0)
        )
        assert isinstance(result, Geometry)

    def test_zero_scale(self, simple_geometry):
        """ゼロスケール変換テスト"""
        result = transform(
            simple_geometry,
            center=(0, 0, 0),
            scale=(0, 1, 1),
            rotate=(0, 0, 0)
        )
        assert isinstance(result, Geometry)

    def test_large_scale(self, simple_geometry):
        """大きなスケール変換テスト"""
        result = transform(
            simple_geometry,
            center=(0, 0, 0),
            scale=(100, 100, 100),
            rotate=(0, 0, 0)
        )
        assert isinstance(result, Geometry)

    def test_full_rotation(self, simple_geometry):
        """1回転変換テスト"""
        result = transform(
            simple_geometry,
            center=(0, 0, 0),
            scale=(1, 1, 1),
            rotate=(0, 0, 1.0)  # 360度
        )
        assert isinstance(result, Geometry)
        # 1回転で元の位置に戻る（浮動小数点の精度を考慮）
        np.testing.assert_allclose(result.coords, simple_geometry.coords, rtol=1e-4, atol=1e-6)

    def test_complex_center(self, simple_geometry):
        """複雑な中心点変換テスト"""
        result = transform(
            simple_geometry,
            center=(10, -5, 3),
            scale=(1.5, 0.8, 2.0),
            rotate=(0.2, -0.1, 0.3)
        )
        assert isinstance(result, Geometry)

    def test_preserves_structure(self, simple_geometry):
        """構造保持テスト"""
        result = transform(
            simple_geometry,
            center=(1, 2, 3),
            scale=(2, 3, 0.5),
            rotate=(0.1, 0.2, 0.3)
        )
        assert isinstance(result, Geometry)
        assert result.coords.shape == simple_geometry.coords.shape
        assert result.offsets.shape == simple_geometry.offsets.shape
        np.testing.assert_array_equal(result.offsets, simple_geometry.offsets)
