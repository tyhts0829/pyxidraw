import numpy as np
import pytest

from api import E, G

try:  # shapely の有無で Voronoi テストを制御
    import shapely  # type: ignore  # noqa: F401

    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False


def _is_closed(line: np.ndarray, eps: float = 1e-6) -> bool:
    if line.shape[0] < 2:
        return False
    return float(np.linalg.norm(line[0] - line[-1])) <= eps


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely が無い環境では Voronoi をスキップ")
def test_voronoi_square_closed():
    # 正方形を Voronoi で分割（サイト 9）
    g = G.polygon(n_sides=4).scale(100.0, 100.0, 1.0)
    pipe = E.pipeline.partition(site_count=9, seed=1).build()
    out = pipe(g)
    coords, offsets = out.as_arrays(copy=False)

    assert offsets.size > 1
    # 全ループが閉じていること
    for i in range(len(offsets) - 1):
        line = coords[offsets[i] : offsets[i + 1]]
        assert _is_closed(line)


def test_partition_then_fill_smoke():
    # 分割 → fill（後段）が問題なく走り、線が生成される
    g = G.polygon(n_sides=5).scale(120.0, 120.0, 1.0)
    pipe = (
        E.pipeline.partition(site_count=12, seed=2).fill(density=10.0, remove_boundary=True).build()
    )
    out = pipe(g)
    coords, offsets = out.as_arrays(copy=False)
    assert coords.shape[0] > 0
    assert offsets.size > 1


def _pnpoly(pt: np.ndarray, poly: np.ndarray) -> bool:
    x, y = float(pt[0]), float(pt[1])
    inside = False
    for i in range(poly.shape[0] - 1):
        x1, y1 = poly[i, 0], poly[i, 1]
        x2, y2 = poly[i + 1, 0], poly[i + 1, 1]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
            if x <= xin:
                inside = not inside
    return inside


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely が無い環境では Voronoi をスキップ")
def test_donut_excludes_inner_hole():
    # 外周と内周（ドーナツ）。穴には三角セルが生成されないこと。
    outer = G.polygon(n_sides=48).scale(200.0, 200.0, 1.0)
    inner = G.polygon(n_sides=48).scale(90.0, 90.0, 1.0)
    g = outer + inner

    pipe = E.pipeline.partition(site_count=9, seed=3).build()
    out = pipe(g)
    coords, offsets = out.as_arrays(copy=False)

    # 参照用に内周輪郭を 2D 取得
    inner_coords, inner_offsets = inner.as_arrays(copy=False)
    inner_ring = inner_coords[inner_offsets[0] : inner_offsets[1], :2]

    # 生成三角の重心が内周の内側に存在しないこと
    for i in range(len(offsets) - 1):
        tri = coords[offsets[i] : offsets[i + 1]]
        if tri.shape[0] < 3:
            continue
        c2d = np.mean(tri[:3, :2], axis=0)
        assert not _pnpoly(c2d, inner_ring)
