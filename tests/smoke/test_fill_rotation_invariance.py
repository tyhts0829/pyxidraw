import pytest

from api import E, G


@pytest.mark.smoke
def test_fill_line_count_rotation_invariant_square():
    # 正方形に対して、Z 回転 0 と 45 度で生成される塗り線本数が概ね同じ（±1）
    base = G.polygon(n_sides=4).scale(100.0)

    density = 50.0
    angle = 0.0  # スキャン角（水平ハッチ）

    pipe0 = (
        E.pipeline.affine(rotation=(0.0, 0.0, 0.0))
        .fill(density=density, angle_sets=1, angle=angle, remove_boundary=True)
        .build()
    )
    pipe45 = (
        E.pipeline.affine(rotation=(0.0, 0.0, 45.0))
        .fill(density=density, angle_sets=1, angle=angle, remove_boundary=True)
        .build()
    )

    g0 = pipe0(base)
    g45 = pipe45(base)

    # arange の端数で ±1 本ぶれることがあるため許容
    assert abs(g0.n_lines - g45.n_lines) <= 1


@pytest.mark.smoke
def test_fill_line_count_tilt_invariant_square_xy():
    # 正方形に対して、X/Y 回転を変えても塗り線本数が概ね一定（±1）
    base = G.polygon(n_sides=4).scale(100.0)

    density = 50.0
    angle = 0.0  # スキャン角（水平ハッチ）

    def _count_lines(rx_deg: float, ry_deg: float) -> int:
        pipe = (
            E.pipeline.affine(rotation=(rx_deg, ry_deg, 0.0))
            .fill(density=density, angle_sets=1, angle=angle, remove_boundary=True)
            .build()
        )
        return pipe(base).n_lines

    base_count = _count_lines(0.0, 0.0)

    # X 軸周りのチルト
    for deg in [15, 30, 45, 60, 75]:
        n = _count_lines(deg, 0.0)
        assert abs(n - base_count) <= 1

    # Y 軸周りのチルト
    for deg in [15, 30, 45, 60, 75]:
        n = _count_lines(0.0, deg)
        assert abs(n - base_count) <= 1

    # X/Y の組み合わせチルト
    combos = [(15, 15), (30, 15), (30, 30), (45, 30), (60, 45), (75, 60)]
    for rx_deg, ry_deg in combos:
        n = _count_lines(rx_deg, ry_deg)
        assert abs(n - base_count) <= 1
