"""
displace エフェクト（Perlin ノイズ変位）

- 3D Perlin ノイズを各頂点に加算し、表面/線を粗く揺らします。
- x/y/z の 3 成分を別位相で生成するため、等方的な乱れになります。

主なパラメータ:
- amplitude_mm: 変位量 [mm]。
- spatial_freq: 空間周波数。float なら等方、Vec3 なら各軸別周波数。
- t_sec: 時間オフセット。値を変えるとアニメ的に流れるノイズへ。

実装メモ:
- Numba 最適化された `perlin_core` を使用し、Permutation/Gradient は `util.constants.NOISE_CONST` を参照。
- 入力が空または変位 0 の場合はコピーを返す純関数です。
"""

from __future__ import annotations

import numpy as np
from numba import njit

from common.types import Vec3
from engine.core.geometry import Geometry
from util.constants import NOISE_CONST

from .registry import effect


@njit(fastmath=True, cache=True)
def fade(t):
    """Perlinノイズ用のフェード関数.

    引数:
        t: 入力値。

    返り値:
        フェード処理された値。
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


@njit(fastmath=True, cache=True)
def lerp(a, b, t):
    """線形補間.

    引数:
        a: 開始値。
        b: 終了値。
        t: 補間パラメータ。

    返り値:
        補間された値。
    """
    return a + t * (b - a)


@njit(fastmath=True, cache=True)
def grad(hash_val, x, y, z, grad3_array):
    """勾配ベクトル計算.

    引数:
        hash_val: ハッシュ値。
        x: X座標。
        y: Y座標。
        z: Z座標。
        grad3_array: 勾配ベクトル配列。

    返り値:
        計算された勾配値。
    """
    # 安全なインデックスアクセス
    idx = int(hash_val) % 12
    g = grad3_array[idx]
    return g[0] * x + g[1] * y + g[2] * z


@njit(fastmath=True, cache=True)
def perlin_noise_3d(x, y, z, perm_table, grad3_array):
    """3次元Perlinノイズ生成"""
    # セル空間上の位置を求める
    X = int(np.floor(x)) & 255
    Y = int(np.floor(y)) & 255
    Z = int(np.floor(z)) & 255

    # 内部点（小数部分）
    x -= np.floor(x)
    y -= np.floor(y)
    z -= np.floor(z)

    # フェード関数
    u = fade(x)
    v = fade(y)
    w = fade(z)

    # Numba最適化: 直接インデックス演算を使用（配列長は未使用）

    A = perm_table[X] + Y
    AA = perm_table[A & 511] + Z  # 511 = 2*256-1
    AB = perm_table[(A + 1) & 511] + Z
    B = perm_table[(X + 1) & 255] + Y
    BA = perm_table[B & 511] + Z
    BB = perm_table[(B + 1) & 511] + Z

    # 8つのコーナーでのグラディエントドット
    gAA = grad(perm_table[AA & 511], x, y, z, grad3_array)
    gBA = grad(perm_table[BA & 511], x - 1, y, z, grad3_array)
    gAB = grad(perm_table[AB & 511], x, y - 1, z, grad3_array)
    gBB = grad(perm_table[BB & 511], x - 1, y - 1, z, grad3_array)
    gAA1 = grad(perm_table[(AA + 1) & 511], x, y, z - 1, grad3_array)
    gBA1 = grad(perm_table[(BA + 1) & 511], x - 1, y, z - 1, grad3_array)
    gAB1 = grad(perm_table[(AB + 1) & 511], x, y - 1, z - 1, grad3_array)
    gBB1 = grad(perm_table[(BB + 1) & 511], x - 1, y - 1, z - 1, grad3_array)

    # trilinear補間
    return lerp(
        lerp(lerp(gAA, gBA, u), lerp(gAB, gBB, u), v),
        lerp(lerp(gAA1, gBA1, u), lerp(gAB1, gBB1, u), v),
        w,
    )


@njit(fastmath=True, cache=True)
def perlin_core(
    vertices: np.ndarray, frequency: tuple, perm_table: np.ndarray, grad3_array: np.ndarray
):
    """コア Perlin ノイズ計算（3次元頂点専用）"""
    n = vertices.shape[0]

    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    result = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        x, y, z = (
            vertices[i, 0] * frequency[0],
            vertices[i, 1] * frequency[1],
            vertices[i, 2] * frequency[2],
        )
        # 3成分のノイズをオフセットを変えて生成
        nx = perlin_noise_3d(x, y, z, perm_table, grad3_array)
        ny = perlin_noise_3d(x + 100.0, y + 100.0, z + 100.0, perm_table, grad3_array)
        nz = perlin_noise_3d(x + 200.0, y + 200.0, z + 200.0, perm_table, grad3_array)
        result[i, 0] = np.float32(nx)
        result[i, 1] = np.float32(ny)
        result[i, 2] = np.float32(nz)

    return result


@njit(fastmath=True, cache=True)
def _apply_noise_to_coords(
    coords: np.ndarray,
    intensity: float,
    frequency: tuple,
    time: float,
    perm_table: np.ndarray,
    grad3_array: np.ndarray,
) -> np.ndarray:
    """座標配列にPerlinノイズを適用します。"""
    if coords.size == 0 or not intensity:
        return coords.copy()

    # 係数調整（提案5: 強度の意図的な倍率を廃止して素直に反映）
    t_offset = np.float32(time * 10 + 1000.0)

    # オフセット付き頂点を作成
    offset_coords = coords + t_offset

    # Perlinノイズ計算
    noise_offset = perlin_core(offset_coords, frequency, perm_table, grad3_array)

    return coords + noise_offset * np.float32(intensity)


# Perlinノイズ用のPermutationテーブルを作成
# 命名: 定数は UPPER_SNAKE_CASE（互換のため旧名も残す）
NOISE_PERMUTATION_TABLE = np.array(NOISE_CONST["PERM"], dtype=np.int32)
NOISE_PERMUTATION_TABLE = np.concatenate(
    [NOISE_PERMUTATION_TABLE, NOISE_PERMUTATION_TABLE]
)  # 0-255 を2回連結

# Perlinノイズで使用するグラディエントベクトル
NOISE_GRADIENTS_3D = np.array(NOISE_CONST["GRAD3"], dtype=np.float32)


@effect()
def displace(
    g: Geometry,
    *,
    amplitude_mm: float = 8.0,
    spatial_freq: float | Vec3 = (0.04, 0.04, 0.04),
    t_sec: float = 0.0,
) -> Geometry:
    """3次元頂点にPerlinノイズを追加（クリーンAPI）。"""
    coords, offsets = g.as_arrays(copy=False)

    # パラメータ解決（新形式のみ）
    amp = float(amplitude_mm)
    freq_val = spatial_freq
    ti = float(t_sec)

    # 周波数の整形
    if isinstance(freq_val, (int, float)):
        freq_tuple = (freq_val, freq_val, freq_val)
    elif len(freq_val) == 1:  # type: ignore[arg-type]
        freq_tuple = (freq_val[0], freq_val[0], freq_val[0])  # type: ignore[index]
    else:
        freq_tuple = tuple(freq_val)  # type: ignore[arg-type]

    new_coords = _apply_noise_to_coords(
        coords,
        amp,
        freq_tuple,
        ti,
        NOISE_PERMUTATION_TABLE,
        NOISE_GRADIENTS_3D,
    )
    return Geometry(new_coords, offsets.copy())


displace.__param_meta__ = {
    "amplitude_mm": {"type": "number", "min": 0.0, "max": 50.0},
    "spatial_freq": {
        "type": "number",
        "min": (0.0, 0.0, 0.0),
        "max": (0.1, 0.1, 0.1),
    },
    "t_sec": {"type": "number", "min": 0.0, "max": 60.0},
}
