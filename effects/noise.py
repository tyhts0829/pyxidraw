from __future__ import annotations

import numpy as np

from util.constants import NOISE_CONST

from .base import BaseEffect


# @njit(fastmath=True, cache=True)
def fade(t):
    """Perlinノイズ用のフェード関数"""
    return t * t * t * (t * (t * 6 - 15) + 10)


# @njit(fastmath=True, cache=True)
def lerp(a, b, t):
    """線形補間"""
    return a + t * (b - a)


# @njit(fastmath=True, cache=True)
def grad(hash_val, x, y, z):
    """勾配ベクトル計算"""
    # 安全なインデックスアクセス
    idx = int(hash_val) % 12
    g = grad3[idx]
    return g[0] * x + g[1] * y + g[2] * z


# @njit(fastmath=True, cache=True)
def perlin_noise_3d(x, y, z, perm_table):
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

    # 安全な配列アクセス（境界チェック）
    perm_len = len(perm_table)
    
    A = perm_table[X % perm_len] + Y
    AA = perm_table[A % perm_len] + Z
    AB = perm_table[(A + 1) % perm_len] + Z
    B = perm_table[(X + 1) % perm_len] + Y
    BA = perm_table[B % perm_len] + Z
    BB = perm_table[(B + 1) % perm_len] + Z

    # 8つのコーナーでのグラディエントドット
    gAA = grad(perm_table[AA % perm_len], x, y, z)
    gBA = grad(perm_table[BA % perm_len], x - 1, y, z)
    gAB = grad(perm_table[AB % perm_len], x, y - 1, z)
    gBB = grad(perm_table[BB % perm_len], x - 1, y - 1, z)
    gAA1 = grad(perm_table[(AA + 1) % perm_len], x, y, z - 1)
    gBA1 = grad(perm_table[(BA + 1) % perm_len], x - 1, y, z - 1)
    gAB1 = grad(perm_table[(AB + 1) % perm_len], x, y - 1, z - 1)
    gBB1 = grad(perm_table[(BB + 1) % perm_len], x - 1, y - 1, z - 1)

    # trilinear補間
    return lerp(lerp(lerp(gAA, gBA, u), lerp(gAB, gBB, u), v), lerp(lerp(gAA1, gBA1, u), lerp(gAB1, gBB1, u), v), w)


# @njit(fastmath=True, cache=True)
def perlin_core(vertices: np.ndarray, frequency: tuple):
    """コア Perlin ノイズ計算（3次元頂点専用）"""
    n = vertices.shape[0]
    
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    result = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):  # prangeからrangeに変更
        try:
            x, y, z = vertices[i, 0] * frequency[0], vertices[i, 1] * frequency[1], vertices[i, 2] * frequency[2]
            # 3成分のノイズをオフセットを変えて生成
            nx = perlin_noise_3d(x, y, z, perm)
            ny = perlin_noise_3d(x + 100.0, y + 100.0, z + 100.0, perm)
            nz = perlin_noise_3d(x + 200.0, y + 200.0, z + 200.0, perm)
            result[i, 0] = np.float32(nx)
            result[i, 1] = np.float32(ny)
            result[i, 2] = np.float32(nz)
        except (IndexError, ValueError) as e:
            # エラー時はそのまま元の値を保持
            if i < len(vertices):
                result[i] = vertices[i]

    return result


# @njit(fastmath=True, cache=True)
def _apply_noise(vertices: np.ndarray, intensity: float, frequency: tuple, t: float) -> np.ndarray:
    """頂点にPerlinノイズを適用します。"""
    try:
        if vertices.size == 0 or not intensity:
            return vertices.astype(np.float32)

        # 入力検証
        if vertices.shape[1] != 3:
            raise ValueError(f"Expected 3D vertices, got shape {vertices.shape}")

        # 係数調整
        intensity = intensity * 0.1
        frequency = (frequency[0] * 10, frequency[1] * 10, frequency[2] * 10)
        t = t * 0.01

        # 入力を一時的にfloat32に変換
        vertices_f32 = vertices.astype(np.float32)

        # Perlinノイズ計算
        noise_offset = perlin_core(vertices_f32 + np.float32(t + 1000), frequency)

        return vertices_f32 + noise_offset * np.float32(intensity)
    
    except Exception as e:
        # エラー時は元の頂点をそのまま返す
        print(f"Warning: Noise effect failed: {e}")
        return vertices.astype(np.float32)


# Perlinノイズ用のPermutationテーブルを作成
perm = np.array(NOISE_CONST["PERM"], dtype=np.int32)
perm = np.concatenate([perm, perm])  # 0-255までの順列を2回連結

# Perlinノイズで使用するグラディエントベクトル
grad3 = np.array(NOISE_CONST["GRAD3"], dtype=np.float32)


class Noise(BaseEffect):
    """3次元頂点にPerlinノイズを追加します。"""

    def apply(
        self,
        vertices_list: list[np.ndarray],
        intensity: float = 0.5,
        frequency: tuple | float = (0.5, 0.5, 0.5),
        t: float = 0.0,
    ) -> list[np.ndarray]:
        """Perlinノイズエフェクトを適用します。

        Args:
            vertices_list: 入力頂点配列（各配列は(N, 3)形状）
            intensity: ノイズの強度
            frequency: ノイズの周波数（tuple or float）
            t: 時間パラメータ

        Returns:
            Perlinノイズが適用された頂点配列
        """
        # 周波数の正規化
        if isinstance(frequency, (int, float)):
            frequency = (frequency, frequency, frequency)
        elif len(frequency) == 1:
            frequency = (frequency[0], frequency[0], frequency[0])

        # Apply Perlin noise to each vertex array
        new_vertices_list = []
        for i, vertices in enumerate(vertices_list):
            try:
                # 空配列の場合はそのまま返す
                if vertices.size == 0:
                    new_vertices_list.append(vertices.astype(np.float32))
                else:
                    noisy_vertices = _apply_noise(vertices, intensity, frequency, t)
                    new_vertices_list.append(noisy_vertices.astype(np.float32))
            except Exception as e:
                print(f"Warning: Failed to apply noise to vertices[{i}]: {e}")
                # エラー時は元の頂点配列をそのまま返す
                new_vertices_list.append(vertices.astype(np.float32))

        return new_vertices_list
