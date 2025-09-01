#!/usr/bin/env python3
"""
チュートリアル 05: カスタムエフェクトの作成

独自のエフェクトを定義して、エフェクトチェーンに組み込む方法を学びます。
デコレータパターンとエフェクト登録の実装。
"""

import numpy as np
from api import E, G
from api.runner import run_sketch
from api.geometry_api import GeometryAPI
from util.constants import CANVAS_SIZES


# カスタムエフェクト1: 波状変形
@E.register()
def wave(geometry, amplitude=0.1, frequency=2.0, axis="z"):
    """
    形状に波状の変形を適用するエフェクト
    
    Args:
        geometry: 入力ジオメトリ
        amplitude: 波の振幅（0-1）
        frequency: 波の周波数
        axis: 波を適用する軸（"x", "y", "z"）
    
    Returns:
        GeometryAPI: 波状に変形されたジオメトリ
    """
    # 頂点座標をコピー
    coords = geometry.coords.copy()
    
    # 軸のインデックスを取得
    axis_map = {"x": 0, "y": 1, "z": 2}
    wave_axis = axis_map.get(axis, 2)
    
    # 基準軸を決定（波の進行方向）
    reference_axis = 0 if wave_axis != 0 else 1
    
    # 各頂点に波を適用
    for i in range(len(coords)):
        # 基準軸の位置に基づいて波を計算
        position = coords[i][reference_axis]
        wave_value = amplitude * np.sin(frequency * position * 0.1)
        
        # 指定された軸に波を適用
        coords[i][wave_axis] += wave_value * 100  # スケール調整
    
    # 新しいジオメトリを作成
    return GeometryAPI(
        coords=coords,
        offsets=geometry.offsets.copy(),
        colors=geometry.colors.copy() if geometry.colors is not None else None,
        normals=geometry.normals.copy() if geometry.normals is not None else None
    )


# カスタムエフェクト2: 爆発効果
@E.register()
def explode(geometry, factor=0.2):
    """
    形状を中心から外側に爆発させるようなエフェクト
    
    Args:
        geometry: 入力ジオメトリ
        factor: 爆発の強度（0-1）
    
    Returns:
        GeometryAPI: 爆発エフェクトを適用したジオメトリ
    """
    coords = geometry.coords.copy()
    
    # ジオメトリの中心を計算
    center = np.mean(coords, axis=0)
    
    # 各頂点を中心から外側に移動
    for i in range(len(coords)):
        # 中心からの方向ベクトル
        direction = coords[i] - center
        
        # 正規化（ゼロ除算を回避）
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
            
            # 外側に移動
            coords[i] += direction * factor * 50  # スケール調整
    
    return GeometryAPI(
        coords=coords,
        offsets=geometry.offsets.copy(),
        colors=geometry.colors.copy() if geometry.colors is not None else None,
        normals=geometry.normals.copy() if geometry.normals is not None else None
    )


# カスタムエフェクト3: ツイスト（ねじれ）
@E.register()
def twist(geometry, angle=45, axis="y"):
    """
    形状をねじるエフェクト
    
    Args:
        geometry: 入力ジオメトリ
        angle: ねじれ角度（度）
        axis: ねじれの軸
    
    Returns:
        GeometryAPI: ねじれたジオメトリ
    """
    coords = geometry.coords.copy()
    
    # 軸のマッピング
    axis_map = {"x": 0, "y": 1, "z": 2}
    twist_axis = axis_map.get(axis, 1)
    
    # ジオメトリの範囲を取得
    min_val = np.min(coords[:, twist_axis])
    max_val = np.max(coords[:, twist_axis])
    range_val = max_val - min_val
    
    if range_val == 0:
        return geometry
    
    # 各頂点にツイストを適用
    for i in range(len(coords)):
        # 軸に沿った位置を0-1に正規化
        t = (coords[i][twist_axis] - min_val) / range_val
        
        # ねじれ角度を計算（ラジアン）
        twist_angle = np.radians(angle * t)
        
        # 回転行列を適用（簡易版）
        if twist_axis == 1:  # Y軸の場合
            x = coords[i][0]
            z = coords[i][2]
            coords[i][0] = x * np.cos(twist_angle) - z * np.sin(twist_angle)
            coords[i][2] = x * np.sin(twist_angle) + z * np.cos(twist_angle)
    
    return GeometryAPI(
        coords=coords,
        offsets=geometry.offsets.copy(),
        colors=geometry.colors.copy() if geometry.colors is not None else None,
        normals=geometry.normals.copy() if geometry.normals is not None else None
    )


# カスタムエフェクト4: カラーグラデーション
@E.register()
def gradient(geometry, color_start=[1, 0, 0], color_end=[0, 0, 1], axis="y"):
    """
    形状にカラーグラデーションを適用
    
    Args:
        geometry: 入力ジオメトリ
        color_start: 開始色 [R, G, B]
        color_end: 終了色 [R, G, B]
        axis: グラデーションの方向
    
    Returns:
        GeometryAPI: カラーグラデーションを適用したジオメトリ
    """
    coords = geometry.coords
    
    # 軸のマッピング
    axis_map = {"x": 0, "y": 1, "z": 2}
    grad_axis = axis_map.get(axis, 1)
    
    # 範囲を取得
    min_val = np.min(coords[:, grad_axis])
    max_val = np.max(coords[:, grad_axis])
    range_val = max_val - min_val
    
    # カラー配列を初期化
    colors = np.zeros((len(coords), 3), dtype=np.float32)
    
    if range_val > 0:
        for i in range(len(coords)):
            # 位置を0-1に正規化
            t = (coords[i][grad_axis] - min_val) / range_val
            
            # 線形補間でカラーを計算
            colors[i] = (1 - t) * np.array(color_start) + t * np.array(color_end)
    
    return GeometryAPI(
        coords=coords.copy(),
        offsets=geometry.offsets.copy(),
        colors=colors,
        normals=geometry.normals.copy() if geometry.normals is not None else None
    )


def draw(t, cc):
    """
    カスタムエフェクトを使用した描画
    """
    # 時間パラメータ
    time_factor = t * 0.01
    
    # ベースとなる形状（グリッド）
    base = G.grid(width=10, height=10).size(150, 150, 150).at(200, 200, 0)
    
    # エフェクトチェーンの構築
    result = E.add(base)
    
    # 1. 波エフェクト（動的）
    wave_amp = 0.1 + 0.05 * np.sin(time_factor)
    result = result.wave(amplitude=wave_amp, frequency=3.0, axis="z")
    
    # 2. ツイストエフェクト
    twist_angle = 30 * np.sin(time_factor * 0.5)
    result = result.twist(angle=twist_angle, axis="y")
    
    # 3. 爆発エフェクト（パルス効果）
    explode_factor = 0.1 * (1 + np.sin(time_factor * 2))
    result = result.explode(factor=explode_factor)
    
    # 4. カラーグラデーション
    result = result.gradient(
        color_start=[1, 0, 0],  # 赤
        color_end=[0, 0, 1],    # 青
        axis="y"
    )
    
    # 5. 標準エフェクトも組み合わせ可能
    result = result.rotate(x=0, y=time_factor * 20, z=0)
    
    return result.result()


def draw_comparison(t, cc):
    """
    エフェクトの個別比較
    """
    combined = G.empty()
    
    # 元の形状
    original = G.polyhedron("cube").size(50, 50, 50).at(100, 100, 0)
    combined = combined.add(original)
    
    # 波エフェクト
    wave = G.polyhedron("cube").size(50, 50, 50).at(200, 100, 0)
    wave = E.add(wave).wave(amplitude=0.2, frequency=2).result()
    combined = combined.add(wave)
    
    # ツイストエフェクト
    twist = G.polyhedron("cube").size(50, 50, 50).at(300, 100, 0)
    twist = E.add(twist).twist(angle=45).result()
    combined = combined.add(twist)
    
    # 爆発エフェクト
    explode = G.polyhedron("cube").size(50, 50, 50).at(150, 200, 0)
    explode = E.add(explode).explode(factor=0.3).result()
    combined = combined.add(explode)
    
    # 複合エフェクト
    complex_fx = G.polyhedron("cube").size(50, 50, 50).at(250, 200, 0)
    complex_fx = E.add(complex_fx)\
        .wave(amplitude=0.1)\
        .twist(angle=30)\
        .gradient()\
        .result()
    combined = combined.add(complex_fx)
    
    return combined


def main():
    """メイン実行関数"""
    print("=== チュートリアル 05: カスタムエフェクトの作成 ===")
    print("独自のエフェクトを定義して適用します：")
    print("- wave: 波状変形エフェクト")
    print("- explode: 爆発エフェクト")
    print("- twist: ねじれエフェクト")
    print("- gradient: カラーグラデーション")
    print("\nエフェクトは E.add().wave().twist() のように連鎖可能")
    print("\n終了するには Ctrl+C を押してください")
    
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    headless = args.headless or os.environ.get("PYXIDRAW_HEADLESS") == "1"

    if headless:
        g = draw(0, {})
        print("Headless OK: points=", len(g.coords), "lines=", len(g.offsets) - 1)
    else:
        import arc
        arc.start(midi=False)
        run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_400"])
        # run_sketch(draw_comparison, canvas_size=CANVAS_SIZES["SQUARE_400"])
        arc.stop()


if __name__ == "__main__":
    main()
