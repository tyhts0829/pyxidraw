#!/usr/bin/env python3
"""
チュートリアル 02: 複数の形状の組み合わせ

複数の形状を組み合わせて、より複雑な構成を作ります。
異なる形状タイプの使い方を学びます。
"""

from api import G
from api.runner import run_sketch
from util.constants import CANVAS_SIZES


def draw(t, cc):
    """
    複数の形状を組み合わせて描画
    
    Args:
        t: 時間（フレーム番号）
        cc: MIDIコントローラーの値
    
    Returns:
        GeometryAPI: 組み合わせた形状
    """
    # 空のジオメトリから開始
    combined = G.empty()
    
    # 1. 中央に球体
    sphere = G.sphere(subdivisions=0.4).size(80, 80, 80).at(200, 200, 0)
    combined = combined.add(sphere)
    
    # 2. 正多面体（立方体）を左上に配置
    cube = G.polyhedron(polygon_type="cube").size(60, 60, 60).at(100, 100, 0)
    combined = combined.add(cube)
    
    # 3. トーラス（ドーナツ型）を右上に配置
    torus = G.torus(
        major_radius=40,  # 大きい半径
        minor_radius=15,  # 小さい半径
        major_segments=16,
        minor_segments=32
    ).at(300, 100, 0)
    combined = combined.add(torus)
    
    # 4. 多角形（六角形）を下部に配置
    hexagon = G.polygon(n_sides=6).size(70, 70, 70).at(200, 300, 0)
    combined = combined.add(hexagon)
    
    # 5. 円錐を左下に配置
    cone = G.cone(
        base_radius=40,
        height=80,
        radial_segments=20
    ).at(100, 300, 0)
    combined = combined.add(cone)
    
    # 6. シリンダーを右下に配置
    cylinder = G.cylinder(
        radius=30,
        height=80,
        radial_segments=16
    ).at(300, 300, 0)
    combined = combined.add(cylinder)
    
    return combined


def main():
    """メイン実行関数"""
    print("=== チュートリアル 02: 複数の形状の組み合わせ ===")
    print("様々な基本形状を組み合わせて表示します：")
    print("- 球体（中央）")
    print("- 立方体（左上）")
    print("- トーラス（右上）")
    print("- 六角形（下中央）")
    print("- 円錐（左下）")
    print("- シリンダー（右下）")
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
        arc.stop()


if __name__ == "__main__":
    main()
