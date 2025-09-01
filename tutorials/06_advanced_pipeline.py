#!/usr/bin/env python3
"""
チュートリアル 06: 高度なパイプライン機能

複雑なエフェクトパイプライン、条件分岐、パフォーマンス最適化、
MIDIコントローラーとの連携などの高度な機能を学びます。
"""

import numpy as np
import arc
from api import E, G
from api.runner import run_sketch
from api.geometry_api import GeometryAPI
from api.effect_pipeline import EffectPipeline
from util.constants import CANVAS_SIZES
import time


class ConditionalPipeline:
    """
    条件付きエフェクトパイプラインのサンプル実装
    """
    
    def __init__(self):
        self.mode = 0
        self.last_switch_time = time.time()
        self.switch_interval = 3.0  # 3秒ごとに切り替え
    
    def apply(self, geometry, t, cc):
        """
        時間や条件に基づいてエフェクトを切り替える
        """
        # モード切り替えロジック
        current_time = time.time()
        if current_time - self.last_switch_time > self.switch_interval:
            self.mode = (self.mode + 1) % 4
            self.last_switch_time = current_time
            print(f"モード切り替え: {self.get_mode_name()}")
        
        # モードに応じたエフェクトチェーン
        if self.mode == 0:
            # モード0: 基本変形
            return E.add(geometry)\
                .noise(intensity=0.1)\
                .rotation(rotate=(0.0, (t * 0.5) / 360.0, 0.0))\
                .result()
        
        elif self.mode == 1:
            # モード1: 波状変形
            return E.add(geometry)\
                .wave(amplitude=0.2, frequency=3)\
                .scaling(scale=(1.2, 0.8, 1.0))\
                .rotation(rotate=(30/360.0, (t * 0.3)/360.0, 0.0))\
                .result()
        
        elif self.mode == 2:
            # モード2: 爆発と収縮
            factor = 0.3 * np.sin(t * 0.02)
            return E.add(geometry)\
                .explode(factor=abs(factor))\
                .twist(angle=factor * 90)\
                .rotation(rotate=(0.0, 0.0, (t * 0.4)/360.0))\
                .result()
        
        else:
            # モード3: 複合エフェクト
            return E.add(geometry)\
                .subdivision(n_divisions=0.5)\
                .noise(intensity=0.05)\
                .wave(amplitude=0.1, frequency=2)\
                .gradient(color_start=[1, 0.5, 0], color_end=[0.5, 0, 1])\
                .rotation(rotate=((t * 0.2)/360.0, (t * 0.3)/360.0, (t * 0.1)/360.0))\
                .result()
    
    def get_mode_name(self):
        """現在のモード名を取得"""
        names = ["基本変形", "波状変形", "爆発と収縮", "複合エフェクト"]
        return names[self.mode]


class PerformanceOptimizedPipeline:
    """
    パフォーマンス最適化されたパイプライン
    """
    
    def __init__(self):
        self.cache = {}
        self.frame_count = 0
        self.use_cache = True
    
    def apply(self, geometry, t, cc):
        """
        キャッシュとLOD（Level of Detail）を使用した最適化
        """
        self.frame_count += 1
        
        # パフォーマンス測定
        start_time = time.time()
        
        # LOD（詳細度）の決定
        # 実際のアプリケーションでは、カメラ距離などに基づいて決定
        lod = self.get_lod_level(t)
        
        # キャッシュキーの生成
        cache_key = f"{lod}_{int(t / 10)}"  # 10フレームごとにキャッシュ
        
        if self.use_cache and cache_key in self.cache:
            result = self.cache[cache_key]
        else:
            # LODに基づいた処理
            if lod == 0:  # 高品質
                result = E.add(geometry)\
                    .subdivide(iterations=2)\
                    .noise(intensity=0.1)\
                    .wave(amplitude=0.15, frequency=4)\
                    .result()
            elif lod == 1:  # 中品質
                result = E.add(geometry)\
                    .subdivide(iterations=1)\
                    .noise(intensity=0.1)\
                    .result()
            else:  # 低品質
                result = E.add(geometry)\
                    .noise(intensity=0.05)\
                    .result()
            
            # キャッシュに保存
            if self.use_cache:
                self.cache[cache_key] = result
        
        # 共通の変形（キャッシュされない）
        result = E.add(result).rotate(y=t * 0.5).result()
        
        # パフォーマンス情報を出力（100フレームごと）
        if self.frame_count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"フレーム {self.frame_count}: {elapsed*1000:.2f}ms (LOD: {lod})")
        
        return result
    
    def get_lod_level(self, t):
        """LODレベルを決定（0=高品質, 1=中品質, 2=低品質）"""
        # デモ用: 時間に基づいて切り替え
        cycle = int(t / 100) % 3
        return cycle


def create_complex_scene(t, cc):
    """
    複数のオブジェクトとパイプラインを組み合わせた複雑なシーン
    """
    combined = G.empty()
    
    # パイプラインインスタンス（永続化のためグローバルに保存）
    if not hasattr(create_complex_scene, 'pipelines'):
        create_complex_scene.pipelines = {
            'conditional': ConditionalPipeline(),
            'optimized': PerformanceOptimizedPipeline()
        }
    
    # 1. 中央のメインオブジェクト（条件付きパイプライン）
    main_obj = G.polyhedron(polygon_type="icosahedron").size(80, 80, 80).at(200, 200, 0)
    main_obj = create_complex_scene.pipelines['conditional'].apply(main_obj, t, cc)
    combined = combined.add(main_obj)
    
    # 2. 周回するサテライトオブジェクト
    num_satellites = 6
    for i in range(num_satellites):
        angle = (2 * np.pi * i / num_satellites) + (t * 0.01)
        radius = 120
        
        x = 200 + radius * np.cos(angle)
        y = 200 + radius * np.sin(angle)
        
        satellite = G.polyhedron(polygon_type="tetrahedron").size(30, 30, 30).at(x, y, 0)
        
        # サテライトごとに異なるエフェクト
        satellite = E.add(satellite)\
            .rotation(rotate=((t * (i + 1))/360.0, (t * 0.5)/360.0, 0.0))\
            .scaling(scale=(1 + 0.3 * np.sin(t * 0.02 + i),) * 3)\
            .result()
        
        combined = combined.add(satellite)
    
    # 3. 背景グリッド（最適化パイプライン）
    background = G.grid(width=20, height=20).size(300, 300, 300).at(200, 200, -50)
    background = create_complex_scene.pipelines['optimized'].apply(background, t, cc)
    combined = combined.add(background)
    
    return combined


def midi_controlled_pipeline(t, cc):
    """
    MIDIコントローラーで制御されるパイプライン
    """
    # MIDIコントローラーの値を取得（デモ用のシミュレーション）
    # 実際の使用時は cc パラメータから値を取得
    simulated_cc = {
        1: 0.5 + 0.5 * np.sin(t * 0.01),      # ノイズ強度
        2: 0.5 + 0.5 * np.cos(t * 0.015),     # 波の振幅
        3: 0.5 + 0.5 * np.sin(t * 0.02),      # 回転速度
        4: 0.5 + 0.5 * np.cos(t * 0.008),     # スケール
    }
    
    # CCの値をマージ（実際のMIDI値を優先）
    for key, value in cc.items():
        simulated_cc[key] = value
    
    # ベース形状
    shape = G.torus(major_radius=50, minor_radius=20).at(200, 200, 0)
    
    # MIDIコントロールによるエフェクトチェーン
    result = E.add(shape)
    
    # CC1: ノイズ強度
    noise_intensity = simulated_cc.get(1, 0.5) * 0.3
    if noise_intensity > 0.01:
        result = result.noise(intensity=noise_intensity)
    
    # CC2: 波エフェクト
    wave_amplitude = simulated_cc.get(2, 0.5) * 0.4
    if wave_amplitude > 0.01:
        result = result.wave(amplitude=wave_amplitude, frequency=3)
    
    # CC3: 回転
    rotation_speed = simulated_cc.get(3, 0.5)
    result = result.rotation(rotate=(
        (t * rotation_speed * 0.5)/360.0,
        (t * rotation_speed)/360.0,
        (t * rotation_speed * 0.3)/360.0,
    ))
    
    # CC4: スケール
    scale_factor = 0.5 + simulated_cc.get(4, 0.5) * 1.5
    result = result.scaling(scale=(scale_factor, scale_factor, scale_factor))
    
    # 情報表示（10フレームごと）
    if t % 10 == 0:
        print(f"MIDI CC値: Noise={noise_intensity:.2f}, "
              f"Wave={wave_amplitude:.2f}, "
              f"Rotation={rotation_speed:.2f}, "
              f"Scale={scale_factor:.2f}")
    
    return result.result()


def main():
    """メイン実行関数"""
    print("=== チュートリアル 06: 高度なパイプライン機能 ===")
    print("\n実装内容：")
    print("1. 条件付きパイプライン（時間で自動切り替え）")
    print("2. パフォーマンス最適化（LODとキャッシュ）")
    print("3. 複雑なシーン構成（複数オブジェクト）")
    print("4. MIDIコントローラー連携（シミュレーション）")
    print("\n3秒ごとにエフェクトモードが切り替わります")
    print("\n終了するには Ctrl+C を押してください")
    
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    headless = args.headless or os.environ.get("PYXIDRAW_HEADLESS") == "1"

    if headless:
        g = create_complex_scene(0, {})
        print("Headless OK: points=", len(g.coords), "lines=", len(g.offsets) - 1)
    else:
        import arc
        arc.start(midi=False)
        run_sketch(create_complex_scene, canvas_size=CANVAS_SIZES["SQUARE_400"])
        # run_sketch(midi_controlled_pipeline, canvas_size=CANVAS_SIZES["SQUARE_400"])
        arc.stop()


if __name__ == "__main__":
    main()
