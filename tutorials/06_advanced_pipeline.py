#!/usr/bin/env python3
"""
チュートリアル 06: 高度なパイプライン機能

複雑なエフェクトパイプライン、条件分岐、パフォーマンス最適化、
MIDIコントローラーとの連携などの高度な機能を学びます。
"""

import os
import logging
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
try:
    while REPO_ROOT in sys.path:
        sys.path.remove(REPO_ROOT)
except ValueError:
    pass
sys.path.insert(0, REPO_ROOT)

import numpy as np
from api import E, G, run
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES
import time
from common.logging import setup_default_logging

logger = logging.getLogger(__name__)


class ConditionalPipeline:
    """
    条件付きエフェクトパイプラインのサンプル実装
    """
    
    def __init__(self):
        self.mode = 0
        self.last_switch_time = time.time()
        self.switch_interval = 3.0  # 3秒ごとに切り替え
    
    def apply(self, geometry: Geometry, t, cc) -> Geometry:
        """
        時間や条件に基づいてエフェクトを切り替える
        """
        # モード切り替えロジック
        current_time = time.time()
        if current_time - self.last_switch_time > self.switch_interval:
            self.mode = (self.mode + 1) % 4
            self.last_switch_time = current_time
            logger.info("モード切り替え: %s", self.get_mode_name())
        
        # モードに応じたエフェクトチェーン
        if self.mode == 0:
            # モード0: 基本変形
            return (E.pipeline.noise(intensity=0.1)
                    .rotation(rotate=(0.0, (t * 0.5) / 360.0, 0.0))
                    .build())(geometry)
        
        elif self.mode == 1:
            # モード1: 波状変形
            return (E.pipeline.wave(amplitude=0.2, frequency=3)
                    .scaling(scale=(1.2, 0.8, 1.0))
                    .rotation(rotate=(30/360.0, (t * 0.3)/360.0, 0.0))
                    .build())(geometry)
        
        elif self.mode == 2:
            # モード2: 爆発と収縮
            factor = 0.3 * np.sin(t * 0.02)
            return (E.pipeline.explode(factor=abs(factor))
                    .twist(angle=factor * 90)
                    .rotation(rotate=(0.0, 0.0, (t * 0.4)/360.0))
                    .build())(geometry)
        
        else:
            # モード3: 複合エフェクト
            return (E.pipeline
                    .noise(intensity=0.05)
                    .wave(amplitude=0.1, frequency=2)
                    .rotation(rotate=((t * 0.2)/360.0, (t * 0.3)/360.0, (t * 0.1)/360.0))
                    .build())(geometry)
    
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
    
    def apply(self, geometry: Geometry, t, cc) -> Geometry:
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
            if lod == 0:  # 高品質（暫定: 細分化は未実装）
                result = (E.pipeline.noise(intensity=0.1)
                          .wave(amplitude=0.15, frequency=4)
                          .build())(geometry)
            elif lod == 1:  # 中品質
                result = (E.pipeline.noise(intensity=0.1).build())(geometry)
            else:  # 低品質
                result = (E.pipeline.noise(intensity=0.05).build())(geometry)
            
            # キャッシュに保存
            if self.use_cache:
                self.cache[cache_key] = result
        
        # 共通の変形（キャッシュされない）
        result = (E.pipeline
                  .rotation(rotate=(0.0, (t * 0.5)/360.0, 0.0))
                  .build())(result)
        
        # パフォーマンス情報を出力（100フレームごと）
        if self.frame_count % 100 == 0:
            elapsed = time.time() - start_time
            logger.info("フレーム %d: %.2fms (LOD: %d)", self.frame_count, elapsed*1000, lod)
        
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
    main_obj = G.polyhedron(polygon_type="icosahedron").scale(80, 80, 80).translate(200, 200, 0)
    main_obj = create_complex_scene.pipelines['conditional'].apply(main_obj, t, cc)
    combined = combined + main_obj
    
    # 2. 周回するサテライトオブジェクト
    num_satellites = 6
    for i in range(num_satellites):
        angle = (2 * np.pi * i / num_satellites) + (t * 0.01)
        radius = 120
        
        x = 200 + radius * np.cos(angle)
        y = 200 + radius * np.sin(angle)
        
        satellite = G.polyhedron(polygon_type="tetrahedron").scale(30, 30, 30).translate(x, y, 0)
        
        # サテライトごとに異なるエフェクト
        satellite = (E.pipeline
                     .rotation(rotate=((t * (i + 1))/360.0, (t * 0.5)/360.0, 0.0))
                     .scaling(scale=(1 + 0.3 * np.sin(t * 0.02 + i),) * 3)
                     .build())(satellite)
        
        combined = combined + satellite
    
    # 3. 背景グリッド（最適化パイプライン）
    background = G.grid(width=20, height=20).scale(300, 300, 300).translate(200, 200, -50)
    background = create_complex_scene.pipelines['optimized'].apply(background, t, cc)
    combined = combined + background
    
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
    shape = G.torus(major_radius=50, minor_radius=20).translate(200, 200, 0)

    # パイプラインを組み立て（CCの値に応じてステップを追加）
    builder = E.pipeline
    noise_intensity = simulated_cc.get(1, 0.5) * 0.3
    if noise_intensity > 0.01:
        builder = builder.noise(intensity=noise_intensity)

    wave_amplitude = simulated_cc.get(2, 0.5) * 0.4
    if wave_amplitude > 0.01:
        builder = builder.wave(amplitude=wave_amplitude, frequency=3)

    rotation_speed = simulated_cc.get(3, 0.5)
    builder = builder.rotation(rotate=(
        (t * rotation_speed * 0.5)/360.0,
        (t * rotation_speed)/360.0,
        (t * rotation_speed * 0.3)/360.0,
    ))

    scale_factor = 0.5 + simulated_cc.get(4, 0.5) * 1.5
    builder = builder.scaling(scale=(scale_factor, scale_factor, scale_factor))

    result = builder.build()(shape)

    # 情報表示（10フレームごと）
    if t % 10 == 0:
        logger.info(
            "MIDI CC値: Noise=%.2f, Wave=%.2f, Rotation=%.2f, Scale=%.2f",
            noise_intensity,
            wave_amplitude,
            rotation_speed,
            scale_factor,
        )
    
    return result


def main():
    """メイン実行関数"""
    setup_default_logging()
    logger.info("=== チュートリアル 06: 高度なパイプライン機能 ===")
    logger.info("実装内容：\n1. 条件付きパイプライン（時間で自動切り替え）\n2. パフォーマンス最適化（LODとキャッシュ）\n3. 複雑なシーン構成（複数オブジェクト）\n4. MIDIコントローラー連携（シミュレーション）")
    logger.info("3秒ごとにエフェクトモードが切り替わります")
    logger.info("終了するには Ctrl+C を押してください")
    
    headless = os.environ.get("PYXIDRAW_HEADLESS") == "1"

    if headless:
        g = create_complex_scene(0, {})
        c, o = g.as_arrays()
        logger.info("Headless OK: points=%d, lines=%d", c.shape[0], max(0, o.shape[0]-1))
    else:
        run(create_complex_scene, canvas_size=CANVAS_SIZES["SQUARE_300"]) 
        # run(midi_controlled_pipeline, canvas_size=CANVAS_SIZES["SQUARE_300"]) 


if __name__ == "__main__":
    main()
