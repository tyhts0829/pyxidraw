#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シリアライズ可能なベンチマークターゲット

pickle問題を解決するため、クロージャを使わないベンチマークターゲット実装。
"""

from typing import Any, Dict, Optional

# グローバル変数として各モジュールをキャッシュ
_cached_modules: Dict[str, Any] = {}


def _get_cached_module(module_name: str, function_name: Optional[str] = None):
    """モジュールまたは関数を遅延インポートしてキャッシュ"""
    cache_key = f"{module_name}.{function_name}" if function_name else module_name
    
    if cache_key not in _cached_modules:
        if module_name == "api.effect_chain":
            from api.effect_chain import E
            if function_name:
                _cached_modules[cache_key] = getattr(E, function_name)
            else:
                _cached_modules[cache_key] = E
        elif module_name == "api.shape_factory":
            from api.shape_factory import G
            if function_name:
                _cached_modules[cache_key] = getattr(G, function_name)
            else:
                _cached_modules[cache_key] = G
        else:
            raise ValueError(f"Unknown module: {module_name}")
    
    return _cached_modules[cache_key]


def init_worker():
    """ProcessPoolExecutorのワーカー初期化時に呼ばれる関数"""
    # よく使われるモジュールを事前にインポート
    try:
        from api.effect_chain import E
        from api.shape_factory import G
        _cached_modules['api.effect_chain'] = E
        _cached_modules['api.shape_factory'] = G
        
        # エフェクトメソッドもキャッシュ（Eはメソッドチェーン形式）
        _cached_modules['api.effect_chain.E'] = E
        
        for func_name in ['polygon', 'grid', 'sphere', 'cylinder', 'cone', 'torus']:
            _cached_modules[f'api.shape_factory.{func_name}'] = getattr(G, func_name)
            
    except ImportError:
        pass  # インポートエラーは後で処理


class SerializableEffectTarget:
    """シリアライズ可能なエフェクトターゲット"""
    
    def __init__(self, effect_type: str, params: Dict[str, Any]):
        self.effect_type = effect_type
        self.params = params
    
    def __call__(self, geom):
        """エフェクトを適用"""
        if self.effect_type == "transform":
            return geom.transform(**self.params)
        elif self.effect_type == "scale":
            scale = self.params.get("scale", (1, 1, 1))
            center = self.params.get("center", (0, 0, 0))
            return geom.scale(scale[0], scale[1], scale[2], center=center)
        elif self.effect_type == "translate":
            translate = self.params.get("translate", (0, 0, 0))
            return geom.translate(translate[0], translate[1], translate[2])
        elif self.effect_type == "rotate":
            rotate = self.params.get("rotate", (0, 0, 0))
            return geom.rotate(rotate[0], rotate[1], rotate[2])
        elif self.effect_type == "noise":
            E = _get_cached_module("api.effect_chain")
            intensity = self.params.get("intensity", 0.5)
            frequency = self.params.get("frequency", 1.0)
            return E.add(geom).noise(intensity=intensity, frequency=frequency).result()
        elif self.effect_type == "subdivision":
            E = _get_cached_module("api.effect_chain")
            level = self.params.get("level", 1)
            return E.add(geom).subdivision(n_divisions=level).result()
        elif self.effect_type == "extrude":
            E = _get_cached_module("api.effect_chain")
            distance = self.params.get("depth", 10.0)
            return E.add(geom).extrude(distance=distance).result()
        elif self.effect_type == "filling":
            E = _get_cached_module("api.effect_chain")
            density = self.params.get("spacing", 10.0) / 20.0  # spacing to density conversion
            angle = self.params.get("angle", 0.0)
            return E.add(geom).filling(density=density, angle=angle).result()
        elif self.effect_type == "buffer":
            E = _get_cached_module("api.effect_chain")
            distance = self.params.get("distance", 5.0) / 10.0  # normalize to 0.0-1.0
            return E.add(geom).buffer(distance=distance).result()
        elif self.effect_type == "array":
            E = _get_cached_module("api.effect_chain")
            count_total = self.params.get("count_x", 2) * self.params.get("count_y", 2)
            n_duplicates = min(count_total / 10.0, 1.0)  # normalize to 0.0-1.0
            spacing_x = self.params.get("spacing_x", 10.0)
            spacing_y = self.params.get("spacing_y", 10.0)
            return E.add(geom).array(n_duplicates=n_duplicates, offset=(spacing_x, spacing_y, 0)).result()
        else:
            raise ValueError(f"Unknown effect type: {self.effect_type}")


class SerializableShapeTarget:
    """シリアライズ可能な形状ターゲット"""
    
    def __init__(self, shape_type: str, params: Dict[str, Any]):
        self.shape_type = shape_type
        self.params = params
    
    def __call__(self):
        """形状を生成"""
        if self.shape_type == "polygon":
            polygon = _get_cached_module("api.shape_factory", "polygon")
            return polygon(**self.params)
        elif self.shape_type == "grid":
            grid = _get_cached_module("api.shape_factory", "grid")
            # G.grid expects 'divisions' parameter instead of 'n_divisions'
            params = self.params.copy()
            if 'n_divisions' in params:
                divisions = params.pop('n_divisions')
                params['divisions'] = divisions[0] if isinstance(divisions, tuple) else divisions
            return grid(**params)
        elif self.shape_type == "sphere":
            sphere = _get_cached_module("api.shape_factory", "sphere")
            return sphere(**self.params)
        elif self.shape_type == "cylinder":
            cylinder = _get_cached_module("api.shape_factory", "cylinder")
            return cylinder(**self.params)
        elif self.shape_type == "cone":
            cone = _get_cached_module("api.shape_factory", "cone")
            return cone(**self.params)
        elif self.shape_type == "torus":
            torus = _get_cached_module("api.shape_factory", "torus")
            return torus(**self.params)
        elif self.shape_type == "capsule":
            capsule = _get_cached_module("api.shape_factory", "capsule")
            return capsule(**self.params)
        elif self.shape_type == "polyhedron":
            polyhedron = _get_cached_module("api.shape_factory", "polyhedron")
            # G.polyhedron expects 'polyhedron_type' as float instead of 'polygon_type'
            params = self.params.copy()
            if 'polygon_type' in params:
                params['polyhedron_type'] = 0.0  # Default mapping
                params.pop('polygon_type')
            return polyhedron(**params)
        elif self.shape_type == "lissajous":
            lissajous = _get_cached_module("api.shape_factory", "lissajous")
            return lissajous(**self.params)
        elif self.shape_type == "attractor":
            attractor = _get_cached_module("api.shape_factory", "attractor")
            return attractor(**self.params)
        elif self.shape_type == "text":
            text = _get_cached_module("api.shape_factory", "text")
            # G.text expects 'text_content' parameter instead of 'text'
            params = self.params.copy()
            if 'text' in params:
                params['text_content'] = params.pop('text')
            return text(**params)
        elif self.shape_type == "asemic_glyph":
            asemic_glyph = _get_cached_module("api.shape_factory", "asemic_glyph")
            return asemic_glyph(**self.params)
        else:
            raise ValueError(f"Unknown shape type: {self.shape_type}")