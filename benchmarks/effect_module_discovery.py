#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
エフェクトモジュール探索・読み込みモジュール

エフェクトモジュールの検出、動的インポート、njit使用状況の分析を行うクラス。
"""

import importlib
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

# Type aliases
VerticesList = List[NDArray[np.float32]]
EffectFunction = Callable[[VerticesList], VerticesList]


class EffectModuleDiscovery:
    """エフェクトモジュールの探索と分析を行うクラス"""

    def __init__(self, effects_path: Path = Path("effects")):
        self.effects_path = effects_path
        self.excluded_files = {"__init__.py", "base.py", "pipeline.py"}
        self.excluded_names = {"annotations", "np", "njit", "Any", "BaseEffect"}

    def get_effect_modules(self) -> List[str]:
        """ベンチマーク対象のエフェクトモジュールリストを取得"""
        return sorted([
            file.stem for file in self.effects_path.glob("*.py")
            if not file.name.startswith("__") and file.name not in self.excluded_files
        ])

    def check_njit_usage(self, module_name: str) -> Dict[str, bool]:
        """モジュール内の関数がnjitデコレータを使用しているかチェック"""
        njit_info: Dict[str, bool] = {}

        try:
            module = importlib.import_module(f"effects.{module_name}")

            for name, obj in inspect.getmembers(module):
                if name.startswith("__") or name in self.excluded_names:
                    continue
                    
                # Check if it's a numba compiled function (CPUDispatcher)
                is_njit = "numba.core.registry.CPUDispatcher" in str(type(obj))
                
                if is_njit or inspect.isfunction(obj):
                    njit_info[name] = is_njit

        except Exception:
            pass

        return njit_info

    def get_effect_function(self, module_name: str) -> Optional[EffectFunction]:
        """エフェクトクラスを検索してインスタンス化し、apply関数を返す"""
        try:
            module = importlib.import_module(f"effects.{module_name}")
            
            # Try to find effect class by name matching
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name.lower() == module_name.lower():
                    try:
                        instance = obj()
                        if hasattr(instance, "apply"):
                            return lambda shapes: instance.apply(shapes)
                    except Exception:
                        pass
            
            # Try any class with apply method
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and hasattr(obj, "apply"):
                    try:
                        instance = obj()
                        return lambda shapes: instance.apply(shapes)
                    except Exception:
                        pass
        
        except Exception:
            pass
            
        return None