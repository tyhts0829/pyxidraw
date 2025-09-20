"""
どこで: `engine.ui.parameters` パッケージの公開入口。
何を: ParameterRuntime/ParameterStore/RangeHint など UI パラメータ機構の主要型を再輸出。
なぜ: 外部から薄いファサードを提供し、内部実装の入れ替えと依存分離を容易にするため。
"""

from .runtime import ParameterRuntime, activate_runtime, deactivate_runtime, get_active_runtime
from .state import (
    OverrideResult,
    ParameterDescriptor,
    ParameterLayoutConfig,
    ParameterStore,
    RangeHint,
)

__all__ = [
    "ParameterRuntime",
    "activate_runtime",
    "deactivate_runtime",
    "get_active_runtime",
    "ParameterDescriptor",
    "ParameterLayoutConfig",
    "ParameterStore",
    "RangeHint",
    "OverrideResult",
]
