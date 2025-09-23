"""
CC バインディング（draw 内で `CC(idx, map=...)` と指定するための軽量ラッパ）。

どこで: engine.ui.parameters
何を: CC のインデックスと任意のマッピング関数を保持する不変オブジェクトを提供。
なぜ: 値ソースが CC であることを明示し、Resolver が `midi_override` として扱えるようにするため。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class CCBinding:
    index: int
    map: Optional[Callable[[float], float]] = None


def CC(index: int, map: Optional[Callable[[float], float]] = None) -> CCBinding:
    return CCBinding(int(index), map)


__all__ = ["CCBinding", "CC"]
