"""
どこで: `engine.ui.hud.config`。
何を: HUD 表示の設定（有効/無効や表示項目、順序、サンプリング周期）を定義する。
なぜ: HUD の表示を宣言的に制御し、オーバーヘッドを必要に応じて抑制するため。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .fields import CACHE_EFFECT, CACHE_SHAPE, CPU, FPS, MEM, VERTEX


@dataclass(frozen=True)
class HUDConfig:
    """HUD の表示設定。

    Parameters
    ----------
    enabled : bool
        HUD 全体の有効/無効。
    show_fps : bool
        FPS 表示の有無。
    show_vertex_count : bool
        頂点数表示の有無。
    show_cpu_mem : bool
        CPU/MEM 表示の有無（未使用時は psutil 呼び出しを抑止）。
    show_cache_status : bool
        CACHE/SHAPE と CACHE/EFFECT の表示と計測を有効化。
    order : list[str] | None
        表示順（None なら既定順）。
    sample_interval : float
        MetricSampler のサンプリング周期（秒）。
    """

    enabled: bool = True
    show_fps: bool = True
    show_vertex_count: bool = True
    show_cpu_mem: bool = True
    show_cache_status: bool = True
    order: Sequence[str] | None = None
    sample_interval: float = 0.5

    def resolved_order(self) -> list[str]:
        """有効フラグに基づく既定順を返す（`order` 指定時はそれを優先）。"""
        if self.order is not None:
            return list(self.order)
        keys: list[str] = []
        if self.show_fps:
            keys.append(FPS)
        if self.show_vertex_count:
            keys.append(VERTEX)
        if self.show_cpu_mem:
            keys.extend([CPU, MEM])
        if self.show_cache_status:
            keys.extend([CACHE_SHAPE, CACHE_EFFECT])
        return keys


__all__ = ["HUDConfig"]
