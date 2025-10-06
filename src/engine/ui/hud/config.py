"""
どこで: `engine.ui.hud.config`。
何を: HUD 表示の設定（有効/無効や表示項目、順序、サンプリング周期）を定義する。
なぜ: HUD の表示を宣言的に制御し、オーバーヘッドを必要に応じて抑制するため。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .fields import CACHE_EFFECT, CACHE_SHAPE, CPU, FPS, LINE, MEM, VERTEX


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
    show_line_count : bool
        ライン数（ポリライン本数）表示の有無。
    show_cpu_mem : bool
        CPU/MEM 表示の有無（未使用時は psutil 呼び出しを抑止）。
    show_cache_status : bool
        CACHE/SHAPE と CACHE/EFFECT の表示と計測を有効化。
    order : list[str] | None
        表示順（None なら既定順）。
    sample_interval : float
        MetricSampler のサンプリング周期（秒）。
    show_meters : bool
        横棒メータの表示有無（テキストの右側に表示）。
    meter_width_px, meter_height_px, meter_gap_px : int
        メータの横幅/高さ/テキストとの間隔（px）。
    meter_alpha_fg, meter_alpha_bg : int
        メータの不透明度（前景/背景, 0..255）。
    meter_color_fg : tuple[int, int, int]
        メータ前景のRGB（単色）。
    smoothing_alpha : float
        メータのEMA平滑化係数（0..1, 大きいほど追従）。
    target_fps : float
        FPS 正規化の基準値（既定 60）。
    mem_scale : Literal['system_total','process_peak','custom']
        メモリ正規化の基準。`custom` の場合は `mem_custom_bytes` を使用。
    mem_custom_bytes : int | None
        メモリ正規化のカスタム上限（バイト）。
    vertex_max : int
        頂点数の固定上限（100%に相当, 既定 10,000,000）。
    line_max : int
        ライン本数の固定上限（100%に相当, 既定 5,000,000）。
    """

    enabled: bool = True
    show_fps: bool = True
    show_vertex_count: bool = True
    show_cpu_mem: bool = True
    show_line_count: bool = True
    show_cache_status: bool = True
    order: Sequence[str] | None = None
    sample_interval: float = 0.5
    # meters
    show_meters: bool = True
    meter_width_px: int = 160
    meter_height_px: int = 6
    meter_gap_px: int = 6
    meter_alpha_fg: int = 220
    meter_alpha_bg: int = 120
    meter_color_fg: tuple[int, int, int] = (0, 120, 220)
    smoothing_alpha: float = 0.5
    target_fps: float = 60.0
    mem_scale: str = "system_total"
    mem_custom_bytes: int | None = None
    vertex_max: int = 10_000_000
    line_max: int = 5_000_000

    def resolved_order(self) -> list[str]:
        """有効フラグに基づく既定順を返す（`order` 指定時はそれを優先）。"""
        if self.order is not None:
            return list(self.order)
        keys: list[str] = []
        if self.show_fps:
            keys.append(FPS)
        if self.show_vertex_count:
            keys.append(VERTEX)
        if self.show_line_count:
            keys.append(LINE)
        if self.show_cpu_mem:
            keys.extend([CPU, MEM])
        if self.show_cache_status:
            keys.extend([CACHE_SHAPE, CACHE_EFFECT])
        return keys


__all__ = ["HUDConfig"]
