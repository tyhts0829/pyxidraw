"""
どこで: `engine.ui.hud` の計測サブモジュール。
何を: SwapBuffer のフロント Geometry から頂点数を読み、プロセスの CPU/MEM とあわせて
      一定間隔でサンプリング（実効FPSを含む）。HUD 描画向けに文字列辞書として保持。
なぜ: 実行時の簡易メトリクスを低コストに観測し、性能/負荷の傾向を把握するため。
"""

from __future__ import annotations

import os
import time
from typing import Callable, Mapping, Optional

from engine.core.lazy_geometry import LazyGeometry

from ...core.tickable import Tickable
from ...runtime.buffer import SwapBuffer
from .config import HUDConfig
from .fields import CPU, FPS, LINE, MEM, VERTEX

try:
    # util は軽量のため engine 層から参照可
    from util.utils import load_config as _load_config  # type: ignore
except Exception:  # pragma: no cover - フォールバック
    _load_config = lambda: {}  # type: ignore[assignment]


class MetricSampler(Tickable):
    """頂点数・CPU・MEM を一定間隔でサンプリングし dict に保持する。

    - `data`: HUD のテキスト表示用にフォーマット済みの文字列を保持。
    - `values`: メータ正規化に使う生値（CPU[%], MEM[bytes], FPS[Hz], VERTEX[int], LINE[int]）。
    """

    def __init__(self, swap: SwapBuffer, config: HUDConfig | None = None):
        self._swap = swap
        self._config = config or HUDConfig()
        self._interval = float(self._config.sample_interval)
        # psutil は必要時のみ遅延 import
        self._proc: Optional["psutil.Process"] = None
        self._psutil_mod = None
        if self._config.show_cpu_mem:
            try:
                import psutil  # type: ignore

                self._psutil_mod = psutil
                self._proc = psutil.Process(os.getpid())
            except Exception:
                self._proc = None
                self._psutil_mod = None
        # MEM 正規化用の上限値（システム総メモリ or カスタム or プロセスピーク）
        self._mem_total_bytes: int = 1
        if self._psutil_mod is not None:
            try:
                self._mem_total_bytes = int(self._psutil_mod.virtual_memory().total)
            except Exception:
                self._mem_total_bytes = 1
        self._mem_peak_bytes: int = 0
        # 最大値（VERTEX/LINE/FPS）: HUDConfig 既定 + 設定ファイルで上書き
        self._target_fps: float = float(self._config.target_fps)
        self._vertex_max: int = int(self._config.vertex_max)
        self._line_max: int = int(self._config.line_max)
        self._mem_scale: str = str(self._config.mem_scale)
        self._mem_custom_bytes: Optional[int] = self._config.mem_custom_bytes
        try:
            cfg = _load_config() or {}
            hud_cfg = cfg.get("hud", {})
            if isinstance(hud_cfg, dict):
                meters = hud_cfg.get("meters", {})
                if isinstance(meters, dict):
                    self._mem_scale = str(meters.get("mem_scale", self._mem_scale))
                    _cm = meters.get("mem_custom_bytes")
                    if _cm is not None:
                        try:
                            self._mem_custom_bytes = int(_cm)
                        except Exception:
                            pass
                    _vx = meters.get("vertex_max")
                    if _vx is not None:
                        try:
                            self._vertex_max = int(_vx)
                        except Exception:
                            pass
                    _lm = meters.get("line_max")
                    if _lm is not None:
                        try:
                            self._line_max = int(_lm)
                        except Exception:
                            pass
        except Exception:
            # 設定読込失敗は安全側: 既定を維持
            pass
        # 前回サンプリング時刻とバージョン（実効FPS算出に使用）
        self._last = 0.0
        self._last_ver = self._swap.version()
        self.data: dict[str, str] = {}
        self.values: dict[str, float] = {}
        # Renderer 公開値（直近アップロードの頂点/ライン数）を取得するためのプロバイダ
        self._counts_provider: Optional[Callable[[], tuple[int, int]]] = None
        # 追加メトリクス（IBO/Indices LRU など）を取得するプロバイダ
        self._extra_metrics_provider: Optional[Callable[[], Mapping[str, int]]] = None
        # IBO/IDX キャッシュ用の前回カウンタ（MISS/HIT 判定に使用）
        self._prev_idx_misses: int | None = None
        self._prev_ibo_uploaded: int | None = None

    # -------- Tickable --------
    def tick(self, dt: float) -> None:
        now = time.time()
        if now - self._last < self._interval:
            return
        dt = now - self._last if self._last > 0.0 else 0.0
        self._last = now

        # 実効FPS: SwapBuffer.version() の増分 / 経過秒
        if self._config.show_fps:
            cur_ver = self._swap.version()
            dv = cur_ver - self._last_ver
            self._last_ver = cur_ver
            fps = (dv / dt) if dt > 0.0 else 0.0
            self.data[FPS] = f"{fps:4.1f}"
            self.values[FPS] = float(fps)
        else:
            self.data.pop(FPS, None)
            self.values.pop(FPS, None)

        # Renderer からの計数（ある場合のみ 1 回だけ呼ぶ）
        counts: Optional[tuple[int, int]] = None
        if self._counts_provider is not None:
            try:
                counts = self._counts_provider()
            except Exception:
                counts = None

        # 頂点数
        if self._config.show_vertex_count:
            verts = (
                int(counts[0]) if counts is not None else self._vertex_count(self._swap.get_front())
            )
            self.data[VERTEX] = f"{verts}"
            self.values[VERTEX] = float(verts)
        else:
            self.data.pop(VERTEX, None)
            self.values.pop(VERTEX, None)

        # ライン数（ポリライン本数）
        if self._config.show_line_count:
            lines = (
                int(counts[1]) if counts is not None else self._line_count(self._swap.get_front())
            )
            self.data[LINE] = f"{lines}"
            self.values[LINE] = float(lines)
        else:
            self.data.pop(LINE, None)
            self.values.pop(LINE, None)

        # CPU/MEM
        if self._config.show_cpu_mem and self._proc is not None:
            try:
                cpu_p = float(self._proc.cpu_percent(0.0))
                rss = float(self._proc.memory_info().rss)
                self.data[CPU] = f"{cpu_p:4.1f}%"
                self.data[MEM] = self._human(rss)
                self.values[CPU] = cpu_p
                self.values[MEM] = rss
                if rss > self._mem_peak_bytes:
                    self._mem_peak_bytes = int(rss)
            except Exception:
                # 計測失敗時は無視（前回値があれば残す）
                pass
        else:
            self.data.pop(CPU, None)
            self.data.pop(MEM, None)
            self.values.pop(CPU, None)
            self.values.pop(MEM, None)

        # 追加メトリクス（キャッシュ統計）
        if self._config.show_cache_status and self._extra_metrics_provider is not None:
            try:
                extra = self._extra_metrics_provider()
            except Exception:
                extra = None
            if isinstance(extra, dict):
                # IBO_CACHE: IBO のアップロードが発生したサンプルを MISS として強調
                try:
                    uploaded_now = int(extra.get("ibo_uploaded", 0))
                    prev_uploaded = self._prev_ibo_uploaded
                    if prev_uploaded is None:
                        # 初回サンプルは「直近コストなし」とみなして HIT 扱い
                        value_ibo = 0.0
                    else:
                        du = uploaded_now - prev_uploaded
                        if du < 0:
                            # カウンタが巻き戻った場合は前回値を維持
                            value_ibo = float(self.values.get("IBO_CACHE", 0.0))
                        elif du > 0:
                            value_ibo = 1.0
                        else:
                            value_ibo = 0.0
                    self._prev_ibo_uploaded = uploaded_now
                    self.data.setdefault("IBO_CACHE", "")
                    self.values["IBO_CACHE"] = float(value_ibo)
                except Exception:
                    # 取得に失敗した場合は値を更新しない（前回値維持）
                    pass

                # IDX_CACHE: Indices LRU の MISS が発生したサンプルを MISS として強調
                try:
                    misses_now = int(extra.get("idx_misses", 0))
                    prev_misses = self._prev_idx_misses
                    if prev_misses is None:
                        value_idx = 0.0
                    else:
                        dm = misses_now - prev_misses
                        if dm < 0:
                            value_idx = float(self.values.get("IDX_CACHE", 0.0))
                        elif dm > 0:
                            value_idx = 1.0
                        else:
                            value_idx = 0.0
                    self._prev_idx_misses = misses_now
                    self.data.setdefault("IDX_CACHE", "")
                    self.values["IDX_CACHE"] = float(value_idx)
                except Exception:
                    pass
            else:
                # プロバイダが有効でない場合は IBO/IDX 行をクリーンアップ
                self.data.pop("IBO_CACHE", None)
                self.data.pop("IDX_CACHE", None)
                self.values.pop("IBO_CACHE", None)
                self.values.pop("IDX_CACHE", None)
        else:
            # キャッシュ表示自体が無効な場合もクリーンアップ
            self.data.pop("IBO_CACHE", None)
            self.data.pop("IDX_CACHE", None)
            self.values.pop("IBO_CACHE", None)
            self.values.pop("IDX_CACHE", None)

    # -------- helpers --------
    @staticmethod
    def _vertex_count(geometry: object | None) -> int:
        if geometry is None:
            return 0
        # LazyGeometry は実体化しない（HUD での暗黙実体化を回避）
        try:
            if isinstance(geometry, LazyGeometry):
                return 0
        except Exception:
            pass
        # 優先: 互換プロパティ
        try:
            nv = getattr(geometry, "n_vertices")
            return int(nv)
        except Exception:
            pass
        # Geometry 互換（coords）
        try:
            return int(len(getattr(geometry, "coords")))
        except Exception:
            pass
        # LazyGeometry 互換（as_arrays）
        try:
            c, _ = getattr(geometry, "as_arrays")(copy=False)
            return int(c.shape[0])
        except Exception:
            return 0

    @staticmethod
    def _human(n: float) -> str:
        for u in "B KB MB GB TB".split():
            if n < 1024:
                return f"{n:4.1f}{u}"
            n /= 1024
        return f"{n:4.1f}PB"

    @staticmethod
    def _line_count(geometry: object | None) -> int:
        if geometry is None:
            return 0
        # LazyGeometry は実体化しない
        try:
            if isinstance(geometry, LazyGeometry):
                return 0
        except Exception:
            pass
        try:
            n_lines = getattr(geometry, "n_lines")
            return int(n_lines)
        except Exception:
            # 念のためのフォールバック
            try:
                offsets = getattr(geometry, "offsets")
                size = int(getattr(offsets, "size", 0))
                if size > 0:
                    shape0 = int(getattr(offsets, "shape", (0,))[0])
                    return max(0, shape0 - 1)
                return 0
            except Exception:
                return 0

    # -------- external provider wiring (Renderer → HUD) --------
    def set_counts_provider(self, provider: Callable[[], tuple[int, int]]) -> None:
        """LineRenderer 等からアップロード済みの頂点/ライン数を受け取るプロバイダを登録する。"""
        self._counts_provider = provider

    def set_extra_metrics_provider(self, provider: Callable[[], Mapping[str, int]]) -> None:
        """追加メトリクス（IBO/Indices LRU 等）を提供するプロバイダを登録する。"""
        self._extra_metrics_provider = provider

    # -------- normalization helpers for OverlayHUD --------
    def mem_max_bytes(self) -> int:
        if self._mem_scale == "custom" and isinstance(self._mem_custom_bytes, int):
            return max(1, int(self._mem_custom_bytes))
        if self._mem_scale == "process_peak":
            return max(1, int(self._mem_peak_bytes))
        # system_total（既定）
        return max(1, int(self._mem_total_bytes))

    def target_fps(self) -> float:
        return float(self._target_fps) if self._target_fps > 0 else 60.0

    def vertex_max(self) -> int:
        return max(1, int(self._vertex_max))

    def line_max(self) -> int:
        return max(1, int(self._line_max))
