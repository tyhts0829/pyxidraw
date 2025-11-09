"""
どこで: `engine.runtime` のバッファ層。
何を: スレッドセーフなダブルバッファ（front/back + version + Event）を提供。
なぜ: 生成スレッドと描画スレッドの間で `Geometry` を安全かつ低オーバーヘッドに受け渡すため。
"""

from __future__ import annotations

import threading

from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry
from engine.render.types import StyledLayer


class SwapBuffer:
    """スレッドセーフなダブルバッファ（front/back + version + Event）。"""

    def __init__(self) -> None:
        """
        _front: 現在描画に使われるデータ
        _back: 新しく生成されたデータを一時的に保管する場所
        _version: データが更新された回数（何回目のデータかを示す）
        _evt (Event): 新しいデータが準備できたかどうかを示すフラグ
        _lock (Lock): スレッド同士が同時に操作しないようにするためのロック
        """
        self._front: (
            Geometry | LazyGeometry | list[StyledLayer] | tuple[StyledLayer, ...] | None
        ) = None
        self._back: Geometry | LazyGeometry | list[StyledLayer] | tuple[StyledLayer, ...] | None = (
            None
        )
        self._version: int = 0
        self._evt = threading.Event()
        self._lock = threading.Lock()

    # ---------- producer (BufferSubsystem) ----------
    def push(
        self, data: Geometry | LazyGeometry | list[StyledLayer] | tuple[StyledLayer, ...]
    ) -> None:
        """BufferSubsystemが呼び出し、新しく生成したデータをセットする"""
        with self._lock:
            self._back = data
            self._version += 1
            self._evt.set()  # 新しいデータが準備できたことを通知

    # ---------- consumer (CanvasController) ----------
    def try_swap(self) -> bool:
        """メインスレッドが定期的に呼び出し、データが準備できていれば front と back を交換する。"""
        if not self._evt.is_set():  # データが準備できていない場合は何もしない
            return False
        with self._lock:  # スレッドセーフにするためのロック
            self._front, self._back = self._back, self._front  # swap
            self._evt.clear()  # データを交換したので、イベントをクリア
        return True

    def get_front(
        self,
    ) -> Geometry | LazyGeometry | list[StyledLayer] | tuple[StyledLayer, ...] | None:
        """現在の front データを取得する。"""
        return self._front

    # ---------- util ----------
    def version(self) -> int:
        """現在のデータのバージョンを取得する。"""
        return self._version

    def is_data_ready(self) -> bool:
        """新しいデータが準備できているかどうかを確認する。"""
        return self._evt.is_set()
