from __future__ import annotations

"""パレット API（公開 `C` オブジェクト）。

どこで: `api.palette`。
何を: Parameter GUI/ワーカーで共有される現在の Palette に対して、
      `C[i]` で色を取り出すための薄いプロキシを提供する。
なぜ: `draw(t)` 内でシンプルに `from api import C; C[0]` のように
      パレットカラーへアクセスできるようにするため。
"""

from typing import List, Sequence, Tuple

from palette import Palette  # type: ignore[import]
from palette.ui_helpers import (  # type: ignore[import]
    ExportFormat,
    export_palette,
)
from util.palette_state import get_palette


class PaletteAPI:
    """現在のパレットに対する読み取り専用ビュー。

    - `C[i]` で i 番目の色を RGBA(0..1) タプルとして返す。
    - `len(C)` は現在の色数を返す（パレット未設定時は 0）。
    - `colors_rgba()` は全色を RGBA(0..1) のリストとして返す。
    - `hex()` は全色を HEX 文字列リストとして返す。
    """

    __slots__ = ()

    def _current_palette(self) -> Palette | None:
        obj = get_palette()
        if isinstance(obj, Palette):
            return obj
        return None

    # --- Python 互換インタフェース ---
    def __len__(self) -> int:
        pal = self._current_palette()
        try:
            return len(pal.colors) if pal is not None else 0
        except Exception:
            return 0

    def __getitem__(self, index: int) -> Tuple[float, float, float, float]:
        pal = self._current_palette()
        if pal is None or not pal.colors:
            raise IndexError("palette is empty")
        colors = pal.colors
        # list と同じインデックス規約を保つ（負インデックス対応）。
        try:
            color = colors[index]
        except Exception as exc:  # IndexError / TypeError 等
            raise IndexError("palette index out of range") from exc
        r, g, b = color.srgb
        return float(r), float(g), float(b), 1.0

    # --- 補助 API ---
    def colors_rgba(self) -> List[Tuple[float, float, float, float]]:
        """現在のパレットを RGBA(0..1) のリストとして返す。"""
        pal = self._current_palette()
        if pal is None:
            return []
        out: list[tuple[float, float, float, float]] = []
        for c in pal.colors:
            try:
                r, g, b = c.srgb
                out.append((float(r), float(g), float(b), 1.0))
            except Exception:
                continue
        return out

    def hex(self) -> List[str]:
        """現在のパレットを HEX 文字列のリストとして返す。"""
        pal = self._current_palette()
        if pal is None:
            return []
        try:
            return [str(c) for c in export_palette(pal, ExportFormat.HEX)]
        except Exception:
            return []

    def export(self, fmt: ExportFormat | str) -> Sequence[object]:
        """指定フォーマットでパレットをエクスポートする。"""
        pal = self._current_palette()
        if pal is None:
            return []
        try:
            return list(export_palette(pal, fmt))
        except Exception:
            return []


# 公開インスタンス
C = PaletteAPI()


__all__ = ["C", "PaletteAPI"]
