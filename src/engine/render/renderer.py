"""
どこで: `engine.render` の高レベル描画。
何を: SwapBuffer の `Geometry` を頂点/インデックスへ変換し、ModernGL に転送して線を描画。
なぜ: 毎フレームのアップロード/描画/リソース寿命を一箇所に集約し、描画処理を単純化するため。
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from engine.core.geometry import Geometry
from util.constants import PRIMITIVE_RESTART_INDEX

from ..core.tickable import Tickable
from ..runtime.buffer import SwapBuffer

# 型参照は文字列注釈で行うため、実行時 import は不要。


class LineRenderer(Tickable):
    """
    SwapBufferからデータを取得し、毎フレームGPUに送り込む作業を管理。
    SwapBuffer（CPU側）からGPUへのデータ転送を明確に管理して、描画の一貫性を保つために必要。
    """

    def __init__(
        self,
        mgl_context: Any,
        projection_matrix: np.ndarray,
        swap_buffer: SwapBuffer,
        line_thickness: float = 0.0006,
        line_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ):
        """
        swap_buffer: GPUへ送る前のデータを管理する仕組み
        gpu: 上記のGPUBufferクラスのインスタンス。データのアップロードを任せる
        """
        self.ctx = mgl_context
        self.swap_buffer = swap_buffer
        self._logger = logging.getLogger(__name__)

        # 遅延 import（optional 依存のない環境でも import 可能にするため）
        from .line_mesh import LineMesh  # local import
        from .shader import Shader  # local import

        # シェーダ初期化
        self.line_program = Shader.create_shader(mgl_context)
        self.line_program["projection"].write(projection_matrix.tobytes())
        # 線幅はクリップ空間（-1..1 基準）で設定する
        self.line_program["line_thickness"].value = float(line_thickness)
        # 線色（RGBA, 0–1）
        from util.color import normalize_color as _normalize_color  # 局所参照（依存を明示）

        try:
            self.line_program["color"].value = _normalize_color(line_color)
        except Exception:  # pragma: no cover - 防御的フォールバック
            self.line_program["color"].value = (0.0, 0.0, 0.0, 1.0)

        # GPUBuffer を保持
        self.gpu = LineMesh(
            ctx=mgl_context,
            program=self.line_program,
            primitive_restart_index=PRIMITIVE_RESTART_INDEX,
        )

    # --------------------------------------------------------------------- #
    # Tickable                                                               #
    # --------------------------------------------------------------------- #
    def tick(self, dt: float) -> None:
        """
        毎フレーム呼ばれ、SwapBufferに新データがあればGPUへ転送。
        """
        if self.swap_buffer.try_swap():
            geometry = self.swap_buffer.get_front()
            self._upload_geometry(geometry)

    # --------------------------------------------------------------------- #
    # Public drawing API                                                    #
    # --------------------------------------------------------------------- #
    def draw(self) -> None:
        """GPUに送ったデータを画面に描画"""
        if self.gpu.index_count > 0:
            # optional 依存（moderngl）はローカル import。未導入環境では描画をスキップ。
            try:  # pragma: no cover - 実行環境依存
                import moderngl as mgl  # type: ignore
            except Exception:
                return
            self.gpu.vao.render(mgl.LINE_STRIP, self.gpu.index_count)
        else:
            # Debug only: nothing to draw this frame
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug("LineRenderer.draw(): no indices (skipped)")

    def clear(self, color: Sequence[float]) -> None:
        """画面を指定色でクリア"""
        self.ctx.clear(*color)  # type: ignore

    def release(self) -> None:
        """GPU リソースを解放。"""
        self.gpu.release()

    def set_line_color(self, rgba: Sequence[float]) -> None:
        """線色（RGBA 0–1）を即時更新する。

        実行時に GUI からの変更を反映する用途を想定。"""
        from util.color import normalize_color as _normalize_color

        try:
            self.line_program["color"].value = _normalize_color(rgba)
        except Exception:
            # mgl 非存在などの環境では黙って無視
            pass

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #
    def _upload_geometry(self, geometry: Geometry | None) -> None:
        """
        front バッファの `geometry` を 1 つの VBO/IBO に統合し GPU へ。
        データが空のときは index_count=0 にして draw() をスキップ。
        """
        if not geometry:
            self.gpu.index_count = 0
            return

        verts, inds = _geometry_to_vertices_indices(geometry, self.gpu.primitive_restart_index)
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(
                "Uploading geometry: verts=%d (%.1f KB), inds=%d (%.1f KB)",
                len(verts),
                verts.nbytes / 1024.0,
                len(inds),
                inds.nbytes / 1024.0,
            )
        self.gpu.upload(verts, inds)


# ---------- utility -------------------------------------------------------- #
def _geometry_to_vertices_indices(
    geometry: Geometry,
    primitive_restart_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Geometry オブジェクトを VBO/IBO に変換。
    GPUは多数のデータを個別に扱うよりも、大きなデータを一括で送った方が高速。
    そのため、この関数でデータをまとめて効率よくGPUに渡す。"""
    # Geometry は生成時に float32/C 連続へ正規化されるため追加変換は不要。
    coords = geometry.coords
    offsets = geometry.offsets

    num_lines = len(offsets) - 1
    total_verts = len(coords)
    total_inds = total_verts + num_lines

    indices = np.empty(total_inds, dtype=np.uint32)

    cursor = 0
    for i in range(num_lines):
        start_idx = offsets[i]
        end_idx = offsets[i + 1]
        line_length = end_idx - start_idx

        indices[cursor : cursor + line_length] = np.arange(start_idx, end_idx, dtype=np.uint32)
        cursor += line_length
        indices[cursor] = primitive_restart_index
        cursor += 1

    return coords, indices
