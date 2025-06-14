"""
📌 全体の流れ（1フレームあたり）
        1.	Rendererのtickが呼ばれる
            •	DoubleBufferに新しいデータがあるか確認
            •	あればGPUへアップロード（_upload_vertices → _merge_vertices_indices → gpu.upload）
        2.	画面描画（Rendererのdraw）
            •	GPUにアップロード済みのデータを使用して描画命令を実行
        3.	終了時にRendererがGPUリソースを解放（release）

⸻

🌟 なぜこのように設計したのか？
        •	GPUとCPUのデータ管理を明確に分け、役割を単一化（Single Responsibility Principle）。
        •	毎フレームの描画データ更新処理をシンプル化（Tickableによる一元管理）。
        •	描画データの管理を1か所で統合し、保守性・拡張性を向上。
"""

from __future__ import annotations

from typing import Sequence

import moderngl as mgl
import numpy as np

from ..core.tickable import Tickable
from ..pipeline.buffer import SwapBuffer
from .line_mesh import LineMesh
from .shader import Shader


class LineRenderer(Tickable):
    """
    DoubleBufferからデータを取得し、毎フレームGPUに送り込む作業を管理。
    DoubleBuffer（CPU側）からGPUへのデータ転送を明確に管理して、描画の一貫性を保つために必要。
    """

    def __init__(
        self,
        mgl_context: mgl.Context,
        projection_matrix: np.ndarray,
        double_buffer: SwapBuffer,
        primitive_restart_index: int = 0xFFFFFFFF,
    ):
        """
        double_buffer: GPUへ送る前のデータを管理する仕組み
        gpu: 上記のGPUBufferクラスのインスタンス。データのアップロードを任せる
        """
        self.ctx = mgl_context
        self.double_buffer = double_buffer

        # シェーダ初期化
        self.line_program = Shader.create_shader(mgl_context)
        self.line_program["projection"].write(projection_matrix.tobytes())

        # GPUBuffer を保持
        self.gpu = LineMesh(
            ctx=mgl_context,
            program=self.line_program,
            primitive_restart_index=primitive_restart_index,
        )

    # --------------------------------------------------------------------- #
    # Tickable                                                               #
    # --------------------------------------------------------------------- #
    def tick(self, dt: float) -> None:
        """
        毎フレーム呼ばれ、DoubleBufferに新データがあればGPUへ転送。
        """
        if self.double_buffer.try_swap():
            vertices_list = self.double_buffer.get_front()
            self._upload_vertices(vertices_list)

    # --------------------------------------------------------------------- #
    # Public drawing API                                                    #
    # --------------------------------------------------------------------- #
    def draw(self) -> None:
        """GPUに送ったデータを画面に描画"""
        if self.gpu.index_count > 0:
            self.gpu.vao.render(mgl.LINE_STRIP, self.gpu.index_count)

    def clear(self, color: Sequence[float]) -> None:
        """画面を指定色でクリア"""
        self.ctx.clear(*color)  # type: ignore

    def release(self) -> None:
        """GPU リソースを解放。"""
        self.gpu.release()

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #
    def _upload_vertices(self, vertices_list: Sequence[np.ndarray] | None) -> None:
        """
        front バッファの `vertices_list` を 1 つの VBO/IBO に統合し GPU へ。
        データが空のときは index_count=0 にして draw() をスキップ。
        """
        if not vertices_list:
            self.gpu.index_count = 0
            return

        verts, inds = _merge_vertices_indices(vertices_list, self.gpu.prim_restart_idx)
        self.gpu.upload(verts, inds)


# ---------- utility -------------------------------------------------------- #
def _merge_vertices_indices(
    vertices_list: Sequence[np.ndarray],
    prim_restart_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    複数ラインを 1 本の VBO/IBO に変換。
    GPUは多数のデータを個別に扱うよりも、大きなデータを一括で送った方が高速。
    そのため、この関数でデータをまとめて効率よくGPUに渡す。"""
    counts = [len(v) for v in vertices_list]  # 頂点の個数を数える
    total_verts = sum(counts)  # 全頂点の合計数
    total_inds = total_verts + len(counts)  # 全インデックス数

    vertices = np.vstack(vertices_list).astype(np.float32)  # 頂点を1つの配列に統合
    indices = np.empty(total_inds, dtype=np.uint32)  # GPUが参照する頂点の順番（索引）を格納する配列を作成

    # 頂点の索引を順番に記録。ラインごとに区切りを入れて描画の際に明確な境界を設ける
    cursor = vert_base = 0
    for cnt in counts:
        indices[cursor : cursor + cnt] = np.arange(vert_base, vert_base + cnt, dtype=np.uint32)
        cursor += cnt
        indices[cursor] = prim_restart_idx
        cursor += 1
        vert_base += cnt

    return vertices, indices
