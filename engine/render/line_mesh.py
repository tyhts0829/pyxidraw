from __future__ import annotations

import moderngl as mgl
import numpy as np


class LineMesh:
    """
    GPUに頂点やインデックスなどの描画データを送り込む作業を管理
    """

    def __init__(
        self,
        ctx: mgl.Context,
        program: mgl.Program,
        initial_reserve: int = 200 * 1024 * 1024,
        primitive_restart_index: int = 0xFFFFFFFF,
    ):
        """
        ctx: GPUへの描画処理を行うためのモダンOpenGL（moderngl）コンテキスト
        program: GPU側で使うシェーダープログラム。
        VBO (Vertex Buffer Object): GPUに送る「頂点データ」を格納するメモリ。
        IBO (Index Buffer Object): GPUに「頂点の順序（描画のための索引）」を送るメモリ。
        VAO (Vertex Array Object): VBOとIBOを関連付けて、描画命令をシンプルに管理する仕組み。
        Primitive Restart Index: 描画時に「ここで一旦区切る」という目印。
        """
        self.ctx = ctx
        self.program = program
        self.initial_reserve = initial_reserve
        # 命名統一: primitive_restart_index に一本化
        self.primitive_restart_index = primitive_restart_index

        # バッファ予約
        self.vbo = ctx.buffer(reserve=initial_reserve, dynamic=True)
        self.ibo = ctx.buffer(reserve=initial_reserve, dynamic=True)
        self.vao = ctx.simple_vertex_array(program, self.vbo, "in_vert", index_buffer=self.ibo)

        # 描画ステート
        self.index_count: int = 0
        self.ctx.primitive_restart = True  # type: ignore
        self.ctx.primitive_restart_index = primitive_restart_index  # type: ignore

    # ---------- バッファ操作 ----------
    def _ensure_capacity(self, vbo_size: int, ibo_size: int) -> None:
        """データが大きくなったらGPUのバッファを再確保"""
        if vbo_size > self.vbo.size:
            self.vbo.release()
            self.vbo = self.ctx.buffer(reserve=max(vbo_size, self.initial_reserve), dynamic=True)

        if ibo_size > self.ibo.size:
            self.ibo.release()
            self.ibo = self.ctx.buffer(reserve=max(ibo_size, self.initial_reserve), dynamic=True)

        # VAO は VBO/IBO が差し替わるたびに張り直す
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, "in_vert", index_buffer=self.ibo)

    def upload(self, vertices: np.ndarray, indices: np.ndarray) -> None:
        """実際にデータをGPUへ送り込む"""
        self._ensure_capacity(vertices.nbytes, indices.nbytes)

        self.vbo.orphan()
        self.vbo.write(vertices.tobytes())

        self.ibo.orphan()
        self.ibo.write(indices.tobytes())

        self.index_count = len(indices)

    def release(self) -> None:
        """GPUのメモリを解放する（終了時に使う）"""
        self.vbo.release()
        self.ibo.release()
        self.vao.release()

    # 後方互換: 旧属性名 prim_restart_idx をプロパティで提供
    @property
    def prim_restart_idx(self) -> int:  # deprecated
        return self.primitive_restart_index

    @prim_restart_idx.setter
    def prim_restart_idx(self, value: int) -> None:  # deprecated
        self.primitive_restart_index = value
