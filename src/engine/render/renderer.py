"""
どこで: `engine.render` の高レベル描画。
何を: SwapBuffer の `Geometry` を頂点/インデックスへ変換し、ModernGL に転送して線を描画。
なぜ: 毎フレームのアップロード/描画/リソース寿命を一箇所に集約し、描画処理を単純化するため。
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any, Sequence

import moderngl as mgl
import numpy as np

from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry
from engine.runtime.frame import RenderFrame
from util.constants import PRIMITIVE_RESTART_INDEX

from ..core.tickable import Tickable
from ..runtime.buffer import SwapBuffer
from .types import Layer

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
        self._base_line_thickness = float(line_thickness)
        self.line_program["line_thickness"].value = float(self._base_line_thickness)
        # 線色（RGBA, 0–1）
        from util.color import normalize_color as _normalize_color  # 局所参照（依存を明示）

        try:
            base_col = _normalize_color(line_color)
        except Exception:  # pragma: no cover - 防御的フォールバック
            base_col = (0.0, 0.0, 0.0, 1.0)
        # 基準色を保持（レイヤー未指定時に使用）
        self._base_line_color: tuple[float, float, float, float] = (
            float(base_col[0]),
            float(base_col[1]),
            float(base_col[2]),
            float(base_col[3]),
        )
        self.line_program["color"].value = self._base_line_color

        # GPUBuffer を保持
        self.gpu = LineMesh(
            ctx=mgl_context,
            program=self.line_program,
            primitive_restart_index=PRIMITIVE_RESTART_INDEX,
        )
        # HUD 連携用: 直近アップロードの頂点/ライン数
        self._last_vertex_count: int = 0
        self._last_line_count: int = 0
        # IBO 固定化用の統計
        self._ibo_uploaded: int = 0
        self._ibo_reused: int = 0
        self._indices_built: int = 0
        # IBO 固定化の有効/ログ（環境変数）
        try:
            from common.settings import get as _get_settings

            settings = _get_settings()
            self._ibo_freeze_enabled = bool(settings.IBO_FREEZE_ENABLED)
            self._ibo_debug = bool(settings.IBO_DEBUG)
        except Exception:
            self._ibo_freeze_enabled = True
            self._ibo_debug = False
        # 受信したフレーム（layers を含む場合は draw() 側で逐次アップロード）
        self._frame: RenderFrame | None = None
        # 直近レイヤーのスナップショット（no-layers フレームで再描画に利用）
        self._last_layers_snapshot: list[Layer] | None = None
        # 直近レイヤーで適用した色（次フレーム以降に再適用するための粘着色）
        self._sticky_color: tuple[float, float, float, float] | None = None
        # subscribe 競合回避用の簡易フレームトラッキング
        self._frame_counter: int = 0
        self._last_layers_frame: int = -1

    # --------------------------------------------------------------------- #
    # Tickable                                                               #
    # --------------------------------------------------------------------- #
    def tick(self, dt: float) -> None:
        """
        毎フレーム呼ばれ、SwapBufferに新データがあればGPUへ転送。
        """
        if self.swap_buffer.try_swap():
            frame = self.swap_buffer.get_front()
            self._frame = frame
            if frame is None:
                return
            if frame.has_layers:
                # レイヤーは draw() 側で逐次アップロード
                return
            if frame.geometry is not None:
                geometry = self._resolve_geometry(frame.geometry)
                self._upload_geometry(geometry)

    # --------------------------------------------------------------------- #
    # Public drawing API                                                    #
    # --------------------------------------------------------------------- #
    def draw(self) -> None:
        """GPUに送ったデータを画面に描画"""
        # on_draw は描画のみを担当（受信は tick で行う）
        self._frame_counter += 1
        frame = self._frame
        # レイヤーが来ている場合は各レイヤーを順描画
        if frame is not None and frame.has_layers and frame.layers is not None:
            self._draw_layers_frame(frame)
            return
        # 通常経路（レイヤーが無いフレーム）
        if self._last_layers_snapshot:
            self._draw_snapshot_layers()
            return
        self._draw_geometry_only()

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
            col = _normalize_color(rgba)
            self.line_program["color"].value = col
        except Exception:
            # mgl 非存在などの環境では黙って無視
            pass

    def set_base_line_color(self, rgba: Sequence[float]) -> None:
        """基準線色（RGBA 0–1）を更新し、レイヤー未指定時/フォールバック用の色として保存する。"""
        from util.color import normalize_color as _normalize_color

        try:
            col = _normalize_color(rgba)
            self._base_line_color = (
                float(col[0]),
                float(col[1]),
                float(col[2]),
                float(col[3]),
            )
            # 現在色も更新しておく（非レイヤー描画時に即時反映）
            self.line_program["color"].value = self._base_line_color
        except Exception:
            # mgl 非存在などの環境では黙って無視
            pass

    def set_line_thickness(self, value: float) -> None:
        """線の太さ（クリップ空間基準）を即時更新する。"""
        try:
            v = float(value)
            self.line_program["line_thickness"].value = v
        except Exception:
            pass

    def set_base_line_thickness(self, value: float) -> None:
        """基準線幅を更新し、即時適用する。"""
        try:
            v = float(value)
        except Exception:
            return
        self._base_line_thickness = v
        try:
            self.line_program["line_thickness"].value = v
        except Exception:
            pass

    def get_base_line_thickness(self) -> float:
        """初期化時の基準線太さを返す。"""
        return float(self._base_line_thickness)

    # ---- subscribe ガード用の公開ヘルパ ----
    def has_pending_layers(self) -> bool:
        return bool(self._frame is not None and self._frame.has_layers)

    def layers_active_this_frame(self) -> bool:
        return self._last_layers_frame == self._frame_counter

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #
    def _draw_layers_frame(self, frame: RenderFrame) -> None:
        """レイヤー付きフレームを描画し、スナップショットと HUD を更新する。"""
        self._last_layers_frame = self._frame_counter

        total_vertices = 0
        total_indices = 0
        snapshot: list[Layer] = []

        for layer in frame.layers or ():
            geometry = self._resolve_geometry(layer.geometry)
            if geometry is None or geometry.is_empty:
                continue

            self._apply_layer_style(layer)
            self._upload_geometry(geometry)
            if self.gpu.index_count > 0:
                self.gpu.vao.render(mgl.LINE_STRIP, self.gpu.index_count)
                total_vertices += self._last_vertex_count
                total_indices += self.gpu.index_count
            snapshot.append(Layer(geometry=geometry, color=layer.color, thickness=layer.thickness))

        if total_vertices and total_indices:
            self._last_vertex_count = total_vertices
            num_lines = max(0, total_indices - total_vertices)
            self._last_line_count = num_lines

        self._last_layers_snapshot = snapshot if snapshot else None
        # 次フレーム以降はスナップショットを再利用する（追加アップロードを避ける）
        self._frame = None

    def _draw_snapshot_layers(self) -> None:
        """直近レイヤーのスナップショットを再描画する。"""
        if not self._last_layers_snapshot:
            return

        for layer in self._last_layers_snapshot:
            self._apply_layer_style(layer)
            geometry = self._resolve_geometry(layer.geometry)
            self._upload_geometry(geometry)
            if self.gpu.index_count > 0:
                self.gpu.vao.render(mgl.LINE_STRIP, self.gpu.index_count)

    def _draw_geometry_only(self) -> None:
        """geometry-only フレームの描画（粘着色を適用）。"""
        if self._sticky_color is not None:
            self.set_line_color(self._sticky_color)
        if self.gpu.index_count > 0:
            self.gpu.vao.render(mgl.LINE_STRIP, self.gpu.index_count)

    def _apply_layer_style(self, layer: Layer) -> None:
        """レイヤーの色と太さを適用し、粘着色を更新する。"""
        if layer.color is not None:
            self.set_line_color(layer.color)
            try:
                r, g, b, a = layer.color  # type: ignore[misc]
                self._sticky_color = (float(r), float(g), float(b), float(a))
            except (TypeError, ValueError):
                self._sticky_color = None
        else:
            self.set_line_color(self._base_line_color)
            base_r, base_g, base_b, base_a = self._base_line_color
            self._sticky_color = (float(base_r), float(base_g), float(base_b), float(base_a))

        if layer.thickness is not None:
            self.set_line_thickness(float(layer.thickness))
        else:
            self.set_line_thickness(self._base_line_thickness)

    def _resolve_geometry(self, geometry: Geometry | LazyGeometry | None) -> Geometry | None:
        """LazyGeometry を含む geometry を実体 Geometry に解決する。"""
        if geometry is None:
            return None
        if isinstance(geometry, LazyGeometry):
            return geometry.realize()
        return geometry

    def _upload_geometry(self, geometry: Geometry | None) -> None:
        """
        front バッファの `geometry` を 1 つの VBO/IBO に統合し GPU へ。
        データが空のときは index_count=0 にして draw() をスキップ。
        """
        if geometry is None or geometry.is_empty:
            # 空データとして扱い、計数もゼロに更新
            self.gpu.index_count = 0
            self._last_vertex_count = 0
            self._last_line_count = 0
            return

        # オフセット署名（不変なら IBO を再利用）
        try:
            offsets = geometry.offsets  # type: ignore[assignment]
            off_bytes = offsets.view(np.uint8)
            h = hashlib.blake2b(digest_size=16)
            h.update(off_bytes.tobytes())
            offsets_sig = h.digest()
        except Exception:
            offsets_sig = None

        # IBO 固定化（実験）: offsets が不変で freeze 有効なら indices を再生成せず VBO のみ更新
        use_freeze = bool(
            self._ibo_freeze_enabled
            and hasattr(self, "_last_offsets_sig")
            and offsets_sig is not None
            and getattr(self, "_last_offsets_sig") == offsets_sig
        )
        if use_freeze:
            verts = geometry.coords
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    "Uploading geometry (VBO only): verts=%d (%.1f KB)",
                    len(verts),
                    verts.nbytes / 1024.0,
                )
            try:
                self.gpu.update_vertices_only(verts)
                self._ibo_reused += 1
            except Exception:
                # フォールバック: 通常経路（念のため indices を作る）
                verts2, inds2 = _geometry_to_vertices_indices(
                    geometry, self.gpu.primitive_restart_index
                )
                self.gpu.upload(verts2, inds2)
                self._indices_built += 1
                self._ibo_uploaded += 1
        else:
            verts, inds = _geometry_to_vertices_indices(geometry, self.gpu.primitive_restart_index)
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    "Uploading geometry: verts=%d (%.1f KB), inds=%d (%.1f KB)",
                    len(verts),
                    verts.nbytes / 1024.0,
                    len(inds),
                    inds.nbytes / 1024.0,
                )
            try:
                self.gpu.upload(verts, inds)
                self._indices_built += 1
                self._ibo_uploaded += 1
                setattr(self, "_last_offsets_sig", offsets_sig)
            except Exception:
                # フォールバック: 何もしない
                pass
        # HUD 参照用の直近アップロード計数を更新
        coords_for_count = geometry.coords
        offsets_for_count = geometry.offsets
        total_verts = int(len(coords_for_count))
        num_lines = max(0, int(len(offsets_for_count)) - 1)
        self._last_vertex_count = total_verts
        self._last_line_count = num_lines

    # HUD 用: 直近アップロードの頂点/ライン数
    def get_last_counts(self) -> tuple[int, int]:
        return int(self._last_vertex_count), int(self._last_line_count)

    # IBO 固定化の実験用カウンタ（HUD/ログから参照可能）
    def get_ibo_stats(self) -> dict[str, int]:
        return {
            "uploaded": int(self._ibo_uploaded),
            "reused": int(self._ibo_reused),
            "indices_built": int(self._indices_built),
        }


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

    # ---- Indices LRU（オフセット署名ベース） ----
    if _INDICES_CACHE_ENABLED:
        logger = logging.getLogger(__name__)
        try:
            off_bytes = offsets.view(np.uint8)
            h = hashlib.blake2b(digest_size=16)
            h.update(off_bytes.tobytes())
            key = (int(primitive_restart_index), int(total_verts), h.digest())
            cached = _INDICES_CACHE.get(key)
            if cached is not None and cached.shape[0] == total_inds:
                # LRU move-to-end
                _ = _INDICES_CACHE.pop(key)
                _INDICES_CACHE[key] = cached
                _IND_HITS_INC()
                return coords, cached
            _IND_MISSES_INC()
        except Exception:
            if _INDICES_DEBUG:
                logger.debug("Indices cache lookup failed", exc_info=True)
    # ベクトル化: 連結 arange + PR マスク挿入
    indices = np.empty(total_inds, dtype=np.uint32)
    # 再始動位置（各ライン終端の直後）: offsets[1:] + 行番号
    restart_pos = offsets[1:].astype(np.int64) + np.arange(num_lines, dtype=np.int64)
    mask = np.zeros(total_inds, dtype=bool)
    mask[restart_pos] = True
    # 非 PR 部分は 0..N-1 を順に割当
    indices[~mask] = np.arange(total_verts, dtype=np.uint32)
    # PR 部分を埋める
    indices[mask] = np.uint32(primitive_restart_index)
    # LRU 保存
    if _INDICES_CACHE_ENABLED:
        logger = logging.getLogger(__name__)
        try:
            _INDICES_CACHE[key] = indices
            if _INDICES_CACHE_MAXSIZE is not None and _INDICES_CACHE_MAXSIZE > 0:
                while len(_INDICES_CACHE) > _INDICES_CACHE_MAXSIZE:
                    _INDICES_CACHE.popitem(last=False)
                    _IND_EVICTS_INC()
            _IND_STORES_INC()
        except Exception:
            if _INDICES_DEBUG:
                logger.debug("Indices cache store failed", exc_info=True)
    return coords, indices


def _load_indices_cache_settings() -> tuple[bool, int | None, bool]:
    """Indices LRU 用の設定を読み込む。"""
    try:
        from common.settings import get as _get_settings

        settings = _get_settings()
        maxsize: int | None = settings.INDICES_CACHE_MAXSIZE
        if maxsize is not None and maxsize < 0:
            maxsize = 0
        return bool(settings.INDICES_CACHE_ENABLED), maxsize, bool(settings.INDICES_DEBUG)
    except Exception:
        return True, 64, False


_INDICES_CACHE_ENABLED, _INDICES_CACHE_MAXSIZE, _INDICES_DEBUG = _load_indices_cache_settings()

_INDICES_CACHE: "OrderedDict[object, np.ndarray]" = OrderedDict()
_IND_HITS = 0
_IND_MISSES = 0
_IND_STORES = 0
_IND_EVICTS = 0


def _IND_HITS_INC() -> None:
    global _IND_HITS
    _IND_HITS += 1


def _IND_MISSES_INC() -> None:
    global _IND_MISSES
    _IND_MISSES += 1


def _IND_STORES_INC() -> None:
    global _IND_STORES
    _IND_STORES += 1


def _IND_EVICTS_INC() -> None:
    global _IND_EVICTS
    _IND_EVICTS += 1


def get_indices_cache_counters() -> dict[str, int | bool]:
    """Indices LRU の集計（HUD/デバッグ用）。"""
    return {
        "enabled": int(_INDICES_CACHE_ENABLED),
        "size": int(len(_INDICES_CACHE)),
        "maxsize": int(_INDICES_CACHE_MAXSIZE or 0),
        "hits": int(_IND_HITS),
        "misses": int(_IND_MISSES),
        "stores": int(_IND_STORES),
        "evicts": int(_IND_EVICTS),
    }
