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
from .types import StyledLayer
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
        # IBO 固定化（実験）用の統計
        self._ibo_uploaded: int = 0
        self._ibo_reused: int = 0
        self._indices_built: int = 0
        # 実験の有効/ログ（環境変数）
        try:
            from common.settings import get as _get_settings

            _s = _get_settings()
            self._ibo_freeze_enabled = bool(_s.IBO_FREEZE_ENABLED)
            self._ibo_debug = bool(_s.IBO_DEBUG)
        except Exception:
            self._ibo_freeze_enabled = True
        self._ibo_debug = False
        # レイヤーフレームバッファ（存在時は draw() 側で逐次アップロード）
        self._frame_layers: list[StyledLayer] | None = None
        # 直近レイヤーのスナップショット（no-layers フレームで再描画に利用）
        self._last_layers_snapshot: list[StyledLayer] | None = None
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
            front = self.swap_buffer.get_front()
            # レイヤー列が来た場合は draw() で逐次アップロード（duck-typing を採用）
            try:
                if isinstance(front, (list, tuple)) and front:
                    fst = front[0]
                    if (
                        hasattr(fst, "geometry")
                        and hasattr(fst, "color")
                        and hasattr(fst, "thickness")
                    ):
                        self._frame_layers = list(front)  # type: ignore[list-item]
                        return
            except Exception:
                pass
            geometry = front  # type: ignore[assignment]
            self._frame_layers = None
            self._upload_geometry(geometry)

    # --------------------------------------------------------------------- #
    # Public drawing API                                                    #
    # --------------------------------------------------------------------- #
    def draw(self) -> None:
        """GPUに送ったデータを画面に描画"""
        # on_draw は描画のみを担当（受信は tick で行う）
        # フレーム番号を増分
        try:
            self._frame_counter += 1
        except Exception:
            self._frame_counter = 0
        # レイヤーが来ている場合は各レイヤーを順描画
        if self._frame_layers:
            # このフレームでレイヤー活動があったことを記録
            self._last_layers_frame = int(self._frame_counter)

            total_vertices = 0
            total_indices = 0
            snapshot: list[StyledLayer] = []
            for layer in self._frame_layers:
                # 色
                if layer.color is not None:
                    try:
                        self.set_line_color(layer.color)
                        # 粘着色を更新（最後に適用した色を保持）
                        try:
                            r, g, b, a = layer.color  # type: ignore[misc]
                            self._sticky_color = (float(r), float(g), float(b), float(a))
                        except Exception:
                            self._sticky_color = None
                    except Exception:
                        pass
                else:
                    # style 未指定レイヤーは基準色に戻す
                    try:
                        self.set_line_color(self._base_line_color)
                        self._sticky_color = tuple(self._base_line_color)
                    except Exception:
                        pass
                # 太さ（倍率）
                if layer.thickness is not None:
                    try:
                        _mul = self._base_line_thickness * float(layer.thickness)
                        self.set_line_thickness(_mul)
                    except Exception:
                        pass
                else:
                    try:
                        self.set_line_thickness(self._base_line_thickness)
                    except Exception:
                        pass
                # アップロード→描画
                self._upload_geometry(layer.geometry)
                if self.gpu.index_count > 0:
                    self.gpu.vao.render(mgl.LINE_STRIP, self.gpu.index_count)
                    try:
                        total_vertices += int(getattr(self, "_last_vertex_count", 0))
                        total_indices += int(self.gpu.index_count)
                    except Exception:
                        pass
                # スナップショット: Lazy を実体化して保存
                try:
                    from engine.core.lazy_geometry import LazyGeometry as _LG

                    g = layer.geometry
                    if isinstance(g, _LG):
                        g = g.realize()
                    snapshot.append(
                        StyledLayer(geometry=g, color=layer.color, thickness=layer.thickness)
                    )
                except Exception:
                    pass
            # HUD 合算（近似。最後の状態を上書き）
            try:
                self._last_vertex_count = int(total_vertices)
                num_lines = max(0, int(total_indices) - int(total_vertices))
                self._last_line_count = int(num_lines)
            except Exception:
                pass
            # レイヤーを消費
            self._frame_layers = None
            # スナップショットを保持（no-layers フレームのフォールバック描画に使用）
            try:
                self._last_layers_snapshot = snapshot if snapshot else None
            except Exception:
                self._last_layers_snapshot = None
            return
        # 通常経路（レイヤーが無いフレーム）
        # フォールバック: 直近レイヤーのスナップショットを再描画
        if self._last_layers_snapshot:
            for layer in self._last_layers_snapshot:
                # 色/太さを適用
                if layer.color is not None:
                    try:
                        self.set_line_color(layer.color)
                    except Exception:
                        pass
                else:
                    try:
                        self.set_line_color(self._base_line_color)
                    except Exception:
                        pass
                if layer.thickness is not None:
                    try:
                        self.set_line_thickness(self._base_line_thickness * float(layer.thickness))
                    except Exception:
                        pass
                else:
                    try:
                        self.set_line_thickness(self._base_line_thickness)
                    except Exception:
                        pass
                # アップロード→描画（実体 Geometry のはず）
                self._upload_geometry(layer.geometry)
                if self.gpu.index_count > 0:
                    self.gpu.vao.render(mgl.LINE_STRIP, self.gpu.index_count)
            return
        # スナップショットが無ければ、従来の geometry-only 描画へ（粘着色を適用）
        try:
            if self._sticky_color is not None:
                self.set_line_color(self._sticky_color)
        except Exception:
            pass
        if self.gpu.index_count > 0:
            self.gpu.vao.render(mgl.LINE_STRIP, self.gpu.index_count)
        else:
            pass

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

    def get_base_line_thickness(self) -> float:
        """初期化時の基準線太さを返す。"""
        try:
            return float(self._base_line_thickness)
        except Exception:
            return 0.0006

    # ---- subscribe ガード用の公開ヘルパ ----
    def has_pending_layers(self) -> bool:
        try:
            return bool(self._frame_layers)
        except Exception:
            return False

    def layers_active_this_frame(self) -> bool:
        try:
            return int(self._last_layers_frame) == int(self._frame_counter)
        except Exception:
            return False

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #
    def _upload_geometry(self, geometry: Geometry | LazyGeometry | None) -> None:
        """
        front バッファの `geometry` を 1 つの VBO/IBO に統合し GPU へ。
        データが空のときは index_count=0 にして draw() をスキップ。
        """
        if geometry is None:
            # 空データとして扱い、計数もゼロに更新
            self.gpu.index_count = 0
            try:
                self._last_vertex_count = 0  # type: ignore[attr-defined]
                self._last_line_count = 0  # type: ignore[attr-defined]
            except Exception:
                pass

            return
        # LazyGeometry はここで実体化
        if isinstance(geometry, LazyGeometry):
            geometry = geometry.realize()
        # 早期スキップ（空）
        try:
            if getattr(geometry, "is_empty", False):  # type: ignore[truthy-bool]
                self.gpu.index_count = 0
                try:
                    self._last_vertex_count = 0  # type: ignore[attr-defined]
                    self._last_line_count = 0  # type: ignore[attr-defined]
                except Exception:
                    pass

                return
        except Exception:
            pass

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
        try:
            total_verts = int(len(verts))
            total_inds = int(len(inds))
            num_lines = max(0, total_inds - total_verts)
            self._last_vertex_count = total_verts  # type: ignore[attr-defined]
            self._last_line_count = num_lines  # type: ignore[attr-defined]
        except Exception:
            pass

    # HUD 用: 直近アップロードの頂点/ライン数
    def get_last_counts(self) -> tuple[int, int]:
        try:
            return int(self._last_vertex_count), int(self._last_line_count)  # type: ignore[attr-defined]
        except Exception:
            # 初期値（未アップロード）
            return 0, 0

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
            pass
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
        try:
            _INDICES_CACHE[key] = indices
            if _INDICES_CACHE_MAXSIZE is not None and _INDICES_CACHE_MAXSIZE > 0:
                while len(_INDICES_CACHE) > _INDICES_CACHE_MAXSIZE:
                    _INDICES_CACHE.popitem(last=False)
                    _IND_EVICTS_INC()
            _IND_STORES_INC()
        except Exception:
            pass
    return coords, indices


# ---- Indices LRU: 設定/カウンタ/取得API -------------------------------------
try:
    from common.settings import get as _get_settings

    _s2 = _get_settings()
    _INDICES_CACHE_ENABLED = bool(_s2.INDICES_CACHE_ENABLED)
    _INDICES_CACHE_MAXSIZE: int | None = _s2.INDICES_CACHE_MAXSIZE
    if _INDICES_CACHE_MAXSIZE is not None and _INDICES_CACHE_MAXSIZE < 0:
        _INDICES_CACHE_MAXSIZE = 0
    _INDICES_DEBUG = bool(_s2.INDICES_DEBUG)
except Exception:
    _INDICES_CACHE_ENABLED = True
    _INDICES_CACHE_MAXSIZE = 64
    _INDICES_DEBUG = False

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
