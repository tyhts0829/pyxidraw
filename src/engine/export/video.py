"""
どこで: `engine.export.video`。
何を: 実行中ウィンドウの内容をフレーム毎に取得して MP4 へ保存する最小リコーダ。
なぜ: ホットキー操作だけで録画を開始/停止できるようにするため（UI は遅くなっても映像はフレーム欠落しない）。

初期実装の方針:
- オーバーレイ含む画面そのまま（`pyglet` のカラーバッファ）を RGBA→RGB で取り出し、同期でエンコード。
- 依存は任意（`imageio-ffmpeg` or `imageio`）。見つからない場合は開始時に明確な RuntimeError を送出。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from util.paths import ensure_video_dir


def _even(v: int) -> int:
    return int(v) & ~1  # 最下位ビットを落として偶数へ


def _default_video_path(width: int, height: int, fps: int, name_prefix: Optional[str]) -> Path:
    out_dir = ensure_video_dir()
    dims = f"{int(width)}x{int(height)}_{int(fps)}fps"
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    if name_prefix and name_prefix.strip():
        base = f"{name_prefix}_{dims}_{ts}"
    else:
        base = f"{dims}_{ts}"
    path = out_dir / f"{base}.mp4"
    # 一意化
    i = 1
    p = path
    while p.exists():
        p = out_dir / f"{base}-{i}.mp4"
        i += 1
    return p


@dataclass
class _Writer:
    close: Any
    append_data: Any


def _open_writer(path: Path, width: int, height: int, fps: int) -> _Writer:
    """imageio / imageio-ffmpeg のどちらかで writer を作成する（存在時のみ）。"""
    # まず imageio-ffmpeg を試す
    try:
        import imageio_ffmpeg as _iioff

        # imageio-ffmpeg は低レベル API のため、シンプルな imageio writer を優先
        # フォールバックとして imageio を試すため、ここでは敢えて使わない
        del _iioff  # noqa: F841
    except Exception:
        pass
    # 次に imageio（v2/v3）を試す
    try:
        import imageio.v3 as iio  # type: ignore

        # v3 は `iio.get_writer(..., plugin='ffmpeg')` が利用可能
        writer = iio.get_writer(str(path), fps=int(fps), codec="libx264", plugin="ffmpeg")  # type: ignore
        return _Writer(close=writer.close, append_data=writer.append_data)
    except Exception:
        try:
            import imageio as iio  # type: ignore

            writer = iio.get_writer(str(path), fps=int(fps))
            return _Writer(close=writer.close, append_data=writer.append_data)
        except Exception as e:
            raise RuntimeError("imageio/imageio-ffmpeg が見つからないため録画できません") from e


class VideoRecorder:
    """ウィンドウの内容を MP4 として保存する最小の同期リコーダ。"""

    def __init__(self) -> None:
        self._writer: _Writer | None = None
        self._size: tuple[int, int] | None = None
        self._fps: int = 60
        self._name_prefix: Optional[str] = None
        self._path: Optional[Path] = None
        self._include_overlay: bool = True
        self._mgl_context: Any | None = None
        self._draw_callable: Any | None = None
        # Pre-allocated FBOs for overlay=False (Shift+V)
        self._samples: int = 0
        self._msaa_fbo: Any | None = None
        self._resolve_fbo: Any | None = None
        # PBO ring for async read (overlay=False)
        self._pbo_ring: list[Any] | None = None
        self._pbo_index: int = 0
        self._pbo_filled: int = 0
        self._pbo_bytes: int = 0
        # Writer-side policy: drop first frame once to avoid initial glitch
        self._drop_initial_frames: int = 0
        # Pre-allocated FBOs for overlay=False (Shift+V)
        self._samples: int = 0
        self._msaa_fbo: Any | None = None
        self._resolve_fbo: Any | None = None

    # ---- state ----
    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    # ---- API ----
    def start(
        self,
        window: Any,
        fps: int,
        *,
        name_prefix: Optional[str] = None,
        include_overlay: bool = True,
        mgl_context: Any | None = None,
        draw: Any | None = None,
    ) -> None:
        """録画を開始する（すでに録画中なら何もしない）。"""
        if self.is_recording:
            return
        # 実フレームバッファのピクセル寸法を優先（HiDPI 対応）
        width, height = 0, 0
        try:
            import pyglet  # type: ignore

            buf = pyglet.image.get_buffer_manager().get_color_buffer()
            img = buf.get_image_data()
            width, height = int(img.width), int(img.height)
        except Exception:
            try:
                width = int(getattr(window, "width", 0))
                height = int(getattr(window, "height", 0))
            except Exception:
                width, height = 0, 0
        if width <= 1 or height <= 1:
            raise RuntimeError("ウィンドウ解像度が不正のため録画できません")
        # H.264 都合で偶数に丸める（切り下げクロップ）
        width2, height2 = _even(width), _even(height)
        if width2 != width or height2 != height:
            # 1px 以内のクロップのみ行うので事前に寸法を保持
            width, height = width2, height2
        path = _default_video_path(width, height, int(fps), name_prefix)
        # include_overlay=False の場合は FBO 経由で描画するため、mgl_context/draw を必須とする
        if not include_overlay and (mgl_context is None or draw is None):
            raise RuntimeError("include_overlay=False では mgl_context と draw が必須です")
        writer = _open_writer(path, width, height, int(fps))
        # 状態保存
        self._writer = writer
        self._size = (width, height)
        self._fps = int(fps)
        self._name_prefix = name_prefix
        self._path = path
        self._include_overlay = bool(include_overlay)
        self._mgl_context = mgl_context
        self._draw_callable = draw
        # drop first captured frame (warmup)
        self._drop_initial_frames = 10
        # Pre-allocate FBOs / PBOs for overlay=False
        if not self._include_overlay:
            try:
                import moderngl  # noqa: F401
            except Exception as e:
                raise RuntimeError(f"ModernGL の利用に失敗: {e}") from e
            assert self._mgl_context is not None
            assert self._size is not None
            w, h = int(self._size[0]), int(self._size[1])
            # 推奨サンプル数（Window 設定があれば利用）
            try:
                self._samples = int(getattr(getattr(window, "config", None), "samples", 0))
            except Exception:
                self._samples = 0
            if self._samples <= 0:
                self._samples = 4
            ctx = self._mgl_context
            # 既存 FBO/PBO の開放
            self._release_fbos()
            self._release_pbos()
            try:
                # MSAA 優先
                try:
                    self._msaa_fbo = ctx.simple_framebuffer((w, h), components=4, samples=int(self._samples))  # type: ignore[attr-defined]
                    self._resolve_fbo = ctx.simple_framebuffer((w, h), components=4)  # type: ignore[attr-defined]
                except Exception:
                    # 非 MSAA にフォールバック
                    self._msaa_fbo = None
                    self._resolve_fbo = ctx.simple_framebuffer((w, h), components=4)  # type: ignore[attr-defined]
                # PBO リング（3枚）
                self._pbo_bytes = int(w * h * 4)
                ring = []
                for _ in range(3):
                    try:
                        ring.append(ctx.buffer(reserve=self._pbo_bytes))  # type: ignore[attr-defined]
                    except Exception:
                        ring = []
                        break
                self._pbo_ring = ring if ring else None
                self._pbo_index = 0
                self._pbo_filled = 0
            except Exception as e:
                self._release_fbos()
                self._release_pbos()
                raise RuntimeError(f"FBO/PBO 初期化に失敗: {e}") from e
        # Pre-allocate FBOs for overlay=False
        if not self._include_overlay:
            try:
                import moderngl  # noqa: F401
            except Exception as e:
                raise RuntimeError(f"ModernGL の利用に失敗: {e}") from e
            assert self._mgl_context is not None
            assert self._size is not None
            w, h = int(self._size[0]), int(self._size[1])
            # 推奨サンプル数（Window 設定があれば利用）
            try:
                self._samples = int(getattr(getattr(window, "config", None), "samples", 0))
            except Exception:
                self._samples = 0
            if self._samples <= 0:
                self._samples = 4
            ctx = self._mgl_context
            # 既存 FBO の開放
            self._release_fbos()
            try:
                # MSAA 優先
                try:
                    self._msaa_fbo = ctx.simple_framebuffer((w, h), components=4, samples=int(self._samples))  # type: ignore[attr-defined]
                    self._resolve_fbo = ctx.simple_framebuffer((w, h), components=4)  # type: ignore[attr-defined]
                except Exception:
                    # 非 MSAA にフォールバック
                    self._msaa_fbo = None
                    self._resolve_fbo = ctx.simple_framebuffer((w, h), components=4)  # type: ignore[attr-defined]
            except Exception as e:
                self._release_fbos()
                raise RuntimeError(f"FBO 初期化に失敗: {e}") from e

    def stop(self) -> Path:
        """録画を停止してファイルを確定する。録画中でなければエラー。"""
        if not self.is_recording:
            raise RuntimeError("録画は開始されていません")
        assert self._writer is not None
        assert self._path is not None
        try:
            self._writer.close()
        finally:
            self._writer = None
            self._release_fbos()
            self._release_pbos()
        return self._path

    # ---- drawing hook ----
    def capture_current_frame(self, window: Any) -> None:
        """現在のウィンドウ内容を 1 フレームとして書き出す（録画中のみ）。"""
        if not self.is_recording:
            return
        if self._include_overlay:
            try:
                import pyglet  # type: ignore
            except Exception as e:  # pragma: no cover - 実行時依存
                raise RuntimeError("pyglet が初期化されていないため録画できません") from e

            # 画面バッファ → ImageData → RGBA bytes
            buf = pyglet.image.get_buffer_manager().get_color_buffer()
            img = buf.get_image_data()
            w, h = int(img.width), int(img.height)
            raw = img.get_data("RGBA", w * 4)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
            # 上下反転 + 偶数寸法へクロップ + RGB へ
            arr = np.flipud(arr)
            even_w, even_h = (_even(w), _even(h))
            if even_w != w or even_h != h:
                arr = arr[:even_h, :even_w, :]
            frame = arr[..., :3]
        else:
            # FBO 経由でラインのみを描画（HUD を含まない）
            assert self._mgl_context is not None
            assert self._draw_callable is not None
            assert self._size is not None
            ctx = self._mgl_context
            width, height = int(self._size[0]), int(self._size[1])

            # ビューポート退避
            try:
                old_viewport = ctx.viewport  # type: ignore[attr-defined]
            except Exception:
                old_viewport = None

            try:
                # ターゲット FBO を選択（MSAA があれば優先）
                target_fbo = self._msaa_fbo if self._msaa_fbo is not None else self._resolve_fbo
                assert target_fbo is not None
                target_fbo.use()
                # 背景色
                bg = getattr(window, "_bg_color", (1.0, 1.0, 1.0, 1.0))
                r, g, b, a = bg
                target_fbo.clear(float(r), float(g), float(b), float(a))
                ctx.viewport = (0, 0, width, height)  # type: ignore[attr-defined]
                # ラインのみ描画
                self._draw_callable()

                # 読み出し
                if self._msaa_fbo is not None and self._resolve_fbo is not None:
                    # MSAA → 単一サンプルへ解像
                    try:
                        try:
                            ctx.copy_framebuffer(self._resolve_fbo, self._msaa_fbo)  # type: ignore[attr-defined]
                        except Exception:
                            self._msaa_fbo.copy_from(self._resolve_fbo)  # type: ignore[attr-defined]
                        # 非同期 read: PBO があれば read_into（次フレームで取り出す）
                        if self._pbo_ring:
                            pbo = self._pbo_ring[self._pbo_index]
                            self._resolve_fbo.read_into(pbo, components=4, alignment=1)  # type: ignore[attr-defined]
                            # 画面へブリット（表示）。失敗時は無視
                            try:
                                ctx.copy_framebuffer(ctx.screen, self._resolve_fbo)  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            # 前フレームの PBO を読み出し
                            if self._pbo_filled > 0:
                                prev = (self._pbo_index - 1) % len(self._pbo_ring)
                                data = self._pbo_ring[prev].read()  # type: ignore[attr-defined]
                            else:
                                # 初回は同期 read
                                data = self._resolve_fbo.read(components=4, alignment=1)
                                self._pbo_filled = 1
                            self._pbo_index = (self._pbo_index + 1) % len(self._pbo_ring)
                        else:
                            # フォールバック: 同期 read
                            data = self._resolve_fbo.read(components=4, alignment=1)
                        # 画面へブリット（表示）。失敗時は無視して録画継続。
                        try:
                            ctx.copy_framebuffer(ctx.screen, self._resolve_fbo)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    except Exception as e:
                        raise RuntimeError(f"MSAA 解像に失敗: {e}") from e
                else:
                    if self._pbo_ring:
                        pbo = self._pbo_ring[self._pbo_index]
                        target_fbo.read_into(pbo, components=4, alignment=1)  # type: ignore[attr-defined]
                        try:
                            ctx.copy_framebuffer(ctx.screen, target_fbo)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        if self._pbo_filled > 0:
                            prev = (self._pbo_index - 1) % len(self._pbo_ring)
                            data = self._pbo_ring[prev].read()  # type: ignore[attr-defined]
                        else:
                            data = target_fbo.read(components=4, alignment=1)
                            self._pbo_filled = 1
                        self._pbo_index = (self._pbo_index + 1) % len(self._pbo_ring)
                    else:
                        data = target_fbo.read(components=4, alignment=1)
                        # 画面へブリット（表示）。失敗時は無視
                        try:
                            ctx.copy_framebuffer(ctx.screen, target_fbo)  # type: ignore[attr-defined]
                        except Exception:
                            pass

                arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
                arr = np.flipud(arr)
                frame = arr[..., :3]
            except Exception as e:
                raise RuntimeError(f"オフスクリーン描画に失敗: {e}") from e
            finally:
                # 後片付け
                try:
                    if old_viewport is not None:
                        ctx.viewport = old_viewport  # type: ignore[attr-defined]
                    ctx.screen.use()  # type: ignore[attr-defined]
                except Exception:
                    pass

        # 書き出し（同期）。初回フレームはドロップしてウォームアップ。
        assert self._writer is not None
        if self._drop_initial_frames > 0:
            self._drop_initial_frames -= 1
        else:
            self._writer.append_data(frame)

    # ---- internal helpers ----
    def _release_fbos(self) -> None:
        for _f in (self._msaa_fbo, self._resolve_fbo):
            try:
                if _f is not None:
                    _f.release()
            except Exception:
                pass
        self._msaa_fbo = None
        self._resolve_fbo = None

    def _release_pbos(self) -> None:
        if not self._pbo_ring:
            return
        for b in self._pbo_ring:
            try:
                if b is not None:
                    b.release()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._pbo_ring = None
        self._pbo_index = 0
        self._pbo_filled = 0
        self._pbo_bytes = 0
