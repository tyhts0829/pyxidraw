"""
どこで: `engine.export.image`。
何を: 現在の描画ウィンドウ内容を PNG として保存するラッパ（最小実装）。
なぜ: ワンアクションでスクリーンショットを得られるようにするため。

Stage 4: オーバーレイ含む低コストパス（window バッファ保存）のみ実装。
高解像度/オーバーレイ除外/透過背景は将来の FBO 経由パスで対応する。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyglet

from util.paths import ensure_screenshots_dir


def save_png(
    window: "pyglet.window.Window",
    path: Path | None = None,
    *,
    scale: float = 1.0,
    include_overlay: bool = True,
    transparent: bool = False,
) -> Path:
    """現在のウィンドウ内容を PNG として保存する。

    Parameters
    ----------
    window : pyglet.window.Window
        対象ウィンドウ。
    path : Path | None
        出力先パス。None の場合は既定の `screenshots/` にタイムスタンプ名で保存。
    scale : float, default 1.0
        画像のスケール。1.0 以外は現段階では未対応（NotImplementedError）。
    include_overlay : bool, default True
        True の場合はウィンドウの見た目そのまま（HUD含む）を保存。
        False は現段階では未対応（NotImplementedError）。
    transparent : bool, default False
        透過背景。現段階では未対応（NotImplementedError）。

    Returns
    -------
    Path
        保存先のファイルパス。
    """
    if scale != 1.0:
        raise NotImplementedError("scale != 1.0 は未対応（将来 FBO にて対応予定）")
    if not include_overlay:
        raise NotImplementedError("include_overlay=False は未対応（将来 FBO にて対応予定）")
    if transparent:
        raise NotImplementedError("transparent=True は未対応（将来 FBO にて対応予定）")

    if path is None:
        out_dir = ensure_screenshots_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        w, h = int(window.width), int(window.height)
        path = _unique_path(out_dir / f"{ts}_{w}x{h}.png")

    # バッファからそのまま保存（RGBA）
    try:
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        buffer.save(str(path))
    except Exception as e:  # pyglet が未初期化/ヘッドレスなど
        raise RuntimeError(f"PNG 保存に失敗: {e}") from e
    return path


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        cand = parent / f"{stem}-{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1
