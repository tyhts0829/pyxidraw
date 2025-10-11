"""
どこで: `engine.ui.hud` の HUD 表示モジュール。
何を: MetricSampler のキー/値ペアを pyglet の Label でオーバーレイ描画する。
なぜ: 実行時メトリクスを即座に可視化し、デバッグ/チューニングのフィードバックを高めるため。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

import pyglet
from pyglet.shapes import Rectangle
from pyglet.window import Window

from ...core.tickable import Tickable
from .config import HUDConfig
from .sampler import MetricSampler

try:
    from util.utils import load_config as _load_config  # type: ignore
except Exception:  # pragma: no cover - フォールバック
    _load_config = lambda: {}  # type: ignore[assignment]


class OverlayHUD(Tickable):
    """MetricSampler が溜めた文字列を pyglet Label で描画する。"""

    def __init__(
        self,
        window: Window,
        sampler: MetricSampler,
        *,
        config: HUDConfig | None = None,
        font_size: int = 8,
        color=(0, 0, 0, 155),
    ):
        self.window = window
        self.sampler = sampler
        self._config = config or HUDConfig()
        self._labels: dict[str, pyglet.text.Label] = {}
        self._color = color
        self._font = "HackGenConsoleNF-Regular"
        self.font_size = font_size
        # --- messages/progress ---
        self._messages: list[tuple[str, float, Literal["info", "warn", "error"]]] = []
        self._progress: dict[str, tuple[int, int]] = {}
        # --- meters (bars) ---
        self._bars_bg: dict[str, Rectangle] = {}
        self._bars_fg: dict[str, Rectangle] = {}
        self._meter_ema: dict[str, float] = {}
        # メータ表示パラメータ（HUDConfig をベースに default.yaml で上書き）
        self._meter_width_px: int = int(self._config.meter_width_px)
        self._meter_height_px: int = int(self._config.meter_height_px)
        self._meter_gap_px: int = int(self._config.meter_gap_px)
        self._meter_alpha_fg: int = int(self._config.meter_alpha_fg)
        self._meter_alpha_bg: int = int(self._config.meter_alpha_bg)
        self._meter_color_fg: tuple[int, int, int] = self._config.meter_color_fg
        self._meter_color_bg: tuple[int, int, int] = (50, 50, 50)
        self._smoothing_alpha: float = float(self._config.smoothing_alpha)
        self._meter_left_margin_px: int = 10
        try:
            cfg = _load_config() or {}
            hud_cfg = cfg.get("hud", {})
            if isinstance(hud_cfg, dict):
                # フォント（任意）: hud.font_name / hud.font_size を優先。互換として status_manager.* を参照。
                try:
                    fn = hud_cfg.get("font_name")
                    if isinstance(fn, str) and fn.strip():
                        self._font = fn.strip()
                except Exception:
                    pass
                try:
                    fs = hud_cfg.get("font_size")
                    if fs is not None:
                        self.font_size = int(fs)
                except Exception:
                    pass
                # テキスト色（任意）: hud.text_color を許容（Hex / 0..1 / 0..255）
                try:
                    text_col = hud_cfg.get("text_color")
                    if text_col is not None:
                        from util.color import to_u8_rgba as _to_u8_rgba

                        r, g, b, a = _to_u8_rgba(text_col)
                        self._color = (int(r), int(g), int(b), int(a))
                except Exception:
                    pass
                meters = hud_cfg.get("meters", {})
                if isinstance(meters, dict):
                    self._meter_width_px = int(meters.get("meter_width_px", self._meter_width_px))
                    self._meter_height_px = int(
                        meters.get("meter_height_px", self._meter_height_px)
                    )
                    self._meter_gap_px = int(meters.get("meter_gap_px", self._meter_gap_px))
                    self._meter_alpha_fg = int(meters.get("meter_alpha_fg", self._meter_alpha_fg))
                    self._meter_alpha_bg = int(meters.get("meter_alpha_bg", self._meter_alpha_bg))
                    col = meters.get("meter_color_fg")
                    if col is not None:
                        try:
                            # Hex / 0..1 / 0..255 のいずれも許容
                            from util.color import to_u8_rgb as _to_u8_rgb

                            r, g, b = _to_u8_rgb(col)
                            self._meter_color_fg = (int(r), int(g), int(b))
                        except Exception:
                            # 互換: 既存の配列形式を最後に試す
                            try:
                                if isinstance(col, (list, tuple)) and len(col) >= 3:
                                    r, g, b = int(col[0]), int(col[1]), int(col[2])
                                    self._meter_color_fg = (r, g, b)
                            except Exception:
                                pass
                    colbg = meters.get("meter_color_bg")
                    if colbg is not None:
                        try:
                            from util.color import to_u8_rgb as _to_u8_rgb

                            r, g, b = _to_u8_rgb(colbg)
                            self._meter_color_bg = (int(r), int(g), int(b))
                        except Exception:
                            try:
                                if isinstance(colbg, (list, tuple)) and len(colbg) >= 3:
                                    r, g, b = int(colbg[0]), int(colbg[1]), int(colbg[2])
                                    self._meter_color_bg = (r, g, b)
                            except Exception:
                                pass
                    self._smoothing_alpha = float(
                        meters.get("smoothing_alpha", self._smoothing_alpha)
                    )
                    self._meter_left_margin_px = int(
                        meters.get("meter_left_margin_px", self._meter_left_margin_px)
                    )
            # 互換フォールバック: status_manager.font / status_manager.font_size
            sm = cfg.get("status_manager", {})
            if isinstance(sm, dict):
                try:
                    if (not self._font) or self._font == "HackGenConsoleNF-Regular":
                        fn = sm.get("font")
                        if isinstance(fn, str) and fn.strip():
                            self._font = fn.strip()
                except Exception:
                    pass
                try:
                    if self.font_size == font_size:  # 未上書きの場合だけ
                        fs = sm.get("font_size")
                        if fs is not None:
                            self.font_size = int(fs)
                except Exception:
                    pass
            # 設定ディレクトリ配下のフォントを pyglet に登録（ベストエフォート）
            try:
                from util.utils import _find_project_root as _root
                from util.utils import load_config as _lc

                ccfg = _lc() or {}
                fcfg = ccfg.get("fonts", {}) if isinstance(ccfg, dict) else {}
                sdirs = fcfg.get("search_dirs", []) if isinstance(fcfg, dict) else []
                if isinstance(sdirs, (str, int)):
                    sdirs = [str(sdirs)]
                root = _root(Path(__file__).parent)
                font_exts = (".ttf", ".otf", ".ttc")
                for s in sdirs:
                    try:
                        p = Path(os.path.expandvars(os.path.expanduser(str(s))))
                        if not p.is_absolute():
                            p = (root / p).resolve()
                        if not p.exists() or not p.is_dir():
                            continue
                        for ext in font_exts:
                            for fp in p.glob(f"**/*{ext}"):
                                try:
                                    pyglet.font.add_file(str(fp))
                                except Exception:
                                    # `.ttc` など非対応の場合もあるため握りつぶし
                                    pass
                    except Exception:
                        continue
            except Exception:
                pass
        except Exception:
            # コンフィグ読み込み失敗時は既定を維持
            pass

    # -------- Tickable --------
    def tick(self, dt: float) -> None:
        # ラベル生成 & 更新（順序は HUDConfig に従う。未知キーは末尾に追加）
        desired = list(self._config.resolved_order())
        # sampler.data に存在するが order に無いキーを後置
        for k in self.sampler.data.keys():
            if k not in desired:
                desired.append(k)

        # 再配置のため、新しいラベル辞書を構築
        new_labels: dict[str, pyglet.text.Label] = {}
        start_y = 10
        line_h = 18
        for i, key in enumerate(desired):
            if key not in self.sampler.data:
                continue
            y = start_y + i * line_h
            lab = self._labels.get(key)
            if lab is None:
                lab = pyglet.text.Label(
                    text="",
                    x=10,
                    y=y,
                    anchor_x="left",
                    anchor_y="bottom",
                    font_name=self._font,
                    font_size=self.font_size,
                    color=self._color,
                )
            else:
                # 位置を更新
                lab.y = y
            lab.text = f"{key} : {self.sampler.data.get(key, '')}"
            new_labels[key] = lab
            # メータの座標・EMA を更新
            if self._config.show_meters:
                bar_w = int(self._meter_width_px)
                bar_h = int(self._meter_height_px)
                # inline（縦位置はラベル行のセンターに揃える）
                bar_x = int(self._meter_left_margin_px)
                bar_y = int(y + (line_h - bar_h) // 2)
                bg = self._bars_bg.get(key)
                fg = self._bars_fg.get(key)
                if bg is None:
                    bg = Rectangle(bar_x, bar_y, bar_w, bar_h, color=(50, 50, 50))
                    bg.opacity = int(self._meter_alpha_bg)
                    self._bars_bg[key] = bg
                else:
                    bg.x = bar_x
                    bg.y = bar_y
                    bg.width = bar_w
                    bg.height = bar_h
                if fg is None:
                    r, g, b = self._meter_color_fg
                    fg = Rectangle(bar_x, bar_y, 0, bar_h, color=(int(r), int(g), int(b)))
                    fg.opacity = int(self._meter_alpha_fg)
                    self._bars_fg[key] = fg
                else:
                    fg.x = bar_x
                    fg.y = bar_y
                    fg.height = bar_h
                # 正規化値と EMA
                ratio = self._normalized_ratio(key)
                if ratio is not None:
                    prev = self._meter_ema.get(key)
                    a = float(self._smoothing_alpha)
                    if prev is None:
                        ema = float(ratio)
                    else:
                        ema = a * float(ratio) + (1.0 - a) * float(prev)
                    self._meter_ema[key] = ema
                    fg.width = int(round(bar_w * max(0.0, min(1.0, ema))))
                else:
                    self._meter_ema.pop(key, None)
                    fg.width = 0
            else:
                # メータ非表示: キャッシュを掃除
                self._bars_bg.pop(key, None)
                self._bars_fg.pop(key, None)
        # 不要になったラベルは破棄（pyglet 側のリソース管理は任せる）
        self._labels = new_labels
        # メッセージの有効期限を掃除
        now = time.monotonic()
        self._messages = [m for m in self._messages if m[1] > now]

    # -------- draw --------
    def draw(self) -> None:
        # メータ（バー）を先に描画して、その上にテキストを重ねる
        if self._config.show_meters:
            for key in self._labels.keys():
                bg = self._bars_bg.get(key)
                fg = self._bars_fg.get(key)
                if bg is not None:
                    try:
                        bg.color = self._meter_color_bg
                        bg.opacity = int(self._meter_alpha_bg)
                    except Exception:
                        pass
                    bg.draw()
                if fg is not None:
                    # メータ色の更新を反映
                    try:
                        fg.color = self._meter_color_fg
                    except Exception:
                        pass
                    fg.draw()
        # テキストメトリクス
        for lab in self._labels.values():
            # テキスト色の更新を反映
            try:
                lab.color = self._color
            except Exception:
                pass
            lab.draw()
        # 進捗 (%表示)
        y = 10 + len(self._labels) * 18 + 8
        for key, (done, total) in self._progress.items():
            pct = 0 if total <= 0 else int(round(100 * done / max(1, total)))
            lbl = pyglet.text.Label(
                text=f"{key} : {pct}%",
                x=10,
                y=y,
                anchor_x="left",
                anchor_y="bottom",
                font_name=self._font,
                font_size=self.font_size,
                color=self._color,
            )
            lbl.draw()
            y += 18
        # 一時メッセージ（最後に重ねて表示）
        for text, _expire, level in self._messages:
            rgba = {
                "info": (0, 0, 0, 200),
                "warn": (200, 120, 0, 230),
                "error": (200, 0, 0, 230),
            }[level]
            lbl = pyglet.text.Label(
                text=text,
                x=10,
                y=self.window.height - 20,
                anchor_x="left",
                anchor_y="top",
                font_name=self._font,
                font_size=self.font_size + 2,
                color=rgba,
            )
            lbl.draw()

    # -------- helpers --------
    def _normalized_ratio(self, key: str) -> float | None:
        # CACHE: MISS で塗りつぶし（1.0）、HIT で 0.0
        if key in ("CACHE/SHAPE", "CACHE/EFFECT"):
            status = str(self.sampler.data.get(key, "")).upper()
            if "MISS" in status:
                return 1.0
            if "HIT" in status:
                return 0.0
            return 0.0
        # Sampler 側で値が無ければ None
        v = self.sampler.values.get(key)
        if v is None:
            return None
        if key == "CPU":
            return max(0.0, min(1.0, float(v) / 100.0))
        if key == "MEM":
            denom = float(max(1, self.sampler.mem_max_bytes()))
            return max(0.0, min(1.0, float(v) / denom))
        if key == "FPS":
            denom = float(max(1e-6, self.sampler.target_fps()))
            return max(0.0, min(1.0, float(v) / denom))
        if key == "VERTEX":
            denom = float(max(1, self.sampler.vertex_max()))
            return max(0.0, min(1.0, float(v) / denom))
        if key == "LINE":
            denom = float(max(1, self.sampler.line_max()))
            return max(0.0, min(1.0, float(v) / denom))
        return None

    # ---- public helpers ----
    def show_message(
        self, text: str, level: Literal["info", "warn", "error"] = "info", timeout_sec: float = 3
    ) -> None:
        expire = time.monotonic() + max(0.1, float(timeout_sec))
        self._messages.append((text, expire, level))

    def set_progress(self, key: str, done: int, total: int) -> None:
        self._progress[key] = (int(done), int(total))

    def clear_progress(self, key: str) -> None:
        self._progress.pop(key, None)

    # --- color setters (runtime) ---
    def set_text_color(self, rgba01: tuple[float, float, float, float]) -> None:
        """HUD テキスト色（0–1 RGBA）を 0–255 に変換して適用する。"""
        try:
            r, g, b, a = rgba01
            self._color = (
                int(round(float(r) * 255)),
                int(round(float(g) * 255)),
                int(round(float(b) * 255)),
                int(round(float(a) * 255)),
            )
        except Exception:
            return

    def set_meter_color(
        self, rgb01: tuple[float, float, float] | tuple[float, float, float, float]
    ) -> None:
        """HUD メータ前景色（0–1 RGB/A）。Alpha は無視。"""
        try:
            r, g, b = float(rgb01[0]), float(rgb01[1]), float(rgb01[2])
            self._meter_color_fg = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
        except Exception:
            return

    def set_meter_bg_color(
        self, rgb01: tuple[float, float, float] | tuple[float, float, float, float]
    ) -> None:
        """HUD メータ背景色（0–1 RGB/A）。Alpha は無視。"""
        try:
            r, g, b = float(rgb01[0]), float(rgb01[1]), float(rgb01[2])
            self._meter_color_bg = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
        except Exception:
            return
