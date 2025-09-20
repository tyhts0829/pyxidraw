from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from fontPens.flattenPen import FlattenPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont
from numba import njit

from engine.core.geometry import Geometry

from .registry import shape

logger = logging.getLogger(__name__)


@njit(fastmath=True, cache=True)
def _get_initial_offset_fast(total_width: float, align_mode: int) -> float:
    """行揃えモードに基づいて初期オフセットを計算する。

    引数:
        total_width: テキスト全体の幅
        align_mode: 0=左寄せ, 1=中央, 2=右寄せ
    """
    if align_mode == 1:  # 中央
        return -total_width / 2
    elif align_mode == 2:  # 右寄せ
        return -total_width
    return 0.0  # 左寄せ


@njit(fastmath=True, cache=True)
def _normalize_vertices_fast(vertices_array: np.ndarray, units_per_em: float) -> np.ndarray:
    """頂点列をユニット座標へ正規化する（njit）。"""
    # 出力配列を作成
    normalized = vertices_array.copy()

    # ユニットサイズへ正規化
    normalized[:, 0] = normalized[:, 0] / units_per_em
    normalized[:, 1] = normalized[:, 1] / units_per_em

    # 垂直方向の中心化と Y 軸反転（フォントは y=0 がベースライン）
    # フォント座標は下→上なので Y を反転する
    normalized[:, 1] = -normalized[:, 1] + 0.5

    return normalized


@njit(fastmath=True, cache=True)
def _apply_text_transforms_fast(vertices: np.ndarray, x_offset: float, size: float) -> np.ndarray:
    """水平方向のオフセットとスケール変換を適用する。"""
    transformed = vertices.copy()

    # 水平オフセットを適用
    transformed[:, 0] += x_offset

    # サイズスケールを適用
    transformed[:, 0] *= size
    transformed[:, 1] *= size
    transformed[:, 2] *= size

    return transformed


@njit(fastmath=True, cache=True)
def _process_vertices_batch_fast(
    vertices_batch: np.ndarray, x_offsets: np.ndarray, size: float
) -> np.ndarray:
    """複数文字の頂点群をバッチで処理する。"""
    batch_size = vertices_batch.shape[0]
    max_vertices = vertices_batch.shape[1]

    # 出力配列を作成
    output = np.empty_like(vertices_batch)

    for i in range(batch_size):
        for j in range(max_vertices):
            # オフセットとスケールを適用
            output[i, j, 0] = (vertices_batch[i, j, 0] + x_offsets[i]) * size
            output[i, j, 1] = vertices_batch[i, j, 1] * size
            output[i, j, 2] = vertices_batch[i, j, 2] * size

    return output


@njit(fastmath=True, cache=True)
def _convert_glyph_commands_to_vertices_fast(
    move_points: np.ndarray, line_points: np.ndarray, close_flags: np.ndarray, units_per_em: float
) -> np.ndarray:
    """グリフのコマンド点を正規化済みの頂点へ変換する（njit）。"""
    num_points = move_points.shape[0] + line_points.shape[0]

    if num_points == 0:
        return np.empty((0, 3), dtype=np.float32)

    # 頂点配列を作成
    vertices = np.empty((num_points, 3), dtype=np.float32)

    # move 点を追加
    move_count = move_points.shape[0]
    for i in range(move_count):
        vertices[i, 0] = move_points[i, 0]
        vertices[i, 1] = move_points[i, 1]
        vertices[i, 2] = 0.0

    # line 点を追加
    line_count = line_points.shape[0]
    for i in range(line_count):
        vertices[move_count + i, 0] = line_points[i, 0]
        vertices[move_count + i, 1] = line_points[i, 1]
        vertices[move_count + i, 2] = 0.0

    # 頂点を正規化
    normalized = _normalize_vertices_fast(vertices, units_per_em)

    return normalized


class TextRenderer:
    """フォントとテキスト描画を管理するシングルトン。"""

    _instance = None
    _fonts = {}  # フォントのキャッシュ
    _glyph_cache = {}  # グリフコマンドのキャッシュ
    _font_paths = None  # フォントパス一覧のキャッシュ
    FONT_DIRS = [
        Path("/Users/tyhts0829/Library/Fonts"),
        Path("/System/Library/Fonts"),
        Path("/System/Library/Fonts/Supplemental"),
        Path("/Library/Fonts"),
    ]
    EXTENSIONS = [".ttf", ".otf", ".ttc"]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_font_path_list(cls) -> list[Path]:
        """利用可能なフォントファイルのパス一覧を取得する。"""
        if cls._font_paths is None:
            font_paths = []
            for font_dir in cls.FONT_DIRS:
                if font_dir.exists():
                    for ext in cls.EXTENSIONS:
                        font_paths.extend(font_dir.glob(f"*{ext}"))
            cls._font_paths = font_paths
        return cls._font_paths

    @classmethod
    def get_font(cls, font_name: str = "Helvetica", font_number: int = 0) -> TTFont:
        """フォントインスタンス（キャッシュ）を取得する。

        引数:
            font_name: フォント名またはパス
            font_number: TTC ファイルのフォント番号

        返り値:
            TTFont インスタンス
        """
        cache_key = f"{font_name}_{font_number}"

        if cache_key not in cls._fonts:
            # フォント名で検索を試みる
            font_paths = cls.get_font_path_list()
            for font_path in font_paths:
                if font_name.lower() in font_path.name.lower():
                    if font_path.suffix == ".ttc":
                        cls._fonts[cache_key] = TTFont(font_path, fontNumber=font_number)
                    else:
                        cls._fonts[cache_key] = TTFont(font_path)
                    return cls._fonts[cache_key]

            # 見つからなければ、直接パスとして解釈
            font_path = Path(font_name)
            if font_path.exists():
                if font_path.suffix == ".ttc":
                    cls._fonts[cache_key] = TTFont(font_path, fontNumber=font_number)
                else:
                    cls._fonts[cache_key] = TTFont(font_path)
                return cls._fonts[cache_key]

            # 見つからなければシステムフォントを既定として使用
            logger.warning("Font '%s' not found, using default font", font_name)
            default_font = Path("/System/Library/Fonts/Helvetica.ttc")
            cls._fonts[cache_key] = TTFont(default_font, fontNumber=0)

        return cls._fonts[cache_key]

    @classmethod
    def get_glyph_commands(cls, char: str, font_name: str, font_number: int) -> tuple:
        """平坦化済みのグリフ描画コマンドを取得する（キャッシュ）。"""
        cache_key = f"{font_name}_{font_number}_{char}"

        if cache_key not in cls._glyph_cache:
            tt_font = cls.get_font(font_name, font_number)

            # フォントからグリフを取得
            cmap = tt_font.getBestCmap()
            if cmap is None:
                cls._glyph_cache[cache_key] = tuple()
                return cls._glyph_cache[cache_key]

            glyph_name = cmap.get(ord(char))
            if glyph_name is None:
                # よくある文字に対するフォールバックを試す
                if char.isascii() and char.isprintable():
                    # グリフ名をそのまま使ってみる
                    glyph_name = char
                else:
                    logger.warning(
                        "Character '%s' (U+%04X) not found in font '%s'.",
                        char,
                        ord(char),
                        font_name,
                    )
                    cls._glyph_cache[cache_key] = tuple()
                    return cls._glyph_cache[cache_key]

            glyph_set = tt_font.getGlyphSet()
            glyph = glyph_set.get(glyph_name)
            if glyph is None:
                logger.warning("Glyph '%s' not found in font '%s'.", glyph_name, font_name)
                cls._glyph_cache[cache_key] = tuple()
                return cls._glyph_cache[cache_key]

            # グリフ描画コマンドを記録
            recording_pen = RecordingPen()
            glyph.draw(recording_pen)

            # 曲線を線分へ平坦化
            flattened_pen = RecordingPen()
            flatten_pen = FlattenPen(flattened_pen, approximateSegmentLength=5, segmentLines=True)
            recording_pen.replay(flatten_pen)

            cls._glyph_cache[cache_key] = tuple(flattened_pen.value)

        return cls._glyph_cache[cache_key]


# パフォーマンスのためのグローバルインスタンス
TEXT_RENDERER = TextRenderer()


@shape
def text(
    *,
    text: str = "HELLO",
    font_size: float = 0.4,
    font: str = "Helvetica",
    font_number: int = 0,
    align: str = "center",
    **params: Any,
) -> Geometry:
    """フォントのアウトラインから線分として文字列を生成します。"""
    tt_font = TEXT_RENDERER.get_font(font, font_number)
    units_per_em = tt_font["head"].unitsPerEm  # type: ignore
    total_width = 0.0
    for ch in text:
        total_width += _get_char_advance(ch, tt_font)
    align_mode = 1 if align == "center" else 2 if align == "right" else 0
    x_offset = _get_initial_offset_fast(total_width, align_mode)
    char_data: list[tuple[np.ndarray, float]] = []
    cur = x_offset
    for ch in text:
        if ch != " ":
            glyph_cmds = TEXT_RENDERER.get_glyph_commands(ch, font, font_number)
            if glyph_cmds:
                for verts in _glyph_commands_to_vertices(list(glyph_cmds), units_per_em):
                    if len(verts) > 0:
                        char_data.append((verts, cur))
        cur += _get_char_advance(ch, tt_font)
    vertices_list: list[np.ndarray] = []
    for verts, xo in char_data:
        vertices_list.append(_apply_text_transforms_fast(verts, xo, font_size))
    return Geometry.from_lines(vertices_list)


def _get_char_advance(char: str, tt_font: TTFont) -> float:
    if char == " ":
        try:
            space_width = tt_font["hmtx"].metrics["space"][0]  # type: ignore
            return space_width / tt_font["head"].unitsPerEm  # type: ignore
        except KeyError:
            return 0.25
    cmap = tt_font.getBestCmap()
    if cmap is None:
        return 0.0
    glyph_name = cmap.get(ord(char))
    if glyph_name is None:
        return 0.0
    try:
        advance_width = tt_font["hmtx"].metrics[glyph_name][0]  # type: ignore
        return advance_width / tt_font["head"].unitsPerEm  # type: ignore
    except KeyError:
        return 0.0


def _glyph_commands_to_vertices(glyph_commands: list, units_per_em: float) -> list[np.ndarray]:
    vertices_list: list[np.ndarray] = []
    current_path: list[list[float]] = []
    for command in glyph_commands:
        cmd_type, cmd_values = command
        if cmd_type == "moveTo":
            if current_path:
                vertices_list.append(
                    _normalize_vertices_fast(np.array(current_path, dtype=np.float32), units_per_em)
                )
                current_path = []
            x, y = cmd_values[0]
            current_path.append([x, y, 0])
        elif cmd_type == "lineTo":
            x, y = cmd_values[0]
            current_path.append([x, y, 0])
        elif cmd_type == "closePath":
            if current_path:
                if len(current_path) > 1 and current_path[0] != current_path[-1]:
                    current_path.append(current_path[0])
                vertices_list.append(
                    _normalize_vertices_fast(np.array(current_path, dtype=np.float32), units_per_em)
                )
                current_path = []
    if current_path:
        vertices_list.append(
            _normalize_vertices_fast(np.array(current_path, dtype=np.float32), units_per_em)
        )
    return vertices_list


text.__param_meta__ = {
    "font_size": {"type": "number", "min": 0.1, "max": 1.0},
    "font": {"type": "string"},
    "font_number": {"type": "integer", "min": 0, "max": 10},
    "align": {"type": "string", "choices": ["left", "center", "right"]},
}
