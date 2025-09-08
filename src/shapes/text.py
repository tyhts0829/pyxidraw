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

from .base import BaseShape
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
class Text(BaseShape):
    """TrueType フォントを用いてテキスト形状を生成する。"""

    def generate(
        self,
        text: str = "HELLO",
        font_size: float = 0.4,
        font: str = "Helvetica",
        font_number: int = 0,
        align: str = "center",
        **params: Any,
    ) -> Geometry:
        """フォントのアウトラインから線分として文字列を生成します。

        引数:
            text: レンダリングする文字列
            font_size: 文字サイズ（キャンバスに対する相対）
            font: フォント名またはパス
            font_number: TTC ファイルのフォント番号
            align: 行揃え（'left' | 'center' | 'right'）
            **params: 追加パラメータ（未使用）

        返り値:
            テキストアウトラインを含む Geometry
        """
        vertices_list = []

        # フォントを取得
        tt_font = TEXT_RENDERER.get_font(font, font_number)
        units_per_em = tt_font["head"].unitsPerEm  # type: ignore

        # 行揃え計算のための全幅を算出
        total_width = 0
        for char in text:
            total_width += self._get_char_advance(char, tt_font)

        # 行揃えに応じて初期オフセットを求める
        x_offset = self._get_initial_offset(total_width, align)

        # バッチ処理のために各文字のデータを集める
        char_data = []
        current_x_offset = x_offset

        for char in text:
            if char != " ":  # スペースは描画しないがオフセットは進める
                char_vertices = self._render_character(char, font, font_number, units_per_em)
                for vertices in char_vertices:
                    if len(vertices) > 0:
                        char_data.append((vertices, current_x_offset))

            # 次の文字用にオフセットを更新
            current_x_offset += self._get_char_advance(char, tt_font)

        # 収集した頂点群を一括処理
        if char_data:
            vertices_list = self._process_character_batch(char_data, font_size)
        else:
            vertices_list = []

        return Geometry.from_lines(vertices_list)

    def _get_initial_offset(self, total_width: float, align: str) -> float:
        """行揃えに基づいて初期オフセットを計算する。"""
        # njit 関数向けに行揃え文字列を整数に変換
        align_mode = 0  # 左寄せ
        if align == "center":
            align_mode = 1
        elif align == "right":
            align_mode = 2

        return _get_initial_offset_fast(total_width, align_mode)

    def _process_character_batch(self, char_data: list, size: float) -> list[np.ndarray]:
        """複数文字の頂点群をバッチ処理し、性能を向上させる。"""
        if not char_data:
            return []

        # データ構造が許す場合はバッチ処理を試みる
        if len(char_data) > 5:  # 長いテキストではバッチ処理を用いる
            try:
                return self._batch_process_vertices(char_data, size)
            except:
                # 失敗した場合は個別処理にフォールバック
                pass

        # 個別処理（元の方法）
        vertices_list = []
        for vertices, x_offset in char_data:
            transformed_vertices = _apply_text_transforms_fast(vertices, x_offset, size)
            vertices_list.append(transformed_vertices)

        return vertices_list

    def _batch_process_vertices(self, char_data: list, size: float) -> list[np.ndarray]:
        """複数文字に対し、njit によるバッチ処理を試みる。"""
        # パディングに用いる最大頂点数を求める
        max_vertices = max(len(vertices) for vertices, _ in char_data)

        if max_vertices == 0:
            return []

        # バッチ処理用のパディング済み配列を作成
        batch_size = len(char_data)
        vertices_batch = np.zeros((batch_size, max_vertices, 3), dtype=np.float32)
        x_offsets = np.zeros(batch_size, dtype=np.float32)
        vertex_counts = np.zeros(batch_size, dtype=np.int32)

        # バッチ配列を埋める
        for i, (vertices, x_offset) in enumerate(char_data):
            vertex_count = len(vertices)
            vertices_batch[i, :vertex_count] = vertices
            x_offsets[i] = x_offset
            vertex_counts[i] = vertex_count

        # njit 関数でバッチ処理
        processed_batch = _process_vertices_batch_fast(vertices_batch, x_offsets, size)

        # 結果をリスト形式へ戻す
        vertices_list = []
        for i in range(batch_size):
            vertex_count = vertex_counts[i]
            if vertex_count > 0:
                vertices_list.append(processed_batch[i, :vertex_count].copy())

        return vertices_list

    def _get_char_advance(self, char: str, tt_font: TTFont) -> float:
        """文字の水平方向アドバンス幅を取得する。"""
        if char == " ":
            try:
                space_width = tt_font["hmtx"].metrics["space"][0]  # type: ignore
                return space_width / tt_font["head"].unitsPerEm  # type: ignore
            except KeyError:
                # 情報が無い場合の既定のスペース幅
                return 0.25

        # cmap から文字に対応するグリフ名を得る
        cmap = tt_font.getBestCmap()
        if cmap is None:
            return 0

        glyph_name = cmap.get(ord(char))
        if glyph_name is None:
            return 0

        try:
            advance_width = tt_font["hmtx"].metrics[glyph_name][0]  # type: ignore
            return advance_width / tt_font["head"].unitsPerEm  # type: ignore
        except KeyError:
            return 0

    def _render_character(
        self, char: str, font_name: str, font_number: int, units_per_em: float
    ) -> list[np.ndarray]:
        """1 文字を線分列としてレンダリングする。"""
        if char == " ":
            return []

        # グリフコマンド（キャッシュ）を取得
        glyph_commands = TEXT_RENDERER.get_glyph_commands(char, font_name, font_number)
        if not glyph_commands:
            return []

        # コマンドを頂点列へ変換
        return self._glyph_commands_to_vertices(list(glyph_commands), units_per_em)

    def _glyph_commands_to_vertices(
        self, glyph_commands: list, units_per_em: float
    ) -> list[np.ndarray]:
        """グリフコマンドを頂点配列へ変換する。"""
        vertices_list = []
        current_path = []

        for command in glyph_commands:
            cmd_type, cmd_values = command

            if cmd_type == "moveTo":
                # 新しいパスを開始
                if current_path:
                    vertices_list.append(self._normalize_vertices(current_path, units_per_em))
                    current_path = []
                x, y = cmd_values[0]
                current_path.append([x, y, 0])

            elif cmd_type == "lineTo":
                # 線分を追加
                x, y = cmd_values[0]
                current_path.append([x, y, 0])

            elif cmd_type == "closePath":
                # 現在のパスを閉じる
                if current_path:
                    # 必要ならば閉路となる終端線分を追加
                    if len(current_path) > 1 and current_path[0] != current_path[-1]:
                        current_path.append(current_path[0])
                    vertices_list.append(self._normalize_vertices(current_path, units_per_em))
                    current_path = []

        # 残っているパスを処理
        if current_path:
            vertices_list.append(self._normalize_vertices(current_path, units_per_em))

        return vertices_list

    def _normalize_vertices(self, vertices: list, units_per_em: float) -> np.ndarray:
        """頂点列をユニット座標へ正規化する。"""
        vertices_np = np.array(vertices, dtype=np.float32)

        # 正規化は njit 最適化した関数を使用
        return _normalize_vertices_fast(vertices_np, units_per_em)
