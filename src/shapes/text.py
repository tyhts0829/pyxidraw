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
    """Calculate initial offset based on alignment mode.

    Args:
        total_width: Total width of text
        align_mode: 0=left, 1=center, 2=right
    """
    if align_mode == 1:  # center
        return -total_width / 2
    elif align_mode == 2:  # right
        return -total_width
    return 0.0  # left alignment


@njit(fastmath=True, cache=True)
def _normalize_vertices_fast(vertices_array: np.ndarray, units_per_em: float) -> np.ndarray:
    """Normalize vertices to unit coords using njit."""
    # Create output array
    normalized = vertices_array.copy()

    # Normalize to unit size
    normalized[:, 0] = normalized[:, 0] / units_per_em
    normalized[:, 1] = normalized[:, 1] / units_per_em

    # Center vertically and flip Y axis (fonts have baseline at y=0)
    # Flip Y axis because font coords are bottom-to-top
    normalized[:, 1] = -normalized[:, 1] + 0.5

    return normalized


@njit(fastmath=True, cache=True)
def _apply_text_transforms_fast(vertices: np.ndarray, x_offset: float, size: float) -> np.ndarray:
    """Apply horizontal offset and scaling transformations."""
    transformed = vertices.copy()

    # Apply horizontal offset
    transformed[:, 0] += x_offset

    # Apply size scaling
    transformed[:, 0] *= size
    transformed[:, 1] *= size
    transformed[:, 2] *= size

    return transformed


@njit(fastmath=True, cache=True)
def _process_vertices_batch_fast(
    vertices_batch: np.ndarray, x_offsets: np.ndarray, size: float
) -> np.ndarray:
    """Process multiple character vertices in batch."""
    batch_size = vertices_batch.shape[0]
    max_vertices = vertices_batch.shape[1]

    # Create output array
    output = np.empty_like(vertices_batch)

    for i in range(batch_size):
        for j in range(max_vertices):
            # Apply offset and scaling
            output[i, j, 0] = (vertices_batch[i, j, 0] + x_offsets[i]) * size
            output[i, j, 1] = vertices_batch[i, j, 1] * size
            output[i, j, 2] = vertices_batch[i, j, 2] * size

    return output


@njit(fastmath=True, cache=True)
def _convert_glyph_commands_to_vertices_fast(
    move_points: np.ndarray, line_points: np.ndarray, close_flags: np.ndarray, units_per_em: float
) -> np.ndarray:
    """Convert glyph command points to normalized vertices using njit."""
    num_points = move_points.shape[0] + line_points.shape[0]

    if num_points == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Create vertices array
    vertices = np.empty((num_points, 3), dtype=np.float32)

    # Add move points
    move_count = move_points.shape[0]
    for i in range(move_count):
        vertices[i, 0] = move_points[i, 0]
        vertices[i, 1] = move_points[i, 1]
        vertices[i, 2] = 0.0

    # Add line points
    line_count = line_points.shape[0]
    for i in range(line_count):
        vertices[move_count + i, 0] = line_points[i, 0]
        vertices[move_count + i, 1] = line_points[i, 1]
        vertices[move_count + i, 2] = 0.0

    # Normalize vertices
    normalized = _normalize_vertices_fast(vertices, units_per_em)

    return normalized


class TextRenderer:
    """Singleton class for font and text rendering management."""

    _instance = None
    _fonts = {}  # Font cache
    _glyph_cache = {}  # Glyph commands cache
    _font_paths = None  # Font paths cache
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
        """Get list of available font paths."""
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
        """Get cached font instance.

        Args:
            font_name: Font name or path
            font_number: Font number for TTC files

        Returns:
            TTFont instance
        """
        cache_key = f"{font_name}_{font_number}"

        if cache_key not in cls._fonts:
            # Try to find font by name
            font_paths = cls.get_font_path_list()
            for font_path in font_paths:
                if font_name.lower() in font_path.name.lower():
                    if font_path.suffix == ".ttc":
                        cls._fonts[cache_key] = TTFont(font_path, fontNumber=font_number)
                    else:
                        cls._fonts[cache_key] = TTFont(font_path)
                    return cls._fonts[cache_key]

            # If not found, try as direct path
            font_path = Path(font_name)
            if font_path.exists():
                if font_path.suffix == ".ttc":
                    cls._fonts[cache_key] = TTFont(font_path, fontNumber=font_number)
                else:
                    cls._fonts[cache_key] = TTFont(font_path)
                return cls._fonts[cache_key]

            # Default to system font
            logger.warning("Font '%s' not found, using default font", font_name)
            default_font = Path("/System/Library/Fonts/Helvetica.ttc")
            cls._fonts[cache_key] = TTFont(default_font, fontNumber=0)

        return cls._fonts[cache_key]

    @classmethod
    def get_glyph_commands(cls, char: str, font_name: str, font_number: int) -> tuple:
        """Get flattened glyph drawing commands (cached)."""
        cache_key = f"{font_name}_{font_number}_{char}"

        if cache_key not in cls._glyph_cache:
            tt_font = cls.get_font(font_name, font_number)

            # Get glyph from font
            cmap = tt_font.getBestCmap()
            if cmap is None:
                cls._glyph_cache[cache_key] = tuple()
                return cls._glyph_cache[cache_key]

            glyph_name = cmap.get(ord(char))
            if glyph_name is None:
                # Try fallback for common characters
                if char.isascii() and char.isprintable():
                    # Try with glyph name directly
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

            # Record glyph drawing commands
            recording_pen = RecordingPen()
            glyph.draw(recording_pen)

            # Flatten curves to line segments
            flattened_pen = RecordingPen()
            flatten_pen = FlattenPen(flattened_pen, approximateSegmentLength=5, segmentLines=True)
            recording_pen.replay(flatten_pen)

            cls._glyph_cache[cache_key] = tuple(flattened_pen.value)

        return cls._glyph_cache[cache_key]


# Global instance for performance
TEXT_RENDERER = TextRenderer()


@shape
class Text(BaseShape):
    """Text shape generator using TrueType font rendering."""

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

        # Get font
        tt_font = TEXT_RENDERER.get_font(font, font_number)
        units_per_em = tt_font["head"].unitsPerEm  # type: ignore

        # Calculate total width for alignment
        total_width = 0
        for char in text:
            total_width += self._get_char_advance(char, tt_font)

        # Get initial offset based on alignment
        x_offset = self._get_initial_offset(total_width, align)

        # Collect all character data for batch processing
        char_data = []
        current_x_offset = x_offset

        for char in text:
            if char != " ":  # Skip spaces for rendering but track offset
                char_vertices = self._render_character(char, font, font_number, units_per_em)
                for vertices in char_vertices:
                    if len(vertices) > 0:
                        char_data.append((vertices, current_x_offset))

            # Update offset for next character
            current_x_offset += self._get_char_advance(char, tt_font)

        # Batch process all character vertices
        if char_data:
            vertices_list = self._process_character_batch(char_data, font_size)
        else:
            vertices_list = []

        return Geometry.from_lines(vertices_list)

    def _get_initial_offset(self, total_width: float, align: str) -> float:
        """Calculate initial offset based on alignment."""
        # Convert alignment string to integer for njit function
        align_mode = 0  # left
        if align == "center":
            align_mode = 1
        elif align == "right":
            align_mode = 2

        return _get_initial_offset_fast(total_width, align_mode)

    def _process_character_batch(self, char_data: list, size: float) -> list[np.ndarray]:
        """Process multiple character vertices in batch for better performance."""
        if not char_data:
            return []

        # Try to use batch processing if data structure allows
        if len(char_data) > 5:  # Use batch processing for longer texts
            try:
                return self._batch_process_vertices(char_data, size)
            except:
                # Fallback to individual processing if batch fails
                pass

        # Individual processing (original method)
        vertices_list = []
        for vertices, x_offset in char_data:
            transformed_vertices = _apply_text_transforms_fast(vertices, x_offset, size)
            vertices_list.append(transformed_vertices)

        return vertices_list

    def _batch_process_vertices(self, char_data: list, size: float) -> list[np.ndarray]:
        """Attempt to use njit batch processing for multiple characters."""
        # Find maximum vertex count for padding
        max_vertices = max(len(vertices) for vertices, _ in char_data)

        if max_vertices == 0:
            return []

        # Create padded arrays for batch processing
        batch_size = len(char_data)
        vertices_batch = np.zeros((batch_size, max_vertices, 3), dtype=np.float32)
        x_offsets = np.zeros(batch_size, dtype=np.float32)
        vertex_counts = np.zeros(batch_size, dtype=np.int32)

        # Fill batch arrays
        for i, (vertices, x_offset) in enumerate(char_data):
            vertex_count = len(vertices)
            vertices_batch[i, :vertex_count] = vertices
            x_offsets[i] = x_offset
            vertex_counts[i] = vertex_count

        # Process batch using njit function
        processed_batch = _process_vertices_batch_fast(vertices_batch, x_offsets, size)

        # Extract results back to list format
        vertices_list = []
        for i in range(batch_size):
            vertex_count = vertex_counts[i]
            if vertex_count > 0:
                vertices_list.append(processed_batch[i, :vertex_count].copy())

        return vertices_list

    def _get_char_advance(self, char: str, tt_font: TTFont) -> float:
        """Get horizontal advance width for a character."""
        if char == " ":
            try:
                space_width = tt_font["hmtx"].metrics["space"][0]  # type: ignore
                return space_width / tt_font["head"].unitsPerEm  # type: ignore
            except KeyError:
                # Default space width if not found
                return 0.25

        # Get character from cmap
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
        """Render a single character as line segments."""
        if char == " ":
            return []

        # Get glyph commands (cached)
        glyph_commands = TEXT_RENDERER.get_glyph_commands(char, font_name, font_number)
        if not glyph_commands:
            return []

        # Convert commands to vertices
        return self._glyph_commands_to_vertices(list(glyph_commands), units_per_em)

    def _glyph_commands_to_vertices(
        self, glyph_commands: list, units_per_em: float
    ) -> list[np.ndarray]:
        """Convert glyph commands to vertex arrays."""
        vertices_list = []
        current_path = []

        for command in glyph_commands:
            cmd_type, cmd_values = command

            if cmd_type == "moveTo":
                # Start new path
                if current_path:
                    vertices_list.append(self._normalize_vertices(current_path, units_per_em))
                    current_path = []
                x, y = cmd_values[0]
                current_path.append([x, y, 0])

            elif cmd_type == "lineTo":
                # Add line segment
                x, y = cmd_values[0]
                current_path.append([x, y, 0])

            elif cmd_type == "closePath":
                # Close current path
                if current_path:
                    # Add closing segment if needed
                    if len(current_path) > 1 and current_path[0] != current_path[-1]:
                        current_path.append(current_path[0])
                    vertices_list.append(self._normalize_vertices(current_path, units_per_em))
                    current_path = []

        # Handle any remaining path
        if current_path:
            vertices_list.append(self._normalize_vertices(current_path, units_per_em))

        return vertices_list

    def _normalize_vertices(self, vertices: list, units_per_em: float) -> np.ndarray:
        """Normalize vertices to unit coords."""
        vertices_np = np.array(vertices, dtype=np.float32)

        # Use njit-optimized function for normalization
        return _normalize_vertices_fast(vertices_np, units_per_em)
