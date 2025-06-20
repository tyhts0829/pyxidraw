from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fontPens.flattenPen import FlattenPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont

from .base import BaseShape


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
            print(f"Font '{font_name}' not found, using default font")
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
                    print(f"Character '{char}' (U+{ord(char):04X}) not found in font '{font_name}'.")
                    cls._glyph_cache[cache_key] = tuple()
                    return cls._glyph_cache[cache_key]

            glyph_set = tt_font.getGlyphSet()
            glyph = glyph_set.get(glyph_name)
            if glyph is None:
                print(f"Glyph '{glyph_name}' not found in font '{font_name}'.")
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


class Text(BaseShape):
    """Text shape generator using TrueType font rendering."""

    def generate(
        self,
        text: str = "HELLO",
        size: float = 0.1,
        font: str = "Helvetica",
        font_number: int = 0,
        align: str = "center",
        **params: Any,
    ) -> list[np.ndarray]:
        """Generate text as line segments from font outlines.

        Args:
            text: Text string to render
            size: Text size (relative to canvas)
            font: Font name or path
            font_number: Font number for TTC files
            align: Text alignment ('left', 'center', 'right')
            **params: Additional parameters (ignored)

        Returns:
            List of vertex arrays for text outlines
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

        # Render each character
        for char in text:
            char_vertices = self._render_character(char, font, font_number, units_per_em)

            # Apply horizontal offset and scale
            for vertices in char_vertices:
                if len(vertices) > 0:
                    # Create a copy to avoid modifying the original
                    vertices_copy = vertices.copy()
                    # Apply offset
                    vertices_copy[:, 0] += x_offset
                    # Apply size scaling
                    vertices_copy *= size
                    vertices_list.append(vertices_copy)

            # Update offset for next character
            x_offset += self._get_char_advance(char, tt_font)

        return vertices_list

    def _get_initial_offset(self, total_width: float, align: str) -> float:
        """Calculate initial offset based on alignment."""
        if align == "center":
            return -total_width / 2
        elif align == "right":
            return -total_width
        return 0.0  # left alignment

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

    def _render_character(self, char: str, font_name: str, font_number: int, units_per_em: float) -> list[np.ndarray]:
        """Render a single character as line segments."""
        if char == " ":
            return []

        # Get glyph commands (cached)
        glyph_commands = TEXT_RENDERER.get_glyph_commands(char, font_name, font_number)
        if not glyph_commands:
            return []

        # Convert commands to vertices
        return self._glyph_commands_to_vertices(list(glyph_commands), units_per_em)

    def _glyph_commands_to_vertices(self, glyph_commands: list, units_per_em: float) -> list[np.ndarray]:
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
        """Normalize vertices to unit coordinates."""
        vertices_np = np.array(vertices, dtype=np.float32)

        # Normalize to unit size
        vertices_np[:, :2] = vertices_np[:, :2] / units_per_em

        # Center vertically and flip Y axis (fonts have baseline at y=0)
        # Flip Y axis because font coordinates are bottom-to-top
        vertices_np[:, 1] = -vertices_np[:, 1] + 0.5

        return vertices_np
