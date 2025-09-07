"""
Shapes module - Internal shape implementations

This module provides the internal shape implementations used by the API layer.

For public API usage, use G from api.shape_factory instead of importing directly.

Example:
    # Recommended public API
    from api import G
    sphere = G.sphere(subdivisions=0.5)
    
    # Direct import (internal use only)
    from shapes.sphere import Sphere
"""

from .asemic_glyph import AsemicGlyph
from .attractor import Attractor
from .base import BaseShape
from .capsule import Capsule
from .cone import Cone
from .cylinder import Cylinder
from .grid import Grid
from .lissajous import Lissajous

# Import all shape classes to register them
from .polygon import Polygon
from .polyhedron import Polyhedron
from .registry import get_shape, is_shape_registered, list_shapes, shape
from .sphere import Sphere
from .text import Text
from .torus import Torus

__all__ = [
    # Base classes and registry
    "BaseShape",
    "shape",
    "get_shape",
    "list_shapes",
    "is_shape_registered",
    # Shape classes
    "Polygon",
    "Sphere",
    "Grid",
    "Polyhedron",
    "Lissajous",
    "Torus",
    "Cylinder",
    "Cone",
    "Capsule",
    "Attractor",
    "Text",
    "AsemicGlyph",
]
