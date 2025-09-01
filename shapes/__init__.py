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

from .base import BaseShape
from .registry import shape, get_shape, list_shapes, is_shape_registered

# Import all shape classes to register them
from .polygon import Polygon
from .sphere import Sphere
from .grid import Grid
from .polyhedron import Polyhedron
from .lissajous import Lissajous
from .torus import Torus
from .cylinder import Cylinder
from .cone import Cone
from .capsule import Capsule
from .attractor import Attractor
from .text import Text
from .asemic_glyph import AsemicGlyph

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