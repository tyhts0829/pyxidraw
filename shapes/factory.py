from __future__ import annotations

from typing import Any, Dict, Type

from .base import BaseShape
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


class ShapeFactory:
    """Factory class for creating shape instances."""
    
    _registry: Dict[str, Type[BaseShape]] = {
        "polygon": Polygon,
        "sphere": Sphere,
        "grid": Grid,
        "polyhedron": Polyhedron,
        "lissajous": Lissajous,
        "torus": Torus,
        "cylinder": Cylinder,
        "cone": Cone,
        "capsule": Capsule,
        "attractor": Attractor,
        "text": Text,
        "asemic_glyph": AsemicGlyph,
    }
    
    @classmethod
    def create(cls, shape_type: str) -> BaseShape:
        """Create a shape instance by type name.
        
        Args:
            shape_type: Name of the shape type (e.g., "polygon", "sphere")
            
        Returns:
            Instance of the requested shape class
            
        Raises:
            ValueError: If shape_type is not registered
        """
        if shape_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown shape type: {shape_type}. Available types: {available}")
        
        shape_class = cls._registry[shape_type]
        return shape_class()
    
    @classmethod
    def register(cls, name: str, shape_class: Type[BaseShape]):
        """Register a new shape type.
        
        Args:
            name: Name to register the shape as
            shape_class: The shape class to register
        """
        cls._registry[name] = shape_class
    
    @classmethod
    def list_shapes(cls) -> list[str]:
        """Get list of available shape types."""
        return list(cls._registry.keys())