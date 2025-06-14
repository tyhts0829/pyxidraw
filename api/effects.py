from __future__ import annotations

from typing import Any

import numpy as np

from effects import (
    EffectPipeline,
    Boldify, Connect, Rotation, Scaling, Translation,
    Dashify, Noise, Subdivision, Culling, Wobble,
    Array, Sweep, Extrude, Filling, Trimming,
    Webify, Desolve, Collapse, Transform, Buffer
)


def boldify(vertices_list: list[np.ndarray],
           offset: float = 1.0,
           num_offset: tuple[float, float, float] = (0.5, 0.5, 0.5),
           method: str = "normal",
           **params: Any) -> list[np.ndarray]:
    """Make lines appear bolder by adding parallel lines.
    
    Args:
        vertices_list: Input vertex arrays
        offset: Thickness coefficient (0.0-1.0, multiplied by 0.1 internally)
        num_offset: Density control for adaptive method
        method: Implementation method ("normal" or "adaptive")
        **params: Additional parameters
        
    Returns:
        Boldified vertex arrays
    """
    effect = Boldify()
    return effect(vertices_list, offset=offset, num_offset=num_offset, method=method, **params)


def connect(vertices_list: list[np.ndarray],
           n_points: float = 0.5,
           alpha: float = 0.0,
           cyclic: bool = False,
           **params: Any) -> list[np.ndarray]:
    """Connect multiple lines smoothly using Catmull-Rom splines.
    
    Args:
        vertices_list: Input vertex arrays
        n_points: Number of interpolation points (0.0-1.0, mapped to 0-50)
        alpha: Spline tension parameter (0.0-1.0, mapped to 0-2)
        cyclic: Whether to connect last line to first
        **params: Additional parameters
        
    Returns:
        Connected vertex arrays
    """
    effect = Connect()
    return effect(vertices_list, n_points=n_points, alpha=alpha, cyclic=cyclic, **params)


def rotation(vertices_list: list[np.ndarray],
            angle_x: float = 0.0,
            angle_y: float = 0.0,
            angle_z: float = 0.0,
            **params: Any) -> list[np.ndarray]:
    """Rotate vertices around specified axes.
    
    Args:
        vertices_list: Input vertex arrays
        angle_x: Rotation angle around X axis in radians
        angle_y: Rotation angle around Y axis in radians
        angle_z: Rotation angle around Z axis in radians
        **params: Additional parameters
        
    Returns:
        Rotated vertex arrays
    """
    effect = Rotation()
    return effect(vertices_list, angle_x=angle_x, angle_y=angle_y, angle_z=angle_z, **params)


def scaling(vertices_list: list[np.ndarray],
           scale_x: float = 1.0,
           scale_y: float = 1.0,
           scale_z: float = 1.0,
           uniform_scale: float | None = None,
           **params: Any) -> list[np.ndarray]:
    """Scale vertices along specified axes.
    
    Args:
        vertices_list: Input vertex arrays
        scale_x: Scale factor for X axis
        scale_y: Scale factor for Y axis
        scale_z: Scale factor for Z axis
        uniform_scale: If provided, overrides individual scale factors
        **params: Additional parameters
        
    Returns:
        Scaled vertex arrays
    """
    effect = Scaling()
    return effect(vertices_list, scale_x=scale_x, scale_y=scale_y, scale_z=scale_z, 
                 uniform_scale=uniform_scale, **params)


def translation(vertices_list: list[np.ndarray],
               offset_x: float = 0.0,
               offset_y: float = 0.0,
               offset_z: float = 0.0,
               **params: Any) -> list[np.ndarray]:
    """Translate vertices by specified offset.
    
    Args:
        vertices_list: Input vertex arrays
        offset_x: Translation offset for X axis
        offset_y: Translation offset for Y axis
        offset_z: Translation offset for Z axis
        **params: Additional parameters
        
    Returns:
        Translated vertex arrays
    """
    effect = Translation()
    return effect(vertices_list, offset_x=offset_x, offset_y=offset_y, offset_z=offset_z, **params)


def dashify(vertices_list: list[np.ndarray],
           dash_length: float = 0.1,
           gap_length: float = 0.05,
           **params: Any) -> list[np.ndarray]:
    """Convert continuous lines into dashed lines.
    
    Args:
        vertices_list: Input vertex arrays
        dash_length: Length of each dash
        gap_length: Length of gap between dashes
        **params: Additional parameters
        
    Returns:
        Dashed vertex arrays
    """
    effect = Dashify()
    return effect(vertices_list, dash_length=dash_length, gap_length=gap_length, **params)


def noise(vertices_list: list[np.ndarray],
         amplitude: float = 0.01,
         seed: int | None = None,
         **params: Any) -> list[np.ndarray]:
    """Add random noise to vertices.
    
    Args:
        vertices_list: Input vertex arrays
        amplitude: Maximum displacement amplitude
        seed: Random seed for reproducibility
        **params: Additional parameters
        
    Returns:
        Noisy vertex arrays
    """
    effect = Noise()
    return effect(vertices_list, amplitude=amplitude, seed=seed, **params)


def subdivision(vertices_list: list[np.ndarray],
               subdivisions: int = 1,
               smoothing: float = 0.0,
               **params: Any) -> list[np.ndarray]:
    """Subdivide lines by adding intermediate points.
    
    Args:
        vertices_list: Input vertex arrays
        subdivisions: Number of subdivision iterations
        smoothing: Smoothing factor (0.0 = linear, 1.0 = maximum smoothing)
        **params: Additional parameters
        
    Returns:
        Subdivided vertex arrays
    """
    effect = Subdivision()
    return effect(vertices_list, subdivisions=subdivisions, smoothing=smoothing, **params)


def culling(vertices_list: list[np.ndarray],
           min_x: float = -1.0, max_x: float = 1.0,
           min_y: float = -1.0, max_y: float = 1.0,
           min_z: float = -1.0, max_z: float = 1.0,
           mode: str = "clip",
           **params: Any) -> list[np.ndarray]:
    """Remove vertices outside specified bounds.
    
    Args:
        vertices_list: Input vertex arrays
        min_x, max_x: X-axis bounds
        min_y, max_y: Y-axis bounds
        min_z, max_z: Z-axis bounds
        mode: "clip" to clip lines at bounds, "remove" to remove entire lines
        **params: Additional parameters
        
    Returns:
        Culled vertex arrays
    """
    effect = Culling()
    return effect(vertices_list, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
                 min_z=min_z, max_z=max_z, mode=mode, **params)


def wobble(vertices_list: list[np.ndarray],
          amplitude: float = 0.05,
          frequency: float = 5.0,
          phase: float = 0.0,
          axis: str = "y",
          **params: Any) -> list[np.ndarray]:
    """Add wobble/wave distortion to lines.
    
    Args:
        vertices_list: Input vertex arrays
        amplitude: Wave amplitude
        frequency: Wave frequency
        phase: Phase offset
        axis: Axis to apply wobble ("x", "y", or "z")
        **params: Additional parameters
        
    Returns:
        Wobbled vertex arrays
    """
    effect = Wobble()
    return effect(vertices_list, amplitude=amplitude, frequency=frequency, 
                 phase=phase, axis=axis, **params)


def array(vertices_list: list[np.ndarray],
         count_x: int = 1, count_y: int = 1, count_z: int = 1,
         spacing_x: float = 1.0, spacing_y: float = 1.0, spacing_z: float = 1.0,
         **params: Any) -> list[np.ndarray]:
    """Create array of copies of the input.
    
    Args:
        vertices_list: Input vertex arrays
        count_x, count_y, count_z: Number of copies in each direction
        spacing_x, spacing_y, spacing_z: Spacing between copies
        **params: Additional parameters
        
    Returns:
        Arrayed vertex arrays
    """
    effect = Array()
    return effect(vertices_list, count_x=count_x, count_y=count_y, count_z=count_z,
                 spacing_x=spacing_x, spacing_y=spacing_y, spacing_z=spacing_z, **params)


def sweep(vertices_list: list[np.ndarray],
         path: np.ndarray | None = None,
         profile: np.ndarray | None = None,
         **params: Any) -> list[np.ndarray]:
    """Sweep a profile along a path.
    
    Args:
        vertices_list: Input vertex arrays (used as path if path not provided)
        path: Path to sweep along
        profile: Profile to sweep (if None, uses simple circular profile)
        **params: Additional parameters
        
    Returns:
        Swept vertex arrays
    """
    effect = Sweep()
    return effect(vertices_list, path=path, profile=profile, **params)


def extrude(vertices_list: list[np.ndarray],
           direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
           distance: float = 1.0,
           **params: Any) -> list[np.ndarray]:
    """Extrude 2D shapes into 3D.
    
    Args:
        vertices_list: Input vertex arrays
        direction: Extrusion direction vector
        distance: Extrusion distance
        **params: Additional parameters
        
    Returns:
        Extruded vertex arrays
    """
    effect = Extrude()
    return effect(vertices_list, direction=direction, distance=distance, **params)


def filling(vertices_list: list[np.ndarray],
           pattern: str = "lines",
           density: float = 0.1,
           angle: float = 0.0,
           **params: Any) -> list[np.ndarray]:
    """Fill closed shapes with hatching patterns.
    
    Args:
        vertices_list: Input vertex arrays (should form closed shapes)
        pattern: Fill pattern type ("lines", "cross", "dots")
        density: Fill density (spacing between pattern elements)
        angle: Pattern angle in radians
        **params: Additional parameters
        
    Returns:
        Filled vertex arrays
    """
    effect = Filling()
    return effect(vertices_list, pattern=pattern, density=density, angle=angle, **params)


def trimming(vertices_list: list[np.ndarray],
            start_param: float = 0.0,
            end_param: float = 1.0,
            **params: Any) -> list[np.ndarray]:
    """Trim lines to specified parameter range.
    
    Args:
        vertices_list: Input vertex arrays
        start_param: Start parameter (0.0 = beginning of line)
        end_param: End parameter (1.0 = end of line)
        **params: Additional parameters
        
    Returns:
        Trimmed vertex arrays
    """
    effect = Trimming()
    return effect(vertices_list, start_param=start_param, end_param=end_param, **params)


def webify(vertices_list: list[np.ndarray],
          connection_probability: float = 0.5,
          max_distance: float = 1.0,
          **params: Any) -> list[np.ndarray]:
    """Create web-like connections between vertices.
    
    Args:
        vertices_list: Input vertex arrays
        connection_probability: Probability of connecting nearby vertices
        max_distance: Maximum distance for connections
        **params: Additional parameters
        
    Returns:
        Webified vertex arrays
    """
    effect = Webify()
    return effect(vertices_list, connection_probability=connection_probability, 
                 max_distance=max_distance, **params)


def desolve(vertices_list: list[np.ndarray],
           factor: float = 0.5,
           seed: int | None = None,
           **params: Any) -> list[np.ndarray]:
    """Dissolve/fragment lines randomly.
    
    Args:
        vertices_list: Input vertex arrays
        factor: Dissolution factor (0.0 = no change, 1.0 = maximum dissolution)
        seed: Random seed for reproducibility
        **params: Additional parameters
        
    Returns:
        Dissolved vertex arrays
    """
    effect = Desolve()
    return effect(vertices_list, factor=factor, seed=seed, **params)


def collapse(vertices_list: list[np.ndarray],
            center: tuple[float, float, float] = (0.0, 0.0, 0.0),
            factor: float = 0.0,
            **params: Any) -> list[np.ndarray]:
    """Collapse vertices toward a center point.
    
    Args:
        vertices_list: Input vertex arrays
        center: Point to collapse toward
        factor: Collapse factor (0.0 = no change, 1.0 = complete collapse)
        **params: Additional parameters
        
    Returns:
        Collapsed vertex arrays
    """
    effect = Collapse()
    return effect(vertices_list, center=center, factor=factor, **params)


def transform(vertices_list: list[np.ndarray],
             matrix: np.ndarray | None = None,
             **params: Any) -> list[np.ndarray]:
    """Apply arbitrary transformation matrix.
    
    Args:
        vertices_list: Input vertex arrays
        matrix: 3x3 or 4x4 transformation matrix (if None, uses identity)
        **params: Additional parameters
        
    Returns:
        Transformed vertex arrays
    """
    effect = Transform()
    if matrix is None:
        matrix = np.eye(4)
    return effect(vertices_list, matrix=matrix, **params)


def buffer(vertices_list: list[np.ndarray],
          distance: float = 0.1,
          join_style: str = "round",
          **params: Any) -> list[np.ndarray]:
    """Create buffer/offset around paths.
    
    Args:
        vertices_list: Input vertex arrays
        distance: Buffer distance (positive = outward, negative = inward)
        join_style: Join style for corners ("round", "miter", "bevel")
        **params: Additional parameters
        
    Returns:
        Buffered vertex arrays
    """
    effect = Buffer()
    return effect(vertices_list, distance=distance, join_style=join_style, **params)


# Create a convenient pipeline function
def pipeline(*effects: Any) -> EffectPipeline:
    """Create an effect pipeline.
    
    Args:
        *effects: Effect instances to add to the pipeline
        
    Returns:
        EffectPipeline instance
    """
    return EffectPipeline(effects)