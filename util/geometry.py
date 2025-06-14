"""Geometry utility functions for 3D transformations."""

from __future__ import annotations

import numpy as np


def transform_to_xy_plane(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Transform vertices to lie on XY plane (z=0).
    
    Rotates the vertices so that their normal vector aligns with Z-axis,
    then translates z-coordinates to 0.
    
    Args:
        vertices: (N, 3) array of 3D points
        
    Returns:
        tuple of:
            - transformed_points: (N, 3) array on XY plane
            - rotation_matrix: (3, 3) rotation matrix used
            - z_offset: z translation amount
    """
    if len(vertices) < 3:
        return vertices.copy(), np.eye(3), 0.0
    
    # Calculate polygon normal vector
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    
    if norm == 0:
        return vertices.copy(), np.eye(3), 0.0
    
    normal = normal / norm  # normalize
    
    # Calculate rotation axis (cross product with Z-axis)
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(normal, z_axis)
    
    if np.linalg.norm(rotation_axis) == 0:
        # Already aligned with Z-axis
        z_offset = vertices[0, 2]
        result = vertices.copy()
        result[:, 2] -= z_offset
        return result, np.eye(3), z_offset
    
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Calculate rotation angle
    cos_theta = np.dot(normal, z_axis)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    
    # Create rotation matrix using Rodrigues' formula
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    # Apply rotation
    transformed_points = vertices @ R.T
    
    # Get z-coordinate and align to z=0
    z_offset = transformed_points[0, 2]
    transformed_points[:, 2] -= z_offset
    
    return transformed_points, R, z_offset


def transform_back(vertices: np.ndarray, rotation_matrix: np.ndarray, z_offset: float) -> np.ndarray:
    """Transform vertices back to original orientation.
    
    Inverse of transform_to_xy_plane function.
    
    Args:
        vertices: (N, 3) array of transformed points
        rotation_matrix: (3, 3) rotation matrix from transform_to_xy_plane
        z_offset: z translation amount from transform_to_xy_plane
        
    Returns:
        (N, 3) array of points in original orientation
    """
    # Restore z-coordinate
    result = vertices.copy()
    result[:, 2] += z_offset
    
    # Apply inverse rotation
    return result @ rotation_matrix