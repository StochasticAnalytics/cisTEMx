"""
cistemx.geometry - Geometric calculations for cryo-EM.

Modules:
    rotations: Euler angle conversions and SO(3) geodesic distances
"""

from cistemx.geometry.rotations import (
    euler_to_rotation_matrix,
    orientation_difference,
)

__all__ = [
    'euler_to_rotation_matrix',
    'orientation_difference',
]
