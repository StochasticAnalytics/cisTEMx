"""
Euler angle conversions and SO(3) geodesic distances for cryo-EM orientations.

This module provides rotation utilities following cisTEM conventions.

cisTEM Euler Angle Convention
=============================
cisTEM uses ZYZ passive (extrinsic) Euler angles, applied in order:
    1. PHI: first rotation about Z axis
    2. THETA: rotation about Y axis
    3. PSI: final rotation about Z axis

For passive (alias) rotations, the matrix transforms coordinates
from the rotated frame back to the original frame.

Argument Order
==============
Functions in this module use (phi, theta, psi) order to match the
mathematical convention of listing rotations in application order.
Note that cisTEM database stores angles as (PSI, THETA, PHI) - callers
must swap the order when loading from database.

Functions
=========
euler_to_rotation_matrix : Convert Euler angles to 3x3 rotation matrix
orientation_difference : Geodesic distance between two orientations on SO(3)
"""

import numpy as np


def euler_to_rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Convert ZYZ passive intrinsic Euler angles to rotation matrix.

    cisTEM convention: ZYZ passive intrinsic
    - PHI: first rotation about Z axis
    - THETA: rotation about Y axis
    - PSI: final rotation about Z axis

    For passive (alias) rotations, the matrix transforms coordinates
    from the rotated frame back to the original frame.

    Args:
        phi, theta, psi: Euler angles in degrees

    Returns:
        3x3 rotation matrix
    """
    # Convert to radians
    phi_r = np.radians(phi)
    theta_r = np.radians(theta)
    psi_r = np.radians(psi)

    # Precompute trig values
    c1, s1 = np.cos(phi_r), np.sin(phi_r)
    c2, s2 = np.cos(theta_r), np.sin(theta_r)
    c3, s3 = np.cos(psi_r), np.sin(psi_r)

    # ZYZ passive intrinsic: R = Rz(phi) @ Ry(theta) @ Rz(psi)
    # Each rotation matrix for passive convention
    R = np.array([
        [c1*c2*c3 - s1*s3,  -c1*c2*s3 - s1*c3,  c1*s2],
        [s1*c2*c3 + c1*s3,  -s1*c2*s3 + c1*c3,  s1*s2],
        [-s2*c3,             s2*s3,              c2]
    ])

    return R


def orientation_difference(phi1: float, theta1: float, psi1: float,
                           phi2: float, theta2: float, psi2: float) -> float:
    """
    Calculate the true angular difference between two orientations.

    Computes the geodesic distance on SO(3) - the minimum rotation angle
    needed to transform one orientation into the other.

    This is more robust than comparing individual Euler angles because:
    - Handles gimbal lock correctly
    - Accounts for equivalent Euler angle representations
    - Gives a single meaningful scalar (rotation angle)

    Args:
        phi1, theta1, psi1: First orientation (degrees)
        phi2, theta2, psi2: Second orientation (degrees)

    Returns:
        Angular difference in degrees [0, 180]
    """
    # Build rotation matrices
    R1 = euler_to_rotation_matrix(phi1, theta1, psi1)
    R2 = euler_to_rotation_matrix(phi2, theta2, psi2)

    # Relative rotation: R_diff = R1^T @ R2
    # This is the rotation that takes orientation 1 to orientation 2
    R_diff = R1.T @ R2

    # Extract rotation angle from trace
    # For a rotation matrix: trace(R) = 1 + 2*cos(angle)
    # Therefore: angle = arccos((trace(R) - 1) / 2)
    trace = np.trace(R_diff)

    # Clamp to [-1, 1] to handle numerical errors
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)

    return np.degrees(angle_rad)
