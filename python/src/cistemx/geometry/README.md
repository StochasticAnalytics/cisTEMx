# cistemx.geometry

Geometric calculations for cryo-EM orientations.

## Modules

### rotations.py

Euler angle conversions and SO(3) geodesic distances.

```python
from cistemx.geometry import euler_to_rotation_matrix, orientation_difference

# Convert Euler angles to rotation matrix
R = euler_to_rotation_matrix(phi=45.0, theta=30.0, psi=90.0)

# Calculate angular difference between orientations
diff = orientation_difference(
    phi1=0, theta1=0, psi1=0,
    phi2=0, theta2=90, psi2=0
)  # Returns 90.0 degrees
```

## cisTEM Euler Convention

ZYZ passive (extrinsic) Euler angles:
1. PHI: rotation about Z axis
2. THETA: rotation about Y' axis
3. PSI: rotation about Z'' axis

**Note**: cisTEM database stores as (PSI, THETA, PHI) - swap order when loading.

## Known Redundancy

`cistemx.calculate_2dtm_p_value.geometry` has similar functions using scipy.Rotation.
This module uses direct numpy implementation from our analysis scripts.
Consolidation pending test coverage for Kexin's package.
