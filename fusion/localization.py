"""Multi-node localisation utilities.

Given N sensor positions and N range measurements, recover the target XYZ
via linear least-squares trilateration.  This is what turns range-only
modalities (acoustic SNR, RF RSSI) from "sphere around the sensor" into
an actual position fix when measurements from multiple spatially-
separated nodes are combined.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class TrilaterationResult:
    x: float
    y: float
    z: float
    uncertainty_m: float       # 1-sigma horizontal uncertainty (metres)
    z_uncertainty_m: float     # vertical uncertainty (large if nodes coplanar)
    residual_m: float          # mean |measured - reconstructed| range error


def _coplanar(positions: Sequence[Tuple[float, float, float]], tol_m: float = 3.0) -> bool:
    zs = [p[2] for p in positions]
    return (max(zs) - min(zs)) < tol_m


def trilaterate(
    positions: Sequence[Tuple[float, float, float]],
    ranges: Sequence[float],
    assumed_altitude_m: Optional[float] = None,
) -> Optional[TrilaterationResult]:
    """Linear-LSQ trilateration.

    For each i>0, subtract sphere equation 0 from i to linearise:
        2(xi-x0)X + 2(yi-y0)Y + 2(zi-z0)Z
            = r0^2 - ri^2 + (xi^2+yi^2+zi^2) - (x0^2+y0^2+z0^2)

    With coplanar nodes (typical small array), Z drops out of the
    difference equations entirely — the linear system solves cleanly
    for (X, Y) and Z must be supplied separately.  With nodes spread
    in altitude (>3 m) we recover all three coordinates.

    Returns None if not enough nodes or the system is singular.
    """
    if len(positions) != len(ranges) or len(positions) < 3:
        return None

    P = np.asarray(positions, dtype=float)
    R = np.asarray(ranges, dtype=float)
    n = len(positions)

    x0, y0, z0 = P[0]
    r0 = R[0]

    A_full = np.zeros((n - 1, 3))
    b = np.zeros(n - 1)
    for i in range(1, n):
        xi, yi, zi = P[i]
        ri = R[i]
        A_full[i - 1] = [2 * (xi - x0), 2 * (yi - y0), 2 * (zi - z0)]
        b[i - 1] = (
            r0 ** 2 - ri ** 2
            + (xi ** 2 + yi ** 2 + zi ** 2)
            - (x0 ** 2 + y0 ** 2 + z0 ** 2)
        )

    coplanar = _coplanar(positions)
    full_3d = (not coplanar) and (n >= 4)

    if full_3d:
        sol, *_ = np.linalg.lstsq(A_full, b, rcond=None)
        x, y, z = float(sol[0]), float(sol[1]), float(sol[2])
        z_unc = 0.0  # will be set from residual below
    else:
        # 2-D solve: hold z at supplied altitude (or array mean) and recover X, Y.
        z_assumed = (
            assumed_altitude_m
            if assumed_altitude_m is not None
            else float(np.mean(P[:, 2]))
        )
        b2 = b - A_full[:, 2] * z_assumed
        A2 = A_full[:, :2]
        sol, *_ = np.linalg.lstsq(A2, b2, rcond=None)
        if sol.size < 2:
            return None
        x, y, z = float(sol[0]), float(sol[1]), z_assumed
        z_unc = 1e3  # marker: vertical is unobservable from coplanar array

    target = np.array([x, y, z])
    reconstructed = np.linalg.norm(P - target, axis=1)
    residual = float(np.mean(np.abs(reconstructed - R)))

    # Sanity check: if the LSQ pushed the source far beyond the largest
    # measured range, one of the inputs was almost certainly an outlier
    # and the solution is meaningless — bail out.
    max_range = float(np.max(R))
    if math.hypot(x, y) > 3.0 * max_range:
        return None

    horiz_unc = max(5.0, residual * 1.5)
    if full_3d:
        z_unc = max(5.0, residual * 2.0)

    return TrilaterationResult(
        x=x,
        y=y,
        z=z,
        uncertainty_m=horiz_unc,
        z_uncertainty_m=z_unc,
        residual_m=residual,
    )


def bearing_and_range_from_origin(x: float, y: float) -> Tuple[float, float]:
    """Convert XY position to (bearing degrees from North, horizontal range)."""
    bearing = math.degrees(math.atan2(x, y)) % 360.0
    rng = math.hypot(x, y)
    return bearing, rng
