"""Array-Stability Model (ASM) equations from Canadillas et al. (2023).

The implementation follows Appendix B, Equations (A3)-(A9).  Turbine thrust
is distributed with the reported streamwise/lateral skew-Gaussian kernel,
the farm-layer ratio is evaluated from Equation (A7), and every grid point is
treated as a downstream-recovery source in Equation (A3).  The minimum of all
source contributions forms the ASM farm-scale speed field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class ArrayStabilityParameters:
    """Numerical and physical constants used by the ASM formulation."""

    von_karman: float = 0.40
    delta_z1_diameters: float = 0.05
    sigma_x_diameters: float = 9.0
    sigma_y_diameters: float = 3.0
    streamwise_skewness: float = 2.0
    minimum_speed_mps: float = 0.10
    recovery_iterations: int = 25
    recovery_tolerance: float = 1.0e-10


DEFAULT_PARAMETERS = ArrayStabilityParameters()


def monin_obukhov_functions(height: float, monin_obukhov_length: float) -> tuple[float, float]:
    """Return the standard Businger-Dyer ``phi_m`` and integrated ``psi_m``.

    These functions close the otherwise unspecified MOST functions in
    Equations (A5) and (A6).  Positive L is stable, negative L is unstable,
    and very large or non-finite L is treated as neutral.
    """

    height = max(float(height), 1.0e-9)
    length = float(monin_obukhov_length)
    if not np.isfinite(length) or abs(length) >= 1.0e8:
        return 1.0, 0.0
    if abs(length) <= 1.0e-12:
        raise ValueError("Monin-Obukhov length must be non-zero.")

    zeta = height / length
    if abs(zeta) <= 1.0e-8:
        return 1.0, 0.0
    if zeta > 0.0:
        return float(1.0 + 5.0 * zeta), float(-5.0 * zeta)

    x = float((1.0 - 16.0 * zeta) ** 0.25)
    phi_m = 1.0 / x
    psi_m = (
        2.0 * np.log((1.0 + x) / 2.0)
        + np.log((1.0 + x * x) / 2.0)
        - 2.0 * np.arctan(x)
        + 0.5 * np.pi
    )
    return float(phi_m), float(psi_m)


def friction_velocity(
    freestream_speed: float,
    height: float,
    roughness: float,
    monin_obukhov_length: float,
    parameters: ArrayStabilityParameters = DEFAULT_PARAMETERS,
) -> tuple[float, float, float]:
    """Evaluate Equation (A6) and return ``u_star``, ``phi_m``, ``psi_m``."""

    height = float(height)
    roughness = float(roughness)
    if height <= 0.0:
        raise ValueError("Hub height must be positive.")
    if not 0.0 < roughness < height:
        raise ValueError("Roughness length must be positive and smaller than hub height.")

    phi_m, psi_m = monin_obukhov_functions(height, monin_obukhov_length)
    profile_term = np.log(height / roughness) - psi_m
    if profile_term <= 1.0e-9:
        raise ValueError("MOST wind-profile denominator is not positive.")

    u_star = parameters.von_karman * max(float(freestream_speed), 0.0) / profile_term
    return float(u_star), phi_m, psi_m


def surface_drag_coefficient(
    height: float,
    roughness: float,
    parameters: ArrayStabilityParameters = DEFAULT_PARAMETERS,
) -> float:
    """Evaluate the logarithmic-profile roughness drag in Equation (A9)."""

    if not 0.0 < float(roughness) < float(height):
        raise ValueError("Roughness length must be positive and smaller than hub height.")
    log_term = np.log(float(height) / float(roughness))
    return float(parameters.von_karman**2 / log_term**2)


def skew_gaussian_thrust_field(
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
    turbine_x_m: np.ndarray,
    turbine_y_m: np.ndarray,
    turbine_ct: np.ndarray,
    rotor_diameter: float,
    parameters: ArrayStabilityParameters = DEFAULT_PARAMETERS,
) -> np.ndarray:
    """Distribute turbine ``C_T`` as the dimensionless ``c'_t(x,y)`` in (A8).

    The streamwise skew-normal and lateral normal densities integrate to one.
    Multiplication by ``D**2`` therefore conserves each turbine's integrated
    dimensionless thrust, i.e. ``integral(c'_t dx dy) = C_T D**2``.
    """

    x_axis_m = np.asarray(x_axis_m, dtype=float)
    y_axis_m = np.asarray(y_axis_m, dtype=float)
    turbine_x_m = np.asarray(turbine_x_m, dtype=float)
    turbine_y_m = np.asarray(turbine_y_m, dtype=float)
    turbine_ct = np.asarray(turbine_ct, dtype=float)
    if not (turbine_x_m.shape == turbine_y_m.shape == turbine_ct.shape):
        raise ValueError("Turbine coordinates and C_T must have matching shapes.")

    diameter = float(rotor_diameter)
    sigma_x = parameters.sigma_x_diameters * diameter
    sigma_y = parameters.sigma_y_diameters * diameter
    if diameter <= 0.0 or sigma_x <= 0.0 or sigma_y <= 0.0:
        raise ValueError("Rotor diameter and Gaussian widths must be positive.")

    normalization = np.sqrt(2.0 * np.pi)
    field = np.zeros((x_axis_m.size, y_axis_m.size), dtype=float)
    for turbine_x, turbine_y, ct in zip(turbine_x_m, turbine_y_m, turbine_ct, strict=True):
        x_standard = (x_axis_m - turbine_x) / sigma_x
        y_standard = (y_axis_m - turbine_y) / sigma_y
        x_density = (
            2.0
            * np.exp(-0.5 * x_standard**2)
            * ndtr(parameters.streamwise_skewness * x_standard)
            / (normalization * sigma_x)
        )
        y_density = np.exp(-0.5 * y_standard**2) / (normalization * sigma_y)
        field += max(float(ct), 0.0) * diameter**2 * x_density[:, None] * y_density[None, :]
    return field


def farm_layer_speed_ratio(
    thrust_density: np.ndarray,
    hub_height: float,
    rotor_diameter: float,
    turbulence_intensity: float,
    phi_m: float,
    drag_coefficient: float,
    parameters: ArrayStabilityParameters = DEFAULT_PARAMETERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``C_t,eff`` in (A8) and the farm-layer ratio in (A7)."""

    intensity = float(turbulence_intensity)
    if intensity < 0.0:
        raise ValueError("Ambient turbulence intensity cannot be negative.")

    delta_z1 = parameters.delta_z1_diameters * float(rotor_diameter)
    if delta_z1 <= 0.0:
        raise ValueError("Delta-z1 must be positive.")

    effective_drag = (np.pi / 8.0) * np.maximum(np.asarray(thrust_density, dtype=float), 0.0)
    effective_drag += float(drag_coefficient)
    ambient_exchange = (float(hub_height) + delta_z1) / delta_z1 * intensity
    stability_factor = float(phi_m) / parameters.von_karman**2
    numerator = ambient_exchange + stability_factor * float(drag_coefficient)
    denominator = ambient_exchange + stability_factor * effective_drag
    ratio = np.divide(numerator, denominator, out=np.ones_like(effective_drag), where=denominator > 0.0)
    return np.clip(ratio, 0.0, 1.0), effective_drag


def downstream_recovery_ratio(
    farm_ratio: np.ndarray,
    x_axis_m: np.ndarray,
    freestream_speed: float,
    exchange_coefficient: float,
    rotor_diameter: float,
    parameters: ArrayStabilityParameters = DEFAULT_PARAMETERS,
) -> np.ndarray:
    """Apply Equation (A3) downstream of every grid point and take its minimum.

    Because the paper defines ``alpha = K / (D**2 U_w)``, each source-target
    pair is solved by fixed-point iteration for its local recovered ``U_w``.
    """

    farm_ratio = np.asarray(farm_ratio, dtype=float)
    x_axis_m = np.asarray(x_axis_m, dtype=float)
    if farm_ratio.ndim != 2 or farm_ratio.shape[0] != x_axis_m.size:
        raise ValueError("farm_ratio must have shape (len(x_axis_m), ny).")
    if np.any(np.diff(x_axis_m) <= 0.0):
        raise ValueError("x_axis_m must be strictly increasing.")

    downstream_distance = x_axis_m[:, None] - x_axis_m[None, :]
    downstream_mask = downstream_distance >= 0.0
    nonnegative_distance = np.maximum(downstream_distance, 0.0)
    diameter_squared = max(float(rotor_diameter) ** 2, 1.0e-12)
    output = np.ones_like(farm_ratio)

    for column in range(farm_ratio.shape[1]):
        source_ratio = np.clip(farm_ratio[:, column], 0.0, 1.0)
        source_ratio_2d = source_ratio[None, :]
        candidates = np.broadcast_to(source_ratio_2d, downstream_distance.shape).copy()
        candidates = np.where(downstream_mask, candidates, 1.0)
        for _ in range(max(int(parameters.recovery_iterations), 1)):
            local_wake_speed = np.maximum(
                float(freestream_speed) * candidates,
                parameters.minimum_speed_mps,
            )
            alpha = float(exchange_coefficient) / (diameter_squared * local_wake_speed)
            updated = 1.0 + (source_ratio_2d - 1.0) * np.exp(
                -nonnegative_distance * alpha
            )
            updated = np.where(downstream_mask, updated, 1.0)
            if float(np.max(np.abs(updated - candidates))) <= parameters.recovery_tolerance:
                candidates = updated
                break
            candidates = updated
        output[:, column] = np.min(candidates, axis=1)

    return np.clip(output, 0.0, 1.0)


def array_stability_ratio_field(
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
    turbine_x_m: np.ndarray,
    turbine_y_m: np.ndarray,
    turbine_ct: np.ndarray,
    freestream_speed: float,
    hub_height: float,
    rotor_diameter: float,
    roughness: float,
    turbulence_intensity: float,
    monin_obukhov_length: float,
    parameters: ArrayStabilityParameters = DEFAULT_PARAMETERS,
) -> dict[str, np.ndarray | float]:
    """Calculate the complete ASM array-scale field for one set of turbine C_T."""

    u_star, phi_m, psi_m = friction_velocity(
        freestream_speed,
        hub_height,
        roughness,
        monin_obukhov_length,
        parameters,
    )
    drag_coefficient = surface_drag_coefficient(hub_height, roughness, parameters)
    exchange_coefficient = parameters.von_karman * u_star * float(hub_height) / phi_m
    thrust_density = skew_gaussian_thrust_field(
        x_axis_m,
        y_axis_m,
        turbine_x_m,
        turbine_y_m,
        turbine_ct,
        rotor_diameter,
        parameters,
    )
    farm_ratio, effective_drag = farm_layer_speed_ratio(
        thrust_density,
        hub_height,
        rotor_diameter,
        turbulence_intensity,
        phi_m,
        drag_coefficient,
        parameters,
    )
    recovered_ratio = downstream_recovery_ratio(
        farm_ratio,
        x_axis_m,
        freestream_speed,
        exchange_coefficient,
        rotor_diameter,
        parameters,
    )
    return {
        "speed_ratio": recovered_ratio,
        "farm_layer_speed_ratio": farm_ratio,
        "thrust_density": thrust_density,
        "effective_drag_coefficient": effective_drag,
        "friction_velocity_mps": u_star,
        "exchange_coefficient_m2ps": exchange_coefficient,
        "surface_drag_coefficient": drag_coefficient,
        "phi_m": phi_m,
        "psi_m": psi_m,
    }
