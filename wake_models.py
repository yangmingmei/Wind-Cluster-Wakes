"""Shared physical models for the WRF, WIPAFF and AWAKEN benchmarks."""
from __future__ import annotations


# Top Down
"""Finite-farm top-down matching and recovery, shared by both benchmarks.

The signed profile term is S=-psi_m. Matching ln(B/z0f)+S=A therefore
requires z0f=B*exp(-A+S). No reference-field blending, stable/neutral
minimum, empirical speed multiplier or result clipping is applied here.

The paired Businger-Dyer functions use coefficients 16 (unstable) and 5
(stable). The finite farm-height cap, internal-layer growth and empirical
recovery law are explicit modelling assumptions, not standard MOST laws.
Dataset READMEs document the retained case settings and observation support.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ImprovedTopDownParameters:
    most_height_fraction: float = 0.10
    stability_strength: float = 1.0
    wake_recovery_scale: float = 1.0
    # Retained for compatibility with saved parameter descriptions only.
    min_speed_fraction: float = 0.0
    max_speed_fraction: float = 1.0

    def __post_init__(self):
        if not 0 < self.most_height_fraction <= 1:
            raise ValueError("most_height_fraction must be in (0, 1]")
        if self.stability_strength != 1.0:
            raise ValueError("The audited model uses the unscaled MOST pair (strength=1).")
        if not math.isfinite(self.wake_recovery_scale) or self.wake_recovery_scale <= 0:
            raise ValueError("Recovery scale must be finite and positive.")
        if (self.min_speed_fraction, self.max_speed_fraction) != (0., 1.):
            raise ValueError("Speed bounds are validity checks, not adjustable corrections.")


TOP_DOWN_DEFAULTS = ImprovedTopDownParameters()


def similarity_functions(zeta: float) -> tuple[float, float]:
    """Return (phi_m, psi_m), satisfying phi_m=1-zeta*d(psi_m)/d(zeta)."""
    if not math.isfinite(zeta):
        raise ValueError("z/L must be finite")
    if zeta >= 0:
        return 1. + 5. * zeta, -5. * zeta
    x = (1. - 16. * zeta) ** .25
    psi = 2. * math.log((1. + x) / 2.) + math.log((1. + x*x) / 2.)
    psi -= 2. * math.atan(x) - math.pi / 2.
    return 1. / x, psi


def most_application_height(z, boundary_layer_height, parameters=TOP_DOWN_DEFAULTS):
    if not (math.isfinite(z) and math.isfinite(boundary_layer_height)
            and z > 0 and boundary_layer_height > 0):
        raise ValueError("MOST heights must be finite and positive")
    return min(z, parameters.most_height_fraction * boundary_layer_height)


def _zeta(z, H, L, parameters):
    if math.isnan(L) or L == 0:
        raise ValueError("L=0 or NaN is not neutral; use infinity for neutral conditions")
    return most_application_height(z, H, parameters) / L


def unstable_psi_m(zeta, parameters=TOP_DOWN_DEFAULTS):
    if zeta > 0:
        raise ValueError("Unstable psi requires zeta <= 0")
    return similarity_functions(zeta)[1]


def stability_profile_term(z, boundary_layer_height, monin_obukhov_length,
                           parameters=TOP_DOWN_DEFAULTS):
    return -similarity_functions(_zeta(z, boundary_layer_height, monin_obukhov_length, parameters))[1]


def mixing_factor(z, boundary_layer_height, monin_obukhov_length,
                  parameters=TOP_DOWN_DEFAULTS):
    return similarity_functions(_zeta(z, boundary_layer_height, monin_obukhov_length, parameters))[0]


def validate_speed(value, free_speed):
    """Reject failed closures instead of replacing them by a reference speed."""
    # Small acceleration can arise in this finite-layer matching closure.
    # Preserve and audit it; clipping to Uinf would conceal that limitation.
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid top-down speed {value} for Uinf={free_speed}")
    return float(value)


def matching_terms(z0, H, L, cft, zh, D, farm_dist, nut, parameters=TOP_DOWN_DEFAULTS):
    """Return the algebraic matching terms for independent residual checks."""
    beta = nut / (1. + nut)
    a = D / (2. * zh)
    z_upper = (zh + nut * (zh + D / 2.)) / (1. + nut)
    z_lower = (zh + nut * (zh - D / 2.)) / (1. + nut)
    s1 = stability_profile_term(z_upper, H, L, parameters)
    s2 = stability_profile_term(z_lower, H, L, parameters)
    q = math.log(zh / z0 * (1. - a) ** beta) + s2
    if q <= 0:
        raise ValueError("Non-positive lower-layer profile denominator")
    A = (cft / (2 * .4**2) + q**-2)**-.5
    B = zh * (1 + a)**beta
    z0_hi = B * math.exp(-A + s1)
    layer = min(zh + D / 2. + .32 * z0_hi * (farm_dist / z0_hi)**.8, H)
    s_hi = stability_profile_term(layer, H, L, parameters)
    hi_num = math.log(layer / z0) + s_hi
    hi_den = math.log(layer / z0_hi) + s_hi
    if min(hi_num, hi_den) <= 0:
        raise ValueError("Non-positive upper-layer matching term")
    lo_num = math.log(zh / z0_hi * (1 + a)**beta) + s1
    return dict(beta=beta, q=q, A=A, B=B, s1=s1, z0_hi=z0_hi,
                layer_height=layer, friction_ratio=(hi_num / hi_den) * (lo_num / q))


def top_down_model(u_inf, z0, H, L, cft, zh, D, farmDist, wakeDist,
                   parameters=TOP_DOWN_DEFAULTS):
    values = (u_inf, z0, H, cft, zh, D, farmDist, wakeDist)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("All physical inputs other than neutral L must be finite")
    if not (u_inf >= 0 and cft >= 0 and 0 < z0 < zh - D / 2.
            and D > 0 and H >= zh + D / 2. and farmDist >= 0 and wakeDist >= 0):
        raise ValueError("Invalid geometry, drag, wind speed or distance")
    S = stability_profile_term(zh, H, L, parameters)
    profile_denom = math.log(zh / z0) + S
    if profile_denom <= 0:
        raise ValueError("Non-positive ambient wind-profile denominator")
    if u_inf == 0 or cft == 0 or farmDist == 0:
        return float(u_inf)
    u_star = u_inf * .4 / profile_denom
    gamma = parameters.wake_recovery_scale * mixing_factor(zh, H, L, parameters) * D**2 * u_inf / (.4 * u_star * zh)
    # Match the farm exit once. Recomputing farm drag with a decaying nut in
    # the turbine-free wake can deepen its deficit; it is not a recovery law.
    # Downstream, transport only the exit deficit through the retained E(s).
    nut = 28. * math.sqrt(.5 * cft)
    m = matching_terms(z0, H, L, cft, zh, D, farmDist, nut, parameters)
    hub_speed = u_star / .4 * m['friction_ratio'] * m['q']
    if wakeDist > 0:
        t = wakeDist / gamma
        recovery = math.exp(-t**max(1 - .6 * t, .4))
        hub_speed = u_inf + (hub_speed - u_inf) * recovery
    return validate_speed(hub_speed, u_inf)


__all__ = ['TOP_DOWN_DEFAULTS', 'ImprovedTopDownParameters', 'top_down_model',
           'matching_terms', 'similarity_functions', 'stability_profile_term',
           'mixing_factor', 'most_application_height', 'unstable_psi_m', 'validate_speed']


# Sequential
"""Sequential finite-farm propagation shared by WRF and WIPAFF."""
from functools import lru_cache

def remaining_deficit_fraction(end_ratio, wake_ratio):
    if abs(1 - end_ratio) < 1e-12:
        if abs(1 - wake_ratio) > 1e-10:
            raise ValueError("Wake generated without an exit deficit")
        return 0.0
    r = (1 - wake_ratio) / (1 - end_ratio)
    if not np.isfinite(r) or r < -1e-10 or r > 1 + 1e-10:
        raise ValueError(f"Invalid remaining deficit fraction {r}")
    return float(r)

def find_farm_segments(cft_row: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive index bounds for positive-Cft runs in one row."""

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.asarray(cft_row, dtype=float)):
        if value > 0.0 and start is None:
            start = index
        elif value <= 0.0 and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(cft_row) - 1))
    return segments

def limited_top_down_speed(
    wind_speed: float,
    roughness: float,
    boundary_layer_height: float,
    monin_obukhov_length: float,
    cft: float,
    hub_height: float,
    rotor_diameter: float,
    farm_dist: float,
    wake_dist: float,
    parameters: ImprovedTopDownParameters,
) -> float:
    """Evaluate the shared core without a stable/neutral override."""

    speed = top_down_model(
        wind_speed,
        roughness,
        boundary_layer_height,
        monin_obukhov_length,
        cft,
        hub_height,
        rotor_diameter,
        farm_dist,
        wake_dist,
        parameters,
    )
    return float(speed)

@lru_cache(maxsize=200000)
def _segment_speed_ratio(
    free_speed: float,
    roughness: float,
    boundary_layer_height: float,
    monin_obukhov_length: float,
    cft: float,
    hub_height: float,
    rotor_diameter: float,
    farm_dist: float,
    wake_dist: float,
    parameters: ImprovedTopDownParameters,
) -> float:
    speed = limited_top_down_speed(
        free_speed,
        roughness,
        boundary_layer_height,
        monin_obukhov_length,
        cft,
        hub_height,
        rotor_diameter,
        farm_dist,
        wake_dist,
        parameters,
    )
    return float(speed / free_speed)

def sequential_row_speed(
    cft_row: np.ndarray,
    free_speed: float,
    dx: float,
    roughness: float,
    boundary_layer_height: float,
    monin_obukhov_length: float,
    hub_height: float,
    rotor_diameter: float,
    parameters: ImprovedTopDownParameters,
) -> np.ndarray:
    """Compose farm segments by propagating the recovered speed downstream."""
    if free_speed <= 0 or dx <= 0 or not np.all(np.isfinite(cft_row)) or np.any(np.asarray(cft_row) < 0):
        raise ValueError("Invalid sequential-row inputs")

    cft_row = np.asarray(cft_row, dtype=float)
    output = np.full(cft_row.shape, free_speed, dtype=float)
    segments = find_farm_segments(cft_row)
    if not segments:
        return output

    previous_end = -1
    previous_end_speed = free_speed
    previous_farm_dist = 0.0
    previous_cft = 0.0

    for segment_index, (start, end) in enumerate(segments):
        entry_speed = free_speed
        if segment_index > 0:
            for index in range(previous_end + 1, start + 1):
                wake_dist = (index - previous_end) * dx
                end_ratio = _segment_speed_ratio(
                    free_speed,
                    roughness,
                    boundary_layer_height,
                    monin_obukhov_length,
                    previous_cft,
                    hub_height,
                    rotor_diameter,
                    previous_farm_dist,
                    0.0,
                    parameters,
                )
                wake_ratio = _segment_speed_ratio(
                    free_speed,
                    roughness,
                    boundary_layer_height,
                    monin_obukhov_length,
                    previous_cft,
                    hub_height,
                    rotor_diameter,
                    previous_farm_dist,
                    wake_dist,
                    parameters,
                )
                denominator = max(1.0 - end_ratio, 1.0e-12)
                recovery = remaining_deficit_fraction(end_ratio, wake_ratio)
                recovered = free_speed + (previous_end_speed - free_speed) * recovery
                if index < start:
                    output[index] = recovered
                else:
                    entry_speed = recovered

        segment_cft = float(np.mean(cft_row[start : end + 1]))
        for index in range(start, end + 1):
            farm_dist = (index - start + 1) * dx
            ratio = _segment_speed_ratio(
                free_speed,
                roughness,
                boundary_layer_height,
                monin_obukhov_length,
                segment_cft,
                hub_height,
                rotor_diameter,
                farm_dist,
                0.0,
                parameters,
            )
            output[index] = entry_speed * ratio

        previous_end = end
        previous_end_speed = output[end]
        previous_farm_dist = (end - start + 1) * dx
        previous_cft = segment_cft

    for index in range(previous_end + 1, len(output)):
        wake_dist = (index - previous_end) * dx
        end_ratio = _segment_speed_ratio(
            free_speed,
            roughness,
            boundary_layer_height,
            monin_obukhov_length,
            previous_cft,
            hub_height,
            rotor_diameter,
            previous_farm_dist,
            0.0,
            parameters,
        )
        wake_ratio = _segment_speed_ratio(
            free_speed,
            roughness,
            boundary_layer_height,
            monin_obukhov_length,
            previous_cft,
            hub_height,
            rotor_diameter,
            previous_farm_dist,
            wake_dist,
            parameters,
        )
        denominator = max(1.0 - end_ratio, 1.0e-12)
        recovery = remaining_deficit_fraction(end_ratio, wake_ratio)
        output[index] = free_speed + (previous_end_speed - free_speed) * recovery

    return output


# Canopy
"""Conservative projection of turbine thrust onto a finite canopy grid.

The support retains the existing nearest-turbine footprint (radius twice the
equal-area cell radius). Its integral, rather than a fitted multiplier,
sets thrust density: sum(Cft * dx * dy) = sum(Ct * rotor_area) for each farm.
"""
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components


def farm_groups(tx, ty):
    points = np.c_[tx, ty]
    tree = cKDTree(points)
    spacing = np.median(tree.query(points, k=2)[0][:, 1])
    graph = tree.sparse_distance_matrix(tree, 3 * spacing, output_type='coo_matrix')
    return connected_components(graph, directed=False)[1]


def conservative_canopy(x, y, tx, ty, diameters, ct, groups=None, support_radius_factor=2/np.sqrt(np.pi)):
    x, y, tx, ty = [np.asarray(a, dtype=float) for a in (x, y, tx, ty)]
    dx, dy = float(x[1]-x[0]), float(y[1]-y[0])
    if not (np.allclose(np.diff(x), dx) and np.allclose(np.diff(y), dy) and min(dx, dy) > 0):
        raise ValueError('The canopy needs uniform increasing axes')
    groups = farm_groups(tx, ty) if groups is None else np.asarray(groups)
    diameters, ct = [np.broadcast_to(np.asarray(a, float), tx.shape) for a in (diameters, ct)]
    spacing = np.empty(len(tx))
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        if len(idx) < 2:
            raise ValueError('Farm spacing requires at least two turbines')
        points = np.c_[tx[idx], ty[idx]]
        spacing[idx] = np.median(cKDTree(points).query(points, k=2)[0][:, 1])
    if not np.isfinite(support_radius_factor) or support_radius_factor <= 0:
        raise ValueError('Support radius factor must be finite and positive')
    radius = support_radius_factor * spacing
    xx, yy = np.meshgrid(x, y, indexing='ij')
    distances, nearest = cKDTree(np.c_[tx, ty]).query(np.c_[xx.ravel(), yy.ravel()])
    support = distances <= radius[nearest]
    thrust_area = ct * np.pi * diameters**2 / 4
    density = np.where(support, thrust_area[nearest] / spacing[nearest]**2, 0.)
    diagnostics = []
    for group in np.unique(groups):
        mask = support & (groups[nearest] == group)
        target = float(thrust_area[groups == group].sum())
        before = float(density[mask].sum() * dx * dy)
        if target > 0 and before <= 0:
            raise ValueError('Farm has no supported canopy cells')
        factor = target / before if before else 1.
        density[mask] *= factor
        diagnostics.append(dict(group=str(group), target_thrust_area_m2=target,
                                original_thrust_area_m2=before, conservation_factor=factor))
    return density.reshape(xx.shape), diagnostics


# Array Stability
"""Array-Stability Model (ASM) equations from Canadillas et al. (2023).

The implementation follows Appendix B, Equations (A3)-(A9).  Turbine thrust
is distributed with the reported streamwise/lateral skew-Gaussian kernel,
the farm-layer ratio is evaluated from Equation (A7), and every grid point is
treated as a downstream-recovery source in Equation (A3).  The minimum of all
source contributions forms the ASM farm-scale speed field.
"""



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


ARRAY_DEFAULTS = ArrayStabilityParameters()


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
    parameters: ArrayStabilityParameters = ARRAY_DEFAULTS,
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
    parameters: ArrayStabilityParameters = ARRAY_DEFAULTS,
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
    parameters: ArrayStabilityParameters = ARRAY_DEFAULTS,
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
    parameters: ArrayStabilityParameters = ARRAY_DEFAULTS,
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
    parameters: ArrayStabilityParameters = ARRAY_DEFAULTS,
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
    parameters: ArrayStabilityParameters = ARRAY_DEFAULTS,
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


# Pywake Models
"""One definition of the two published PyWake baseline configurations."""
GAUSSIAN_WAKE_EXPANSION = 0.0324555
AMBIENT_TI = 0.09


def validate_pywake_speed(values):
    """Preserve finite baseline predictions, explicitly flagging negative speeds.

    A superposed engineering field can be negative close to turbines. This is
    a model failure, not a reason to clip its output or omit those score cells.
    Non-finite output still prevents scoring and raises an error.
    """
    import warnings
    import numpy as np
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError('Non-finite PyWake output; no silent replacement')
    if np.any(values < -1e-10):
        warnings.warn(f'Unmodified PyWake output has {np.count_nonzero(values < -1e-10)}/{values.size} '
                      f'negative values below -1e-10; minimum={values.min():.6g}. Retain and audit model failure.',
                      RuntimeWarning, stacklevel=2)
    return values


def build_pywake_model(model_key, site, turbines, gaussian_k=GAUSSIAN_WAKE_EXPANSION):
    if model_key == 'gaussian':
        from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
        return Bastankhah_PorteAgel_2014(site, turbines, k=float(gaussian_k))
    if model_key == 'turbopark':
        from py_wake.literature import Nygaard_2022
        return Nygaard_2022(site, turbines)
    raise ValueError(f'Unknown PyWake model {model_key}')


# Site Parameters
"""Declared site configurations. AWAKEN values are internal calibration results.

These effective closure parameters are not concurrent ABL measurements.
The WRF and WIPAFF adapters retain their existing site inputs and r=1.
"""
AWAKEN_ONSHORE = dict(H=350., z0=.01, recovery_scale=9.)


# Awaken
"""AWAKEN prediction adapter; fitting is separate in the calibration driver.

Defaults reproduce the original transfer. Pass **AWAKEN_ONSHORE from
site_parameters for the documented terrestrial configuration. Radar fields
are not inputs to hub_field.
"""
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

DIAMETERS = np.array([127., 103., 116., 100.])
TYPE_IDS = {'GE2.82-127': 0, 'GE1.7-103': 1, 'GE1.715-103': 1,
            'GE2.3-116': 2, 'GE1.79-100': 3}


def inventory_types(frame, unknown_type=2):
    """Explicit curve mapping, independent of inventory row order.

    Garfield has no supplied type: use a declared 2.3 MW proxy for all 23
    turbines, with an alternate 1.79 MW proxy evaluated as a sensitivity.
    """
    unknown = frame.t_model.isna().to_numpy()
    unsupported = set(frame.loc[~unknown, 't_model']) - set(TYPE_IDS)
    if unsupported:
        raise ValueError(f'Unknown turbine labels: {unsupported}')
    types = frame.t_model.map(TYPE_IDS).fillna(unknown_type).to_numpy(int)
    return types, unknown


def project(x, y, wind_from):
    angle = np.deg2rad(wind_from)
    return -x*np.sin(angle)-y*np.cos(angle), x*np.cos(angle)-y*np.sin(angle)


def hub_field(x, y, tx, ty, types, groups, speed, wind_from, L, curves,
              H=1000., z0=.03, step=250., recovery_scale=1., most_height_fraction=.1):
    """Conserve thrust separately for named plants, then propagate segments."""
    s, n = project(np.asarray(x), np.asarray(y), wind_from)
    ts, tn = project(np.asarray(tx), np.asarray(ty), wind_from)
    # Padding includes the entire canopy support; fixed grid anchored at zero.
    from scipy.spatial import cKDTree
    spacing = max(np.median(cKDTree(np.c_[ts[groups==g],tn[groups==g]]).query(
        np.c_[ts[groups==g],tn[groups==g]],k=2)[0][:,1]) for g in np.unique(groups))
    padding = 2*spacing/np.sqrt(np.pi)+step
    axes = [np.arange(np.floor(min(a.min(),b.min())/step)*step-padding,
                      np.ceil(max(a.max(),b.max())/step)*step+padding+step, step)
            for a,b in [(s,ts),(n,tn)]]
    ct = np.array([np.interp(speed,c[0],c[1],left=0,right=0) for c in curves])[types]
    canopy, audit = conservative_canopy(*axes,ts,tn,DIAMETERS[types],ct,groups)
    field = np.empty_like(canopy)
    parameters = ImprovedTopDownParameters(wake_recovery_scale=recovery_scale,
                                           most_height_fraction=most_height_fraction)
    for j in range(len(axes[1])):
        field[:,j] = sequential_row_speed(canopy[:,j],speed,step,z0,H,L,85.,110.,parameters)
    q = RegularGridInterpolator(axes,field/speed,bounds_error=True)(np.c_[s.ravel(),n.ravel()]).reshape(s.shape)
    if not np.isfinite(q).all() or np.any(q<0):
        raise ValueError('Invalid top-down field')
    actual = canopy.sum()*step**2
    target = sum(a['target_thrust_area_m2'] for a in audit)
    if not np.isclose(actual,target,rtol=1e-12):
        raise ValueError('Thrust conservation failure')
    return q, dict(thrust_area_target_m2=target,thrust_area_grid_m2=actual,
                   hub_ratio_min=q.min(),hub_ratio_max=q.max(),solver_step_m=step)


def height_projection(q, height, sigma=55.):
    """Declared observation operator, not a vertically resolved wake solution."""
    if sigma <= 0:
        raise ValueError('Positive projection width required')
    return 1-(1-np.asarray(q))*np.exp(-.5*((np.asarray(height)-85.)/sigma)**2)


def common_metrics(observed, predictions, mask, s, n, mean_valid, full_valid, limit=8.):
    """Score all models on identical observed cells and profile supports.

    Fail on missing predictions within the observation mask. Do not shrink
    the evaluation region to conceal a model's non-finite output. Crosswind
    integrals use the OBSERVED edge criterion for every model.
    """
    mask = np.asarray(mask,bool) & np.isfinite(observed)
    if any(np.shape(q)!=np.shape(observed) for q in predictions.values()):
        raise ValueError('Model/radar shape mismatch')
    for name,q in predictions.items():
        if not np.isfinite(q[mask]).all():
            raise ValueError(f'{name}: missing prediction on radar support')
    domain = (s>0)&(s<=limit)
    means, integrals = {}, {}
    for name,q in {'radar':observed, **predictions}.items():
        means[name] = np.array([np.mean((1-q[i])[m]) if m.any() else np.nan for i,m in enumerate(mask)])
        integrals[name] = np.array([np.trapezoid((1-q[i])[m],n[m]) if m.sum()>1 else np.nan for i,m in enumerate(mask)])
    pm = domain & mean_valid & np.isfinite(means['radar'])
    im = domain & full_valid & np.isfinite(integrals['radar'])
    fm = mask & domain[:,None]
    rows = []
    for name,q in predictions.items():
        error = q[fm]-observed[fm]
        rows.append(dict(model=name, field_mae_pp=np.abs(error).mean()*100 if error.size else np.nan,
            field_bias_pp=error.mean()*100 if error.size else np.nan,
            field_rmse_pp=np.sqrt(np.mean(error**2))*100 if error.size else np.nan,
            field_cells=int(fm.sum()),profile_points=int(pm.sum()),integral_points=int(im.sum()),
            profile_mae_pp=np.abs(means[name][pm]-means['radar'][pm]).mean()*100 if pm.any() else np.nan,
            integral_mae_km=np.abs(integrals[name][im]-integrals['radar'][im]).mean() if im.any() else np.nan,
            negative_speed_cells=int(np.sum(q[fm]<-1e-10)),maximum_ratio=float(q[fm].max()) if error.size else np.nan))
    return rows, means, integrals
