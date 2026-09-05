"""Top-down core with bounded MOST height and full-range unstable similarity.

The unstable branch blends the usual Businger-Dyer relation into the
convective-limit form used by COARE/Fairall-style full-instability functions.
The blend begins when ``z/L`` falls below the configured Businger-Dyer limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPS = 1.0e-9
_NEUTRAL_L = 1.0e8
_MIN_PROFILE_DENOM = 0.25
_MAX_STABILITY_TERM = 50.0


@dataclass(frozen=True)
class ImprovedTopDownParameters:
    """Numerical and similarity-function controls for the improved core."""

    most_height_fraction: float = 0.10
    bd_limit_zeta: float = -2.0
    convective_transition_width: float = 1.0
    stability_strength: float = 0.50
    wake_recovery_scale: float = 1.0
    min_speed_fraction: float = 0.0
    max_speed_fraction: float = 1.10

    def __post_init__(self) -> None:
        if not 0.0 < self.most_height_fraction <= 1.0:
            raise ValueError("most_height_fraction must be in (0, 1]")
        if self.bd_limit_zeta >= 0.0:
            raise ValueError("bd_limit_zeta must be negative")
        if self.convective_transition_width <= 0.0:
            raise ValueError("convective_transition_width must be positive")
        if self.stability_strength < 0.0:
            raise ValueError("stability_strength must be non-negative")
        if self.wake_recovery_scale <= 0.0:
            raise ValueError("wake_recovery_scale must be positive")
        if not 0.0 <= self.min_speed_fraction <= self.max_speed_fraction:
            raise ValueError("invalid speed-fraction bounds")


DEFAULT_PARAMETERS = ImprovedTopDownParameters()


def _positive_finite(value: float, fallback: float) -> float:
    value = float(value)
    if np.isfinite(value) and value > _EPS:
        return value
    return float(fallback)


def most_application_height(z: float, boundary_layer_height: float, parameters: ImprovedTopDownParameters) -> float:
    """Return the height at which the surface-layer correction is evaluated."""

    z = _positive_finite(z, _EPS)
    boundary_layer_height = _positive_finite(boundary_layer_height, z)
    return float(min(z, parameters.most_height_fraction * boundary_layer_height))


def _businger_dyer_psi_m(zeta: float) -> float:
    x = (1.0 - 16.0 * zeta) ** 0.25
    value = 2.0 * np.log((1.0 + x) / 2.0)
    value += np.log((1.0 + x * x) / 2.0)
    value -= 2.0 * np.arctan(x) - np.pi / 2.0
    return float(max(value, 0.0))


def _convective_psi_m(zeta: float) -> float:
    """Integrated one-third-power convective-limit momentum function."""

    x = (1.0 - 10.15 * zeta) ** (1.0 / 3.0)
    sqrt3 = np.sqrt(3.0)
    value = 1.5 * np.log((1.0 + x + x * x) / 3.0)
    value -= sqrt3 * np.arctan((1.0 + 2.0 * x) / sqrt3)
    value += 4.0 * np.arctan(1.0) / sqrt3
    return float(max(value, 0.0))


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def unstable_psi_m(zeta: float, parameters: ImprovedTopDownParameters = DEFAULT_PARAMETERS) -> float:
    """Return a continuous BD-to-convective integrated momentum function."""

    zeta = min(float(zeta), 0.0)
    psi_bd = _businger_dyer_psi_m(zeta)
    zeta_sq = zeta * zeta
    psi_full = (psi_bd + zeta_sq * _convective_psi_m(zeta)) / (1.0 + zeta_sq)
    transition = (-zeta + parameters.bd_limit_zeta) / parameters.convective_transition_width
    weight = _smoothstep(transition)
    return float((1.0 - weight) * psi_bd + weight * psi_full)


def _unstable_phi_m(zeta: float, parameters: ImprovedTopDownParameters) -> float:
    phi_bd = (1.0 - 16.0 * zeta) ** -0.25
    phi_convective = (1.0 - 10.15 * zeta) ** (-1.0 / 3.0)
    zeta_sq = zeta * zeta
    phi_full = (phi_bd + zeta_sq * phi_convective) / (1.0 + zeta_sq)
    transition = (-zeta + parameters.bd_limit_zeta) / parameters.convective_transition_width
    weight = _smoothstep(transition)
    return float((1.0 - weight) * phi_bd + weight * phi_full)


def stability_profile_term(
    z: float,
    boundary_layer_height: float,
    monin_obukhov_length: float,
    parameters: ImprovedTopDownParameters = DEFAULT_PARAMETERS,
) -> float:
    """Return the signed MOST wind-profile term using a bounded height."""

    z_most = most_application_height(z, boundary_layer_height, parameters)
    length = float(monin_obukhov_length)
    if not np.isfinite(length) or abs(length) < _EPS or abs(length) >= _NEUTRAL_L:
        return 0.0
    zeta = z_most / length
    if length > 0.0:
        value = 4.7 * zeta
    else:
        value = -unstable_psi_m(zeta, parameters)
    value *= parameters.stability_strength
    return float(np.clip(value, -_MAX_STABILITY_TERM, _MAX_STABILITY_TERM))


def mixing_factor(
    z: float,
    boundary_layer_height: float,
    monin_obukhov_length: float,
    parameters: ImprovedTopDownParameters = DEFAULT_PARAMETERS,
) -> float:
    """Return the bounded dimensionless momentum-gradient factor."""

    z_most = most_application_height(z, boundary_layer_height, parameters)
    length = float(monin_obukhov_length)
    if not np.isfinite(length) or abs(length) < _EPS or abs(length) >= _NEUTRAL_L:
        return 1.0
    zeta = z_most / length
    if length > 0.0:
        value = 1.0 + parameters.stability_strength * 4.7 * zeta
    else:
        unstable = _unstable_phi_m(zeta, parameters)
        value = 1.0 + parameters.stability_strength * (unstable - 1.0)
    return float(np.clip(value, 0.25, 20.0))


def _safe_divisor(value: float, minimum: float = _MIN_PROFILE_DENOM) -> float:
    value = float(value)
    if not np.isfinite(value):
        return float(minimum)
    if abs(value) < minimum:
        return float(np.copysign(minimum, value if value != 0.0 else 1.0))
    return value


def _bounded_speed(value: float, u_inf: float, parameters: ImprovedTopDownParameters) -> float:
    u_ref = max(float(u_inf), 0.0)
    if not np.isfinite(value):
        return u_ref
    return float(
        np.clip(
            value,
            parameters.min_speed_fraction * u_ref,
            parameters.max_speed_fraction * u_ref,
        )
    )


def top_down_model(
    u_inf: float,
    z0: float,
    H: float,
    L: float,
    cft: float,
    zh: float,
    D: float,
    farmDist: float,
    wakeDist: float,
    parameters: ImprovedTopDownParameters = DEFAULT_PARAMETERS,
) -> float:
    """Return improved top-down hub-height wind speed."""

    kappa = 0.4
    uinf = float(u_inf)
    if not np.isfinite(uinf) or uinf <= 0.0:
        return 0.0

    z0 = _positive_finite(z0, 1.0e-4)
    H = _positive_finite(H, 1000.0)
    zh = _positive_finite(zh, 100.0)
    D = _positive_finite(D, zh)
    L = float(L) if np.isfinite(L) else _NEUTRAL_L
    cft = max(float(cft), 0.0) if np.isfinite(cft) else 0.0
    farmDist = max(float(farmDist), 0.0)
    wakeDist = max(float(wakeDist), 0.0)

    profile_denom = np.log(zh / z0) + stability_profile_term(zh, H, L, parameters)
    u_star = uinf * kappa / _safe_divisor(profile_denom)
    rbyz = float(np.clip(D / (2.0 * zh), 0.0, 0.95))
    z_tip = zh + D / 2.0

    if wakeDist > 0.0:
        gamma = mixing_factor(zh, H, L, parameters) * D**2 * uinf
        gamma /= kappa * max(u_star, _EPS) * zh
        gamma *= parameters.wake_recovery_scale
        gamma = max(float(gamma), 0.25 * D)
    else:
        gamma = 1.0

    friction_ratio = 1.0
    if farmDist > 0.0:
        nut = 28.0 * np.sqrt(0.5 * cft)
        if wakeDist > 0.0:
            nut *= np.exp(-wakeDist / gamma)
        beta = nut / (1.0 + nut)
        z_upper = (zh + nut * (zh + D / 2.0)) / (1.0 + nut)
        z_lower = max((zh + nut * (zh - D / 2.0)) / (1.0 + nut), z0 * 1.01)
        s1 = stability_profile_term(z_upper, H, L, parameters)
        s2 = stability_profile_term(z_lower, H, L, parameters)

        psi = np.log(zh / z0 * (1.0 - rbyz) ** beta) + s2
        psi = _safe_divisor(psi, minimum=0.05)
        roughness_exp = -(cft / (2.0 * kappa**2) + psi ** -2.0) ** -0.5 - s1
        z0_hi = zh * (1.0 + rbyz) ** beta * np.exp(np.clip(roughness_exp, -50.0, 50.0))
        z0_hi = _positive_finite(z0_hi, z0)

        farm_layer_height = z_tip + 0.32 * z0_hi * (farmDist / z0_hi) ** 0.8
        farm_layer_height = min(max(float(farm_layer_height), z_tip), H)
        hi_stability = stability_profile_term(farm_layer_height, H, L, parameters)
        hi_num = np.log(farm_layer_height / z0) + hi_stability
        hi_den = np.log(farm_layer_height / z0_hi) + hi_stability
        upper_layer_ratio = _safe_divisor(hi_num) / _safe_divisor(hi_den)
        lo_num = np.log(zh / z0_hi * (1.0 + rbyz) ** beta) + s1
        lo_den = np.log(zh / z0 * (1.0 - rbyz) ** beta) + s2
        lower_layer_ratio = _safe_divisor(lo_num) / _safe_divisor(lo_den)
        friction_ratio = upper_layer_ratio * lower_layer_ratio
    else:
        nut, beta = 0.0, 0.0

    profile = np.log((zh / z0) * (1.0 - rbyz) ** beta)
    z_lower = (zh + nut * (zh - D / 2.0)) / (1.0 + nut)
    profile += stability_profile_term(max(z_lower, z0 * 1.01), H, L, parameters)
    hub_speed = u_star / kappa * friction_ratio * profile

    if wakeDist > 0.0:
        zeta_shape = max((0.4 - 1.0) / gamma * wakeDist + 1.0, 0.4)
        recovery = np.exp(-((wakeDist / gamma) ** zeta_shape))
        hub_speed = uinf * (1.0 + (hub_speed / uinf - 1.0) * recovery)

    return _bounded_speed(hub_speed, uinf, parameters)


__all__ = [
    "DEFAULT_PARAMETERS",
    "ImprovedTopDownParameters",
    "mixing_factor",
    "most_application_height",
    "stability_profile_term",
    "top_down_model",
    "unstable_psi_m",
]
