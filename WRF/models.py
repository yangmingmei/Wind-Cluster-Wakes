"""Numerical routines used by this dataset; extracted from the audited implementation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from wake_models import ImprovedTopDownParameters, top_down_model, mixing_factor
from wake_models import ArrayStabilityParameters, array_stability_ratio_field
from wake_models import sequential_row_speed

from collections.abc import Callable
from typing import Any
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from dataclasses import dataclass

def _metadata_value(metadata: pd.DataFrame, key: str, default: float) -> float:
    """Read one numeric metadata value from the DTU 10 MW curve workbook."""

    rows = metadata.loc[metadata.iloc[:, 0].astype(str).str.contains(key, regex=False, na=False)]
    if rows.empty:
        return default
    value = pd.to_numeric(rows.iloc[0, 1], errors="coerce")
    if pd.isna(value):
        return default
    return float(value)

def load_dtu10mw_curve(curve_file: Path) -> dict[str, np.ndarray | float]:
    """Load the DTU 10 MW Ct/Cp/power curve and turbine metadata."""

    curve = pd.read_excel(curve_file, sheet_name=0, header=2)
    curve = curve.iloc[:, :4].copy()
    curve.columns = ["wind_speed", "ct", "cp", "power_kw"]
    curve = curve.apply(pd.to_numeric, errors="coerce").dropna()

    metadata = pd.read_excel(curve_file, sheet_name=1, header=None)
    hub_height = _metadata_value(metadata, "轮毂高度", 119.0)
    rotor_diameter = _metadata_value(metadata, "叶轮直径", 178.0)
    rated_power_mw = _metadata_value(metadata, "额定功率", 10.0)

    return {
        "wind_speed": curve["wind_speed"].to_numpy(float),
        "ct": curve["ct"].to_numpy(float),
        "cp": curve["cp"].to_numpy(float),
        "power_kw": curve["power_kw"].to_numpy(float),
        "hub_height": hub_height,
        "rotor_diameter": rotor_diameter,
        "rated_power_mw": rated_power_mw,
    }

def curve_value(curve: dict[str, np.ndarray | float], wind_speed: np.ndarray | float, key: str) -> np.ndarray:
    """Interpolate one DTU 10 MW curve field by wind speed."""

    ws_curve = np.asarray(curve["wind_speed"], dtype=float)
    values = np.asarray(curve[key], dtype=float)
    return np.interp(wind_speed, ws_curve, values, left=0.0, right=0.0)

def infer_planform_area(x_m: np.ndarray, y_m: np.ndarray) -> tuple[float, float]:
    """Infer a representative turbine-cell planform area from nearest spacing."""

    points = np.column_stack((x_m, y_m))
    distances, _ = cKDTree(points).query(points, k=2)
    median_spacing = float(np.median(distances[:, 1]))
    return median_spacing**2, median_spacing

def rotate_to_wind_frame(
    x_m: np.ndarray,
    y_m: np.ndarray,
    wind_direction_deg: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotate map coordinates so the rotated x-axis points downstream."""

    theta = np.radians(270.0 - wind_direction_deg)
    x_rot = x_m * np.cos(theta) + y_m * np.sin(theta)
    y_rot = -x_m * np.sin(theta) + y_m * np.cos(theta)
    return x_rot, y_rot, float(theta)

def wrf_center_coordinates(metadata: WRFMetadata) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one-dimensional axes and two-dimensional WRF center grids in meters."""

    x_m = grid_centers_km(metadata.nx, metadata.dx_m) * 1000.0
    y_m = grid_centers_km(metadata.ny, metadata.dy_m) * 1000.0
    x_grid_m, y_grid_m = np.meshgrid(x_m, y_m, indexing="xy")
    return x_m, y_m, x_grid_m, y_grid_m

def turbine_coordinates_m(turbines: TurbineTable, metadata: WRFMetadata) -> tuple[np.ndarray, np.ndarray]:
    """Convert turbine i/j grid indices to WRF map coordinates in meters."""

    x_m = turbines.i0.astype(float) * metadata.dx_m
    y_m = turbines.j0.astype(float) * metadata.dy_m
    return x_m, y_m

def build_dtu10mw_pywake_turbines(curve: dict[str, np.ndarray | float]) -> Any:
    """Create the single-type PyWake DTU 10 MW turbine from the curve workbook."""

    from py_wake.wind_turbines import WindTurbine, WindTurbines
    from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

    wind_speed = np.asarray(curve["wind_speed"], dtype=float)
    power_kw = np.asarray(curve["power_kw"], dtype=float)
    ct = np.asarray(curve["ct"], dtype=float)

    power_ct = PowerCtTabular(
        wind_speed,
        power_kw,
        "kW",
        ct,
        ws_cutin=float(np.nanmin(wind_speed)),
        ws_cutout=float(np.nanmax(wind_speed)),
    )
    turbine = WindTurbine(
        name="DTU 10 MW",
        diameter=float(curve["rotor_diameter"]),
        hub_height=float(curve["hub_height"]),
        powerCtFunction=power_ct,
    )
    return WindTurbines.from_WindTurbine_lst([turbine])

def squeeze_single_case_array(values: object, expected_ndim: int, label: str) -> np.ndarray:
    """Convert a single wind-direction/speed PyWake result to a NumPy array."""

    squeezed = np.asarray(values).squeeze()
    if squeezed.ndim != expected_ndim:
        raise ValueError(f"{label} should have {expected_ndim} dimensions after squeeze, got {squeezed.shape}")
    return squeezed

def clean_limited_speed(speed_2d: np.ndarray, freestream_speed: float) -> np.ndarray:
    """Validate PyWake outputs without modifying their values."""

    from wake_models import validate_pywake_speed
    return validate_pywake_speed(speed_2d)

def simulate_pywake_on_wrf_domain(
    turbines: TurbineTable,
    metadata: WRFMetadata,
    wind_speed: float,
    wind_direction: float,
    curve: dict[str, np.ndarray | float],
    pywake_turbines: Any,
    model_factory: Callable[[Any, Any], Any],
    turbulence_intensity: float,
    model_key: str,
) -> dict[str, np.ndarray | float]:
    """Run one PyWake model on the WRF mass grid and return a standard result."""

    from py_wake.flow_map import HorizontalGrid
    from py_wake.site import UniformSite

    hub_height = float(curve["hub_height"])
    turbine_x_m, turbine_y_m = turbine_coordinates_m(turbines, metadata)
    x_axis_m, y_axis_m, x_grid_m, y_grid_m = wrf_center_coordinates(metadata)

    site = UniformSite(p_wd=[1.0], ti=float(turbulence_intensity))
    wf_model = model_factory(site, pywake_turbines)
    turbine_type = np.zeros(len(turbine_x_m), dtype=int)

    sim_res = wf_model(
        turbine_x_m,
        turbine_y_m,
        wd=[float(wind_direction)],
        ws=[float(wind_speed)],
        type=turbine_type,
        TI=float(turbulence_intensity),
    )
    flow_map = sim_res.flow_map(
        grid=HorizontalGrid(x=x_axis_m, y=y_axis_m, h=hub_height),
        wd=[float(wind_direction)],
        ws=[float(wind_speed)],
    )

    speed_2d = clean_limited_speed(squeeze_single_case_array(flow_map.WS_eff, 2, "flow_map.WS_eff"), wind_speed)
    normalized_speed = speed_2d / max(float(wind_speed), 1.0e-12)
    turbine_ws = squeeze_single_case_array(sim_res.WS_eff, 1, "sim_res.WS_eff")
    turbine_power_kw = squeeze_single_case_array(sim_res.Power, 1, "sim_res.Power") / 1000.0

    if turbine_ws.size != len(turbine_x_m):
        raise ValueError(f"Expected {len(turbine_x_m)} turbine values, got {turbine_ws.size}")

    return {
        f"{model_key}_wind_speed_mps": speed_2d,
        f"{model_key}_normalized_wind_speed": normalized_speed,
        "x_axis_m": x_axis_m,
        "y_axis_m": y_axis_m,
        "x_grid_m": x_grid_m,
        "y_grid_m": y_grid_m,
        "turbine_x_m": turbine_x_m,
        "turbine_y_m": turbine_y_m,
        "turbine_ws_mps": turbine_ws,
        "turbine_power_kw": turbine_power_kw,
        "turbine_ct": curve_value(curve, turbine_ws, "ct"),
        "turbine_cp": curve_value(curve, turbine_ws, "cp"),
        "ct_at_inflow": float(curve_value(curve, wind_speed, "ct")),
        "power_at_inflow_kw": float(curve_value(curve, wind_speed, "power_kw")),
    }

def axis_covering(values: np.ndarray, spacing_m: float) -> np.ndarray:
    """Create a regular axis that covers all supplied coordinates."""

    axis_min = np.floor(float(np.nanmin(values)) / spacing_m) * spacing_m
    axis_max = np.ceil(float(np.nanmax(values)) / spacing_m) * spacing_m
    return np.arange(axis_min, axis_max + 0.5 * spacing_m, spacing_m, dtype=float)

def sample_wrf_grid_at_turbines(
    speed_wrf: np.ndarray,
    turbine_x_m: np.ndarray,
    turbine_y_m: np.ndarray,
    metadata: WRFMetadata,
    fallback_speed: float,
) -> np.ndarray:
    """Sample a WRF-shaped speed field at turbine map coordinates."""

    x_m, y_m, _, _ = wrf_center_coordinates(metadata)
    interpolator = RegularGridInterpolator(
        (y_m, x_m),
        speed_wrf,
        bounds_error=False,
        fill_value=float(fallback_speed),
    )
    turbine_points = np.column_stack((turbine_y_m, turbine_x_m))
    return interpolator(turbine_points)

@dataclass(frozen=True)
class WRFMetadata:
    """Store the WRF metadata needed for plotting."""

    nx: int
    ny: int
    dx_m: float
    dy_m: float
    time_label: str

@dataclass(frozen=True)
class TurbineTable:
    """Store original turbine indices and zero-based indices used by NumPy arrays."""

    i: np.ndarray
    j: np.ndarray
    type_id: np.ndarray
    i0: np.ndarray
    j0: np.ndarray

def grid_centers_km(n: int, spacing_m: float) -> np.ndarray:
    """Build WRF mass-grid cell-center coordinates in kilometers."""

    return np.arange(n, dtype=float) * spacing_m / 1000.0

def simulate_top_down_on_wrf_domain(
    turbines: TurbineTable,
    metadata: WRFMetadata,
    wind_speed: float,
    wind_direction: float,
    curve: dict[str, np.ndarray | float],
    roughness: float,
    boundary_layer_height: float,
    monin_obukhov_length: float,
    cft_scale: float,
    parameters: ImprovedTopDownParameters,
    reference_speed: np.ndarray | None = None,
    reference_blend_weight: float = 0.0,
    canopy_spacing_m: float = 250.0,
    support_radius_factor: float = 1 / np.sqrt(2),
) -> dict[str, np.ndarray | float]:
    """Run the improved model and interpolate it back to the WRF grid."""

    if cft_scale != 1.0 or reference_blend_weight != 0.0:
        raise ValueError("Audited predictions require Cft scale=1 and no reference blending")
    hub_height = float(curve["hub_height"])
    rotor_diameter = float(curve["rotor_diameter"])
    ct = float(curve_value(curve, wind_speed, "ct"))
    turbine_x_m, turbine_y_m = turbine_coordinates_m(turbines, metadata)
    planform_area, median_spacing = infer_planform_area(turbine_x_m, turbine_y_m)
    influence_radius = np.sqrt(planform_area / np.pi) * 2.0
    rotor_area = np.pi * rotor_diameter**2 / 4.0
    turbine_cft = cft_scale * ct * rotor_area / planform_area

    _, _, x_grid_wrf_m, y_grid_wrf_m = wrf_center_coordinates(metadata)
    x_grid_rot_centers, y_grid_rot_centers, _ = rotate_to_wind_frame(
        x_grid_wrf_m, y_grid_wrf_m, wind_direction
    )
    turbine_x_rot, turbine_y_rot, _ = rotate_to_wind_frame(turbine_x_m, turbine_y_m, wind_direction)
    if not np.isfinite(canopy_spacing_m) or canopy_spacing_m <= 0:
        raise ValueError("Canopy spacing must be finite and positive")
    x_axis = axis_covering(x_grid_rot_centers, canopy_spacing_m)
    y_axis = axis_covering(y_grid_rot_centers, canopy_spacing_m)
    x_grid_rot, y_grid_rot = np.meshgrid(x_axis, y_axis, indexing="ij")

    u_free = float(
        top_down_model(
            wind_speed,
            roughness,
            boundary_layer_height,
            monin_obukhov_length,
            0.0,
            hub_height,
            rotor_diameter,
            0.0,
            0.0,
            parameters,
        )
    )
    from wake_models import conservative_canopy
    cft_field, drag_audit = conservative_canopy(x_axis, y_axis, turbine_x_rot,
                                               turbine_y_rot, rotor_diameter, ct,
                                               support_radius_factor=support_radius_factor)
    u_field_rot = np.empty_like(x_grid_rot, dtype=float)
    for column in range(u_field_rot.shape[1]):
        u_field_rot[:, column] = sequential_row_speed(
            cft_field[:, column],
            u_free,
            float(canopy_spacing_m),
            roughness,
            boundary_layer_height,
            monin_obukhov_length,
            hub_height,
            rotor_diameter,
            parameters,
        )
    if not np.all(np.isfinite(u_field_rot)):
        raise ValueError("Non-finite top-down field")

    interpolator = RegularGridInterpolator(
        (x_axis, y_axis), u_field_rot, bounds_error=False, fill_value=u_free
    )
    wrf_points = np.column_stack((x_grid_rot_centers.ravel(), y_grid_rot_centers.ravel()))
    u_field_wrf_raw = interpolator(wrf_points).reshape(metadata.ny, metadata.nx)

    u_field_wrf_blended = u_field_wrf_raw.copy()

    turbine_ws = sample_wrf_grid_at_turbines(
        u_field_wrf_raw, turbine_x_m, turbine_y_m, metadata, u_free
    )
    turbine_ws = np.clip(turbine_ws, 0.0, None)
    turbine_power_kw = curve_value(curve, turbine_ws, "power_kw")
    turbine_cp = curve_value(curve, turbine_ws, "cp")

    return {
        "drag_audit": drag_audit,
        "canopy_spacing_m": float(canopy_spacing_m),
        "support_radius_factor": float(support_radius_factor),
        "u_field_wrf": u_field_wrf_raw,
        "normalized_u_wrf": u_field_wrf_raw / max(u_free, 1.0e-12),
        "u_field_rot": u_field_rot,
        "u_field_wrf_raw": u_field_wrf_raw,
        "normalized_u_wrf_raw": u_field_wrf_raw / max(u_free, 1.0e-12),
        "u_field_wrf_blended": u_field_wrf_blended,
        "normalized_u_wrf_blended": u_field_wrf_blended / max(u_free, 1.0e-12),
        "cft_field_rot": cft_field,
        "x_axis_rot_m": x_axis,
        "y_axis_rot_m": y_axis,
        "x_grid_wrf_m": x_grid_wrf_m,
        "y_grid_wrf_m": y_grid_wrf_m,
        "turbine_x_m": turbine_x_m,
        "turbine_y_m": turbine_y_m,
        "turbine_ws": turbine_ws,
        "turbine_ws_mps": turbine_ws,
        "turbine_power_kw": turbine_power_kw,
        "turbine_cp": turbine_cp,
        "u_free": u_free,
        "ct_at_inflow": ct,
        "turbine_cft": turbine_cft,
        "planform_area_m2": planform_area,
        "median_spacing_m": median_spacing,
        "influence_radius_m": influence_radius,
        "reference_blend_weight": float(reference_blend_weight),
    }

def simulate_turbopark_on_wrf_domain(
    turbines: TurbineTable,
    metadata: WRFMetadata,
    wind_speed: float,
    wind_direction: float,
    curve: dict[str, np.ndarray | float],
    pywake_turbines: object,
    turbulence_intensity: float,
) -> dict[str, np.ndarray | float]:
    """Run the TurboPark wake model on the WRF mass grid."""

    from wake_models import build_pywake_model

    return simulate_pywake_on_wrf_domain(
        turbines=turbines,
        metadata=metadata,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        curve=curve,
        pywake_turbines=pywake_turbines,
        model_factory=lambda site, wt: build_pywake_model("turbopark", site, wt),
        turbulence_intensity=turbulence_intensity,
        model_key="turbopark",
    )

def simulate_gaussian_on_wrf_domain(
    turbines: TurbineTable,
    metadata: WRFMetadata,
    wind_speed: float,
    wind_direction: float,
    curve: dict[str, np.ndarray | float],
    pywake_turbines: object,
    wake_expansion_k: float,
    turbulence_intensity: float,
) -> dict[str, np.ndarray | float]:
    """Run the Gaussian wake model on the WRF mass grid."""

    from wake_models import build_pywake_model

    return simulate_pywake_on_wrf_domain(
        turbines=turbines,
        metadata=metadata,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        curve=curve,
        pywake_turbines=pywake_turbines,
        model_factory=lambda site, wt: build_pywake_model("gaussian", site, wt, gaussian_k=wake_expansion_k),
        turbulence_intensity=turbulence_intensity,
        model_key="gaussian",
    )

MODEL_KEY = 'array_stability'

def calculate_iterated_asm_field(
    x_axis_rot_m: np.ndarray,
    y_axis_rot_m: np.ndarray,
    turbine_x_rot_m: np.ndarray,
    turbine_y_rot_m: np.ndarray,
    wind_speed: float,
    curve: dict[str, np.ndarray | float],
    roughness: float,
    turbulence_intensity: float,
    monin_obukhov_length: float,
    ct_scale: float,
    ct_iterations: int,
    ct_tolerance: float,
    parameters: ArrayStabilityParameters,
) -> dict[str, np.ndarray | float]:
    """Iterate ASM turbine inflow and C_T until the thrust field converges."""

    hub_height = float(curve["hub_height"])
    rotor_diameter = float(curve["rotor_diameter"])
    turbine_ct = np.full(
        turbine_x_rot_m.shape,
        float(curve_value(curve, wind_speed, "ct")),
        dtype=float,
    )
    turbine_points = np.column_stack((turbine_x_rot_m, turbine_y_rot_m))
    residual = np.inf
    iterations_used = 0

    for iteration in range(max(int(ct_iterations), 1)):
        asm = array_stability_ratio_field(
            x_axis_rot_m,
            y_axis_rot_m,
            turbine_x_rot_m,
            turbine_y_rot_m,
            ct_scale * turbine_ct,
            wind_speed,
            hub_height,
            rotor_diameter,
            roughness,
            turbulence_intensity,
            monin_obukhov_length,
            parameters,
        )
        interpolator = RegularGridInterpolator(
            (x_axis_rot_m, y_axis_rot_m),
            np.asarray(asm["speed_ratio"]),
            bounds_error=False,
            fill_value=1.0,
        )
        turbine_asm_speed = wind_speed * np.clip(interpolator(turbine_points), 0.0, 1.0)
        updated_ct = np.asarray(curve_value(curve, turbine_asm_speed, "ct"), dtype=float)
        residual = float(np.max(np.abs(updated_ct - turbine_ct))) if turbine_ct.size else 0.0
        iterations_used = iteration + 1
        if residual <= float(ct_tolerance):
            turbine_ct = updated_ct
            break
        turbine_ct = 0.5 * turbine_ct + 0.5 * updated_ct

    asm = array_stability_ratio_field(
        x_axis_rot_m,
        y_axis_rot_m,
        turbine_x_rot_m,
        turbine_y_rot_m,
        ct_scale * turbine_ct,
        wind_speed,
        hub_height,
        rotor_diameter,
        roughness,
        turbulence_intensity,
        monin_obukhov_length,
        parameters,
    )
    interpolator = RegularGridInterpolator(
        (x_axis_rot_m, y_axis_rot_m),
        np.asarray(asm["speed_ratio"]),
        bounds_error=False,
        fill_value=1.0,
    )
    asm["turbine_speed_ratio"] = np.clip(interpolator(turbine_points), 0.0, 1.0)
    asm["turbine_ct_for_asm"] = turbine_ct
    asm["ct_iteration_count"] = iterations_used
    asm["ct_iteration_residual"] = residual
    return asm

def simulate_array_stability_on_wrf_domain(
    turbines: TurbineTable,
    metadata: WRFMetadata,
    wind_speed: float,
    wind_direction: float,
    curve: dict[str, np.ndarray | float],
    pywake_turbines: object,
    roughness: float,
    turbulence_intensity: float,
    monin_obukhov_length: float,
    ct_scale: float,
    ct_iterations: int,
    ct_tolerance: float,
    parameters: ArrayStabilityParameters,
) -> dict[str, np.ndarray | float]:
    """Run ASM and its spatial-inflow TurboPark calculation on the WRF grid."""

    import xarray as xr
    from py_wake.flow_map import HorizontalGrid
    from py_wake.literature import Nygaard_2022
    from py_wake.site import XRSite

    hub_height = float(curve["hub_height"])
    turbine_x_m, turbine_y_m = turbine_coordinates_m(turbines, metadata)
    x_axis_m, y_axis_m, x_grid_m, y_grid_m = wrf_center_coordinates(metadata)
    x_grid_rot_m, y_grid_rot_m, _ = rotate_to_wind_frame(x_grid_m, y_grid_m, wind_direction)
    turbine_x_rot_m, turbine_y_rot_m, _ = rotate_to_wind_frame(
        turbine_x_m, turbine_y_m, wind_direction
    )
    grid_spacing_m = min(float(metadata.dx_m), float(metadata.dy_m))
    x_axis_rot_m = axis_covering(x_grid_rot_m, grid_spacing_m)
    y_axis_rot_m = axis_covering(y_grid_rot_m, grid_spacing_m)

    asm = calculate_iterated_asm_field(
        x_axis_rot_m,
        y_axis_rot_m,
        turbine_x_rot_m,
        turbine_y_rot_m,
        wind_speed,
        curve,
        roughness,
        turbulence_intensity,
        monin_obukhov_length,
        ct_scale,
        ct_iterations,
        ct_tolerance,
        parameters,
    )
    asm_interpolator = RegularGridInterpolator(
        (x_axis_rot_m, y_axis_rot_m),
        np.asarray(asm["speed_ratio"]),
        bounds_error=False,
        fill_value=1.0,
    )
    wrf_rotated_points = np.column_stack((x_grid_rot_m.ravel(), y_grid_rot_m.ravel()))
    asm_ratio_wrf = asm_interpolator(wrf_rotated_points).reshape(metadata.ny, metadata.nx)
    asm_ratio_wrf = np.clip(np.nan_to_num(asm_ratio_wrf, nan=1.0), 0.0, 1.0)
    asm_speed_wrf = float(wind_speed) * asm_ratio_wrf

    site_data = xr.Dataset(
        data_vars={
            "Speedup": (("x", "y"), asm_ratio_wrf.T),
            "P": 1.0,
            "TI": float(turbulence_intensity),
        },
        coords={"x": x_axis_m, "y": y_axis_m},
    )
    site = XRSite(site_data, interp_method="linear")
    wake_model = Nygaard_2022(site, pywake_turbines)
    turbine_type = np.zeros(len(turbine_x_m), dtype=int)
    sim_res = wake_model(
        turbine_x_m,
        turbine_y_m,
        wd=[float(wind_direction)],
        ws=[float(wind_speed)],
        type=turbine_type,
        TI=float(turbulence_intensity),
    )
    flow_map = sim_res.flow_map(
        grid=HorizontalGrid(x=x_axis_m, y=y_axis_m, h=hub_height),
        wd=[float(wind_direction)],
        ws=[float(wind_speed)],
    )
    turbopark_speed_wrf = clean_limited_speed(
        squeeze_single_case_array(flow_map.WS_eff, 2, "flow_map.WS_eff"),
        wind_speed,
    )
    combined_speed_wrf = np.minimum(asm_speed_wrf, turbopark_speed_wrf)
    combined_speed_wrf = clean_limited_speed(combined_speed_wrf, wind_speed)

    turbopark_turbine_speed = squeeze_single_case_array(sim_res.WS_eff, 1, "sim_res.WS_eff")
    asm_turbine_speed = float(wind_speed) * np.asarray(asm["turbine_speed_ratio"])
    turbine_ws = np.minimum(turbopark_turbine_speed, asm_turbine_speed)
    turbine_power_kw = curve_value(curve, turbine_ws, "power_kw")

    return {
        f"{MODEL_KEY}_wind_speed_mps": combined_speed_wrf,
        f"{MODEL_KEY}_normalized_wind_speed": combined_speed_wrf / max(float(wind_speed), 1.0e-12),
        "asm_wind_speed_mps": asm_speed_wrf,
        "asm_normalized_wind_speed": asm_ratio_wrf,
        "turbopark_with_asm_inflow_wind_speed_mps": turbopark_speed_wrf,
        "x_axis_m": x_axis_m,
        "y_axis_m": y_axis_m,
        "x_grid_m": x_grid_m,
        "y_grid_m": y_grid_m,
        "x_axis_rot_m": x_axis_rot_m,
        "y_axis_rot_m": y_axis_rot_m,
        "turbine_x_m": turbine_x_m,
        "turbine_y_m": turbine_y_m,
        "turbine_ws_mps": turbine_ws,
        "turbine_power_kw": turbine_power_kw,
        "turbine_ct": curve_value(curve, turbine_ws, "ct"),
        "turbine_cp": curve_value(curve, turbine_ws, "cp"),
        "ct_at_inflow": float(curve_value(curve, wind_speed, "ct")),
        "power_at_inflow_kw": float(curve_value(curve, wind_speed, "power_kw")),
        "turbine_ct_for_asm": np.asarray(asm["turbine_ct_for_asm"]),
        "thrust_density_rot": np.asarray(asm["thrust_density"]),
        "farm_layer_speed_ratio_rot": np.asarray(asm["farm_layer_speed_ratio"]),
        "asm_speed_ratio_rot": np.asarray(asm["speed_ratio"]),
        "effective_drag_coefficient_rot": np.asarray(asm["effective_drag_coefficient"]),
        "friction_velocity_mps": float(asm["friction_velocity_mps"]),
        "exchange_coefficient_m2ps": float(asm["exchange_coefficient_m2ps"]),
        "surface_drag_coefficient": float(asm["surface_drag_coefficient"]),
        "phi_m": float(asm["phi_m"]),
        "psi_m": float(asm["psi_m"]),
        "ct_iteration_count": int(asm["ct_iteration_count"]),
        "ct_iteration_residual": float(asm["ct_iteration_residual"]),
    }
