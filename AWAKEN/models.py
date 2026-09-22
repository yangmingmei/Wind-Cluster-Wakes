"""Numerical routines used by this dataset; extracted from the audited implementation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from wake_models import ImprovedTopDownParameters, top_down_model, mixing_factor
from wake_models import ArrayStabilityParameters, array_stability_ratio_field
from wake_models import sequential_row_speed

import math
from typing import Any
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from py_wake.examples.data.iea37 import IEA37Site
from py_wake.flow_map import Points
from py_wake.literature import Nygaard_2022
from py_wake.wind_turbines import WindTurbine, WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from dataclasses import dataclass
import xarray as xr
from py_wake.site import XRSite
from wake_models import array_stability_ratio_field
from wake_models import DIAMETERS, project, height_projection
from wake_models import build_pywake_model, validate_pywake_speed

CT_CURVE_CACHE: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)

def selected_geometry_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, float]:
    columns = [safe_text(column) for column in data["selected_geometry_columns"]]
    values = np.asarray(data["selected_geometry_values"], dtype=float)
    geometry = {column: float(value) for column, value in zip(columns, values)}
    required = [
        "s_upstream_km",
        "s_exit_km",
        "projected_farm_width_km",
        "farm_crosswind_center_km",
    ]
    missing = [key for key in required if key not in geometry or not np.isfinite(geometry[key])]
    if missing:
        raise ValueError(f"Selected geometry is missing required fields: {missing}")
    return geometry

def load_ct_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path not in CT_CURVE_CACHE:
        performance = pd.read_csv(path)
        wind_speed = pd.to_numeric(performance["V"], errors="coerce").to_numpy(float)
        ct = pd.to_numeric(performance["Ct"], errors="coerce").to_numpy(float)
        valid = np.isfinite(wind_speed) & np.isfinite(ct)
        if not np.any(valid):
            raise ValueError(f"No valid Ct curve data found in {path}")
        order = np.argsort(wind_speed[valid])
        CT_CURVE_CACHE[path] = wind_speed[valid][order], ct[valid][order]
    return CT_CURVE_CACHE[path]

def native_model_georef(x_grid_km: np.ndarray, y_grid_km: np.ndarray, model_shape: tuple[int, int], source: RadarGeoreference) -> RadarGeoreference:
    mesh_x_km, mesh_y_km = np.meshgrid(x_grid_km, y_grid_km, indexing="xy")
    return RadarGeoreference(
        status=source.status,
        method=source.method,
        crs=source.crs,
        crs_text=source.crs_text,
        x_m=x_grid_km * 1000.0,
        y_m=y_grid_km * 1000.0,
        x_km=x_grid_km,
        y_km=y_grid_km,
        mesh_x_km=mesh_x_km,
        mesh_y_km=mesh_y_km,
        radar_lon=np.array([]),
        radar_lat=np.array([]),
        origin_x_m=np.nan,
        origin_y_m=np.nan,
        dx_km=float(np.nanmedian(np.diff(x_grid_km))) if len(x_grid_km) > 1 else np.nan,
        dy_km=float(np.nanmedian(np.diff(y_grid_km))) if len(y_grid_km) > 1 else np.nan,
        wsf_dims=model_shape,
        diagnostics={},
    )

def define_pywake_turbine_quiet(path_txt: Path, path_csv: Path, verbose: bool):
    """Build a PyWake turbine from physically dimensional AWAKEN CSV data."""
    del path_txt
    performance = pd.read_csv(path_csv)
    model_specs = {
        "2.8MW": ("GE2.82-127", 127.0, 89.0),
        "1.7MW": ("GE1.7-103", 103.0, 80.0),
        "2.3MW": ("GE2.3-116", 116.0, 90.0),
        "1.8MW": ("GE1.79-100", 100.0, 80.0),
    }
    name, diameter_m, hub_height_m = model_specs[path_csv.parent.name]
    power_ct = PowerCtTabular(
        performance["V"].to_numpy(dtype=float),
        performance["gen power [kW]"].to_numpy(dtype=float),
        "kW",
        performance["Ct"].to_numpy(dtype=float),
        ws_cutin=3.0,
        ws_cutout=25.0,
    )
    turbine = WindTurbine(name, diameter_m, hub_height_m, power_ct)
    if verbose:
        print(f"Loaded {name}: D={diameter_m:.0f} m, H={hub_height_m:.0f} m")
    return {}, turbine

def initialize_pywake_model(verbose: bool):
    _, nrel_2p8 = define_pywake_turbine_quiet(
        LEGACY_MODEL_DIR / "2.8MW" / "NREL-2p8-127_Cp_Ct_Cq.txt",
        LEGACY_MODEL_DIR / "2.8MW" / "NREL-2.82-127_performance.csv",
        verbose,
    )
    _, nrel_1p7 = define_pywake_turbine_quiet(
        LEGACY_MODEL_DIR / "1.7MW" / "NREL-1p72-103_Cp_Ct_Cq.txt",
        LEGACY_MODEL_DIR / "1.7MW" / "NREL-1.72-103_performance.csv",
        verbose,
    )
    _, nrel_2p3 = define_pywake_turbine_quiet(
        LEGACY_MODEL_DIR / "2.3MW" / "NREL-2p3-116_Cp_Ct_Cq.txt",
        LEGACY_MODEL_DIR / "2.3MW" / "NREL-2.3-116_performance.csv",
        verbose,
    )
    _, nrel_1p8 = define_pywake_turbine_quiet(
        LEGACY_MODEL_DIR / "1.8MW" / "NREL-1p79-100_Cp_Ct_Cq.txt",
        LEGACY_MODEL_DIR / "1.8MW" / "NREL-1.79-100_performance.csv",
        verbose,
    )

    wind_turbines = WindTurbines.from_WindTurbine_lst([nrel_2p8, nrel_1p7, nrel_2p3, nrel_1p8])
    return Nygaard_2022(IEA37Site(16), wind_turbines)

def simulate_native_model(
    wake_model: Any,
    turbine_x_m: np.ndarray,
    turbine_y_m: np.ndarray,
    turbine_type: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    wind_speed_ms: float,
    wind_from_deg: float,
    turbulence_intensity: float,
    native_chunk_y: int,
    radar_height_agl_m: np.ndarray,
) -> np.ndarray:
    if not np.isfinite(wind_speed_ms) or wind_speed_ms <= 0.0:
        raise ValueError(f"Invalid wind speed: {wind_speed_ms}")
    if not np.isfinite(wind_from_deg):
        raise ValueError(f"Invalid wind direction: {wind_from_deg}")

    sim_res = wake_model(
        turbine_x_m,
        turbine_y_m,
        wd=[float(wind_from_deg)],
        ws=[float(wind_speed_ms)],
        type=turbine_type.astype(int),
        TI=float(turbulence_intensity),
    )

    x_m = np.asarray(x_km, dtype=float) * 1000.0
    y_km = np.asarray(y_km, dtype=float)
    model_ws = np.full((len(y_km), len(x_m)), np.nan, dtype=float)
    radar_height_agl_m = np.asarray(radar_height_agl_m, dtype=float)
    if radar_height_agl_m.shape != model_ws.shape:
        raise ValueError(
            f"Radar AGL shape {radar_height_agl_m.shape} does not match model grid {model_ws.shape}."
        )
    chunk = max(1, int(native_chunk_y))

    for start in range(0, len(y_km), chunk):
        stop = min(start + chunk, len(y_km))
        y_m_chunk = y_km[start:stop] * 1000.0
        x_chunk, y_chunk = np.meshgrid(x_m, y_m_chunk, indexing="xy")
        h_chunk = radar_height_agl_m[start:stop]
        flow_map = sim_res.flow_map(
            grid=Points(x_chunk.ravel(), y_chunk.ravel(), h_chunk.ravel()),
            wd=[float(wind_from_deg)],
            ws=[float(wind_speed_ms)],
        )
        model_ws[start:stop, :] = np.asarray(flow_map.WS_eff, dtype=float).reshape(
            -1, order="C"
        ).reshape(stop - start, len(x_m))

    return model_ws

@dataclass
class RadarGeoreference:
    status: str
    method: str
    crs: object
    crs_text: str
    x_m: np.ndarray
    y_m: np.ndarray
    x_km: np.ndarray
    y_km: np.ndarray
    mesh_x_km: np.ndarray
    mesh_y_km: np.ndarray
    radar_lon: np.ndarray
    radar_lat: np.ndarray
    origin_x_m: float
    origin_y_m: float
    dx_km: float
    dy_km: float
    wsf_dims: tuple
    diagnostics: dict

def flow_basis(wind_from_deg):
    flow_to_deg = (float(wind_from_deg) + 180.0) % 360.0
    theta = math.radians(flow_to_deg)
    e_s = np.array([math.sin(theta), math.cos(theta)], dtype=float)
    e_n = np.array([-math.cos(theta), math.sin(theta)], dtype=float)
    return flow_to_deg, e_s, e_n

def flow_to_xy(s_abs, n_abs, wind_from_deg):
    _, e_s, e_n = flow_basis(wind_from_deg)
    return s_abs * e_s[0] + n_abs * e_n[0], s_abs * e_s[1] + n_abs * e_n[1]

def bilinear_sample(x_coords, y_coords, values_yx, x_points, y_points):
    x = np.asarray(x_coords, dtype=float)
    y = np.asarray(y_coords, dtype=float)
    values = np.asarray(values_yx, dtype=float)
    xp = np.asarray(x_points, dtype=float)
    yp = np.asarray(y_points, dtype=float)
    out = np.full(xp.shape, np.nan, dtype=float)

    if len(x) < 2 or len(y) < 2:
        return out
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError("bilinear_sample requires increasing x/y coordinates.")

    inside = (xp >= x[0]) & (xp <= x[-1]) & (yp >= y[0]) & (yp <= y[-1])
    if not np.any(inside):
        return out

    xi = np.searchsorted(x, xp[inside], side="right") - 1
    yi = np.searchsorted(y, yp[inside], side="right") - 1
    xi = np.clip(xi, 0, len(x) - 2)
    yi = np.clip(yi, 0, len(y) - 2)

    x0 = x[xi]
    x1 = x[xi + 1]
    y0 = y[yi]
    y1 = y[yi + 1]
    wx = np.where(x1 != x0, (xp[inside] - x0) / (x1 - x0), 0.0)
    wy = np.where(y1 != y0, (yp[inside] - y0) / (y1 - y0), 0.0)

    v00 = values[yi, xi]
    v10 = values[yi, xi + 1]
    v01 = values[yi + 1, xi]
    v11 = values[yi + 1, xi + 1]
    sampled = (
        v00 * (1.0 - wx) * (1.0 - wy)
        + v10 * wx * (1.0 - wy)
        + v01 * (1.0 - wx) * wy
        + v11 * wx * wy
    )
    valid_corners = np.isfinite(v00) & np.isfinite(v10) & np.isfinite(v01) & np.isfinite(v11)
    sampled[~valid_corners] = np.nan
    out[inside] = sampled
    return out

def flow_grid_xy(s_grid, n_grid, geom, wind_from_deg):
    s_mesh, n_mesh = np.meshgrid(s_grid, n_grid, indexing="ij")
    s_abs = s_mesh + geom["s_exit_km"]
    n_abs = n_mesh + geom["farm_crosswind_center_km"]
    x, y = flow_to_xy(s_abs, n_abs, wind_from_deg)
    return x, y

def sample_native_on_flow_grid(native_values, georef, geom, wind_from_deg, s_grid, n_grid):
    x, y = flow_grid_xy(s_grid, n_grid, geom, wind_from_deg)
    return bilinear_sample(georef.x_km, georef.y_km, native_values, x, y)

DEFAULT_LAND_ROUGHNESS_M = 0.03

def neutral_log_profile_ratio(
    target_height_m: float | np.ndarray,
    reference_height_m: float | np.ndarray,
    roughness_length_m: float = DEFAULT_LAND_ROUGHNESS_M,
) -> np.ndarray:
    """Return U(target)/U(reference) from a neutral land log profile."""

    z0 = max(float(roughness_length_m), 1.0e-4)
    target = np.maximum(np.asarray(target_height_m, dtype=float), 1.01 * z0)
    reference = np.maximum(np.asarray(reference_height_m, dtype=float), 1.01 * z0)
    return np.log(target / z0) / np.log(reference / z0)

def representative_hub_speed(
    radar_speed_ms: float,
    radar_height_agl_m: np.ndarray,
    hub_height_m: float,
    roughness_length_m: float = DEFAULT_LAND_ROUGHNESS_M,
) -> tuple[float, float, float]:
    """Map the fixed-elevation radar inflow to a representative hub speed."""

    finite_height = np.asarray(radar_height_agl_m, dtype=float)
    finite_height = finite_height[np.isfinite(finite_height)]
    if finite_height.size == 0:
        raise ValueError("No finite radar AGL heights are available.")
    reference_height = float(np.nanmedian(finite_height))
    factor = float(
        neutral_log_profile_ratio(hub_height_m, reference_height, roughness_length_m)
    )
    return float(radar_speed_ms) * factor, reference_height, factor

def array_field(xx, yy, tx, ty, types, speed, wd, length, curves, turbines,
                height, simulate, step=250., roughness=.03, ti=.09):
    s, n = project(xx, yy, wd)
    ts, tn = project(tx, ty, wd)
    diameter = 110.
    site_axes = [np.arange(np.floor(min(a.min(), b.min())/step)*step-step,
                          np.ceil(max(a.max(), b.max())/step)*step+2*step, step)
                 for a, b in [(xx, tx), (yy, ty)]]
    gx, gy = np.meshgrid(*site_axes, indexing='ij')
    gs, gn = project(gx, gy, wd)
    axes = [np.arange(np.floor(min(a.min(), b.min()-padding)/step)*step,
                      np.ceil(max(a.max(), b.max()+padding)/step)*step+step, step)
            for a, b, padding in [(gs, ts, 8*9*diameter), (gn, tn, 8*3*diameter)]]
    points = np.c_[ts, tn]
    def thrust(u):
        return np.array([np.interp(v, curves[t][0], curves[t][1], left=0., right=0.)
                         for t, v in zip(types, np.broadcast_to(u, types.shape))])
    ct = thrust(speed)
    area_scale = (DIAMETERS[types]/diameter)**2
    def solve(values):
        return array_stability_ratio_field(*axes, ts, tn, values*area_scale,
            speed, 85., diameter, roughness, ti, length)['speed_ratio']
    for iteration in range(100):
        q = solve(ct)
        interp = RegularGridInterpolator(axes, q, bounds_error=True)
        updated = thrust(speed*interp(points))
        residual = float(np.max(np.abs(updated-ct)))
        if residual <= 1.e-4:
            ct = updated
            break
        ct = .5*(ct+updated)
    if residual > 1.e-4:
        raise RuntimeError(f'ASM CT iteration did not converge: {residual}')
    q = solve(ct)
    interp = RegularGridInterpolator(axes, q, bounds_error=True)
    native_hub = interp(np.c_[s.ravel(), n.ravel()]).reshape(s.shape)
    # The geographical site and its entire interpolation stencil are covered
    # by the solved wind-aligned grid. No neutral fill or extrapolation.
    site_q = interp(np.c_[gs.ravel(), gn.ravel()]).reshape(gs.shape)
    site = XRSite(xr.Dataset(data_vars={'Speedup': (('x','y'), site_q), 'P': 1., 'TI': ti},
                            coords={'x': site_axes[0], 'y': site_axes[1]}), interp_method='linear')
    model = build_pywake_model('turbopark', site, turbines)
    valid = np.isfinite(height)
    turbo = simulate(model, tx, ty, types, xx[0]/1000, yy[:,0]/1000,
                     speed, wd, ti, 32, np.where(valid, height, 85.))/speed
    validate_pywake_speed(turbo[valid])
    result = np.minimum(height_projection(native_hub, height), turbo)
    return np.where(valid, result, np.nan), dict(asm_ct_iterations=iteration+1,
        asm_ct_residual=residual, asm_step_m=step, asm_nx=len(axes[0]), asm_ny=len(axes[1]),
        asm_representative_diameter_m=diameter, asm_roughness_m=roughness)

SCRIPT_DIR = Path(__file__).resolve().parent

LEGACY_MODEL_DIR = SCRIPT_DIR / 'data/turbines'
