"""Numerical routines used by this dataset; extracted from the audited implementation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from wake_models import ImprovedTopDownParameters, top_down_model, mixing_factor
from wake_models import ArrayStabilityParameters, array_stability_ratio_field
from wake_models import sequential_row_speed

import math
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from py_wake.site import XRSite

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CURVE_FILE = SCRIPT_DIR / 'data/10MW_turbine_curve.xlsx'

MODEL_LABELS = {'gaussian': 'Gaussian', 'top_down': 'Top-Down', 'turbopark': 'TurboPark', 'array_stability': 'Array-Stability'}

def load_turbine_curve(curve_file: Path = DEFAULT_CURVE_FILE) -> dict[str, np.ndarray | float]:
    """Read the verified 10 MW curve and retain Ct while scaling rated power by turbine type."""

    curve = pd.read_excel(curve_file, sheet_name="CT_CP曲线", header=2).iloc[:, :4].copy()
    curve.columns = ["wind_speed", "ct", "cp", "power_kw"]
    curve = curve.apply(pd.to_numeric, errors="coerce").dropna()
    metadata = pd.read_excel(curve_file, sheet_name="说明", header=None)

    def metadata_value(fragment: str, default: float) -> float:
        row = metadata.loc[metadata.iloc[:, 0].astype(str).str.contains(fragment, regex=False, na=False)]
        if row.empty:
            return default
        value = pd.to_numeric(row.iloc[0, 1], errors="coerce")
        return default if pd.isna(value) else float(value)

    return {
        "wind_speed": curve["wind_speed"].to_numpy(float),
        "ct": curve["ct"].to_numpy(float),
        "cp": curve["cp"].to_numpy(float),
        "power_kw": curve["power_kw"].to_numpy(float),
        "source_hub_height_m": metadata_value("轮毂高度", 119.0),
        "source_rotor_diameter_m": metadata_value("叶轮直径", 178.0),
        "source_rated_power_mw": metadata_value("额定功率", 10.0),
    }

def curve_value(curve: dict[str, np.ndarray | float], wind_speed: np.ndarray | float, key: str) -> np.ndarray:
    ws = np.asarray(curve["wind_speed"], dtype=float)
    values = np.asarray(curve[key], dtype=float)
    return np.interp(wind_speed, ws, values, left=0.0, right=0.0)

def lonlat_to_local_km(
    longitude: np.ndarray,
    latitude: np.ndarray,
    origin_longitude: float,
    origin_latitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert WGS84 coordinates to a local east/north tangent approximation."""

    east_km = (np.asarray(longitude, dtype=float) - origin_longitude) * 111.320 * math.cos(math.radians(origin_latitude))
    north_km = (np.asarray(latitude, dtype=float) - origin_latitude) * 111.320
    return east_km, north_km

def rotate_to_wind_frame(east_km: np.ndarray, north_km: np.ndarray, wind_from_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return downstream and west-positive crosswind coordinates."""

    theta = math.radians(float(wind_from_deg))
    downstream = -np.sin(theta) * east_km - np.cos(theta) * north_km
    crosswind = np.cos(theta) * east_km - np.sin(theta) * north_km
    return downstream, crosswind

def layout_in_wind_frame(layout: pd.DataFrame, wind_from_deg: float) -> tuple[np.ndarray, np.ndarray]:
    origin_latitude = float(layout["latitude"].mean())
    origin_longitude = float(layout["longitude"].mean())
    east, north = lonlat_to_local_km(
        layout["longitude"].to_numpy(float),
        layout["latitude"].to_numpy(float),
        origin_longitude,
        origin_latitude,
    )
    return rotate_to_wind_frame(east, north, wind_from_deg)

def pywake_turbopark_speed_ratio(
    layout: pd.DataFrame,
    curve: dict[str, np.ndarray | float],
    turbine_s_km: np.ndarray,
    turbine_n_km: np.ndarray,
    query_s_km: np.ndarray,
    query_n_km: np.ndarray,
    query_height_m: np.ndarray,
    u_ref_mps: float,
    turbulence_intensity: float,
    model_key: str = "turbopark",
    spatial_site=None,
) -> np.ndarray:
    """Evaluate PyWake's Nygaard_2022 (TurboPark) implementation at aircraft points."""

    try:
        from py_wake.flow_map import Points
        from py_wake.literature import Nygaard_2022
        from py_wake.site import UniformSite
        from py_wake.wind_turbines import WindTurbine, WindTurbines
        from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
    except ImportError as exc:
        raise RuntimeError(
            "TurboPark requires PyWake. Run this script with the pywake_cluster conda environment."
        ) from exc

    source_rated_mw = float(curve["source_rated_power_mw"])
    wind_turbine_list = []
    turbine_type_index: dict[str, int] = {}
    for turbine_type, group in layout.groupby("turbine_type", sort=False):
        row = group.iloc[0]
        scaled_power_kw = np.asarray(curve["power_kw"], dtype=float) * (
            float(row["rated_power_mw"]) / source_rated_mw
        )
        power_ct = PowerCtTabular(
            np.asarray(curve["wind_speed"], dtype=float),
            scaled_power_kw,
            "kW",
            np.asarray(curve["ct"], dtype=float),
            ws_cutin=float(np.min(curve["wind_speed"])),
            ws_cutout=float(np.max(curve["wind_speed"])),
        )
        turbine_type_index[str(turbine_type)] = len(wind_turbine_list)
        wind_turbine_list.append(
            WindTurbine(
                str(turbine_type),
                diameter=float(row["rotor_diameter_m"]),
                hub_height=float(row["hub_height_m"]),
                powerCtFunction=power_ct,
            )
        )

    wind_turbines = WindTurbines.from_WindTurbine_lst(wind_turbine_list)
    turbine_types = layout["turbine_type"].map(turbine_type_index).to_numpy(int)
    site = UniformSite(p_wd=[1.0], ti=float(turbulence_intensity), ws=float(u_ref_mps))
    if spatial_site is not None:
        site = spatial_site
    from wake_models import build_pywake_model
    wake_model = build_pywake_model(model_key, site, wind_turbines)
    simulation = wake_model(
        x=np.asarray(turbine_s_km, dtype=float) * 1000.0,
        y=np.asarray(turbine_n_km, dtype=float) * 1000.0,
        type=turbine_types,
        wd=[270.0],
        ws=[float(u_ref_mps)],
    )
    points = Points(
        np.asarray(query_s_km, dtype=float) * 1000.0,
        np.asarray(query_n_km, dtype=float) * 1000.0,
        np.asarray(query_height_m, dtype=float),
    )
    flow_map = simulation.flow_map(grid=points, wd=[270.0], ws=[float(u_ref_mps)])
    ratio = np.asarray(flow_map.WS_eff, dtype=float).squeeze() / float(u_ref_mps)
    from wake_models import validate_pywake_speed
    return validate_pywake_speed(ratio)

def top_down_speed_ratio(
    layout: pd.DataFrame,
    turbine_s_km: np.ndarray,
    turbine_n_km: np.ndarray,
    query_s_km: np.ndarray,
    query_n_km: np.ndarray,
    ct: float,
    turbulence_intensity: float,
    wind_speed_mps: float,
    monin_obukhov_length_m: float = 1.0e9,
    roughness_m: float = 0.0002,
    boundary_layer_height_m: float = 1000.0,
    stability_strength: float = 1.0,
    wake_recovery_scale: float = 1.0,
    streamwise_step_m: float = 250.0,
) -> np.ndarray:
    """Latest improved top-down core on sequential turbine-canopy ribbons."""

    del turbulence_intensity
    parameters = ImprovedTopDownParameters(
        stability_strength=float(stability_strength),
        wake_recovery_scale=float(wake_recovery_scale),
    )
    from wake_models import conservative_canopy
    from scipy.interpolate import RegularGridInterpolator
    tx, ty, qx, qy = [np.asarray(a) * 1000 for a in (turbine_s_km, turbine_n_km, query_s_km, query_n_km)]
    dx = float(streamwise_step_m)
    dy = dx
    x = np.arange(np.floor((min(tx.min(), qx.min()) - 3000) / dx) * dx,
                  max(tx.max(), qx.max()) + 3000 + dx, dx)
    y = np.arange(np.floor((min(ty.min(), qy.min()) - 3000) / dy) * dy,
                  max(ty.max(), qy.max()) + 3000 + dy, dy)
    cft_field, drag_audit = conservative_canopy(x, y, tx, ty,
        layout['rotor_diameter_m'].to_numpy(float), ct, layout['farm_name'].to_numpy())
    hub = float(np.average(layout['hub_height_m'], weights=layout['rated_power_mw']))
    diameter = float(np.average(layout['rotor_diameter_m'], weights=layout['rated_power_mw']))
    speed = np.empty_like(cft_field)
    for j in range(len(y)):
        speed[:, j] = sequential_row_speed(cft_field[:, j], 1., dx, roughness_m,
            boundary_layer_height_m, monin_obukhov_length_m, hub, diameter, parameters)
    return RegularGridInterpolator((x, y), speed, bounds_error=True)(np.c_[qx, qy])

def predict_array_stability(layout, curve, turbine_s, turbine_n, query_s, query_n,
                            height, u_ref, ti, length):
    diameter = float(np.average(layout.rotor_diameter_m, weights=layout.rated_power_mw))
    hub = float(np.average(layout.hub_height_m, weights=layout.rated_power_mw))
    tx, ty, qx, qy = [np.asarray(a) * 1000 for a in (turbine_s, turbine_n, query_s, query_n)]
    x = np.arange(np.floor(min(tx.min()-8*9*diameter, qx.min()-500)/250)*250,
                  max(tx.max()+8*9*diameter, qx.max()+500)+250, 250)
    y = np.arange(np.floor(min(ty.min()-8*3*diameter, qy.min()-500)/200)*200,
                  max(ty.max()+8*3*diameter, qy.max()+500)+200, 200)
    ct = np.full(tx.shape, float(curve_value(curve, u_ref, 'ct')))
    # Same damped fixed-point iteration and defaults as the V1 ASM driver.
    for iteration in range(20):
        field = array_stability_ratio_field(x, y, tx, ty, ct, u_ref, hub, diameter, .0002, ti, length)
        interp = RegularGridInterpolator((x,y), field['speed_ratio'], bounds_error=True)
        updated = np.asarray(curve_value(curve, u_ref*np.clip(interp(np.c_[tx,ty]),0,1), 'ct'))
        if np.max(np.abs(updated-ct)) <= 1.e-4:
            ct = updated
            break
        ct = .5*(ct+updated)
    field = array_stability_ratio_field(x, y, tx, ty, ct, u_ref, hub, diameter, .0002, ti, length)
    interp = RegularGridInterpolator((x,y), field['speed_ratio'], bounds_error=True)
    site = XRSite(xr.Dataset(data_vars={'Speedup': (('x','y'), field['speed_ratio']),
                                       'P': 1., 'TI': float(ti)}, coords={'x':x,'y':y}), interp_method='linear')
    turbo = pywake_turbopark_speed_ratio(layout, curve, turbine_s, turbine_n, query_s, query_n,
                                        height, u_ref, ti, spatial_site=site)
    # Keep the same unmodified pointwise minimum as the WRF adapter.
    return np.minimum(interp(np.c_[qx,qy]), turbo)

def deficit_geometry(crosswind: np.ndarray, ratio: np.ndarray) -> tuple[float, float, float]:
    order = np.argsort(crosswind)
    x = np.asarray(crosswind, dtype=float)[order]
    deficit = np.clip(1.0 - np.asarray(ratio, dtype=float)[order], 0.0, None)
    area = float(np.trapezoid(deficit, x)) if len(x) > 1 else 0.0
    weight = float(np.sum(deficit))
    if weight <= 1.0e-12:
        return area, float("nan"), 0.0
    center = float(np.sum(x * deficit) / weight)
    width = float(2.0 * np.sqrt(np.sum(deficit * (x - center) ** 2) / weight))
    return area, center, width

def calculate_metrics(profiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["case_id", "flight_label", "transect_role", "leg_id", "model", "model_label"]
    for values, group in profiles.groupby(keys, sort=False):
        case_id, flight_label, role, leg_id, model, model_label = values
        if role != "downstream wake":
            continue
        evaluated = group.loc[np.abs(group["crosswind_bin_km"]) <= 12.0 + 1e-9].copy()
        if len(evaluated) != 121:
            raise ValueError(f'Expected 121 common bins in +/-12 km: {case_id}, leg {leg_id}, got {len(evaluated)}')
        observed = evaluated["observed_u_over_uref"].to_numpy(float)
        modeled = evaluated["model_u_over_uref"].to_numpy(float)
        error = modeled - observed
        obs_area, obs_center, obs_width = deficit_geometry(evaluated["crosswind_bin_km"], observed)
        mod_area, mod_center, mod_width = deficit_geometry(evaluated["crosswind_bin_km"], modeled)
        rows.append(
            {
                "case_id": case_id,
                "flight_label": flight_label,
                "transect_role": role,
                "leg_id": int(leg_id),
                "nominal_distance_km": float(group["nominal_downstream_distance_km"].iloc[0]),
                "actual_distance_km": float(group["actual_downstream_distance_km"].iloc[0]),
                "model": model,
                "model_label": model_label,
                "sample_count": len(evaluated),
                "evaluation_half_width_km": 12.,
                "normalization": "same section U_ref for observations and models",
                "rmse_speed_mps": float(np.sqrt(np.mean((error * evaluated.u_ref_mps)**2))),
                "rmse_background_normalized_speed": float(np.sqrt(np.mean(((evaluated.model_speed_mps - evaluated.observed_speed_mps) / evaluated.background_speed_mps)**2))),
                "rmse_shape_only_legacy": float(np.sqrt(np.mean((modeled - evaluated.observed_u_over_background)**2))),
                "rmse_normalized_speed": float(np.sqrt(np.mean(error**2))),
                "mae_normalized_speed": float(np.mean(np.abs(error))),
                "bias_normalized_speed": float(np.mean(error)),
                "observed_max_deficit": float(np.max(np.clip(1.0 - observed, 0.0, None))),
                "modeled_max_deficit": float(np.max(np.clip(1.0 - modeled, 0.0, None))),
                "observed_deficit_area_km": obs_area,
                "modeled_deficit_area_km": mod_area,
                "observed_deficit_center_km": obs_center,
                "modeled_deficit_center_km": mod_center,
                "observed_deficit_width_km": obs_width,
                "modeled_deficit_width_km": mod_width,
            }
        )
    return pd.DataFrame(rows)

def calculate_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    downstream = metrics.loc[metrics["transect_role"] == "downstream wake"]
    ranking = (
        downstream.groupby(["model", "model_label"], as_index=False)
        .agg(
            downstream_mean_rmse=("rmse_normalized_speed", "mean"),
            downstream_mean_mae=("mae_normalized_speed", "mean"),
            downstream_mean_abs_bias=("bias_normalized_speed", lambda values: float(np.mean(np.abs(values)))),
            downstream_sections=("leg_id", "size"),
        )
        .sort_values("downstream_mean_rmse")
        .reset_index(drop=True)
    )
    ranking.insert(0, "downstream_rank", np.arange(1, len(ranking) + 1))
    return ranking
