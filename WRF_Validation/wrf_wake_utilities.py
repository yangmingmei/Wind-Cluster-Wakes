"""
Shared utilities for wake-model simulations driven by processed WRF cases.

This module holds the code that is common to the Gaussian, TurboPark, and
top-down drivers, plus the standalone offshore DTU 10 MW top-down workflow:
  - reading compact 119 m wind fields and inflow metadata from NPZ files,
  - converting turbine grid indices to WRF map coordinates,
  - plotting and saving model outputs as NPZ, CSV, and optional PNG files.

The model-specific scripts should stay small: they choose the wake model and
its parameters, then call the common batch helpers here.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
from improved_top_down_model import top_down_model  # noqa: E402
from wrf_processed_io import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    TurbineTable,
    WRFMetadata,
    colorbar_ticks_with_endpoints,
    compact_number_formatter,
    grid_centers_km,
    grid_edges_km,
    load_turbines,
    parse_case_label,
    plot_heatmap_png,
    resolve_wake_color_limits,
    axis_ticks_with_endpoints,
    wake_adaptive_x_limits_km,
    wake_axis_ticks_km,
)


DEFAULT_CURVE_FILE = SCRIPT_DIR / "10MW_turbine_curve.xlsx"
DEFAULT_LOCATION_FILE = SCRIPT_DIR / "offshore_turbine_location.xlsx"
DEFAULT_OFFSHORE_OUTPUT_DIR = SCRIPT_DIR / "top_down_results"
DEFAULT_TURBINES = DEFAULT_RESULTS_DIR / "windturbines-ij.txt"
STABLE_NEUTRAL_DEFICIT_LENGTH = 1.0e9


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


def load_offshore_locations(location_file: Path) -> pd.DataFrame:
    """Load offshore turbine coordinates from the original location workbook."""

    turbines_raw = pd.read_excel(location_file, sheet_name=0, header=3)
    if turbines_raw.shape[1] < 5:
        raise ValueError(f"{location_file} must contain at least five columns.")

    turbines = turbines_raw.iloc[:, :5].copy()
    turbines.columns = ["turbine_id", "farm_id", "farm_name", "x_m", "y_m"]
    turbines["x_m"] = pd.to_numeric(turbines["x_m"], errors="coerce")
    turbines["y_m"] = pd.to_numeric(turbines["y_m"], errors="coerce")
    turbines = turbines.dropna(subset=["x_m", "y_m"]).reset_index(drop=True)
    if turbines.empty:
        raise ValueError(f"No turbine coordinates were found in {location_file}")
    return turbines


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


def configure_matplotlib(font_size: float, title_font_size: float) -> None:
    """Apply a compact, consistent Matplotlib style for batch figures."""

    for font_file in [
        Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\timesbd.ttf"),
        Path(r"C:\Windows\Fonts\timesi.ttf"),
        Path(r"C:\Windows\Fonts\timesbi.ttf"),
    ]:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.titlesize"] = title_font_size
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    plt.rcParams["figure.titlesize"] = title_font_size
    plt.rcParams["axes.linewidth"] = 0.45
    plt.rcParams["xtick.major.width"] = 0.45
    plt.rcParams["ytick.major.width"] = 0.45
    plt.rcParams["xtick.major.size"] = 2.0
    plt.rcParams["ytick.major.size"] = 2.0


def safe_output_stem(wrfout_file: Path, target_height_m: float, model_suffix: str) -> str:
    """Build a filesystem-safe output stem that is unique per case/model."""

    safe_source_name = re.sub(
        r"_\d+m_wind_speed$",
        "",
        wrfout_file.stem.replace(":", "-"),
    )
    return f"{safe_source_name}_{target_height_m:.0f}m_{model_suffix}"


def find_wrfout_files(results_dir: Path, pattern: str) -> list[Path]:
    """Find processed WRF NPZ files in deterministic filename order."""

    wrfout_files = sorted(path for path in results_dir.glob(pattern) if path.is_file())
    if not wrfout_files:
        raise FileNotFoundError(f"No processed WRF files matched {results_dir / pattern}")
    return wrfout_files


def parse_monin_obukhov_length(wrfout_file: Path, neutral_value: float) -> float:
    """Parse Lm+500, Lm-200, Lm12p5, or Lmneutral from a WRF filename."""

    match = re.search(r"Lm(neutral|[+-]\d+(?:p\d+)?)", wrfout_file.name)
    if not match:
        return neutral_value

    raw_value = match.group(1)
    if raw_value == "neutral":
        return neutral_value

    return float(raw_value.replace("p", "."))


def uv_to_met_direction_deg(u_mps: float, v_mps: float) -> float:
    """Convert east/north wind components to meteorological direction."""

    return float((270.0 - np.degrees(np.arctan2(v_mps, u_mps))) % 360.0)


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


def rotate_to_real_frame(x_rot: np.ndarray, y_rot: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate downstream-frame coordinates back to the original map frame."""

    x_real = x_rot * np.cos(theta) - y_rot * np.sin(theta)
    y_real = x_rot * np.sin(theta) + y_rot * np.cos(theta)
    return x_real, y_real


def simulate_top_down(
    x_m: np.ndarray,
    y_m: np.ndarray,
    wind_speed: float,
    wind_direction: float,
    curve: dict[str, np.ndarray | float],
    grid_resolution: float,
    padding: float,
    roughness: float,
    boundary_layer_height: float,
    monin_obukhov_length: float,
    cft_scale: float,
) -> dict[str, np.ndarray | float]:
    """Run the standalone offshore DTU 10 MW top-down simulation."""

    hub_height = float(curve["hub_height"])
    rotor_diameter = float(curve["rotor_diameter"])
    ct = float(curve_value(curve, wind_speed, "ct"))

    planform_area, median_spacing = infer_planform_area(x_m, y_m)
    influence_radius = np.sqrt(planform_area / np.pi) * 2.0
    rotor_area = np.pi * rotor_diameter**2 / 4.0
    turbine_cft = cft_scale * ct * rotor_area / planform_area

    tx_rot, ty_rot, theta = rotate_to_wind_frame(x_m, y_m, wind_direction)
    x_axis = np.arange(tx_rot.min() - padding, tx_rot.max() + padding + grid_resolution, grid_resolution)
    y_axis = np.arange(ty_rot.min() - padding, ty_rot.max() + padding + grid_resolution, grid_resolution)
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
            -1.0,
            -1.0,
        )
    )

    tree = cKDTree(np.column_stack((tx_rot, ty_rot)))
    grid_points = np.column_stack((x_grid_rot.ravel(), y_grid_rot.ravel()))
    distances, _ = tree.query(grid_points)
    cft_field = np.where(distances <= influence_radius, turbine_cft, 0.0).reshape(x_grid_rot.shape)

    u_field = np.full_like(x_grid_rot, u_free, dtype=float)
    dx = float(grid_resolution)
    nx, ny = x_grid_rot.shape

    for j in range(ny):
        cft_row = cft_field[:, j]
        in_farm = False
        starts: list[int] = []
        ends: list[int] = []

        for i in range(nx):
            if cft_row[i] > 0.0 and not in_farm:
                in_farm = True
                starts.append(i)
            elif cft_row[i] <= 0.0 and in_farm:
                in_farm = False
                ends.append(i - 1)
        if in_farm:
            ends.append(nx - 1)

        for i in range(nx):
            deficit = 0.0
            for start, end in zip(starts, ends):
                if start > i:
                    break
                segment_end = min(end, i)
                farm_dist = (segment_end - start + 1) * dx
                wake_dist = (i - end) * dx if i > end else 0.0
                cft_avg = float(np.mean(cft_row[start : segment_end + 1]))
                u_top_down = top_down_model(
                    wind_speed,
                    roughness,
                    boundary_layer_height,
                    monin_obukhov_length,
                    cft_avg,
                    hub_height,
                    rotor_diameter,
                    farm_dist,
                    wake_dist,
                )
                deficit += u_free - float(u_top_down)
            u_field[i, j] = u_free - deficit

    interpolator = RegularGridInterpolator(
        (x_axis, y_axis),
        u_field,
        bounds_error=False,
        fill_value=float(u_free),
    )
    turbine_ws = interpolator(np.column_stack((tx_rot, ty_rot)))
    turbine_power_kw = curve_value(curve, turbine_ws, "power_kw")
    turbine_cp = curve_value(curve, turbine_ws, "cp")

    x_grid_real, y_grid_real = rotate_to_real_frame(x_grid_rot, y_grid_rot, theta)

    return {
        "u_field": u_field,
        "u_free": float(u_free),
        "normalized_u": u_field / float(u_free),
        "x_grid_m": x_grid_real,
        "y_grid_m": y_grid_real,
        "x_grid_rot_m": x_grid_rot,
        "y_grid_rot_m": y_grid_rot,
        "turbine_ws": turbine_ws,
        "turbine_power_kw": turbine_power_kw,
        "turbine_cp": turbine_cp,
        "ct_at_inflow": ct,
        "turbine_cft": turbine_cft,
        "planform_area_m2": planform_area,
        "median_spacing_m": median_spacing,
        "influence_radius_m": influence_radius,
    }


def save_outputs(
    output_dir: Path,
    turbines: pd.DataFrame,
    result: dict[str, np.ndarray | float],
    wind_speed: float,
    wind_direction: float,
    make_plot: bool,
) -> None:
    """Save standalone offshore top-down CSV, NPZ, and optional SVG outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    case_name = f"top_down_ws{wind_speed:.2f}_wd{wind_direction:.1f}".replace(".", "p")

    turbine_results = turbines.copy()
    turbine_results["effective_ws_mps"] = np.asarray(result["turbine_ws"])
    turbine_results["cp"] = np.asarray(result["turbine_cp"])
    turbine_results["power_kw"] = np.asarray(result["turbine_power_kw"])
    turbine_results.to_csv(output_dir / f"{case_name}_turbine_results.csv", index=False, encoding="utf-8-sig")

    np.savez(
        output_dir / f"{case_name}_flow_data.npz",
        avg_matrix=np.asarray(result["normalized_u"]),
        u_field=np.asarray(result["u_field"]),
        X=np.asarray(result["x_grid_m"]),
        Y=np.asarray(result["y_grid_m"]),
        x_array=turbines["x_m"].to_numpy(float),
        y_array=turbines["y_m"].to_numpy(float),
        turbine_ws=np.asarray(result["turbine_ws"]),
        turbine_power_kw=np.asarray(result["turbine_power_kw"]),
        wsp=wind_speed,
        direction=wind_direction,
        u_free=float(result["u_free"]),
        ct_at_inflow=float(result["ct_at_inflow"]),
        turbine_cft=float(result["turbine_cft"]),
        planform_area_m2=float(result["planform_area_m2"]),
        influence_radius_m=float(result["influence_radius_m"]),
    )

    if not make_plot:
        return

    colors_bg = ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#ffffff"]
    cmap_wind = LinearSegmentedColormap.from_list("top_down_blues", colors_bg, N=256)

    for font_file in [
        Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\timesbd.ttf"),
        Path(r"C:\Windows\Fonts\timesi.ttf"),
        Path(r"C:\Windows\Fonts\timesbi.ttf"),
    ]:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor="white")
    ax.set_aspect("equal")
    pc = ax.pcolormesh(
        np.asarray(result["x_grid_m"]) / 1000.0,
        np.asarray(result["y_grid_m"]) / 1000.0,
        np.asarray(result["normalized_u"]),
        cmap=cmap_wind,
        vmin=0.8,
        vmax=1.0,
        shading="nearest",
        rasterized=True,
    )
    ax.scatter(turbines["x_m"] / 1000.0, turbines["y_m"] / 1000.0, s=12, c="#d62728", zorder=5, label="DTU 10 MW")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title(
        f"Offshore top-down model: wind speed={wind_speed:.2f} m/s, "
        f"wind direction={wind_direction:.1f}°"
    )
    ax.legend(loc="upper right", frameon=True)
    cb = fig.colorbar(pc, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label(
        "Normalized\nwind speed",
        labelpad=1.5,
        rotation=0,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{case_name}_flow_map.svg", dpi=600, bbox_inches="tight")
    plt.close(fig)


def extract_hub_wind_components(
    wrfout_file: Path,
    time_index: int,
    target_height_m: float,
) -> tuple[np.ndarray, np.ndarray, WRFMetadata]:
    """Reconstruct constant-direction U/V fields from a processed speed NPZ."""

    del time_index  # Each compact file contains one already-selected time slice.
    with np.load(wrfout_file, allow_pickle=False) as data:
        speed = np.asarray(data["wind_speed_mps"], dtype=float)
        stored_height = float(np.asarray(data["height_agl_m"]).item())
        if not np.isclose(stored_height, target_height_m):
            raise ValueError(
                f"{wrfout_file.name} contains {stored_height:g} m data, "
                f"but --height={target_height_m:g} was requested"
            )
        direction = float(np.asarray(data["freestream_direction_deg"]).item())
        metadata = WRFMetadata(
            nx=int(speed.shape[1]),
            ny=int(speed.shape[0]),
            dx_m=float(np.asarray(data["dx_m"]).item()),
            dy_m=float(np.asarray(data["dy_m"]).item()),
            time_label=str(np.asarray(data["time_label"]).item()),
        )

    direction_rad = np.radians(direction)
    u_hub = -speed * np.sin(direction_rad)
    v_hub = -speed * np.cos(direction_rad)
    return u_hub, v_hub, metadata


def extract_boundary_layer_height(wrfout_file: Path, time_index: int) -> float:
    """Read the domain-mean PBLH retained in a processed WRF case."""

    del time_index
    with np.load(wrfout_file, allow_pickle=False) as data:
        if "boundary_layer_height_m" not in data.files:
            raise KeyError(
                f"{wrfout_file} has no boundary_layer_height_m metadata; "
                "pass --boundary-layer-height explicitly"
            )
        value = float(np.asarray(data["boundary_layer_height_m"]).item())
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{wrfout_file} has an invalid boundary-layer height: {value}")
    return value


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


def estimate_freestream_from_wrf(
    u_hub: np.ndarray,
    v_hub: np.ndarray,
    turbines: TurbineTable,
    metadata: WRFMetadata,
    upstream_buffer_m: float,
    neutral_direction_fallback_deg: float = 270.0,
) -> dict[str, float | int]:
    """Estimate case inflow from WRF cells upstream of the first turbine row."""

    domain_u = float(np.nanmean(u_hub))
    domain_v = float(np.nanmean(v_hub))
    if np.hypot(domain_u, domain_v) <= 1.0e-12:
        initial_direction = neutral_direction_fallback_deg
    else:
        initial_direction = uv_to_met_direction_deg(domain_u, domain_v)

    _, _, x_grid_m, y_grid_m = wrf_center_coordinates(metadata)
    turbine_x_m, turbine_y_m = turbine_coordinates_m(turbines, metadata)
    x_rot, _, _ = rotate_to_wind_frame(x_grid_m, y_grid_m, initial_direction)
    turbine_x_rot, _, _ = rotate_to_wind_frame(turbine_x_m, turbine_y_m, initial_direction)

    upstream_limit = float(np.nanmin(turbine_x_rot) - upstream_buffer_m)
    mask = x_rot <= upstream_limit

    # Keep the estimate robust for small domains or unusual wind directions.
    if int(np.count_nonzero(mask)) < 10:
        fallback_limit = float(np.nanpercentile(x_rot, 15.0))
        mask = x_rot <= fallback_limit
    if int(np.count_nonzero(mask)) < 10:
        mask = np.isfinite(u_hub) & np.isfinite(v_hub)

    u_free = float(np.nanmean(np.where(mask, u_hub, np.nan)))
    v_free = float(np.nanmean(np.where(mask, v_hub, np.nan)))
    wind_speed = float(np.hypot(u_free, v_free))
    wind_direction = uv_to_met_direction_deg(u_free, v_free)

    return {
        "wind_speed_mps": wind_speed,
        "wind_direction_deg": wind_direction,
        "u_mps": u_free,
        "v_mps": v_free,
        "sample_count": int(np.count_nonzero(mask)),
        "initial_direction_deg": float(initial_direction),
        "upstream_limit_m": upstream_limit,
    }


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
    """Replace non-finite values and clip PyWake numerical over/undershoots."""

    upper = max(float(freestream_speed) * 1.05, 0.1)
    cleaned = np.nan_to_num(speed_2d, nan=float(freestream_speed), posinf=upper, neginf=0.0)
    return np.clip(cleaned, 0.0, upper)


def resolve_pywake_color_limits(
    plot_data: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    freestream_speed: float,
    plot_normalized: bool,
) -> tuple[float, float]:
    """Use WRF-plot color limits, with fixed comparable bounds for normalized maps."""

    if plot_normalized:
        resolved_vmin = 0.8 if vmin is None else float(vmin)
        resolved_vmax = 1.0 if vmax is None else float(vmax)
        if resolved_vmax <= resolved_vmin:
            resolved_vmax = resolved_vmin + 0.01
        return resolved_vmin, resolved_vmax

    ref = float(freestream_speed)
    if np.isfinite(ref) and ref > 0.0:
        resolved_vmin = 0.8 * ref if vmin is None else float(vmin)
        resolved_vmax = ref if vmax is None else float(vmax)
        if resolved_vmax <= resolved_vmin:
            resolved_vmax = resolved_vmin + max(0.01, 0.01 * abs(resolved_vmin))
        return resolved_vmin, resolved_vmax

    return resolve_wake_color_limits(plot_data, vmin=vmin, vmax=vmax)


def comparable_speed_color_limits(
    reference_speed: float,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float | None, float | None]:
    """Return comparable speed limits, unless the user supplied explicit bounds."""

    ref = float(reference_speed)
    if not (np.isfinite(ref) and ref > 0.0):
        return vmin, vmax

    resolved_vmin = 0.8 * ref if vmin is None else float(vmin)
    resolved_vmax = ref if vmax is None else float(vmax)
    if resolved_vmax <= resolved_vmin:
        resolved_vmax = resolved_vmin + max(0.01, 0.01 * abs(resolved_vmin))
    return resolved_vmin, resolved_vmax


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


def plot_pywake_png(
    output_file: Path,
    result: dict[str, np.ndarray | float],
    turbines: TurbineTable,
    metadata: WRFMetadata,
    target_height_m: float,
    case_label: str,
    freestream_speed: float,
    freestream_direction: float,
    turbine_name: str,
    vmin: float | None,
    vmax: float | None,
    figure_width_in: float,
    figure_height_in: float,
    font_size: float,
    title_font_size: float,
    cmap: str,
    x_min_km: float,
    dpi: int,
    plot_normalized: bool,
    model_key: str,
    model_title: str,
) -> None:
    """Save one PyWake heatmap PNG using the standard visual style."""

    configure_matplotlib(font_size=font_size, title_font_size=title_font_size)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if plot_normalized:
        plot_data = np.asarray(result[f"{model_key}_normalized_wind_speed"], dtype=float)
        colorbar_title = "Normalized wind speed"
        colorbar_label_rotation = 0
        plot_figure_width_in = figure_width_in
    else:
        plot_data = np.asarray(result[f"{model_key}_wind_speed_mps"], dtype=float)
        colorbar_title = "Wind speed " + r"($\mathrm{m/s}$)"
        colorbar_label_rotation = 90
        plot_figure_width_in = figure_width_in + 0.35

    vmin, vmax = resolve_pywake_color_limits(
        plot_data,
        vmin=vmin,
        vmax=vmax,
        freestream_speed=float(freestream_speed),
        plot_normalized=plot_normalized,
    )

    x_edges = grid_edges_km(metadata.nx, metadata.dx_m)
    y_edges = grid_edges_km(metadata.ny, metadata.dy_m)
    x_centers = grid_centers_km(metadata.nx, metadata.dx_m)
    y_centers = grid_centers_km(metadata.ny, metadata.dy_m)

    fig, ax = plt.subplots(figsize=(plot_figure_width_in, figure_height_in), facecolor="white")
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        plot_data,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.scatter(
        x_centers[turbines.i0],
        y_centers[turbines.j0],
        s=9.0,
        marker="^",
        c="#d62728",
        edgecolors="white",
        linewidths=0.25,
        zorder=5,
        label=turbine_name,
    )

    ax.set_aspect("equal", adjustable="box")
    x_plot_min, x_plot_max = wake_adaptive_x_limits_km(turbines, metadata, freestream_direction, x_min_km)
    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_xlabel(r"$\mathit{x}$ (km)", labelpad=1.5)
    ax.set_ylabel(r"$\mathit{y}$ (km)", labelpad=1.5)
    ax.set_xticks(wake_axis_ticks_km(x_plot_min, x_plot_max))
    ax.set_yticks(axis_ticks_with_endpoints(float(y_edges[0]), float(y_edges[-1])))
    ax.xaxis.set_major_formatter(FuncFormatter(compact_number_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_number_formatter))
    ax.tick_params(axis="both", pad=1.5)

    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.030, pad=0.018, aspect=18)
    colorbar.set_ticks(colorbar_ticks_with_endpoints(vmin, vmax))
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(compact_number_formatter))
    colorbar.ax.tick_params(labelsize=font_size, width=0.45, length=2.0, pad=1.5)
    colorbar.outline.set_linewidth(0.45)
    colorbar.set_label(
        colorbar_title,
        labelpad=8.0,
        rotation=colorbar_label_rotation,
    )
    ax.legend(loc="upper right", frameon=False, handlelength=0.8, handletextpad=0.3, borderpad=0.0)

    fig.subplots_adjust(left=0.125, right=0.870, bottom=0.245, top=0.835)
    fig.savefig(output_file, format="png", dpi=dpi)
    plt.close(fig)


def write_turbine_results_csv(
    output_file: Path,
    turbines: TurbineTable,
    result: dict[str, np.ndarray | float],
    include_ct: bool,
) -> None:
    """Write one turbine-level CSV for a model result dictionary."""

    data: dict[str, Any] = {
        "turbine_id": np.arange(1, len(turbines.i0) + 1),
        "i": turbines.i,
        "j": turbines.j,
        "type_id": turbines.type_id,
        "x_m": np.asarray(result["turbine_x_m"]),
        "y_m": np.asarray(result["turbine_y_m"]),
        "effective_ws_mps": np.asarray(result["turbine_ws_mps"]),
    }
    if include_ct:
        data["ct"] = np.asarray(result["turbine_ct"])
    data["cp"] = np.asarray(result["turbine_cp"])
    data["power_kw"] = np.asarray(result["turbine_power_kw"])

    pd.DataFrame(data).to_csv(output_file, index=False, encoding="utf-8-sig")


def save_pywake_case_outputs(
    output_dir: Path,
    wrfout: Path,
    metadata: WRFMetadata,
    turbines: TurbineTable,
    result: dict[str, np.ndarray | float],
    reference_speed: np.ndarray,
    freestream: dict[str, float | int],
    monin_obukhov_length: float,
    target_height_m: float,
    plot_settings: argparse.Namespace,
    model_key: str,
    model_title: str,
    extra_npz: dict[str, Any] | None = None,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, str | float | int]:
    """Save one PyWake case as PNG, NPZ, turbine CSV, and summary row."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = safe_output_stem(wrfout, target_height_m, model_key)
    png_file = output_dir / f"{output_stem}.png"
    npz_file = output_dir / f"{output_stem}.npz"
    turbine_csv = output_dir / f"{output_stem}_turbines.csv"

    plot_pywake_png(
        output_file=png_file,
        result=result,
        turbines=turbines,
        metadata=metadata,
        target_height_m=target_height_m,
        case_label=parse_case_label(wrfout),
        freestream_speed=float(freestream["wind_speed_mps"]),
        freestream_direction=float(freestream["wind_direction_deg"]),
        turbine_name=plot_settings.turbine_name,
        vmin=plot_settings.vmin,
        vmax=plot_settings.vmax,
        figure_width_in=plot_settings.figure_width,
        figure_height_in=plot_settings.figure_height,
        font_size=plot_settings.font_size,
        title_font_size=plot_settings.title_font_size,
        cmap=plot_settings.cmap,
        x_min_km=plot_settings.x_min_km,
        dpi=plot_settings.dpi,
        plot_normalized=plot_settings.plot_normalized,
        model_key=model_key,
        model_title=model_title,
    )
    write_turbine_results_csv(turbine_csv, turbines, result, include_ct=True)

    model_speed = np.asarray(result[f"{model_key}_wind_speed_mps"])
    npz_payload: dict[str, Any] = {
        "wrfout_file": str(wrfout),
        "case_label": parse_case_label(wrfout),
        "time_label": metadata.time_label,
        "hub_height_m": target_height_m,
        "nx": metadata.nx,
        "ny": metadata.ny,
        "dx_m": metadata.dx_m,
        "dy_m": metadata.dy_m,
        "freestream_speed_mps": float(freestream["wind_speed_mps"]),
        "freestream_direction_deg": float(freestream["wind_direction_deg"]),
        "freestream_u_mps": float(freestream["u_mps"]),
        "freestream_v_mps": float(freestream["v_mps"]),
        "freestream_sample_count": int(freestream["sample_count"]),
        "monin_obukhov_length_m": monin_obukhov_length,
        f"{model_key}_wind_speed_mps": model_speed,
        f"{model_key}_normalized_wind_speed": np.asarray(result[f"{model_key}_normalized_wind_speed"]),
        "wrf_reference_wind_speed_mps": reference_speed,
        "turbine_i": turbines.i,
        "turbine_j": turbines.j,
        "turbine_i0": turbines.i0,
        "turbine_j0": turbines.j0,
        "turbine_x_m": np.asarray(result["turbine_x_m"]),
        "turbine_y_m": np.asarray(result["turbine_y_m"]),
        "turbine_ws_mps": np.asarray(result["turbine_ws_mps"]),
        "turbine_power_kw": np.asarray(result["turbine_power_kw"]),
        "turbine_ct": np.asarray(result["turbine_ct"]),
        "turbine_cp": np.asarray(result["turbine_cp"]),
        "x_axis_m": np.asarray(result["x_axis_m"]),
        "y_axis_m": np.asarray(result["y_axis_m"]),
        "x_grid_m": np.asarray(result["x_grid_m"]),
        "y_grid_m": np.asarray(result["y_grid_m"]),
        "ct_at_inflow": float(result["ct_at_inflow"]),
        "power_at_inflow_kw": float(result["power_at_inflow_kw"]),
        "turbulence_intensity": float(plot_settings.turbulence_intensity),
    }
    if extra_npz:
        npz_payload.update(extra_npz)
    np.savez_compressed(npz_file, **npz_payload)

    row: dict[str, Any] = {
        "wrfout_file": wrfout.name,
        "time_label": metadata.time_label,
        "hub_height_m": target_height_m,
        "freestream_speed_mps": float(freestream["wind_speed_mps"]),
        "freestream_direction_deg": float(freestream["wind_direction_deg"]),
        "freestream_u_mps": float(freestream["u_mps"]),
        "freestream_v_mps": float(freestream["v_mps"]),
        "freestream_sample_count": int(freestream["sample_count"]),
        "monin_obukhov_length_m": monin_obukhov_length,
        "turbulence_intensity": float(plot_settings.turbulence_intensity),
        f"{model_key}_u_min_mps": float(np.nanmin(model_speed)),
        f"{model_key}_u_mean_mps": float(np.nanmean(model_speed)),
        f"{model_key}_u_max_mps": float(np.nanmax(model_speed)),
        "wrf_reference_u_min_mps": float(np.nanmin(reference_speed)),
        "wrf_reference_u_mean_mps": float(np.nanmean(reference_speed)),
        "wrf_reference_u_max_mps": float(np.nanmax(reference_speed)),
        "mean_turbine_ws_mps": float(np.nanmean(result["turbine_ws_mps"])),
        "total_power_mw": float(np.nansum(result["turbine_power_kw"]) / 1000.0),
        "png_file": str(png_file),
        "npz_file": str(npz_file),
        "turbine_csv": str(turbine_csv),
    }
    if extra_summary:
        row.update(extra_summary)
    return row


def write_summary_csv(output_file: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of summary dictionaries to CSV."""

    if not rows:
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_common_wrf_model_args(parser: argparse.ArgumentParser, output_dir: Path, output_kind: str) -> None:
    """Register CLI arguments shared by the four model drivers."""

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing processed WRF NPZ files.",
    )
    parser.add_argument(
        "--pattern",
        default="*_wind_speed.npz",
        help="Filename pattern used inside results-dir.",
    )
    parser.add_argument("--turbines", type=Path, default=DEFAULT_TURBINES, help="Whitespace txt file with i j type columns.")
    parser.add_argument("--curve-file", type=Path, default=DEFAULT_CURVE_FILE, help="DTU 10 MW curve workbook.")
    parser.add_argument("--output-dir", type=Path, default=output_dir, help=f"Output directory for {output_kind} files.")
    parser.add_argument("--height", type=float, default=119.0, help="Hub height above ground level in meters.")
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Reserved for CLI compatibility; processed files contain one time slice.",
    )
    parser.add_argument("--index-base", type=int, choices=(0, 1), default=1, help="Index base used by the turbine i/j file.")
    parser.add_argument("--upstream-buffer", type=float, default=5000.0, help="Upstream sampling distance before first turbine, in meters.")
    parser.add_argument("--neutral-monin-obukhov-length", type=float, default=1.0e9, help="L value used for Lmneutral filenames.")
    parser.add_argument("--monin-obukhov-length", type=float, default=None, help="Override L saved for every case.")
    parser.add_argument("--turbine-name", default="DTU 10 MW", help="Name used in plot legend.")
    parser.add_argument("--vmin", type=float, default=None, help="Optional heatmap lower color limit.")
    parser.add_argument("--vmax", type=float, default=None, help="Optional heatmap upper color limit.")
    parser.add_argument("--figure-width", type=float, default=3.80, help="Figure width in inches.")
    parser.add_argument("--figure-height", type=float, default=1.80, help="Figure height in inches.")
    parser.add_argument("--font-size", type=float, default=8.0, help="Main plot font size in points.")
    parser.add_argument("--title-font-size", type=float, default=8.2, help="Title font size in points.")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name.")
    parser.add_argument("--x-min-km", type=float, default=40.0, help="Lower x-axis limit in kilometers.")
    parser.add_argument("--dpi", type=int, default=600, help="Output resolution.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of processed cases to run.")


def resolve_case_monin_obukhov_length(args: argparse.Namespace, wrfout: Path) -> float:
    """Return the user override or the L value parsed from the WRF filename."""

    if args.monin_obukhov_length is not None:
        return float(args.monin_obukhov_length)
    return parse_monin_obukhov_length(wrfout, args.neutral_monin_obukhov_length)


def load_wrf_case_inputs(
    wrfout: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, WRFMetadata, TurbineTable, dict[str, float | int]]:
    """Load one processed wind field, turbine table, and saved inflow metadata."""

    u_hub, v_hub, metadata = extract_hub_wind_components(wrfout, args.time_index, args.height)
    turbines = load_turbines(args.turbines.resolve(), metadata.nx, metadata.ny, args.index_base)
    with np.load(wrfout, allow_pickle=False) as data:
        freestream = {
            "wind_speed_mps": float(np.asarray(data["freestream_speed_mps"]).item()),
            "wind_direction_deg": float(np.asarray(data["freestream_direction_deg"]).item()),
            "u_mps": float(np.asarray(data["freestream_u_mps"]).item()),
            "v_mps": float(np.asarray(data["freestream_v_mps"]).item()),
            "sample_count": int(np.asarray(data["freestream_sample_count"]).item()),
            "initial_direction_deg": float(np.asarray(data["freestream_direction_deg"]).item()),
            "upstream_limit_m": float("nan"),
        }
    return u_hub, v_hub, metadata, turbines, freestream


def get_limited_wrfout_files(args: argparse.Namespace) -> list[Path]:
    """Resolve and optionally truncate the processed WRF case list."""

    wrfout_files = find_wrfout_files(args.results_dir.resolve(), args.pattern)
    if args.limit is not None:
        wrfout_files = wrfout_files[: max(args.limit, 0)]
    return wrfout_files


def axis_covering(values: np.ndarray, spacing_m: float) -> np.ndarray:
    """Create a regular axis that covers all supplied coordinates."""

    axis_min = np.floor(float(np.nanmin(values)) / spacing_m) * spacing_m
    axis_max = np.ceil(float(np.nanmax(values)) / spacing_m) * spacing_m
    return np.arange(axis_min, axis_max + 0.5 * spacing_m, spacing_m, dtype=float)


def speed_bounds_from_reference(
    fallback_speed: float,
    reference_speed: np.ndarray | None = None,
) -> tuple[float, float]:
    """Build conservative scalar wind-speed bounds for post-processing."""

    fallback = max(float(fallback_speed), 0.0)
    lower = 0.0
    upper = max(0.1, 1.10 * fallback)

    if reference_speed is not None:
        ref = np.asarray(reference_speed, dtype=float)
        finite_ref = ref[np.isfinite(ref) & (ref >= 0.0)]
        if finite_ref.size:
            ref_hi = float(np.nanpercentile(finite_ref, 99.9))
            upper = max(upper, ref_hi + max(0.5, 0.05 * ref_hi))

    return lower, upper


def clean_speed_field(
    speed: np.ndarray,
    fallback_speed: float,
    reference_speed: np.ndarray | None = None,
) -> np.ndarray:
    """Replace non-finite values and clip negative or extreme wind speeds."""

    lower, upper = speed_bounds_from_reference(fallback_speed, reference_speed)
    cleaned = np.asarray(speed, dtype=float).copy()
    cleaned = np.nan_to_num(cleaned, nan=fallback_speed, posinf=upper, neginf=lower)
    return np.clip(cleaned, lower, upper)


def blend_with_reference_speed(
    top_down_speed: np.ndarray,
    reference_speed: np.ndarray | None,
    fallback_speed: float,
    reference_weight: float,
) -> np.ndarray:
    """Optionally blend the top-down field toward the WRF hub-height speed field."""

    top_down_clean = clean_speed_field(top_down_speed, fallback_speed, reference_speed)
    if reference_speed is None or reference_weight <= 0.0:
        return top_down_clean

    weight = float(np.clip(reference_weight, 0.0, 1.0))
    reference_clean = clean_speed_field(reference_speed, fallback_speed, reference_speed)
    blended = (1.0 - weight) * top_down_clean + weight * reference_clean
    return clean_speed_field(blended, fallback_speed, reference_speed)


def stable_limited_top_down_speed(
    wind_speed: float,
    roughness: float,
    boundary_layer_height: float,
    monin_obukhov_length: float,
    cft: float,
    hub_height: float,
    rotor_diameter: float,
    farm_dist: float,
    wake_dist: float,
) -> float:
    """Return top-down speed while preventing stable-stratification acceleration artifacts."""

    u_top_down = float(
        top_down_model(
            wind_speed,
            roughness,
            boundary_layer_height,
            monin_obukhov_length,
            cft,
            hub_height,
            rotor_diameter,
            farm_dist,
            wake_dist,
        )
    )

    if (
        np.isfinite(u_top_down)
        and np.isfinite(monin_obukhov_length)
        and 0.0 < float(monin_obukhov_length) < STABLE_NEUTRAL_DEFICIT_LENGTH
        and cft > 0.0
        and farm_dist > 0.0
    ):
        neutral_speed = float(
            top_down_model(
                wind_speed,
                roughness,
                boundary_layer_height,
                STABLE_NEUTRAL_DEFICIT_LENGTH,
                cft,
                hub_height,
                rotor_diameter,
                farm_dist,
                wake_dist,
            )
        )
        if np.isfinite(neutral_speed):
            u_top_down = min(u_top_down, neutral_speed)

    return u_top_down


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


def save_top_down_case_outputs(
    output_dir: Path,
    wrfout: Path,
    metadata: WRFMetadata,
    turbines: TurbineTable,
    result: dict[str, np.ndarray | float],
    freestream: dict[str, float | int],
    wrf_reference_speed: np.ndarray,
    monin_obukhov_length: float,
    boundary_layer_height_m: float,
    cft_scale: float,
    target_height_m: float,
    plot_settings: argparse.Namespace,
) -> dict[str, str | float | int]:
    """Save one top-down case as PNG, NPZ, turbine CSV, and summary row."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = safe_output_stem(wrfout, target_height_m, "top_down")
    png_file = output_dir / f"{output_stem}.png"
    npz_file = output_dir / f"{output_stem}.npz"
    turbine_csv = output_dir / f"{output_stem}_turbines.csv"

    u_field_wrf_raw = np.asarray(result["u_field_wrf_raw"])
    u_field_wrf_blended = np.asarray(result["u_field_wrf_blended"])
    plot_vmin, plot_vmax = comparable_speed_color_limits(
        reference_speed=float(result["u_free"]),
        vmin=plot_settings.vmin,
        vmax=plot_settings.vmax,
    )
    if not getattr(plot_settings, "no_png", False):
        plot_heatmap_png(
            output_file=png_file,
            speed_2d=u_field_wrf_raw,
            turbines=turbines,
            metadata=metadata,
            target_height_m=target_height_m,
            case_label=parse_case_label(wrfout),
            turbine_name=plot_settings.turbine_name,
            vmin=plot_vmin,
            vmax=plot_vmax,
            figure_width_in=plot_settings.figure_width,
            figure_height_in=plot_settings.figure_height,
            font_size=plot_settings.font_size,
            title_font_size=plot_settings.title_font_size,
            cmap=plot_settings.cmap,
            x_min_km=plot_settings.x_min_km,
            dpi=plot_settings.dpi,
            wind_direction_deg=float(freestream["wind_direction_deg"]),
            color_reference_speed=float(result["u_free"]),
        )
    write_turbine_results_csv(turbine_csv, turbines, result, include_ct=False)

    np.savez_compressed(
        npz_file,
        wrfout_file=str(wrfout),
        case_label=parse_case_label(wrfout),
        time_label=metadata.time_label,
        hub_height_m=target_height_m,
        nx=metadata.nx,
        ny=metadata.ny,
        dx_m=metadata.dx_m,
        dy_m=metadata.dy_m,
        freestream_speed_mps=float(freestream["wind_speed_mps"]),
        freestream_direction_deg=float(freestream["wind_direction_deg"]),
        freestream_u_mps=float(freestream["u_mps"]),
        freestream_v_mps=float(freestream["v_mps"]),
        freestream_sample_count=int(freestream["sample_count"]),
        monin_obukhov_length_m=monin_obukhov_length,
        boundary_layer_height_m=boundary_layer_height_m,
        cft_scale=cft_scale,
        reference_blend_weight=float(result["reference_blend_weight"]),
        wrf_reference_speed_mps=np.asarray(wrf_reference_speed),
        top_down_wind_speed_mps=np.asarray(result["u_field_wrf_raw"]),
        top_down_normalized_wind_speed=np.asarray(result["normalized_u_wrf_raw"]),
        top_down_wind_speed_raw_mps=np.asarray(result["u_field_wrf_raw"]),
        top_down_wind_speed_blended_mps=np.asarray(result["u_field_wrf_blended"]),
        top_down_normalized_wind_speed_raw=np.asarray(result["normalized_u_wrf_raw"]),
        top_down_normalized_wind_speed_blended=np.asarray(result["normalized_u_wrf_blended"]),
        top_down_u_free_mps=float(result["u_free"]),
        turbine_i=turbines.i,
        turbine_j=turbines.j,
        turbine_i0=turbines.i0,
        turbine_j0=turbines.j0,
        turbine_x_m=np.asarray(result["turbine_x_m"]),
        turbine_y_m=np.asarray(result["turbine_y_m"]),
        turbine_ws_mps=np.asarray(result["turbine_ws_mps"]),
        turbine_power_kw=np.asarray(result["turbine_power_kw"]),
        x_grid_wrf_m=np.asarray(result["x_grid_wrf_m"]),
        y_grid_wrf_m=np.asarray(result["y_grid_wrf_m"]),
        x_axis_rot_m=np.asarray(result["x_axis_rot_m"]),
        y_axis_rot_m=np.asarray(result["y_axis_rot_m"]),
        u_field_rot_mps=np.asarray(result["u_field_rot"]),
        cft_field_rot=np.asarray(result["cft_field_rot"]),
        ct_at_inflow=float(result["ct_at_inflow"]),
        turbine_cft=float(result["turbine_cft"]),
        planform_area_m2=float(result["planform_area_m2"]),
        median_spacing_m=float(result["median_spacing_m"]),
        influence_radius_m=float(result["influence_radius_m"]),
    )

    return {
        "wrfout_file": wrfout.name,
        "time_label": metadata.time_label,
        "hub_height_m": target_height_m,
        "freestream_speed_mps": float(freestream["wind_speed_mps"]),
        "freestream_direction_deg": float(freestream["wind_direction_deg"]),
        "freestream_u_mps": float(freestream["u_mps"]),
        "freestream_v_mps": float(freestream["v_mps"]),
        "freestream_sample_count": int(freestream["sample_count"]),
        "monin_obukhov_length_m": monin_obukhov_length,
        "boundary_layer_height_m": boundary_layer_height_m,
        "cft_scale": cft_scale,
        "reference_blend_weight": float(result["reference_blend_weight"]),
        "raw_min_mps": float(np.nanmin(u_field_wrf_raw)),
        "raw_mean_mps": float(np.nanmean(u_field_wrf_raw)),
        "raw_max_mps": float(np.nanmax(u_field_wrf_raw)),
        "blended_min_mps": float(np.nanmin(u_field_wrf_blended)),
        "blended_mean_mps": float(np.nanmean(u_field_wrf_blended)),
        "blended_max_mps": float(np.nanmax(u_field_wrf_blended)),
        "top_down_u_min_mps": float(np.nanmin(u_field_wrf_raw)),
        "top_down_u_mean_mps": float(np.nanmean(u_field_wrf_raw)),
        "top_down_u_max_mps": float(np.nanmax(u_field_wrf_raw)),
        "wrf_reference_u_min_mps": float(np.nanmin(wrf_reference_speed)),
        "wrf_reference_u_mean_mps": float(np.nanmean(wrf_reference_speed)),
        "wrf_reference_u_max_mps": float(np.nanmax(wrf_reference_speed)),
        "total_power_mw": float(np.sum(result["turbine_power_kw"]) / 1000.0),
        "png_file": str(png_file),
        "npz_file": str(npz_file),
        "turbine_csv": str(turbine_csv),
    }


def parse_offshore_top_down_args() -> argparse.Namespace:
    """Parse command-line arguments for the standalone offshore top-down case."""

    parser = argparse.ArgumentParser(description="Top-down model for the offshore DTU 10 MW wind farm.")
    parser.add_argument("--wind-speed", type=float, default=8.0, help="Inflow wind speed in m/s.")
    parser.add_argument(
        "--wind-direction",
        type=float,
        default=270.0,
        help="Wind direction in degrees, following the existing TDM convention.",
    )
    parser.add_argument("--curve-file", type=Path, default=DEFAULT_CURVE_FILE)
    parser.add_argument("--location-file", type=Path, default=DEFAULT_LOCATION_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OFFSHORE_OUTPUT_DIR)
    parser.add_argument("--grid-resolution", type=float, default=250.0, help="Flow-field grid resolution in m.")
    parser.add_argument("--padding", type=float, default=15000.0, help="Upstream/downstream plot padding in m.")
    parser.add_argument("--roughness", type=float, default=0.0002, help="Surface roughness length z0 in m.")
    parser.add_argument("--boundary-layer-height", type=float, default=1000.0, help="Boundary layer height H in m.")
    parser.add_argument("--monin-obukhov-length", type=float, default=10000.0, help="Monin-Obukhov length L in m.")
    parser.add_argument("--cft-scale", type=float, default=1.0, help="Multiplier for the Ct-derived planform thrust coefficient.")
    parser.add_argument("--no-plot", action="store_true", help="Skip SVG flow-map generation.")
    return parser.parse_args()


def run_offshore_top_down(args: argparse.Namespace | None = None) -> None:
    """Run the standalone offshore top-down workflow formerly in its own script."""

    if args is None:
        args = parse_offshore_top_down_args()

    curve = load_dtu10mw_curve(args.curve_file)
    turbines = load_offshore_locations(args.location_file)

    result = simulate_top_down(
        x_m=turbines["x_m"].to_numpy(float),
        y_m=turbines["y_m"].to_numpy(float),
        wind_speed=args.wind_speed,
        wind_direction=args.wind_direction,
        curve=curve,
        grid_resolution=args.grid_resolution,
        padding=args.padding,
        roughness=args.roughness,
        boundary_layer_height=args.boundary_layer_height,
        monin_obukhov_length=args.monin_obukhov_length,
        cft_scale=args.cft_scale,
    )
    save_outputs(
        output_dir=args.output_dir,
        turbines=turbines,
        result=result,
        wind_speed=args.wind_speed,
        wind_direction=args.wind_direction,
        make_plot=not args.no_plot,
    )

    total_power_mw = float(np.sum(result["turbine_power_kw"]) / 1000.0)
    mean_ws = float(np.mean(result["turbine_ws"]))
    print(f"Loaded turbines: {len(turbines)}")
    print(f"DTU 10 MW: D={curve['rotor_diameter']:.1f} m, hub height={curve['hub_height']:.1f} m")
    print(f"Inflow: WS={args.wind_speed:.3f} m/s, WD={args.wind_direction:.3f} deg")
    print(f"Top-down free-stream hub-height speed: {result['u_free']:.3f} m/s")
    print(f"Ct at inflow: {result['ct_at_inflow']:.4f}")
    print(f"Planform Cft per turbine cell: {result['turbine_cft']:.6f}")
    print(f"Mean turbine effective wind speed: {mean_ws:.3f} m/s")
    print(f"Total farm power: {total_power_mw:.3f} MW")
    print(f"Saved results to: {args.output_dir}")


def main() -> None:
    """Entry point retained for the merged standalone offshore workflow."""

    run_offshore_top_down()


if __name__ == "__main__":
    main()
