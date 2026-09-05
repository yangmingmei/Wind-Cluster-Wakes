"""Small data and plotting helpers for the processed WRF workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "WRF_processed"


@dataclass(frozen=True)
class WRFMetadata:
    """Grid metadata stored with a processed WRF case."""

    nx: int
    ny: int
    dx_m: float
    dy_m: float
    time_label: str


@dataclass(frozen=True)
class TurbineTable:
    """Original and zero-based turbine grid indices."""

    i: np.ndarray
    j: np.ndarray
    type_id: np.ndarray
    i0: np.ndarray
    j0: np.ndarray


def parse_case_label(case_file: Path) -> str:
    """Extract a compact case label from a processed WRF filename."""

    match = re.search(r"(ws\d+_wd\d+(?:\.\d+)?_Lm(?:neutral|[+-]\d+))", case_file.name)
    if match:
        return match.group(1)
    return case_file.stem


def load_turbines(turbine_file: Path, nx: int, ny: int, index_base: int) -> TurbineTable:
    """Load whitespace-separated turbine ``i j type`` positions."""

    values = np.loadtxt(turbine_file, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] < 2:
        raise ValueError(f"{turbine_file} must contain at least i and j columns")

    i = values[:, 0]
    j = values[:, 1]
    type_id = values[:, 2] if values.shape[1] >= 3 else np.ones(values.shape[0])
    i0 = np.rint(i - index_base).astype(int)
    j0 = np.rint(j - index_base).astype(int)
    invalid = (i0 < 0) | (i0 >= nx) | (j0 < 0) | (j0 >= ny)
    if np.any(invalid):
        rows = (np.flatnonzero(invalid)[:10] + 1).tolist()
        raise ValueError(
            f"{turbine_file} has indices outside the processed grid for index base "
            f"{index_base}; first invalid rows: {rows}"
        )
    return TurbineTable(i=i, j=j, type_id=type_id, i0=i0, j0=j0)


def configure_matplotlib(font_size: float, title_font_size: float) -> None:
    """Use Times New Roman and a compact style for saved figures."""

    for font_file in (
        Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\timesbd.ttf"),
        Path(r"C:\Windows\Fonts\timesi.ttf"),
        Path(r"C:\Windows\Fonts\timesbi.ttf"),
    ):
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": font_size,
            "axes.titlesize": title_font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": title_font_size,
            "axes.linewidth": 0.45,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
        }
    )


def grid_edges_km(n: int, spacing_m: float) -> np.ndarray:
    return (np.arange(n + 1, dtype=float) - 0.5) * spacing_m / 1000.0


def grid_centers_km(n: int, spacing_m: float) -> np.ndarray:
    return np.arange(n, dtype=float) * spacing_m / 1000.0


def compact_number_formatter(value: float, _position: int | None = None) -> str:
    if np.isclose(value, round(value), atol=1.0e-8):
        return f"{value:.0f}"
    if np.isclose(value * 10.0, round(value * 10.0), atol=1.0e-8):
        return f"{value:.1f}"
    return f"{value:.2f}"


def axis_ticks_with_endpoints(axis_min: float, axis_max: float) -> np.ndarray:
    width = max(float(axis_max) - float(axis_min), 1.0e-12)
    step = 10.0 if width <= 40.0 else 20.0 if width <= 100.0 else 40.0
    inner = np.arange(np.ceil(axis_min / step) * step, axis_max + 1.0e-9, step)
    inner = inner[(inner > axis_min + 0.2 * step) & (inner < axis_max - 0.2 * step)]
    return np.unique(np.round(np.concatenate(([axis_min], inner, [axis_max])), 8))


def wake_axis_ticks_km(axis_min: float, axis_max: float) -> np.ndarray:
    return axis_ticks_with_endpoints(axis_min, axis_max)


def colorbar_ticks_with_endpoints(vmin: float, vmax: float) -> np.ndarray:
    if np.isclose(vmin, vmax):
        return np.asarray([vmin], dtype=float)
    return np.linspace(float(vmin), float(vmax), 5)


def resolve_wake_color_limits(
    speed_2d: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    reference_speed: float | None = None,
) -> tuple[float, float]:
    """Choose robust wind-speed bounds while retaining a shared reference."""

    finite = np.asarray(speed_2d, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        finite = np.asarray([float(reference_speed or 1.0)])
    reference = (
        float(reference_speed)
        if reference_speed is not None and np.isfinite(reference_speed) and reference_speed > 0.0
        else None
    )
    resolved_min = float(np.nanpercentile(finite, 1.0) if vmin is None else vmin)
    resolved_max = float((reference or np.nanpercentile(finite, 99.0)) if vmax is None else vmax)
    if reference is not None and vmin is None:
        resolved_min = max(0.8 * reference, min(resolved_min, 0.98 * resolved_max))
    if resolved_max <= resolved_min:
        resolved_max = resolved_min + max(0.01, 0.01 * abs(resolved_min))
    return resolved_min, resolved_max


def wake_adaptive_x_limits_km(
    turbines: TurbineTable,
    metadata: WRFMetadata,
    wind_direction_deg: float | None,
    fallback_x_min_km: float,
) -> tuple[float, float]:
    """Keep the farm and the downstream portion of the domain visible."""

    edges = grid_edges_km(metadata.nx, metadata.dx_m)
    centers = grid_centers_km(metadata.nx, metadata.dx_m)
    turbine_x = centers[turbines.i0]
    domain_min, domain_max = float(edges[0]), float(edges[-1])
    turbine_min, turbine_max = float(np.min(turbine_x)), float(np.max(turbine_x))
    pad = max(5.0, 2.0 * metadata.dx_m / 1000.0)

    if wind_direction_deg is None or not np.isfinite(wind_direction_deg):
        limits = (float(fallback_x_min_km), domain_max)
    else:
        downstream_x = float(np.cos(np.radians(270.0 - wind_direction_deg)))
        if downstream_x > 0.15:
            limits = (min(float(fallback_x_min_km), turbine_min - pad), domain_max)
        elif downstream_x < -0.15:
            limits = (domain_min, turbine_max + pad)
        else:
            crosswind_pad = max(10.0, 0.35 * max(turbine_max - turbine_min, 1.0))
            limits = (turbine_min - crosswind_pad, turbine_max + crosswind_pad)

    x_min = max(domain_min, limits[0])
    x_max = min(domain_max, limits[1])
    return (domain_min, domain_max) if x_max <= x_min else (x_min, x_max)


def plot_heatmap_png(
    output_file: Path,
    speed_2d: np.ndarray,
    turbines: TurbineTable,
    metadata: WRFMetadata,
    target_height_m: float,
    case_label: str,
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
    wind_direction_deg: float | None = None,
    color_reference_speed: float | None = None,
) -> tuple[float, float]:
    """Render a processed-grid wind-speed map."""

    configure_matplotlib(font_size, title_font_size)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = resolve_wake_color_limits(speed_2d, vmin, vmax, color_reference_speed)
    x_edges = grid_edges_km(metadata.nx, metadata.dx_m)
    y_edges = grid_edges_km(metadata.ny, metadata.dy_m)
    x_centers = grid_centers_km(metadata.nx, metadata.dx_m)
    y_centers = grid_centers_km(metadata.ny, metadata.dy_m)

    fig, ax = plt.subplots(figsize=(figure_width_in, figure_height_in))
    mesh = ax.pcolormesh(x_edges, y_edges, speed_2d, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.scatter(
        x_centers[turbines.i0],
        y_centers[turbines.j0],
        s=9.0,
        marker="^",
        c="#d62728",
        edgecolors="white",
        linewidths=0.25,
        label=turbine_name,
    )
    x_plot_min, x_plot_max = wake_adaptive_x_limits_km(
        turbines, metadata, wind_direction_deg, x_min_km
    )
    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    ax.set_xticks(wake_axis_ticks_km(x_plot_min, x_plot_max))
    ax.set_yticks(axis_ticks_with_endpoints(float(y_edges[0]), float(y_edges[-1])))
    ax.xaxis.set_major_formatter(FuncFormatter(compact_number_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_number_formatter))
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title(f"{case_label} at {target_height_m:g} m")
    ax.legend(loc="best", frameon=True)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.025)
    colorbar.set_label("Wind speed [m s$^{-1}$]")
    colorbar.set_ticks(colorbar_ticks_with_endpoints(vmin, vmax))
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(compact_number_formatter))
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return vmin, vmax


__all__ = [
    "DEFAULT_RESULTS_DIR",
    "TurbineTable",
    "WRFMetadata",
    "axis_ticks_with_endpoints",
    "colorbar_ticks_with_endpoints",
    "compact_number_formatter",
    "configure_matplotlib",
    "grid_centers_km",
    "grid_edges_km",
    "load_turbines",
    "parse_case_label",
    "plot_heatmap_png",
    "resolve_wake_color_limits",
    "wake_adaptive_x_limits_km",
    "wake_axis_ticks_km",
]
