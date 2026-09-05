"""
Run PyWake's TurboPark model for every WRF wrfout case.

The driver reads WRF hub-height inflow, runs Nygaard_2022 TurboPark on the WRF
mass grid, and saves one PNG, NPZ, and turbine CSV per case plus a summary CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wrf_wake_utilities import (
    DEFAULT_CURVE_FILE,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TURBINES,
    TurbineTable,
    WRFMetadata,
    add_common_wrf_model_args,
    build_dtu10mw_pywake_turbines,
    clean_limited_speed,
    configure_matplotlib,
    curve_value,
    find_wrfout_files,
    get_limited_wrfout_files,
    load_dtu10mw_curve,
    load_wrf_case_inputs,
    parse_case_label,
    parse_monin_obukhov_length,
    plot_pywake_png,
    resolve_case_monin_obukhov_length,
    safe_output_stem,
    save_pywake_case_outputs,
    simulate_pywake_on_wrf_domain,
    squeeze_single_case_array,
    turbine_coordinates_m,
    wrf_center_coordinates,
    write_summary_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "turbopark"


def wrf_center_axes_m(metadata: WRFMetadata) -> tuple[np.ndarray, np.ndarray]:
    """Build WRF mass-grid center axes in meters."""

    x_m, y_m, _, _ = wrf_center_coordinates(metadata)
    return x_m, y_m


def clean_turbopark_speed(speed_2d: np.ndarray, freestream_speed: float) -> np.ndarray:
    """Backward-compatible alias for the shared PyWake speed cleaner."""

    return clean_limited_speed(speed_2d, freestream_speed)


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

    from py_wake.literature import Nygaard_2022

    return simulate_pywake_on_wrf_domain(
        turbines=turbines,
        metadata=metadata,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        curve=curve,
        pywake_turbines=pywake_turbines,
        model_factory=lambda site, wt: Nygaard_2022(site, wt),
        turbulence_intensity=turbulence_intensity,
        model_key="turbopark",
    )


def plot_turbopark_png(
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
) -> None:
    """Save one TurboPark wake heatmap as a PNG file."""

    plot_pywake_png(
        output_file=output_file,
        result=result,
        turbines=turbines,
        metadata=metadata,
        target_height_m=target_height_m,
        case_label=case_label,
        freestream_speed=freestream_speed,
        freestream_direction=freestream_direction,
        turbine_name=turbine_name,
        vmin=vmin,
        vmax=vmax,
        figure_width_in=figure_width_in,
        figure_height_in=figure_height_in,
        font_size=font_size,
        title_font_size=title_font_size,
        cmap=cmap,
        x_min_km=x_min_km,
        dpi=dpi,
        plot_normalized=plot_normalized,
        model_key="turbopark",
        model_title="TurboPark",
    )


def save_case_outputs(
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
) -> dict[str, str | float | int]:
    """Save one TurboPark case as PNG, NPZ, and turbine CSV."""

    return save_pywake_case_outputs(
        output_dir=output_dir,
        wrfout=wrfout,
        metadata=metadata,
        turbines=turbines,
        result=result,
        reference_speed=reference_speed,
        freestream=freestream,
        monin_obukhov_length=monin_obukhov_length,
        target_height_m=target_height_m,
        plot_settings=plot_settings,
        model_key="turbopark",
        model_title="TurboPark",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for WRF-driven TurboPark batch processing."""

    parser = argparse.ArgumentParser(
        description="Run PyWake TurboPark for every processed WRF case."
    )
    add_common_wrf_model_args(parser, DEFAULT_OUTPUT_DIR, "PNG/NPZ/CSV")
    parser.add_argument("--turbulence-intensity", type=float, default=0.09, help="Ambient turbulence intensity passed to PyWake.")
    parser.add_argument("--plot-normalized", action="store_true", help="Plot normalized wind speed instead of m/s.")
    return parser.parse_args()


def main() -> None:
    """Run the full TurboPark batch workflow."""

    args = parse_args()
    curve = load_dtu10mw_curve(args.curve_file.resolve())
    pywake_turbines = build_dtu10mw_pywake_turbines(curve)
    wrfout_files = get_limited_wrfout_files(args)
    output_dir = args.output_dir.resolve()

    print(f"Found {len(wrfout_files)} processed WRF cases in {args.results_dir.resolve()}")
    print(f"Saving TurboPark PNG/NPZ/CSV outputs to {output_dir}")

    summary_rows: list[dict[str, str | float | int]] = []
    for file_index, wrfout in enumerate(wrfout_files, start=1):
        print(f"[{file_index}/{len(wrfout_files)}] Processing {wrfout.name}")

        u_hub, v_hub, metadata, turbines, freestream = load_wrf_case_inputs(wrfout, args)
        reference_speed = np.hypot(u_hub, v_hub)
        monin_obukhov_length = resolve_case_monin_obukhov_length(args, wrfout)

        result = simulate_turbopark_on_wrf_domain(
            turbines=turbines,
            metadata=metadata,
            wind_speed=float(freestream["wind_speed_mps"]),
            wind_direction=float(freestream["wind_direction_deg"]),
            curve=curve,
            pywake_turbines=pywake_turbines,
            turbulence_intensity=args.turbulence_intensity,
        )
        row = save_case_outputs(
            output_dir=output_dir,
            wrfout=wrfout,
            metadata=metadata,
            turbines=turbines,
            result=result,
            reference_speed=reference_speed,
            freestream=freestream,
            monin_obukhov_length=monin_obukhov_length,
            target_height_m=args.height,
            plot_settings=args,
        )
        summary_rows.append(row)

        print(
            "  Inflow "
            f"WS={row['freestream_speed_mps']:.3f} m/s, "
            f"WD={row['freestream_direction_deg']:.2f} deg; "
            f"TurboPark min/mean/max={row['turbopark_u_min_mps']:.3f}/"
            f"{row['turbopark_u_mean_mps']:.3f}/{row['turbopark_u_max_mps']:.3f} m/s; "
            f"power={row['total_power_mw']:.3f} MW"
        )

    summary_file = output_dir / "turbopark_from_wrf_summary.csv"
    write_summary_csv(summary_file, summary_rows)
    print(f"Saved summary: {summary_file}")
    print("Done.")


if __name__ == "__main__":
    main()
