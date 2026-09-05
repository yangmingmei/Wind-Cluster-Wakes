"""Run the paper's Array-Stability Model plus TurboPark for WRF cases.

The farm-scale implementation follows Canadillas et al. (2023), Appendix B:
spatial turbine thrust is used in Equations (A7)-(A8), far-wake recovery uses
Equation (A3), and the resulting spatial inflow drives PyWake's Nygaard_2022
TurboPark model.  The final speed is the pointwise minimum of the ASM farm
field and the TurboPark field, as specified by the paper's four-step method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from array_stability_model import ArrayStabilityParameters, array_stability_ratio_field
from wrf_wake_utilities import (
    TurbineTable,
    WRFMetadata,
    add_common_wrf_model_args,
    axis_covering,
    build_dtu10mw_pywake_turbines,
    clean_limited_speed,
    curve_value,
    get_limited_wrfout_files,
    load_dtu10mw_curve,
    load_wrf_case_inputs,
    resolve_case_monin_obukhov_length,
    rotate_to_wind_frame,
    save_pywake_case_outputs,
    squeeze_single_case_array,
    turbine_coordinates_m,
    wrf_center_coordinates,
    write_summary_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "array_stability"
MODEL_KEY = "array_stability"


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
    """Save one ASM case using the project's standard PNG/NPZ/CSV layout."""

    extra_npz = {
        key: value
        for key, value in result.items()
        if key
        in {
            "asm_wind_speed_mps",
            "asm_normalized_wind_speed",
            "turbopark_with_asm_inflow_wind_speed_mps",
            "x_axis_rot_m",
            "y_axis_rot_m",
            "turbine_ct_for_asm",
            "thrust_density_rot",
            "farm_layer_speed_ratio_rot",
            "asm_speed_ratio_rot",
            "effective_drag_coefficient_rot",
            "friction_velocity_mps",
            "exchange_coefficient_m2ps",
            "surface_drag_coefficient",
            "phi_m",
            "psi_m",
            "ct_iteration_count",
            "ct_iteration_residual",
        }
    }
    extra_npz.update(
        {
            "roughness_length_m": plot_settings.roughness,
            "ct_scale": plot_settings.ct_scale,
            "delta_z1_diameters": plot_settings.delta_z1_diameters,
            "sigma_x_diameters": plot_settings.sigma_x_diameters,
            "sigma_y_diameters": plot_settings.sigma_y_diameters,
            "streamwise_skewness": plot_settings.streamwise_skewness,
        }
    )
    extra_summary = {
        "roughness_length_m": plot_settings.roughness,
        "ct_scale": plot_settings.ct_scale,
        "phi_m": float(result["phi_m"]),
        "psi_m": float(result["psi_m"]),
        "friction_velocity_mps": float(result["friction_velocity_mps"]),
        "exchange_coefficient_m2ps": float(result["exchange_coefficient_m2ps"]),
        "ct_iteration_count": int(result["ct_iteration_count"]),
        "ct_iteration_residual": float(result["ct_iteration_residual"]),
    }
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
        model_key=MODEL_KEY,
        model_title="Array-Stability + TurboPark",
        extra_npz=extra_npz,
        extra_summary=extra_summary,
    )


def parse_args() -> argparse.Namespace:
    """Parse WRF-driven ASM batch arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_wrf_model_args(parser, DEFAULT_OUTPUT_DIR, "ASM PNG/NPZ/CSV")
    parser.add_argument("--roughness", type=float, default=0.0002, help="Surface roughness z0 in meters.")
    parser.add_argument(
        "--turbulence-intensity",
        type=float,
        default=0.045,
        help="Ambient turbulence intensity; paper Table 1 uses 0.045.",
    )
    parser.add_argument("--ct-scale", type=float, default=1.0, help="Multiplier for ASM distributed C_T.")
    parser.add_argument("--ct-iterations", type=int, default=20)
    parser.add_argument("--ct-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--delta-z1-diameters", type=float, default=0.05)
    parser.add_argument("--sigma-x-diameters", type=float, default=9.0)
    parser.add_argument("--sigma-y-diameters", type=float, default=3.0)
    parser.add_argument("--streamwise-skewness", type=float, default=2.0)
    parser.add_argument("--plot-normalized", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the paper-faithful ASM plus TurboPark workflow for all WRF cases."""

    args = parse_args()
    parameters = ArrayStabilityParameters(
        delta_z1_diameters=args.delta_z1_diameters,
        sigma_x_diameters=args.sigma_x_diameters,
        sigma_y_diameters=args.sigma_y_diameters,
        streamwise_skewness=args.streamwise_skewness,
    )
    curve = load_dtu10mw_curve(args.curve_file.resolve())
    pywake_turbines = build_dtu10mw_pywake_turbines(curve)
    wrfout_files = get_limited_wrfout_files(args)
    output_dir = args.output_dir.resolve()

    print(f"Found {len(wrfout_files)} processed WRF cases in {args.results_dir.resolve()}")
    print(f"Saving Array-Stability PNG/NPZ/CSV outputs to {output_dir}")
    summary_rows: list[dict[str, str | float | int]] = []

    for file_index, wrfout in enumerate(wrfout_files, start=1):
        print(f"[{file_index}/{len(wrfout_files)}] Processing {wrfout.name}")
        u_hub, v_hub, metadata, turbines, freestream = load_wrf_case_inputs(wrfout, args)
        reference_speed = np.hypot(u_hub, v_hub)
        monin_obukhov_length = resolve_case_monin_obukhov_length(args, wrfout)
        result = simulate_array_stability_on_wrf_domain(
            turbines=turbines,
            metadata=metadata,
            wind_speed=float(freestream["wind_speed_mps"]),
            wind_direction=float(freestream["wind_direction_deg"]),
            curve=curve,
            pywake_turbines=pywake_turbines,
            roughness=args.roughness,
            turbulence_intensity=args.turbulence_intensity,
            monin_obukhov_length=monin_obukhov_length,
            ct_scale=args.ct_scale,
            ct_iterations=args.ct_iterations,
            ct_tolerance=args.ct_tolerance,
            parameters=parameters,
        )
        row = save_case_outputs(
            output_dir,
            wrfout,
            metadata,
            turbines,
            result,
            reference_speed,
            freestream,
            monin_obukhov_length,
            args.height,
            args,
        )
        summary_rows.append(row)
        print(
            f"  L={monin_obukhov_length:g} m, phi_m={row['phi_m']:.4f}; "
            f"ASM+TP min/mean/max={row['array_stability_u_min_mps']:.3f}/"
            f"{row['array_stability_u_mean_mps']:.3f}/"
            f"{row['array_stability_u_max_mps']:.3f} m/s; "
            f"power={row['total_power_mw']:.3f} MW"
        )

    summary_file = output_dir / "array_stability_from_wrf_summary.csv"
    write_summary_csv(summary_file, summary_rows)
    print(f"Saved summary: {summary_file}")
    print("Done.")


if __name__ == "__main__":
    main()
