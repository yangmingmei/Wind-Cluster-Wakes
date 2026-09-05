"""WRF-driven top-down model with strong-instability and cluster fixes.

Changes relative to ``analytical_top_down_from_wrf_freestream.py``:

* MOST corrections are evaluated no higher than a fraction of the ABL depth.
* Strongly unstable ``z/L`` uses a smooth Businger-Dyer/convective blend.
* Multiple farm segments are composed by propagating wind speed downstream,
  instead of adding independent deficits referenced to the same free stream.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from improved_top_down_model import ImprovedTopDownParameters, top_down_model
from wrf_wake_utilities import (
    DEFAULT_CURVE_FILE,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TURBINES,
    TurbineTable,
    WRFMetadata,
    add_common_wrf_model_args,
    axis_covering,
    blend_with_reference_speed,
    clean_speed_field,
    curve_value,
    extract_boundary_layer_height,
    get_limited_wrfout_files,
    infer_planform_area,
    load_dtu10mw_curve,
    load_wrf_case_inputs,
    resolve_case_monin_obukhov_length,
    rotate_to_wind_frame,
    sample_wrf_grid_at_turbines,
    save_top_down_case_outputs,
    turbine_coordinates_m,
    wrf_center_coordinates,
    write_summary_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "top_down"
NEUTRAL_LENGTH = 1.0e9


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
    """Evaluate the improved core and retain the legacy stable safeguard."""

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
    if 0.0 < monin_obukhov_length < NEUTRAL_LENGTH and cft > 0.0 and farm_dist > 0.0:
        neutral_speed = top_down_model(
            wind_speed,
            roughness,
            boundary_layer_height,
            NEUTRAL_LENGTH,
            cft,
            hub_height,
            rotor_diameter,
            farm_dist,
            wake_dist,
            parameters,
        )
        speed = min(speed, neutral_speed)
    return float(speed)


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
    return float(np.clip(speed / max(free_speed, 1.0e-12), 0.0, parameters.max_speed_fraction))


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
                recovery = float(np.clip((1.0 - wake_ratio) / denominator, 0.0, 1.0))
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
        recovery = float(np.clip((1.0 - wake_ratio) / denominator, 0.0, 1.0))
        output[index] = free_speed + (previous_end_speed - free_speed) * recovery

    return output


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
) -> dict[str, np.ndarray | float]:
    """Run the improved model and interpolate it back to the WRF grid."""

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
    x_axis = axis_covering(x_grid_rot_centers, metadata.dx_m)
    y_axis = axis_covering(y_grid_rot_centers, metadata.dy_m)
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
            parameters,
        )
    )
    tree = cKDTree(np.column_stack((turbine_x_rot, turbine_y_rot)))
    distances, _ = tree.query(np.column_stack((x_grid_rot.ravel(), y_grid_rot.ravel())))
    cft_field = np.where(distances <= influence_radius, turbine_cft, 0.0).reshape(x_grid_rot.shape)

    u_field_rot = np.empty_like(x_grid_rot, dtype=float)
    for column in range(u_field_rot.shape[1]):
        u_field_rot[:, column] = sequential_row_speed(
            cft_field[:, column],
            u_free,
            float(metadata.dx_m),
            roughness,
            boundary_layer_height,
            monin_obukhov_length,
            hub_height,
            rotor_diameter,
            parameters,
        )
    u_field_rot = clean_speed_field(u_field_rot, u_free)

    interpolator = RegularGridInterpolator(
        (x_axis, y_axis), u_field_rot, bounds_error=False, fill_value=u_free
    )
    wrf_points = np.column_stack((x_grid_rot_centers.ravel(), y_grid_rot_centers.ravel()))
    u_field_wrf_raw = interpolator(wrf_points).reshape(metadata.ny, metadata.nx)
    u_field_wrf_raw = clean_speed_field(u_field_wrf_raw, u_free)
    u_field_wrf_blended = blend_with_reference_speed(
        u_field_wrf_raw, reference_speed, u_free, reference_blend_weight
    )

    turbine_ws = sample_wrf_grid_at_turbines(
        u_field_wrf_raw, turbine_x_m, turbine_y_m, metadata, u_free
    )
    turbine_ws = np.clip(turbine_ws, 0.0, None)
    turbine_power_kw = curve_value(curve, turbine_ws, "power_kw")
    turbine_cp = curve_value(curve, turbine_ws, "cp")

    return {
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
        "reference_blend_weight": float(np.clip(reference_blend_weight, 0.0, 1.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_wrf_model_args(parser, DEFAULT_OUTPUT_DIR, "improved top-down PNG/NPZ/CSV")
    parser.set_defaults(figure_width=3.50, figure_height=1.55, title_font_size=8.5)
    parser.add_argument("--roughness", type=float, default=0.0002)
    parser.add_argument(
        "--boundary-layer-height",
        type=float,
        default=None,
        help="Optional fixed H override in meters; by default each case uses domain-mean WRF PBLH.",
    )
    parser.add_argument("--cft-scale", type=float, default=1.0)
    parser.add_argument("--most-height-fraction", type=float, default=0.10)
    parser.add_argument("--bd-limit-zeta", type=float, default=-2.0)
    parser.add_argument("--convective-transition-width", type=float, default=1.0)
    parser.add_argument("--stability-strength", type=float, default=0.50)
    parser.add_argument("--wake-recovery-scale", type=float, default=1.0)
    parser.add_argument("--reference-blend-weight", type=float, default=0.0)
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument("--write-png", dest="no_png", action="store_false", help="Render PNG heatmaps.")
    plot_group.add_argument("--no-png", dest="no_png", action="store_true", help="Skip PNG rendering.")
    parser.set_defaults(no_png=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters = ImprovedTopDownParameters(
        most_height_fraction=args.most_height_fraction,
        bd_limit_zeta=args.bd_limit_zeta,
        convective_transition_width=args.convective_transition_width,
        stability_strength=args.stability_strength,
        wake_recovery_scale=args.wake_recovery_scale,
    )
    curve = load_dtu10mw_curve(args.curve_file.resolve())
    wrfout_files = get_limited_wrfout_files(args)
    output_dir = args.output_dir.resolve()
    summary_rows: list[dict[str, str | float | int]] = []

    print(f"Found {len(wrfout_files)} processed WRF cases; output={output_dir}")
    for file_index, wrfout in enumerate(wrfout_files, start=1):
        u_hub, v_hub, metadata, turbines, freestream = load_wrf_case_inputs(wrfout, args)
        reference_speed = np.hypot(u_hub, v_hub)
        monin_obukhov_length = resolve_case_monin_obukhov_length(args, wrfout)
        boundary_layer_height = (
            float(args.boundary_layer_height)
            if args.boundary_layer_height is not None
            else extract_boundary_layer_height(wrfout, args.time_index)
        )
        case_cft_scale = float(args.cft_scale)
        result = simulate_top_down_on_wrf_domain(
            turbines,
            metadata,
            float(freestream["wind_speed_mps"]),
            float(freestream["wind_direction_deg"]),
            curve,
            args.roughness,
            boundary_layer_height,
            monin_obukhov_length,
            case_cft_scale,
            parameters,
            reference_speed,
            args.reference_blend_weight,
        )
        row = save_top_down_case_outputs(
            output_dir,
            wrfout,
            metadata,
            turbines,
            result,
            freestream,
            reference_speed,
            monin_obukhov_length,
            boundary_layer_height,
            case_cft_scale,
            args.height,
            args,
        )
        row.update(
            {
                "most_height_fraction": args.most_height_fraction,
                "bd_limit_zeta": args.bd_limit_zeta,
                "convective_transition_width": args.convective_transition_width,
                "stability_strength": args.stability_strength,
                "wake_recovery_scale": args.wake_recovery_scale,
            }
        )
        summary_rows.append(row)
        print(
            f"[{file_index}/{len(wrfout_files)}] {wrfout.name}: "
            f"L={monin_obukhov_length:g}, H={boundary_layer_height:.1f} m, "
            f"Cft-scale={case_cft_scale:.3f}, "
            f"min/mean={row['raw_min_mps']:.3f}/{row['raw_mean_mps']:.3f} m/s"
        )

    summary_file = output_dir / "top_down_unstable_improved_from_wrf_summary.csv"
    write_summary_csv(summary_file, summary_rows)
    print(f"Saved summary: {summary_file}")


if __name__ == "__main__":
    main()
