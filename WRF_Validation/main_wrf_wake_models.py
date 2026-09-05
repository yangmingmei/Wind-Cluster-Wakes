"""
Main entry point for the WRF wake-model workflow.

By default this script runs:
  TurboPark, Gaussian, improved top-down, and Array-Stability simulations
  from the compact cases in WRF_processed.

Examples
--------
Run all four models:
    python main_wrf_wake_models.py

Smoke-test the pipeline on one WRF case:
    python main_wrf_wake_models.py --limit 1

"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_SCRIPTS = [
    ("TurboPark", SCRIPT_DIR / "analytical_turbopark_from_wrf_freestream.py"),
    ("Gaussian", SCRIPT_DIR / "analytical_gaussian_from_wrf_freestream.py"),
    (
        "Improved top-down",
        SCRIPT_DIR / "analytical_top_down_unstable_improved_from_wrf_freestream.py",
    ),
    (
        "Array-Stability + TurboPark",
        SCRIPT_DIR / "analytical_array_stability_from_wrf_freestream.py",
    ),
]
def add_optional_path_arg(command: list[str], flag: str, value: Path | None) -> None:
    """Append a path argument only when the user supplied it."""

    if value is not None:
        command.extend([flag, str(value)])


def add_optional_scalar_arg(command: list[str], flag: str, value: object | None) -> None:
    """Append a scalar argument only when the user supplied it."""

    if value is not None:
        command.extend([flag, str(value)])


def run_command(label: str, command: list[str], continue_on_error: bool) -> None:
    """Run a child workflow step and handle failure consistently."""

    print(f"\n=== {label} ===")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=SCRIPT_DIR, check=False)
    if completed.returncode != 0:
        message = f"{label} failed with exit code {completed.returncode}"
        if continue_on_error:
            print(message, file=sys.stderr)
            return
        raise SystemExit(message)


def build_model_command(script: Path, args: argparse.Namespace) -> list[str]:
    """Build the common command line passed to each model driver."""

    command = [sys.executable, str(script)]
    add_optional_path_arg(command, "--results-dir", args.results_dir)
    add_optional_scalar_arg(command, "--pattern", args.pattern)
    add_optional_path_arg(command, "--turbines", args.turbines)
    add_optional_path_arg(command, "--curve-file", args.curve_file)
    add_optional_scalar_arg(command, "--height", args.height)
    add_optional_scalar_arg(command, "--time-index", args.time_index)
    add_optional_scalar_arg(command, "--index-base", args.index_base)
    add_optional_scalar_arg(command, "--upstream-buffer", args.upstream_buffer)
    add_optional_scalar_arg(command, "--limit", args.limit)
    return command


def parse_args() -> argparse.Namespace:
    """Parse the workflow-level arguments passed through to child scripts."""

    parser = argparse.ArgumentParser(description="Run four wake models from compact processed WRF inputs.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue to later steps when one child command fails.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory containing processed WRF NPZ files.")
    parser.add_argument("--pattern", default=None, help="Filename pattern used inside results-dir.")
    parser.add_argument("--turbines", type=Path, default=None, help="Whitespace txt file with i j type columns.")
    parser.add_argument("--curve-file", type=Path, default=None, help="DTU 10 MW curve workbook.")
    parser.add_argument("--height", type=float, default=None, help="Hub/reference height in meters.")
    parser.add_argument("--time-index", type=int, default=None, help="Reserved for compatibility with processed inputs.")
    parser.add_argument("--index-base", type=int, choices=(0, 1), default=None, help="Index base used by the turbine i/j file.")
    parser.add_argument("--upstream-buffer", type=float, default=None, help="Upstream sampling distance in meters.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of processed cases per model.")
    return parser.parse_args()


def main() -> None:
    """Run all four wake-model simulations."""

    args = parse_args()

    for label, script in MODEL_SCRIPTS:
        run_command(label, build_model_command(script, args), args.continue_on_error)


if __name__ == "__main__":
    main()
