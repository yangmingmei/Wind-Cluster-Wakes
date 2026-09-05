# WRF Wake-Model Validation

[Back to the project overview](../README.md)

This directory is the first integrated validation workflow in **Wind Cluster Wakes**. It compares the improved Top-down model and three reference models with 30 compact WRF cases at 119 m above ground level. Processed inputs and a curated result figure are included; raw WRF outputs and generated model predictions are not tracked.

## Reported results

![Mean absolute error of four wake models across 30 WRF cases](assets/model_validation_summary.png)

| Model | Global field MAE (m/s) | Turbine effective wind-speed MAE (m/s) | Wake-region MAE (m/s) |
|---|---:|---:|---:|
| **Improved Top-down** | **0.100** | **0.203** | **0.213** |
| Gaussian | 0.291 | 0.705 | 1.237 |
| TurboPark | 0.337 | 0.531 | 1.319 |
| Array Stability | 0.542 | 0.474 | 1.930 |

These values reproduce the supplied summary figure, which reports errors averaged across 30 cases. Improved Top-down ranks first for all three metrics in this set. The figure labels this model “Top-down” and the Array Stability + TurboPark implementation “Array Stability.”

For an evaluation set of N samples, mean absolute error is

\[
\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|U_{\mathrm{model},i}-U_{\mathrm{WRF},i}|.
\]

The three reported comparisons concern the global field, turbine effective wind speeds, and the wake-region field. **The repository currently includes the summary image but no metric aggregation script or explicit wake-region mask.** The table is therefore a recorded result, not a newly recomputed benchmark. Exact sample selection, missing-value handling, and case weighting should be archived with the future evaluation workflow.

## Dataset and inputs

| Property | Included cases |
|---|---|
| Case count | 30 = 3 speeds × 2 directions × 5 stability conditions |
| Nominal wind speed | 8, 10, 12 m/s |
| Wind direction | 90°, 270° |
| Monin–Obukhov length labels | −500, −200, +200, +500 m; neutral |
| Stored reference height | 119 m AGL |
| Grid | 162 × 42 cells, with 1,000 m spacing in both directions |
| Selected time | `2000-01-01_14:00:00`, one stored slice per case |
| Turbine curve | DTU 10 MW workbook |

- [WRF_processed/manifest.csv](WRF_processed/manifest.csv) lists each case, its source filename, grid and height, wind-speed statistics, and NPZ filename.
- `WRF_processed/*.npz` contains compact wind-speed fields and metadata used by the drivers, including inflow direction and boundary-layer height.
- [WRF_processed/windturbines-ij.txt](WRF_processed/windturbines-ij.txt) contains turbine `i j type` entries. The default index base is **1**.
- [10MW_turbine_curve.xlsx](10MW_turbine_curve.xlsx) supplies the turbine performance curve.

The nominal speed in a case name is distinct from the upstream speed estimated from its WRF field. The current loader reconstructs horizontal velocity components from the stored speed and a single inflow direction. The drivers estimate freestream conditions upstream of the farm, using a default 5,000 m buffer. Top-down and Array Stability use the stored domain-mean boundary-layer height unless overridden.

Each NPZ already represents one height and time. `--height` must match the stored height; it does not interpolate a new level. `--time-index` is retained for compatibility and does not select another time from these inputs. Raw WRF data and the preprocessing workflow are not included.

## Environment and execution

Use the shared [root environment.yml](../environment.yml). From the **repository root**:

```bash
conda env create -f environment.yml
conda activate pywake_cluster
python WRF_Validation/main_wrf_wake_models.py --limit 1
python WRF_Validation/main_wrf_wake_models.py
```

The first Python command runs one case per model; the second runs all cases through all four models. For an existing environment, use `conda env update -f environment.yml`.

If your shell is already inside `WRF_Validation/`, use:

```bash
conda env create -f ../environment.yml
conda activate pywake_cluster
python main_wrf_wake_models.py --limit 1
```

The environment retains the name `pywake_cluster`. Direct dependency versions reflect the installed local environment used for the documentation smoke test; this file is not a complete transitive dependency lock.

Validation on 2026-09-05: all four models completed the one-case smoke test in the existing Conda environment. A fresh environment installation and the full 30-case benchmark were not rerun for this documentation update. On Windows, activate Conda as shown above so its DLL paths are loaded. The local environment also emitted an h5py/HDF5 version-mismatch warning, although the smoke test completed.

## Models and individual runs

| Model | Driver | Default output directory |
|---|---|---|
| Improved Top-down | `analytical_top_down_unstable_improved_from_wrf_freestream.py` | `outputs/top_down/` |
| Gaussian | `analytical_gaussian_from_wrf_freestream.py` | `outputs/gaussian/` |
| TurboPark | `analytical_turbopark_from_wrf_freestream.py` | `outputs/turbopark/` |
| Array Stability + TurboPark | `analytical_array_stability_from_wrf_freestream.py` | `outputs/array_stability/` |

The formulations live in `improved_top_down_model.py` and `array_stability_model.py`; shared utilities live in `wrf_wake_utilities.py` and `wrf_processed_io.py`.

To run only improved Top-down and request its PNG heatmap, from the repository root:

```bash
python WRF_Validation/analytical_top_down_unstable_improved_from_wrf_freestream.py --limit 1 --write-png
```

Improved Top-down skips PNG rendering by default. The other three drivers generate PNGs by default. For common workflow options and model-specific settings:

```bash
python WRF_Validation/main_wrf_wake_models.py --help
python WRF_Validation/analytical_top_down_unstable_improved_from_wrf_freestream.py --help
```

The main runner accepts input selection options such as `--results-dir`, `--pattern`, and `--limit`, and stops on a failed model unless `--continue-on-error` is supplied. Use individual drivers for `--output-dir`, plotting controls, and model-specific parameters. When supplying paths through the main runner, use absolute paths: child processes run with `WRF_Validation/` as their working directory.

## Generated outputs

All default output paths are relative to this directory, regardless of where the default command is launched. Each model writes compressed NPZ fields, turbine CSV tables, and a run-summary CSV under `outputs/<model>/`, plus PNGs when enabled. These summaries describe predictions and case metadata; they do not compute the three validation MAEs above.

Output names are deterministic: rerunning the same case can overwrite its files, and the run-summary CSV describes the current run. Use a separate `--output-dir` with an individual driver to retain parameter comparisons. Generated `outputs/` files are ignored by Git; the curated figure under `assets/` is versioned.

## Continuing this validation

- Keep NPZ inputs and `manifest.csv` synchronized when adding or replacing cases.
- Check one case through all four models after dependency or model changes, then run the full set before refreshing results.
- Archive the evaluation script, masks, per-case metrics, model parameters, code revision, and resolved environment alongside the next summary update.
- Document WRF configuration, data provenance, preprocessing, and reuse terms to make the full path from simulation to validation traceable.
