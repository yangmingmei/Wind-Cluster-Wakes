# WIPAFF validation

Four wake models compared with **Flights 01 and 07**, five downstream transects per flight, and the **208-turbine** Amrumbank West / Nordsee Ost / Meerwind Sued/Ost cluster. Use the [shared environment](../README.md#one-environment-for-all-datasets), then run from the repository root:

```bash
python WIPAFF/main.py
```

This recalculates all four models, scores the new profiles, and draws article Figs. 4–6. Add `--check-reference` to check publication predictions, or `--reference` for a quick redraw from retained profiles.

## Inputs and settings

`data/` contains the selected `turbine_layout.csv`, two unsmoothed `*_profiles.csv` inputs, two `*_selected_seconds.csv` tracks for the context plot, `reference_profiles.csv`, and `10MW_turbine_curve.xlsx`. Full flight exports and the global turbine database are excluded.

- Observations are quality-controlled, 0.2 km crosswind bins. Each transect retains its wind-from direction, reference speed, height, and actual downstream distance.
- Top-down: H = 1000 m, z0 = 0.0002 m, 250 m streamwise grid, recovery scale 1.
- Flight 01: L = 1e9 m (near neutral). Flight 07: prescribed L = +500 m, a modeling scenario rather than a surface-flux measurement.
- Engineering baselines: TI = 0.09. Turbine heights and diameters follow the layout; the supplied CT/CP workbook is a surrogate curve.

Change case settings in `main.py`, dataset adapters in `models.py`, or shared physics in `../wake_models.py`.

## Results

| Model | Mean section RMSE | Mean section MAE | Mean absolute section bias |
|---|---:|---:|---:|
| Top-down | 0.073345 | 0.061975 | 0.018295 |
| Array-stability + TurboPark | 0.075846 | 0.063293 | 0.019999 |
| TurboPark | 0.078180 | 0.066793 | 0.024846 |
| Gaussian | 0.089479 | 0.080618 | 0.039406 |

Errors use normalized speed U/Uref, unsmoothed profiles, and equal weight for each of the ten sections. Absolute bias is calculated within each section before averaging. Flight 41 is outside this downstream benchmark.

## Generated files

- `outputs/profiles.csv`: newly predicted profiles.
- `outputs/metrics_by_section.csv`, `metrics_summary.csv`: section errors and model ranking.
- `outputs/figures/wipaff_context.{pdf,png,svg}`: Fig. 4, tracks and turbine layout.
- `outputs/figures/wipaff_profiles.{pdf,png,svg}`: Fig. 5, observations and new model profiles.
- `outputs/figures/wipaff_downstream_deficit.{pdf,png,svg}`: Fig. 6, downstream deficit diagnostics.
- `outputs/figure_data/`: plotted numerical data; `outputs/verification.json`: reference differences.

`plotting.py::make_figures` draws all three figures. Display curves use the same centered five-bin median for observations and every model. The deficit diagram is a display diagnostic; error scores remain unsmoothed. `assets/` preserves the submitted PDFs and previews. Campaign attribution is in [DATA_SOURCES.md](../DATA_SOURCES.md).
