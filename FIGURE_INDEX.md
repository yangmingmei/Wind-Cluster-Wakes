# Publication plotting files · 8 October 2026

Each dataset's `main.py` calculates its benchmark and produces figures from the results. `--reference` redraws retained predictions. Generated PDF/PNG/SVG files are in `outputs/figures/`; publication PDFs and README previews are in `assets/`.

| Paper figure | Plotting source | Output stem |
|---|---|---|
| 3 · WRF, 50 m analytical output, 3+2 panels | [WRF/cluster_figure.py](WRF/cluster_figure.py), `predict()` / `draw()` | `wrf_model_heatmap` |
| 4 · WIPAFF tracks and layout | [WIPAFF/plotting.py](WIPAFF/plotting.py) | `wipaff_context` |
| 5 · WIPAFF profiles | Same WIPAFF function | `wipaff_profiles` |
| 6 · AWAKEN context and scores | [AWAKEN/plotting.py](AWAKEN/plotting.py), `summarize()` / `context()` | `awaken_context_and_scores` |
| 7 · AWAKEN multi-farm case, 25 m | [AWAKEN/cluster_figure.py](AWAKEN/cluster_figure.py), `predict()` / `draw()` | `awaken_representative_comparison` |

The additional `wipaff_downstream_deficit` plot remains available as a diagnostic, but is not numbered in the current manuscript. Figures 1–2 are conceptual schematics maintained as Visio sources in the manuscript workspace.

Figure 3 uses `ws10_wd270_Lmneutral`, with 50 m analytical output, 250 m top-down/ASM internal grids and the native 1 km WRF reference. Figure 7 uses `split_551`, with 25 m output/display grids, 250 m internal grids and interpolated 100 m height data. These illustrations are separate from the unchanged ensemble scoring grids. See the dataset READMEs for single-case commands, retained inputs and measured computational costs.
