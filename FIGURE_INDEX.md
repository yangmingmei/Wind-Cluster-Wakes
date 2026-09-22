# Publication plotting files

Each dataset's `main.py` calculates its models, scores the new predictions, and calls its `plotting.py`. Add `--reference` to draw from retained publication predictions. Generated figures are PDF, PNG, and SVG in each dataset's `outputs/figures/`; submitted PDFs and previews remain in `assets/`.

| Paper figure | Plotting source | Output stem |
|---|---|---|
| 3 · WRF flow fields | [WRF/plotting.py](WRF/plotting.py), `make_figures()` | `wrf_model_heatmap` |
| 4 · WIPAFF tracks and layout | [WIPAFF/plotting.py](WIPAFF/plotting.py), `make_figures()` | `wipaff_context` |
| 5 · WIPAFF profiles | Same WIPAFF function | `wipaff_profiles` |
| 6 · WIPAFF downstream deficits | Same WIPAFF function | `wipaff_downstream_deficit` |
| 7 · AWAKEN context and scores | [AWAKEN/plotting.py](AWAKEN/plotting.py), `summarize()` / `context()` | `awaken_context_and_scores` |
| 8 · AWAKEN multi-farm case | [AWAKEN/cluster_figure.py](AWAKEN/cluster_figure.py), `predict()` / `draw()`, called by `plotting.py` | `awaken_representative_comparison` |

Shared fonts are configured by [plot_style.py](plot_style.py). Fig. 3 uses the fixed `ws10_wd270_Lmneutral` case. Fig. 8 uses the documented favorable-case selection rule and is separate from the ensemble ranking.

Paper Figs. 1–2 are conceptual schematics. Their original PDF, Visio/SVG files, and generating scripts remain in the full manuscript workspace under `editable_figures/` and `workflow/scripts/`; they are not needed by the three numerical entrypoints. The manuscript's `FIGURE_SOURCES.md` lists all eight original figure sources.

Fig. 8 uses Times New Roman titles and case `split_551`. All panels show the maximum available terrain-supported observation footprint, spanning −7.3 to 14.4 km from the King Plains exit and including farm interiors and regions between farms. The 2026-09-16 revision contains six field panels (a)-(f); the lateral-mean panel and observed-width strip have been removed. Numerical profile diagnostics remain in the exported CSV, including values, interpolation flags, observed-cell counts and observed width. Complete selected input, reference fields and the selection table are in `AWAKEN/data/cluster_figure/`; the fixed 35-period benchmark remains separate.
