# Data sources and attribution

The compact package begins with processed observations and retained numerical states. Each dataset's `data/` directory contains everything required by its `main.py`; the commands do not retrieve the original campaigns or integrate WRF.

## WRF

The 30 supplied WRF v4.6.0 final states were processed to a 119 m reference grid with 165 turbines. Compact NPZ files preserve the grid, turbine positions, upstream inflow, diagnosed L and boundary-layer height. No public download DOI or redistribution license for these study inputs was supplied. Namelists, initial/boundary conditions and the WRF source build are outside this distribution.

## WIPAFF and DeepOWT

Original aircraft metadata identifies the [WIPAFF series](https://doi.org/10.1594/PANGAEA.902845), [Flight 01](https://doi.org/10.1594/PANGAEA.902843) and [Flight 07](https://doi.org/10.1594/PANGAEA.902914), with Creative Commons Attribution 4.0. Processed profiles and selected seconds retain this attribution. Associated study: [Platis et al. (2018)](https://www.nature.com/articles/s41598-018-20389-y).

The selected layout derives from DeepOWT 1.21.2, 2016 Q3 state, [Zenodo record 5933967](https://doi.org/10.5281/zenodo.5933967). The original workflow checked GeoJSON MD5 `e18481283285cb7f50086f06e84a1881` and selected 208 positions. The compact package retains the selected CSV instead of the global GeoJSON. Source description: [Hoeser, Feuerstein and Kuenzer (2022)](https://essd.copernicus.org/articles/14/4251/2022/). The separate data-release license was not independently established in the earlier workspace; the journal article license was not used to infer it.

## AWAKEN

Provenance begins at 35 supplied corrected, lag-aware ensembles. Compact cases retain the observation field, common support, terrain height, grids, turbine geometry, inflow/stability metadata and normalized predictions. `AWAKEN/data/provenance/compact_derivation.json` maps every case to its original research-workspace sources and the four geometry/identity arrays added during compaction. Source paths in this provenance record describe the historical inputs, not runtime dependencies.

Campaign/radar references retained from the study: [Moriarty et al. (2024)](https://doi.org/10.1063/5.0141683), [Abraham et al. (2024)](https://doi.org/10.1088/1742-6596/2767/9/092037). Operational status and some Garfield turbine types were not supplied. Raw radar retrieval and parameter-search caches are outside the compact workflow.

## Turbine curves and code license

WRF/WIPAFF turbine workbooks name `case_neutral/wind-turbine-1.tbl` as their source; the original `.tbl` files were absent. WIPAFF uses the supplied curve as a surrogate with type-specific geometry. AWAKEN uses four supplied NREL/GE proxy performance CSVs. Manufacturer certification and reuse permissions were not independently established in the original workspace.

No project-wide source-code license was present in the supplied workspace. This preparation preserves provenance without assigning a new license or publishing the repository. Existing dataset attribution remains applicable to derivatives.
