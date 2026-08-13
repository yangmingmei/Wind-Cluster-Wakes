# Wind Cluster Wakes

Open-source analytical modeling and validation of inter-farm wake effects in wind farm clusters.

Wind Cluster Wakes provides a computationally efficient analytical framework for predicting long-range wake interactions between wind farms.

Validation and case studies include:

- Airborne observations from WIPAFF
- Doppler-radar observations from the AWAKEN experiment
- Numerical simulations using WRF

## AWAKEN Doppler-radar observations

![Normalized wind-speed heatmap from AWAKEN Doppler-radar observations](docs/images/awaken-radar-wake-heatmap.png)

The normalized wind-speed heatmap highlights turbine wakes observed by Doppler radar at the King Plains wind farm during the AWAKEN experiment. It illustrates the spatial extent of the merged in-farm and downstream wake region, together with the turbine layout and inflow conditions. Terrain effect significantly influence the wakes

## WIPAFF airborne observations

![Observed and modeled wake cross-sections from a WIPAFF flight](docs/images/wipaff-flight-cross-sections.png)

Crosswind profiles at downstream distances from 5 to 45 km compare normalized wind speed measured during a WIPAFF research flight with predictions from an existing top-down model. The profiles illustrate the persistence and gradual recovery of the observed long-distance wake deficit.

## WRF numerical simulations

WRF simulations provide a numerical reference for assessing the analytical model across wind-farm layouts, atmospheric conditions, and inter-farm separation distances.
