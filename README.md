
# File outline
bisg_raking.ipynb and synthetic.ipynb are the two jupyter notebooks used to replicate our experiments.

The other files - decision_functions.py, metrics.py, viz.py, and utils.py - contain helper functions to discretize probabilities, calculate accuracy and fidelity, plotting functions for visualizations, and miscellaneous helper functions respectively.

# Data
 
bisg_raking.ipynb uses data from two files, to be placed in the `data/` folder. 

### Data Points
The file of data points, labels, and posteriors is `data/df_agg_nc2020_dataverse.pkl`, retrivable online from [here](https://dataverse.harvard.edu/file.xhtml?fileId=7053328&version=1.0). The full data citation is:
```
Philip Greengard and Andrew Gelman. Replication Data for: BISG: When inferring race or ethnicity, does
it matter that people often live near their relatives?, 2023. URL https://doi.org/10.7910/DVN/QIM4UF.
```

### Map Shapefile
The shapefile used for the geographical map visualization of the counties in North Carolina is `data/cb_2022_us_county_500k/cb_2022_us_county_500k.shp`. These cartographic boundaries are sourced from the [US Census Bureau](https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html). Specifically, use the .shp file from the 2022 counties shapefile, at the 500,000 scale [zip link here](https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip).

