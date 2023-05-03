import sys
from copy import deepcopy as dc
import glob
import xarray as xr
import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import Polygon, MultiPolygon
from pyproj import Proj, transform
pd.set_option('display.max_columns', 10)

# Custom packages
sys.path.append('.')
import gcpy as gc
import inversion_settings as s

## ------------------------------------------------------------------------ ##
## Directories
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
data_dir = base_dir + 'inversion_data/'

## ------------------------------------------------------------------------ ##
## Get overlap between metropolitan statistical areas and grid cells
## ------------------------------------------------------------------------ ##
# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()
nstate = int(clusters.values.max())

# Open populations
# pop = pd.read_csv(f'{data_dir}cities/urban_areas_pop.csv', header=0)

# Open the air basin boundaries
airbasin = shapefile.Reader(f'{data_dir}states/ca_air_basins/CaAirBasin_WGS84_R2.shp')

# Create storage for the masks
w_ca = pd.DataFrame(columns=[s.record[4] for s in airbasin.shapeRecords()])

# Iterate through each airbasin
for j, shape in enumerate(airbasin.shapeRecords()):
    # Add a column to w_city
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    w_ca[shape.record[4]] = gc.grid_shape_overlap(clusters, x, y, 
                                                    shape.record[4])

# # And do the Cusworth domain
# x = [-119, -119, -117, -117]
# y = [33.5, 34.5, 34.5, 33.5]
# w_ca['Cusworth et al. (2020)'] = gc.grid_shape_overlap(clusters, x, y)

# Save out w_city
w_ca.to_csv(f'{data_dir}states/airbasin_mask.csv', header=True, index=False)
