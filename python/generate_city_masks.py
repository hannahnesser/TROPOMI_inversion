import sys
from copy import deepcopy as dc
import glob
import xarray as xr
import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import Polygon, MultiPolygon
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

# Open the 2019 city boundaries
city = shapefile.Reader(f'{data_dir}cities/2019_tl_us_cbsa/tl_2019_us_cbsa.shp')

# Create a numpy array for the information content analysis
w_city = pd.DataFrame(
             columns=[s.record[3] for s in city.shapeRecords()
                      if (s.record[5] == 'M1') and 
                      (s.record[3].split(', ')[-1] not in ['AK', 'HI', 'PR'])])

# Iterate through each city
for j, shape in enumerate(city.shapeRecords()):
    if shape.record[3] in w_city.columns:
        # Add a column to w_city
        w_city[shape.record[3]] = gc.grid_shape_overlap(clusters, shape)

# Save out w_city
w_city.to_csv(f'{data_dir}cities/cities_mask.csv', header=True, index=False)