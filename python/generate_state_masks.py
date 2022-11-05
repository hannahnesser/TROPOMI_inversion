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
## Get overlap between states and grid cells
## ------------------------------------------------------------------------ ##
# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()
nstate = int(clusters.values.max())

# Open the 2019 state boundaries
state = shapefile.Reader(f'{data_dir}states/2019_tl_us_state/tl_2019_us_state.shp')

# Create a numpy array for the information content analysis
w_state = pd.DataFrame(columns=[s.record[6] for s in state.shapeRecords()
                                if s.record[5] not in ['AK', 'HI', 'MP', 'GU',
                                                       'AS', 'PR', 'VI', 'DC']])
print(w_state)

# Iterate through each city
for j, shape in enumerate(state.shapeRecords()):
    if shape.record[6] in w_state.columns:
        # Add a row to w_state
        w_state[shape.record[6]] = gc.grid_shape_overlap(clusters, shape)

# Save out w_city
w_state.to_csv(f'{data_dir}states/states_mask.csv', header=True, index=False)
