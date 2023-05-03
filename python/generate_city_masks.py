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

# Open populations
pop = pd.read_csv(f'{data_dir}cities/urban_areas_pop.csv', header=0)

# Open the 2019 city boundaries
city = shapefile.Reader(f'{data_dir}cities/2019_tl_urban_areas/tl_2019_us_uac10_buffered.shp')

# Create storage for the masks
w_city = pd.DataFrame(
             columns=[s.record[2] for s in city.shapeRecords()
                      if (s.record[3][-4:] == 'Area') and 
                      (s.record[2] in pop['Name'].values) and
                      (s.record[2].split(', ')[-1] not in ['AK', 'HI', 'PR'])])

# Iterate through each city
for j, shape in enumerate(city.shapeRecords()):
    if shape.record[2] in w_city.columns:
        # Add a column to w_city
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        w_city[shape.record[2]] = gc.grid_shape_overlap(clusters, x, y, 
                                                        shape.record[2])

# Save out w_city
w_city.to_csv(f'{data_dir}cities/urban_areas_mask.csv', header=True, index=False)
