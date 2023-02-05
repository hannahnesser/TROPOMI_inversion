import sys
from copy import deepcopy as dc
import glob
import xarray as xr
import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
pd.set_option('display.max_columns', 10)

# Custom packages
sys.path.append('.')
import gcpy as gc
import format_plots as fp
import inversion_settings as s

## ------------------------------------------------------------------------ ##
## Directories
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
data_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'

## ------------------------------------------------------------------------ ##
## Get overlap between states and grid cells
## ------------------------------------------------------------------------ ##
# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()
nstate = int(clusters.values.max())

# Open the 2019 state boundaries
ong_folders = glob.glob(f'{data_dir}ong/?_*/')
ong_folders.sort()

# Create storage for the masks
w_ong = pd.DataFrame(
            columns=[s.split('/')[-2][2:].lower() for s in ong_folders])

# Iterate through basin folders
for i, folder in enumerate(ong_folders):
    # Get name
    name = folder.split('/')[-2][2:].lower()

    # List all relevant files
    ong_files = glob.glob(f'{folder}*.csv')
    ong_files.sort()

    # Get the union of all files
    poly = []
    for file in ong_files:
        boundaries = pd.read_csv(file)
        x = boundaries['lon'].values
        y = boundaries['lat'].values
        poly.append(Polygon(np.column_stack((x, y))))
    poly = unary_union(poly)
    x, y = poly.exterior.coords.xy

    # Add a column to the mask
    w_ong[name] = gc.grid_shape_overlap(clusters, x, y, name)

# Save out w_city
w_ong.to_csv(f'{data_dir}ong/ong_mask.csv', header=True, index=False)