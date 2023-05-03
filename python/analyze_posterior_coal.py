import sys
from copy import deepcopy as dc
import glob
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import shapefile
from shapely.geometry import Polygon, MultiPolygon
pd.set_option('display.max_columns', 10)

# Custom packages
sys.path.append('.')
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
import gcpy as gc
import troppy as tp
import invpy as ip
import format_plots as fp
import inversion_settings as s

# Define colormaps
sf_cmap_1 = plt.cm.PuOr_r(np.linspace(0.2, 0.5, 256))
sf_cmap_2 = plt.cm.PuOr_r(np.linspace(0.5, 1, 256))
sf_cmap = np.vstack((sf_cmap_1, sf_cmap_2))
sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
div_norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=1, vmax=3)

## ------------------------------------------------------------------------ ##
## Directories
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# DOFS_filter
DOFS_filter = 0.05

# Define basins of interest
interest = {'Illinois Basin' : ([36.25, 41.25, -91.5, -84.5], [0, 40], None),
            'Appalachia' : ([36.25, 42.5, -84.5, -77], [0, 40], None),
            'Powder River Basin': ([41, 46, -111.5, -103.5], [0, 40], None)}

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()

# Load area (km2)
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load ensemble members (relative posterior and DOFS), all of which are
# previously filtered on DOFS and have the BC elements removed as needed
dofs = pd.read_csv(f'{data_dir}ensemble/dofs.csv', index_col=0)
xhat = pd.read_csv(f'{data_dir}ensemble/xhat.csv', index_col=0)
ensemble = xhat.columns

# Load weighting matrices in units Mg/yr
w = pd.read_csv(f'{data_dir}sectors/w.csv')['coal'].T

# Get the posterior xhat_abs (this is n x 15)
xhat_diff_abs = (w.values[:, None]*(xhat - 1))
xhat_abs = (w.values[:, None]*xhat)

# Iterate through the regions
for name, reg in interest.items():
    c = clusters.where((clusters.lat > reg[0][0]) &
                       (clusters.lat < reg[0][1]) &
                       (clusters.lon > reg[0][2]) &
                       (clusters.lon < reg[0][3]), drop=True)
    c_idx = (c.values[c.values > 0] - 1).astype(int)

    tt_prior = w.values[c_idx].sum()*1e-6
    tt_post = xhat_abs.values[c_idx, :].sum(axis=0)*1e-6
    print(f'{name:<20s}:   {tt_prior:.2f}     {tt_post.mean():4.2f} ({tt_post.min():4.2f}, {tt_post.max():4.2f}) Tg/yr     {(tt_post.mean()/tt_prior - 1)*100:4.2f}')
