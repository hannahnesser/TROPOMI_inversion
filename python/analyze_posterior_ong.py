from os.path import join
from os import listdir
import sys
import glob
import copy
import math
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
from scipy.stats import probplot as qq
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Patch as patch
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import imageio
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

## ------------------------------------------------------------------------ ##
## Directories
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'

## ------------------------------------------------------------------------ ##
## Set plotting preferences
## ------------------------------------------------------------------------ ##
# Colormaps
plasma_trans = fp.cmap_trans('plasma')
plasma_trans_r = fp.cmap_trans('plasma_r')
rdbu_trans = fp.cmap_trans_center('RdBu_r', nalpha=70)
r_trans = fp.cmap_trans('Reds', nalpha=100)
yor_trans = fp.cmap_trans('YlOrRd', nalpha=100)
viridis_trans_r = fp.cmap_trans('viridis_r')
viridis_trans = fp.cmap_trans('viridis')
magma_trans = fp.cmap_trans('magma')
# print(viridis_trans)

# sf_cmap_1 = plt.cm.Reds(np.linspace(0, 0.5, 256))
# sf_cmap_2 = plt.cm.Blues(np.linspace(0.5, 1, 256))
# sf_cmap = np.vstack((sf_cmap_1, sf_cmap_2))
# sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
# div_norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=6)

sf_cmap_1 = plt.cm.PuOr_r(np.linspace(0.2, 0.5, 256))
sf_cmap_2 = plt.cm.PuOr_r(np.linspace(0.5, 1, 256))
sf_cmap = np.vstack((sf_cmap_1, sf_cmap_2))
sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
div_norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=1, vmax=3)

diff_cmap_1 = plt.cm.bwr(np.linspace(0, 0.5, 256))
diff_cmap_2 = plt.cm.bwr(np.linspace(0.5, 1, 256))
diff_cmap = np.vstack((diff_cmap_1, diff_cmap_2))
diff_cmap = colors.LinearSegmentedColormap.from_list('diff_cmap', diff_cmap)
diff_div_norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=1)

# Small (i.e. non-default) figure settings
small_map_kwargs = {'draw_labels' : False}
small_fig_kwargs = {'max_width' : 4,
                    'max_height' : 3.5}

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# DOFS_filter
DOFS_filter = 0.05

# Define file names
f = 'rg2rt_10t_w404_rf0.25_sax0.75_poi80.0'
xa_abs_file = 'xa_abs_w404.nc'
w_file = 'w_w404.csv'
optimize_BC = False

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc')

# Load prior (Mg/km2/yr)
xa_abs = xr.open_dataarray(f'{data_dir}{xa_abs_file}').values.reshape((-1, 1))
xa_abs_base = xr.open_dataarray(f'{data_dir}xa_abs.nc').values.reshape((-1, 1))
xa_ratio = xa_abs/xa_abs_base
xa_ratio[(xa_abs_base == 0) & (xa_abs == 0)] = 1 # Correct for the grid cell with 0 emisisons
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load weighting matrix
w = pd.read_csv(f'{data_dir}{w_file}')
w_rel = w.div(w.sum(axis=1), axis=0)
w_rel = w_rel.fillna(0)

# Create a mask using Lu Shen's definition of "oil and natural gas" grid
# cells from his 2022 ACP paper
w_mask = copy.deepcopy(w)
w_mask = w_mask.where(w*area/1e3 > 0.5, 0)
w_mask = w_mask.where(w*area/1e3 <= 0.5, 1)

# ?
w['total'] = w.sum(axis=1)

# Load posterior and DOFS
dofs = np.load(f'{data_dir}posterior/dofs2_{f}.npy').reshape((-1, 1))
xhat = np.load(f'{data_dir}posterior/xhat_fr2_{f}.npy').reshape((-1, 1))
# dofs = np.nan_to_num(dofs, 0)

# BC alteration
if optimize_BC:
    print('-'*30)
    print('Boundary condition optimization')
    print(' N E S W')
    print('xhat : ', xhat[-4:])
    print('dofs : ', dofs[-4:])
    print('-'*30)
    xhat = xhat[:-4]
    dofs = dofs[:-4]

# Print information
print('-'*30)
print(f'We optimize {(dofs >= DOFS_filter).sum():d} grid cells, including {xa_abs[dofs >= DOFS_filter].sum():.2E}/{xa_abs.sum():.2E} = {(xa_abs[dofs >= DOFS_filter].sum()/xa_abs.sum()*100):.2f}% of prior emissions. This\nproduces {dofs[dofs >= DOFS_filter].sum():.2f} ({dofs.sum():.2f}) DOFS with an xhat range of {xhat.min():.2f} to {xhat.max():.2f}. There are {len(xhat[xhat < 0]):d} negative values.')
print('-'*30)

# Filter on DOFS filter
xhat[dofs < DOFS_filter] = 1
dofs[dofs < DOFS_filter] = 0

# Calculate xhat abs
xhat_abs = (xhat*xa_abs)

# print(f'{f}, maximum: {xhat.max():.2f}, minimum: {xhat.min():.2f} DOFS: {dofs.sum():.2f} ({dofs[dofs >= DOFS_filter].sum():.2f}), negative values: {len(xhat[xhat < 0])}')

# Get county outlines for high resolution results
reader = shpreader.Reader(f'{data_dir}counties/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

# Get basins

