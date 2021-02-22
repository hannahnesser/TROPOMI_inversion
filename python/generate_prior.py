from os.path import join
from os import listdir
import sys

import pickle
import numpy as np
import xarray as xr

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python'
data_dir = base_dir + 'inversion_data'
plot_dir = base_dir + 'plots'

# The emissions can either be a list of files or a single file
# with an annual average
year = 2019
months = np.arange(1, 12, 1) # excluding December for now
days = np.arange(1, 32, 1)

# Files
emis_file = f'{base_dir}/prior/total_emissions/HEMCO_diagnostics.{year}.nc'
clusters = f'{data_dir}/clusters_0.25x0.3125.nc'

# Set relative prior error covariance value
rel_err = 0.5


# Information on the grid
lat_bins = np.arange(10, 65, 5)
lat_min = 9.75
lat_max = 60
lat_delta = 0.25
lon_min = -130
lon_max = -60
lon_delta = 0.3125
buffers = [3, 3, 3, 3]

## ------------------------------------------------------------------------ ##
## Import custom packages
## ------------------------------------------------------------------------ ##
# Custom packages
sys.path.append(code_dir)
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
import gcpy as gc
import troppy as tp
import invpy as ip
import format_plots as fp

## -------------------------------------------------------------------------##
## Save out absolute and relative priors
## -------------------------------------------------------------------------##

## ------------------------------------------------------------------------ ##
## Define the inversion grid
## ------------------------------------------------------------------------ ##
# Get information on the lats and lons (edges of the domain) (this means
# removing the buffer grid cells)
lat_e, lon_e = gc.adjust_grid_bounds(lat_min, lat_max, lat_delta,
                                     lon_min, lon_max, lon_delta, buffers)

## -------------------------------------------------------------------------##
## Load raw emissions data
## -------------------------------------------------------------------------##
if type(emis_file) == list:
    # Open files
    emis_file = [join(data_dir, f) for f in emis_file
                 if f in listdir(data_dir)]
    emis = xr.open_mfdataset(emis_file)
else:
    emis = xr.open_dataset(join(data_dir, emis_file))

# Remove emissions from buffer grid cells
emis = gc.subset_data_latlon(emis, *lat_e, *lon_e)

if 'time' in emis.dims:
    # Average over time
    emis = emis.mean(dim='time')

# Select total emissions
emis = emis['EmisCH4_Total']

# Adjust units to Mg/km2/yr
emis *= 0.001*60*60*24*365*1000*1000

print('The minimum positive emission is',
      np.abs(emis.where(emis > 0).min()).values)

## -------------------------------------------------------------------------##
## Open clusters
## -------------------------------------------------------------------------##
clusters = xr.open_dataarray(clusters)
nstate = int(clusters.max().values)

## -------------------------------------------------------------------------##
## Generate relative prior and prior error covariance
## -------------------------------------------------------------------------##
xa = np.ones(nstate)
sa = rel_err*np.ones(nstate)

## -------------------------------------------------------------------------##
## Generate the absolute prior and prior error covariance
## -------------------------------------------------------------------------##
xa_abs = ip.clusters_2d_to_1d(clusters, emis)
sa_abs = sa*xa_abs

## -------------------------------------------------------------------------##
## Save out
## -------------------------------------------------------------------------##
gc.save_obj(xa, join(data_dir, 'xa.pkl'))
gc.save_obj(xa_abs, join(data_dir, 'xa_abs.pkl'))

gc.save_obj(sa, join(data_dir, 'sa.pkl'))
gc.save_obj(sa_abs, join(data_dir, 'sa_abs.pkl'))

