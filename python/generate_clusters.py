'''
This python script generates the clusters for an analytic inversion

Inputs:
    emis      This contains the prior run emissions. It can either be a list
              of monthly files or a yearly average.
'''

## -------------------------------------------------------------------------##
## Load packages and set environment defaults
## -------------------------------------------------------------------------##
import xarray as xr
import matplotlib.pyplot as plt
import math
import numpy as np

from os.path import join
from os import listdir
import sys

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'prior/total_emissions/'
output_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'

# The emissions can either be a list of files or a single file
# with an annual average
year = 2019
months = np.arange(1, 13, 1) # excluding December for now
days = np.arange(1, 32, 1)
emis_file = [f'{data_dir}HEMCO_diagnostics.{year}{mm:02d}010000.nc'
             for mm in months]
# emis_file = f'{data_dir}HEMCO_diagnostics.{year}.nc'

# We also need to define a land cover file
land_file = f'{base_dir}gc_inputs/GEOSFP.20200101.CN.025x03125.NA.nc'

# Set emission threshold in Mg/km2/yr (we use anthropogenic emissions
# only)
emis_threshold = 0.1

# Set the land threshold
land_threshold = 0.25

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
import format_plots as fp

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
emis = gc.load_files(emis_file, data_dir=data_dir)

# Remove emissions from buffer grid cells
emis = gc.subset_data_latlon(emis, *lat_e, *lon_e)

# Separate out anthropogenic methane emissions
emis = (emis['EmisCH4_OtherAnth'] + emis['EmisCH4_Rice'] +
        emis['EmisCH4_Wastewater'] + emis['EmisCH4_Coal'] +
        emis['EmisCH4_Landfills'] + emis['EmisCH4_Gas'] +
        emis['EmisCH4_Livestock'] + emis['EmisCH4_Oil'])

# Average over time
if 'time' in emis.dims:
    emis = emis.mean(dim='time').load()

# Adjust units to Mg/km2/yr
emis *= 0.001*60*60*24*365*1000*1000

print('The minimum positive emission is',
      np.abs(emis.where(emis > 0).min()).values)

# Create histogram of prior emissions
fig, ax = fp.get_figax(aspect=1.75)
ax.hist(emis.values.reshape(-1,), bins=np.arange(0, 7, 0.25),
        color=fp.color(4))
ax.axvline(emis_threshold, color=fp.color(7), ls='--')
ax = fp.add_labels(ax, r'Emissions (Mg km$^2$ yr$^{-1}$)', 'Count')
ax = fp.add_title(ax, 'Distribution of Prior Emissions')
fp.save_fig(fig, plot_dir, 'prior_emis_distribution')

## -------------------------------------------------------------------------##
## Load raw land cover data
## -------------------------------------------------------------------------##
lc = xr.open_dataset(land_file)

# Subset to lat/lon grid
lc = gc.subset_data_latlon(lc, *lat_e, *lon_e)

# Group together
lc = (lc['FRLAKE'] + lc['FRLAND'] + lc['FRLANDIC']).drop('time').squeeze()

## -------------------------------------------------------------------------##
## Define clusters
## -------------------------------------------------------------------------##
# Where the emissions are larger than the threshold, set the values to nan
# so that we can use iterate through them. Elsewhere, set the value to 0.
emis = emis.where((emis < emis_threshold) & (lc < land_threshold))
emis = emis.where(emis.isnull(), 0)

# Fill in the cluster values
emis.values[emis.isnull()] = np.arange(1, emis.isnull().values.sum()+1)[::-1]

# Print information about clusters
print(f'The inversion will optimize {int(emis.max().values)} clusters.')

# Format for HEMCO
emis = gc.define_HEMCO_std_attributes(emis, name='Clusters')
emis = gc.define_HEMCO_var_attributes(emis, 'Clusters',
                                      long_name='Clusters generated for analytical inversion',
                                      units='none')
emis.attrs = {'Title' : 'Clusters generated for analytical inversion'}

# Save out clusters
gc.save_HEMCO_netcdf(emis, output_dir, 'clusters_0.25x0.3125.nc')

## -------------------------------------------------------------------------##
## Plot the result
## -------------------------------------------------------------------------##
fig, ax = fp.get_figax(maps=True, lats=emis.lat, lons=emis.lon)
cax = fp.add_cax(fig, ax, cbar_pad_inches=0.5)
ax = fp.format_map(ax, lats=emis.lat, lons=emis.lon)
c = emis['Clusters'].plot(ax=ax, cmap=fp.cmap_trans('jet_r', nalpha=5),
                          add_colorbar=False, vmin=1)
cb = fig.colorbar(c, cax=cax)
cb = fp.format_cbar(cb, cbar_title=r'Cluster Number')
ax = fp.add_title(ax, 'Clusters')
fp.save_fig(fig, plot_dir, 'clusters')
