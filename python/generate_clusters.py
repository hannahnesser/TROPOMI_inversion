'''
This script generates a cluster file that meets HEMCO requirements for use in inversions. The file defines a unique integer key for every grid cell contained in the state vector. A grid cell is included in the state vector if it meets either the `emis_threshold` or `land_threshold` criteria.

   **Inputs**

   | ----------------- | -------------------------------------------------- |
   | Input             | Description                                        |
   | ----------------- | -------------------------------------------------- |
   | emis_file         | A file or files containing information on methane  |
   |                   | emissions from the prior run. This is typically    |
   |                   | given by HEMCO_diagnostics. The input here can be  |
   |                   | either a list of monthly files or a single file    |
   |                   | with an annual average.                            |
   | ----------------- | -------------------------------------------------- |
   | land_file         | A file containing information on land cover for    |
   |                   | inversion domain. This can be provided by the land |
   |                   | cover file referenced by HEMCO_Config.rc.          |
   | ----------------- | -------------------------------------------------- |
   | emis_threshold    | An emission threshold in Mg/km2/yr that gives the  |
   |                   | minimum anthropogenic emissions needed in a grid   |
   |                   | cell for its inclusion in the cluster file. The    |
   |                   | default value is 0.1.                              |
   | ----------------- | -------------------------------------------------- |
   | land_threshold    | A fractional land threshold that gives the minimum |
   |                   | fraction of a grid cell that must be land covered  |
   |                   | for inclusion in the cluster file. The default     |
   |                   | value is 0.25.                                     |
   | ----------------- | -------------------------------------------------- |

   **Outputs**

   | ----------------- | -------------------------------------------------- |
   | Output            | Description                                        |
   | ----------------- | -------------------------------------------------- |
   | clusters.nc       | A HEMCO-ready cluster file that contains 0s in all |
   |                   | grid cells not contained in the state vector and   |
   |                   | a unique integer key in every other grid cell.     |
   | ----------------- | -------------------------------------------------- |
   | clusters.png      | A plot of the clusters.                            |
   | ----------------- | -------------------------------------------------- |
'''

## -------------------------------------------------------------------------##
## Load packages and set environment defaults
## -------------------------------------------------------------------------##
from os.path import join
import sys

import math
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

# Custom packages
sys.path.append('.')
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
import gcpy as gc
import troppy as tp
import format_plots as fp
import inversion_settings as settings

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
# emis_file = [f'{data_dir}HEMCO_diagnostics.{settings.year}{mm:02d}010000.nc'
#              for mm in settings.months]
emis_file = f'{data_dir}HEMCO_diagnostics.{settings.year}.nc'

# We also need to define a land cover file
land_file = f'{base_dir}gc_inputs/GEOSFP.20200101.CN.025x03125.NA.nc'

# Set emission threshold in Mg/km2/yr (anthropogenic emissions only)
emis_threshold = 0.1

# Set the land threshold
land_threshold = 0.25

## -------------------------------------------------------------------------##
## Load raw emissions data
## -------------------------------------------------------------------------##
emis = gc.read_netcdf_file(emis_file)

# Remove emissions from buffer grid cells
emis = gc.subset_data_latlon(emis, *settings.lats, *settings.lons)

# Average over time
if 'time' in emis.dims:
    emis = emis.mean(dim='time')

# Separate out anthropogenic methane emissions
emis['EmisCH4_Anthro'] = (emis['EmisCH4_OtherAnth'] + emis['EmisCH4_Rice'] +
                          emis['EmisCH4_Wastewater'] + emis['EmisCH4_Coal'] +
                          emis['EmisCH4_Landfills'] + emis['EmisCH4_Gas'] +
                          emis['EmisCH4_Livestock'] + emis['EmisCH4_Oil'])
emis = emis['EmisCH4_Anthro']

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
lc = gc.subset_data_latlon(lc, *settings.lats, *settings.lons)

# Group together
lc = (lc['FRLAKE'] + lc['FRLAND'] + lc['FRLANDIC']).drop('time').squeeze()

## -------------------------------------------------------------------------##
## Define clusters
## -------------------------------------------------------------------------##
# Where the emissions are larger than the threshold, set the values to nan
# so that we can use iterate through them. Elsewhere, set the value to 0.
clusters = emis.where((emis < emis_threshold) & (lc < land_threshold))
clusters = clusters.where(clusters.isnull(), 0)

# Fill in the cluster values
clusters_vals = clusters.values
clusters_vals[clusters.isnull()] = np.arange(1, clusters.isnull().sum()+1)[::-1]
clusters.values = clusters_vals

# Print information about clusters
print(f'The inversion will optimize {int(clusters.max().values)} clusters.')

# Format for HEMCO
clusters = gc.define_HEMCO_std_attributes(clusters, name='Clusters')
clusters = gc.define_HEMCO_var_attributes(clusters, 'Clusters',
                                      long_name='Clusters generated for analytical inversion',
                                      units='none')
clusters.attrs = {'Title' : 'Clusters generated for analytical inversion'}

# Save out clusters
gc.save_HEMCO_netcdf(clusters, output_dir, 'clusters.nc')

# Print information on clusters
emis_tot = emis.sum().values
cluster_tot = emis.where(clusters['Clusters'] > 0).sum().values
cluster_min = emis.where(clusters['Clusters'] > 0).min().values
print(cluster_tot, emis_tot, cluster_tot/emis_tot)
print(cluster_min)

## -------------------------------------------------------------------------##
## Plot the result
## -------------------------------------------------------------------------##
fig, ax = fp.get_figax(maps=True, lats=clusters.lat, lons=clusters.lon)
ax = fp.format_map(ax, lats=clusters.lat, lons=clusters.lon)
c = clusters['Clusters'].plot(ax=ax, cmap=fp.cmap_trans('jet_r', nalpha=5),
                          add_colorbar=False, vmin=1)
cax = fp.add_cax(fig, ax, cbar_pad_inches=0.5)
cb = fig.colorbar(c, cax=cax)
cb = fp.format_cbar(cb, cbar_title=r'Cluster Number')
ax = fp.add_title(ax, 'Clusters')
fp.save_fig(fig, plot_dir, 'clusters')
