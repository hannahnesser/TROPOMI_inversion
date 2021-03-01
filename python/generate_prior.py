from os.path import join
from os import listdir
import sys
import math

import pickle
import numpy as np
import xarray as xr

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams, colorbar, colors
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom
import cartopy.crs as ccrs
import cartopy

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
months = np.arange(1, 13, 1) # excluding December for now
days = np.arange(1, 32, 1)

# Files
emis_file = [f'{base_dir}/prior/total_emissions/\
HEMCO_diagnostics.{year:04d}{mm:02d}010000.nc'
             for mm in months]
# emis_file = f'{base_dir}/prior/total_emissions/HEMCO_diagnostics.{year}.nc'
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
# config.SCALE = config.PRES_SCALE
# config.BASE_WIDTH = config.PRES_WIDTH
# config.BASE_HEIGHT = config.PRES_HEIGHT
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
    emis = xr.open_mfdataset(emis_file)
else:
    emis = xr.open_dataset(join(data_dir, emis_file))

# Remove emissions from buffer grid cells
emis = gc.subset_data_latlon(emis, *lat_e, *lon_e)

if 'time' in emis.dims:
    # Average over time
    emis = emis.mean(dim='time')

    # Save summary file
    name = 'HEMCO_diagnostics.2019.nc'
    emis.to_netcdf(f'{base_dir}/prior/total_emissions/{name}')

# Print a summary table
summ = emis[[var for var in emis.keys() if var[:4] == 'Emis']]*emis['AREA']
summ *= 1e-9*(60*60*24*365) # Adjust units to Tg/yr
summ = summ.sum(dim=['lat', 'lon']).values
print(summ)

# Adjust units to Mg/km2/yr
emis *= 1e-3*(60*60*24*365)*(1000*1000)

## ---------------------------------- ##
## Plot
## ---------------------------------- ##
if plot_dir is not None:
    emissions = ['Wetlands', 'Livestock',
                ['Coal', 'Oil', 'Gas'],
                ['Wastewater', 'Landfills'],
                ['Termites', 'Seeps', 'BiomassBurn', 'Lakes'],
                ['Rice', 'OtherAnth']]
    titles = ['Wetlands', 'Livestock', 'Coal, Oil, and\nNatural Gas',
              'Wastewater\nand Landfills', 'Other Biogenic\nSources',
              'Other Anthropogenic\nSources']

    # Set colormap
    colormap = fp.cmap_trans('viridis')

    ncategory = len(emissions)
    fig, ax = fp.get_figax(rows=2, cols=math.ceil(ncategory/2),
                           maps=True, lats=emis.lat, lons=emis.lon)
    plt.subplots_adjust(hspace=0.5)
    cax = fp.add_cax(fig, ax, cbar_pad_inches=0.5)
    for i, axis in enumerate(ax.flatten()):
        axis = fp.format_map(axis, lats=emis.lat, lons=emis.lon)
        if type(emissions[i]) == str:
            e = emis['EmisCH4_%s' % emissions[i]].squeeze()
        elif type(emissions[i] == list):
            e = sum(emis['EmisCH4_%s' % em].squeeze()
                    for em in emissions [i])

        c = e.plot(ax=axis, cmap=colormap, vmin=0, vmax=5,
                   add_colorbar=False)
        cb = fig.colorbar(c, cax=cax, ticks=np.arange(0, 6, 1))
        cb = fp.format_cbar(cb, cbar_title=r'Emissions (Mg km$^2$ a$^{-1}$)')
        axis = fp.add_title(axis, titles[i])

    fp.save_fig(fig, plot_dir, 'prior_emissions_2019')

# Select total emissions
emis = emis['EmisCH4_Total']

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

