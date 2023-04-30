from os.path import join
from os import listdir
import sys
import copy
import calendar as cal
import xarray as xr
import numpy as np
from numpy.polynomial import polynomial as p
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# from matplotlib import cm
pd.set_option('display.max_columns', 15)

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
local = True

if local:
    # # Local preferences
    base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
    code_dir = base_dir + 'python'
    data_dir = base_dir + 'inversion_data'
    output_dir = base_dir + 'inversion_data'
    plot_dir = base_dir + 'plots'
else:
    # Cannon long-path preferences
    base_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_final/'
    code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
    data_dir = f'{base_dir}ProcessedDir'
    output_dir = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/inversion_data'

# Cannon preferences
# code_dir = sys.argv[1]
# base_dir = sys.argv[2]
# data_dir = f'{base_dir}ProcessedDir'
# output_dir = sys.argv[3]
# plot_dir = None

# Import Custom packages
sys.path.append(code_dir)
import config
# config.SCALE = config.PRES_SCALE
# config.BASE_WIDTH = config.PRES_WIDTH
# config.BASE_HEIGHT = config.PRES_HEIGHT
import gcpy as gc
import troppy as tp
import format_plots as fp
import inversion_settings as settings

# The prior_run can either be a list of files or a single file
# with all of the data for simulation
prior_run = f'{settings.year}_full.pkl'
suffix = '_w37_edf'
mod_suffix = '_w37_edf'
# prior_run = [f'{settings.year}{mm:02d}{dd:02d}_GCtoTROPOMI.pkl'
#              for mm in settings.months
#              for dd in settings.days]
# prior_run.sort()

# Compare to GOSAT?
compare_gosat = True
# gosat_dir = '/n/jacob_lab/Lab/seasasfs02/CH4_inversion/InputData/Obs/ULGOSAT_v9/2019'
# gosat = [f'{gosat_dir}/UoL-GHG-L2-CH4-GOSAT-OCPR-{settings.year}{mm:02d}{dd:02d}-fv9.0.nc'
#          for mm in settings.months for dd in settings.days]
gosat = None

# Define the blended albedo threshold
filter_on_blended_albedo = True
blended_albedo_threshold = 0.75
blended_albedo_bins = np.arange(0, 2.5, 0.1)

# Define a plain old albedo threshold
filter_on_albedo = True
albedo_threshold = 0.05
albedo_bins = np.arange(0, 1.1, 0.05)

# Define a seasonal latitudinal filter
filter_on_seasonal_latitude = True

# Remove latitudinal bias
remove_latitudinal_bias = False
remove_mean_bias = True
lat_bins = np.arange(10, 65, 5)

# Which analyses do you wish to perform?
analyze_biases = False

# Calculate the error variances?
calculate_so = False
err_min = 10

## ------------------------------------------------------------------------ ##
## Get grid information
## ------------------------------------------------------------------------ ##
lats, lons = gc.create_gc_grid(*settings.lats, settings.lat_delta,
                               *settings.lons, settings.lon_delta,
                               centers=False, return_xarray=False)

# GOSAT comparison grid
lats_l, lons_l = gc.create_gc_grid(*settings.lats, 2, *settings.lons, 2,
                                   centers=False, return_xarray=False)

groupby = ['LAT_CENTER_L', 'LON_CENTER_L', 'SEASON']
err_suffix = '_rg2rt'

## ------------------------------------------------------------------------ ##
## Load GOSAT data
## ------------------------------------------------------------------------ ##
print('-'*70)
print(f'Opening GOSAT data in {output_dir}')
gosat_data = gc.read_file(join(output_dir, f'{settings.year}_gosat.pkl'))
gosat_grid = gc.read_file(join(output_dir,
                               f'{settings.year}_gosat_gridded.nc'))

## ------------------------------------------------------------------------ ##
## Load TROPOMI data
## ------------------------------------------------------------------------ ##
print(f'Opening TROPOMI data in {output_dir}/{prior_run}')
data = gc.load_obj(join(output_dir, prior_run))

# Prnt summary
print('-'*70)
print('Original data (pre-filtering) is loaded.')
print(f'm = {data.shape[0]}')

# Update DIFF to apply to the mod_suffix
data[f'DIFF'] = (data[f'MOD{mod_suffix.upper()}'] - 
                                     data['OBS'])

## ------------------------------------------------------------------------ ##
## Process TROPOMI data
## ------------------------------------------------------------------------ ##
# Grid the TROPOMI data
trop_grid = data.groupby(['LAT_CENTER_L', 'LON_CENTER_L',
                          'DATE']).mean()[['OBS', 'BLENDED_ALBEDO']]
trop_grid = trop_grid.to_xarray().rename({'LAT_CENTER_L' : 'lats',
                                          'LON_CENTER_L' : 'lons',
                                          'DATE' : 'date'})

# Take the difference and save out some time information
trop_grid['DIFF'] = (trop_grid['OBS'] - gosat_grid)
trop_grid['GOSAT'] = gosat_grid
trop_grid = trop_grid.rename({'OBS' : 'TROPOMI'})
trop_grid['date'] = pd.to_datetime(trop_grid['date'].values,
                                   format='%Y%m%d')
trop_grid['MONTH'] = trop_grid['date'].dt.month
trop_grid['SEASON'] = trop_grid['date'].dt.season

# Create a dataframe
diff_grid = trop_grid.to_dataframe()[['GOSAT', 'TROPOMI', 
                                      'DIFF',
                                      'MONTH', 'SEASON', 'BLENDED_ALBEDO']]
diff_grid = diff_grid.dropna().reset_index()

# Cut into blended albedo bins and latitude bins
diff_grid['BLENDED_ALBEDO_BIN'] = pd.cut(diff_grid['BLENDED_ALBEDO'],
                                         blended_albedo_bins)
diff_grid['LAT_BIN'] = pd.cut(diff_grid['lats'], lat_bins)

# Also do a spatial analysis, grouped by month
trop_grid = trop_grid.groupby('date.season').mean()

## ------------------------------------------------------------------------ ##
## Initialize the plot
## ------------------------------------------------------------------------ ##
fig, ax = fp.get_figax(aspect=1, rows=2, cols=4, sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.1, hspace=0.1)

dmin = min(np.min(diff_grid['GOSAT'].values), 
           np.min(diff_grid['TROPOMI'].values))
dmax = max(np.max(diff_grid['GOSAT'].values), 
           np.max(diff_grid['TROPOMI'].values))

# Plot unfiltered data
for i, s in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
    d = diff_grid[diff_grid['SEASON'] == s]
    m, b, r, bias, std = gc.comparison_stats(d['GOSAT'].values,
                                             d['TROPOMI'].values)
    systematic_bias = d['DIFF'].std()
    print(s)
    # print(f'  y = {m}x + {b}')
    print(f'  R2            : {(r**2):.2f}')
    print(f'  bias          : {bias:.2f}')
    print(f'  std           : {std:.2f}')
    print(f'  regional bias : {systematic_bias:.2f}')
    ax[0, i] = fp.add_title(ax[0, i], s)
    c = ax[0, i].hexbin(d['GOSAT'].values, d['TROPOMI'].values, gridsize=20, 
                        cmap=fp.cmap_trans('inferno'), vmin=0, vmax=50, 
                        linewidths=0)
    ax[0, i].plot([dmin - 5, dmax + 5], [dmin - 5, dmax + 5], c='0.5', ls=':')
    _, _, r, _, _ = gc.comparison_stats(d['GOSAT'].values, d['TROPOMI'].values)
    spatial_bias = (d['TROPOMI'].values - d['GOSAT'].values).std()
    ax[0, i] = gc.add_stats_text(ax[0, i], r, spatial_bias)
    ax[0, i].set_xlim(dmin - 5, dmax + 5)
    ax[0, i].set_ylim(dmin - 5, dmax + 5)

## ------------------------------------------------------------------------ ##
## Make and apply observational mask
## ------------------------------------------------------------------------ ##
print('-'*70)

# Create a vector for storing the observational filter
obs_filter = np.ones(data.shape[0], dtype=bool)

# We always filter on clouds (this variable is a count of the number
# of nan values in the cloud_fraction variable, so where it is greater
# than zero, we wish to filter out the data)
cloud_filter = (data['CLOUDS'] == 0)
obs_filter = (obs_filter & cloud_filter)
m = obs_filter.sum()

# Incorporate other filters 
# filter_on_blended_albedo:
BAF_filter = ((data['MONTH'].isin(np.arange(6, 9, 1))) |
              (data['BLENDED_ALBEDO'] < blended_albedo_threshold))
obs_filter = (obs_filter & BAF_filter)

# filter_on_albedo:
albedo_filter = (data['ALBEDO_SWIR'] > albedo_threshold)
obs_filter = (obs_filter & albedo_filter)

# filter_on_seasonal_latitude:
latitude_filter = ((data['MONTH'].isin(np.arange(3, 12, 1))) |
                   (data['LAT'] <= 50))
obs_filter = (obs_filter & latitude_filter)

# Apply filter
data = data[obs_filter]

# Print summary
print('-'*70)
print('Data is filtered.')
print(f'm = {data.shape[0]}')
print('Mean bias = ', data['DIFF'].mean())
print('-'*70)

## ------------------------------------------------------------------------ ##
## Re-process TROPOMI data
## ------------------------------------------------------------------------ ##
# Grid the TROPOMI data
trop_grid = data.groupby(['LAT_CENTER_L', 'LON_CENTER_L',
                          'DATE']).mean()[['OBS', 'BLENDED_ALBEDO']]
trop_grid = trop_grid.to_xarray().rename({'LAT_CENTER_L' : 'lats',
                                          'LON_CENTER_L' : 'lons',
                                          'DATE' : 'date'})

# Take the difference and save out some time information
trop_grid['DIFF'] = (trop_grid['OBS'] - gosat_grid)
trop_grid['GOSAT'] = gosat_grid
trop_grid = trop_grid.rename({'OBS' : 'TROPOMI'})
trop_grid['date'] = pd.to_datetime(trop_grid['date'].values,
                                   format='%Y%m%d')
trop_grid['MONTH'] = trop_grid['date'].dt.month
trop_grid['SEASON'] = trop_grid['date'].dt.season

# Create a dataframe
diff_grid = trop_grid.to_dataframe()[['GOSAT', 'TROPOMI', 
                                      'DIFF',
                                      'MONTH', 'SEASON', 'BLENDED_ALBEDO']]
diff_grid = diff_grid.dropna().reset_index()

# Cut into blended albedo bins and latitude bins
diff_grid['BLENDED_ALBEDO_BIN'] = pd.cut(diff_grid['BLENDED_ALBEDO'],
                                         blended_albedo_bins)
diff_grid['LAT_BIN'] = pd.cut(diff_grid['lats'], lat_bins)

# Also do a spatial analysis, grouped by month
trop_grid = trop_grid.groupby('date.season').mean()

## ------------------------------------------------------------------------ ##
## Finalize the plot
## ------------------------------------------------------------------------ ##
# Plot unfiltered data
for i, s in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
    d = diff_grid[diff_grid['SEASON'] == s]
    m, b, r, bias, std = gc.comparison_stats(d['GOSAT'].values,
                                             d['TROPOMI'].values)
    systematic_bias = d['DIFF'].std()
    print(s)
    # print(f'  y = {m}x + {b}')
    print(f'  R2            : {(r**2):.2f}')
    print(f'  bias          : {bias:.2f}')
    print(f'  std           : {std:.2f}')
    print(f'  regional bias : {systematic_bias:.2f}')
    c = ax[1, i].hexbin(d['GOSAT'].values, d['TROPOMI'].values, gridsize=20,
                        cmap=fp.cmap_trans('inferno'), vmin=0, vmax=50, 
                        linewidths=0)
    ax[1, i].plot([dmin - 5, dmax + 5], [dmin - 5, dmax + 5], c='0.5', ls=':')
    _, _, r, _, _ = gc.comparison_stats(d['GOSAT'].values, d['TROPOMI'].values)
    spatial_bias = (d['TROPOMI'].values - d['GOSAT'].values).std()
    ax[1, i] = gc.add_stats_text(ax[1, i], r, spatial_bias)
    ax[1, i].set_xlim(dmin - 5, dmax + 5)
    ax[1, i].set_ylim(dmin - 5, dmax + 5)
    ax[1, i].set_xlabel('GOSAT XCH4', 
                        fontsize=config.LABEL_FONTSIZE*config.SCALE,
                        labelpad=config.LABEL_PAD/2)

# Add x axis labels
for i in range(2):
    for j in range(4):
        ax[i, j].tick_params(axis='both', which='both', 
                             labelsize=config.TICK_FONTSIZE*config.SCALE)

ax[0, 0].set_ylabel('TROPOMI XCH4',
                    fontsize=config.LABEL_FONTSIZE*config.SCALE,
                    labelpad=config.LABEL_PAD/2)

ax[1, 0].set_ylabel('Filtered TROPOMI XCH4',
                    fontsize=config.LABEL_FONTSIZE*config.SCALE,
                    labelpad=config.LABEL_PAD/2)

# Add colorbar
cax = fp.add_cax(fig, ax)
cb = fig.colorbar(c, cax=cax)
cb = fp.format_cbar(cb, cbar_title='Count')

fp.save_fig(fig, plot_dir, 'gosat_tropomi_comparison')

