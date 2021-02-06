import pickle
from os.path import join
from os import listdir
import sys
import datetime

import xarray as xr
import numpy as np
import pandas as pd
from itertools import product

import matplotlib.pyplot as plt

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
# Perform the raw data analysis?
Analysis = False

# Make plots?
Plots = True

# Set directories
code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
# code_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/python'
data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/ProcessedDir'
processed_data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/prior/summary'
plot_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/plots'

# Define file name format
files = 'YYYYMMDD_GCtoTROPOMI.pkl'

# Set years, months, and dates
year = 2019
months = np.arange(1, 12, 1) # Excluding December for now
days = np.arange(1, 32, 1)

# Information on the grid
lat_bins = np.arange(10, 65, 5)
lat_min = 9.75
lat_max = 60
lat_delta = 0.25
lon_min = -130
lon_max = -60
lon_delta = 0.3125
buffers = [3, 3, 3, 3]

## -------------------------------------------------------------------------##
## Import additional packages
## -------------------------------------------------------------------------##
sys.path.append(code_dir)
import config
config.SCALE = config.PRES_SCALE
import gcpy as gc
import troppy as tp
import format_plots as fp

## -------------------------------------------------------------------------##
## Define functions
## -------------------------------------------------------------------------##
def cum_stats(d1, d2):
    # Count
    count = d1['count'] + d2['count']

    # Mean
    mean = (d1['count']*d1['mean'] + d2['count']*d2['mean'])/count

    # Standard deviation
    std = d1['count']*(d1['std']**2 + (d1['mean'] - mean)**2)
    std += d2['count']*(d2['std']**2 + (d2['mean'] - mean)**2)
    std = np.sqrt(std/count)

    # RMSE
    rmse = (d1['count']*d1['rmse']**2 + d2['count']*d2['rmse']**2)/count
    rmse = np.sqrt(rmse)

    stats = pd.DataFrame({'index' : d1['index'],
                         'count' : count, 'mean' : mean,
                         'std' : std, 'rmse' : rmse})
    stats = stats.fillna(0)
    return stats

## -------------------------------------------------------------------------##
## Analysis
## -------------------------------------------------------------------------##
if Analysis:
    # Get information on the lats and lons (edges of the domain)
    lat_edges, lon_edges = gc.adjust_grid_bounds(lat_min, lat_max, lat_delta,
                                                 lon_min, lon_max, lon_delta,
                                                 buffers)

    # Generate a grid on which to save out average difference
    # diff_grid = gc.create_gc_grid(*lat_edges, lat_delta, *lon_edges, lon_delta,
    #                                centers=False, return_xarray=True)
    # diff_grid = xr.Dataset({'count' : diff_grid, 'sum' : diff_grid})

    # Loop through the data
        #  Create a dataframe filled with 0s to store summary yearly data
        # lat_intervals = pd.cut(lat_bins, lat_bins)[1:]
        # year_summ = pd.DataFrame(index=lat_intervals,
        #                          columns=['count', 'mean',
        #                                   'std', 'rmse'])
        # year_summ = year_summ.reset_index()
        # year_summ = year_summ.append({'index' : 'Total'}, ignore_index=True)
        # year_summ = year_summ.replace(np.nan, 0)

    # Load data for the year
    data = np.array([]).reshape(0, 11)
    for m in months:
        print('Analyzing data for month %d' % m)
        # Iterate through the days in the month, beginning by initializing
        # on day 1
        # data = np.array([]).reshape(0, 11)
        for d in days:
            # Create the file name for that date
            file = files.replace('YYYY', '%04d' % year)
            file = file.replace('MM', '%02d' % m)
            file = file.replace('DD', '%02d' % d)

            # Check if that file is in the data directory
            if file not in listdir(data_dir):
                print('Data for %04d-%02d-%02d is not in the data directory.'
                      % (year, m, d))
                continue

            # Load the data
            # The columns are: 0 OBS, 1 MOD, 2 LON, 3 LAT,
            # 4 iGC, 5 jGC, 6 PRECISION, 7 ALBEDO_SWIR,
            # 8 ALBEDO_NIR, 9 AOD, 10 MOD_COL
            new_data = gc.load_obj(join(data_dir, file))['obs_GC']
            data = np.concatenate((data,
                                   gc.load_obj(join(data_dir,
                                                    file))['obs_GC']))

    # Create a dataframe from the data
    data = pd.DataFrame(data, columns=['OBS', 'MOD', 'LON', 'LAT',
                                       'iGC', 'jGC', 'PRECISION',
                                       'ALBEDO_SWIR', 'ALBEDO_NIR',
                                       'AOD', 'MOD_COL'])

    # Create a column for the blended albedo filter
    data['FILTER'] = tp.blended_albedo_filter(data, data['ALBEDO_SWIR'],)

    # Subset data
    data = data[['LON', 'LAT', 'OBS', 'MOD', 'DIFF']]

    # Calculate model - observation
    data['DIFF'] = data['MOD'] - data['OBS']


    # Take the difference and apply it to a grid so that we can track
    # the spatial distribution of the average difference (this will be
    # an annual average)

    # We begin by calculating the nearest latitude center
    nearest_lat = gc.nearest_loc(data['LAT'].values,
                                 diff_grid.lats.values)
    nearest_lat = diff_grid.lats.values[nearest_lat]
    data['LAT_CENTER'] = nearest_lat

    # And the nearest longitude center
    nearest_lon = gc.nearest_loc(data['LON'].values,
                                 diff_grid.lons.values)
    nearest_lon = diff_grid.lons.values[nearest_lon]
    data['LON_CENTER'] = nearest_lon

    # We then group by the latitude and longitude centers and
    # convert it to an xarray that is added to diff_grid,
    #which tracks the totals
    data_grid = data.groupby(['LAT_CENTER', 'LON_CENTER']).mean()['DIFF']
    # data_grid = data_grid.agg({'DIFF' : ['count', 'sum']})['DIFF']
    # data_grid = data_grid.to_xarray().rename({'LAT_CENTER' : 'lats',
    #                                           'LON_CENTER' : 'lons'})

    # data_grid = data_grid.fillna(0)
    # diff_grid += data_grid
    # diff_grid = diff_grid.fillna(0)

    # Now return to tracking other statistics. This time we are
    # interested in latitude bin instead of grid cell.
    # Group the difference between model and observation by
    # latitude bin
    data['LAT_BIN'] = pd.cut(data['LAT'], lat_bins)
    month_summ = data.groupby('LAT_BIN')
    month_summ = month_summ.agg({'DIFF' : ['count', 'mean',
                                           'std', gc.rmse]})
    month_summ = month_summ['DIFF'].reset_index()

    # Add in a total summary for that month
    month_summ = month_summ.append({'LAT_BIN' : 'Total',
                        'count' : data['DIFF'].shape[0],
                        'mean' : data['DIFF'].mean(),
                        'std' : data['DIFF'].std(),
                        'rmse' : gc.rmse(data['DIFF'])},
                         ignore_index=True)

    # Save out monthly summary
    file_name = '%04d%02d_summary.csv' % (y, m)
    month_summ.to_csv(join(data_dir, file_name))

    # Add monthly summary to yearly summary
    month_summ = month_summ.fillna(0)
    year_summ = cum_stats(year_summ, month_summ)

    # Now we will calculate the average gridded difference
    diff_grid = diff_grid['sum']/diff_grid['count']

    # Now save out yearly summary
    print('Code Complete.')
    print(year_summ)
    file_name = '%04d_summary' % y
    year_summ.to_csv(join(data_dir, file_name + '.csv'))
    diff_grid.to_netcdf(join(data_dir, file_name + '.nc'))

## -------------------------------------------------------------------------##
## Plots
## -------------------------------------------------------------------------##
for y in years:
    ## ---------------------------------------------------------------------##
    ## Seasonal bias summary
    ## ---------------------------------------------------------------------##
    seas_bias = pd.DataFrame(columns=['LAT_BIN', 'count', 'mean',
                                      'std', 'rmse'])
    for m in months:
        month_summ = pd.read_csv(join(processed_data_dir,
                                      '%04d%02d_summary.csv' % (y, m)),
                                 index_col=0)
        month_summ.iloc[-1, 0] = m
        seas_bias = seas_bias.append(month_summ.iloc[-1, :])

    seas_bias = seas_bias.rename(columns={'LAT_BIN' : 'month'})
    seas_bias['month'] = pd.to_datetime(seas_bias['month'], format='%m')
    seas_bias['month'] = seas_bias['month'].dt.month_name().str[:3]

    # Make figure
    fig, ax = fp.get_figax(aspect=1.75)
    ax.errorbar(seas_bias['month'], seas_bias['mean'],
                yerr=seas_bias['std'], color=fp.color(4))
    ax = fp.add_labels(ax, 'Month', 'Model - Observation')
    ax = fp.add_title(ax, 'Seasonal Bias in Prior Run')
    fp.save_fig(fig, plot_dir, 'prior_seasonal_bias')

    ## ---------------------------------------------------------------------##
    ## Latitudinal bias summary
    ## ---------------------------------------------------------------------##
    lat_bias = pd.read_csv(join(processed_data_dir, '%04d_summary.csv' % y),
                           index_col=0)
    print(lat_bias)
    lat_bias = lat_bias.iloc[:-1, :]
    lat_bias['index'] = (lat_bias['index'].str.split(r'\D', expand=True)
                         .iloc[:, [1, 3]]
                         .astype(int).mean(axis=1))

    # Make figure
    fig, ax = fp.get_figax(aspect=1.75)
    ax.errorbar(lat_bias['index'], lat_bias['mean'],
                yerr=lat_bias['std'], color=fp.color(4))
    ax.set_xticks(np.arange(10, 70, 10))
    ax = fp.add_labels(ax, 'Month', 'Model - Observation')
    ax = fp.add_title(ax, 'Latitudinal Bias in Prior Run')
    fp.save_fig(fig, plot_dir, 'prior_latitudinal_bias')

    ## ---------------------------------------------------------------------##
    ## Area bias summary
    ## ---------------------------------------------------------------------##
    spat_bias = xr.open_dataarray(join(processed_data_dir,
                                       '%04d_summary.nc' % y))
    print(spat_bias.where(spat_bias.lats > 55, drop=True).max())

    fig, ax = fp.get_figax(maps=True, lats=spat_bias.lats, lons=spat_bias.lons)
    c = spat_bias.plot(ax=ax, cmap='RdBu_r', vmin=-30, vmax=30,
                   add_colorbar=False)
    cax = fp.add_cax(fig, ax)
    cb = fig.colorbar(c, ax=ax, cax=cax)
    cb = fp.format_cbar(cb, 'Model - Observation')
    ax = fp.format_map(ax, spat_bias.lats, spat_bias.lons)
    ax = fp.add_title(ax, 'Spatial Distribution of Average Bias')

    fp.save_fig(fig, plot_dir, 'prior_spatial_bias')

    ## ---------------------------------------------------------------------##
    ## Albedo bias summary
    ## ---------------------------------------------------------------------##


    ## ---------------------------------------------------------------------##
    ## Scatter plot or at least R2 (memory might be tough on scatter plot)
    ## ---------------------------------------------------------------------##
