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

# Set directories (this needs to be amended)
if Analysis:
    code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
    data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/ProcessedDir'
    processed_data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/SummaryDir'

if Plots:
    code_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/python'
    processed_data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/prior/prior_run'
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

# albedo bins
albedo_bins = np.arange(0, 1.1, 0.1)

## -------------------------------------------------------------------------##
## Import additional packages
## -------------------------------------------------------------------------##
sys.path.append(code_dir)
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
import gcpy as gc
import troppy as tp
import format_plots as fp

## -------------------------------------------------------------------------##
## Define functions
## -------------------------------------------------------------------------##
def filter_data(data, groupby, quantity='DIFF',
                stats=['count', 'mean', 'std', gc.rmse]):
    d = data.groupby(groupby)
    d = d.agg({quantity : stats})
    d = d[quantity].reset_index()
    return d
## -------------------------------------------------------------------------##
## Analysis
## -------------------------------------------------------------------------##
if Analysis:
    # Get information on the lats and lons (edges of the domain)
    lat_edges, lon_edges = gc.adjust_grid_bounds(lat_min, lat_max, lat_delta,
                                                 lon_min, lon_max, lon_delta,
                                                 buffers)

    # Generate a grid on which to save out average difference
    lats, lons = gc.create_gc_grid(*lat_edges, lat_delta,
                                   *lon_edges, lon_delta,
                                   centers=False, return_xarray=False)

    ## ----------------------------------------- ##
    ## Load data for the year
    ## ----------------------------------------- ##
    data = np.array([]).reshape(0, 12)
    for m in months:
        print('Analyzing data for month %d' % m)
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
            new_data = np.insert(new_data, 11, m, axis=1) # add month

            data = np.concatenate((data, new_data))

    ## ----------------------------------------- ##
    ## Basic data formatting
    ## ----------------------------------------- ##
    # Create a dataframe from the data
    data = pd.DataFrame(data, columns=['OBS', 'MOD', 'LON', 'LAT',
                                       'iGC', 'jGC', 'PRECISION',
                                       'ALBEDO_SWIR', 'ALBEDO_NIR',
                                       'AOD', 'MOD_COL', 'MONTH'])

    # Create a column for the blended albedo filter
    data['FILTER'] = tp.blended_albedo_filter(data, data['ALBEDO_SWIR'],
                                              data['ALBEDO_NIR'])

    # Subset data
    data = data[['MONTH', 'LON', 'LAT', 'OBS', 'MOD',
                 'ALBEDO_SWIR', 'FILTER']]

    # Calculate model - observation
    data['DIFF'] = data['MOD'] - data['OBS']

    # Calculate the nearest latitude & longitude center
    data['LAT_CENTER'] = lats[gc.nearest_loc(data['LAT'].values, lats)]
    data['LON_CENTER'] = lons[gc.nearest_loc(data['LON'].values, lons)]

    # Group the difference between model and observation by latitude bin
    data['LAT_BIN'] = pd.cut(data['LAT'], lat_bins)

    # Group by albedo bin
    data['ALBEDO_BIN'] = pd.cut(data['ALBEDO_SWIR'], albedo_bins)

    # Save the data
    gc.save_obj(data, join(processed_data_dir, '%d.pkl' % year))

    # Create a dictionary of total and filtered data
    data_dict = {'Total' :data, 'Filtered' : data[data['FILTER']]}

    ## ----------------------------------------- ##
    ## Calculate spatial bias
    ## ----------------------------------------- ##
    # Apply DIFF to a grid so that we can track the spatial
    # distribution of the average difference annually.

    # We group by the latitude and longitude centers and convert it
    # to an xarray. We do this for filtered and unfiltered data.
    data_grid = {}
    for k, d in data_dict.items():
        d_g = d.groupby(['LAT_CENTER', 'LON_CENTER'])
        d_g = d_g.mean()['DIFF']
        d_g = d_g.to_xarray().rename({'LAT_CENTER' : 'lats',
                                      'LON_CENTER' : 'lons'})
        d_g = d_g.rename(k)
        data_grid[k] = d_g
    data_grid = xr.Dataset(data_grid)

    file_name = '%04d_spatial_summary.nc' % year
    data_grid.to_netcdf(join(processed_data_dir, file_name))

    ## ----------------------------------------- ##
    ## Calculate latitudinal bias
    ## ----------------------------------------- ##
    lat_bias = pd.DataFrame(columns={'FILTER', 'LAT_BIN', 'count', 'mean',
                                     'std', 'rmse'})
    for k, d in data_dict.items():
        l_b = filter_data(d, groupby=['LAT_BIN'])
        l_b['FILTER'] = k
        lat_bias = lat_bias.append(l_b)

    # Save out latitudinal bias summary
    file_name = '%04d_lat_summary.csv' % year
    lat_bias.to_csv(join(processed_data_dir, file_name))

    ## ----------------------------------------- ##
    ## Calculate monthly bias
    ## ----------------------------------------- ##
    month_bias = pd.DataFrame(columns={'FILTER', 'MONTH', 'count', 'mean',
                                       'std', 'rmse'})
    for k, d in data_dict.items():
        m_b = filter_data(d, groupby=['MONTH'])
        m_b['FILTER'] = k
        month_bias = month_bias.append(m_b)

    # Save out latitudinal bias summary
    file_name = '%04d_month_summary.csv' % year
    month_bias.to_csv(join(processed_data_dir, file_name))

    ## ----------------------------------------- ##
    ## Calculate albedo bias
    ## ----------------------------------------- ##
    albedo_bias = pd.DataFrame(columns={'FILTER', 'ALBEDO_BIN', 'count',
                                        'mean', 'std', 'rmse'})
    for k, d in data_dict.items():
        a_b = filter_data(d, groupby=['ALBEDO_BIN'])
        a_b['FILTER'] = k
        albedo_bias = albedo_bias.append(a_b)

    # Save out latitudinal bias summary
    file_name = '%04d_albedo_summary.csv' % year
    albedo_bias.to_csv(join(processed_data_dir, file_name))

    ## ----------------------------------------- ##
    ## Complete
    ## ----------------------------------------- ##
    print('Code Complete.')

## -------------------------------------------------------------------------##
## Plots
## -------------------------------------------------------------------------##
if Plots:
    ## ----------------------------------------- ##
    ## Total summary
    ## ----------------------------------------- ##
    data = gc.load_obj(join(processed_data_dir, '%04d.pkl' % year))

    for f in ['Total', 'Filtered']:
        if f == 'Filtered':
            d = data[data['FILTER']]
        else:
            d = data

        fig, ax, c = gc.plot_comparison(d['OBS'].values, d['MOD'].values,
                                        lims=[1750, 1950],
                                        xlabel='Observation',
                                        ylabel='Model',
                                        vmin=0, vmax=3e4)
        ax.set_xticks(np.arange(1750, 2000, 100))
        ax.set_yticks(np.arange(1750, 2000, 100))
        fp.save_fig(fig, plot_dir, 'prior_bias_%s' % f.lower())
        # print(data.head())

    ## ----------------------------------------- ##
    ## Spatial bias
    ## ----------------------------------------- ##
    spat_bias = xr.open_dataset(join(processed_data_dir,
                                '%04d_spatial_summary.nc' % year))

    # Make figures
    for f in ['Total', 'Filtered']:
        d = spat_bias[f]
        fig, ax = fp.get_figax(maps=True,
                               lats=spat_bias.lats, lons=spat_bias.lons)
        c = d.plot(ax=ax, cmap='RdBu_r', vmin=-30, vmax=30, add_colorbar=False)
        cax = fp.add_cax(fig, ax)
        cb = fig.colorbar(c, ax=ax, cax=cax)
        cb = fp.format_cbar(cb, 'Model - Observation')
        ax = fp.format_map(ax, spat_bias.lats, spat_bias.lons)
        ax = fp.add_title(ax, 'Spatial Bias in Prior Run')
        fp.save_fig(fig, plot_dir, 'prior_spatial_bias_%s' % f.lower())

    ## ----------------------------------------- ##
    ## Latitudinal bias
    ## ----------------------------------------- ##
    lat_bias = pd.read_csv(join(processed_data_dir,
                                '%04d_lat_summary.csv' % year),
                           index_col=0)
    lat_bias['LAT'] = (lat_bias['LAT_BIN'].str.split(r'\D', expand=True)
                       .iloc[:, [1, 3]]
                       .astype(float).mean(axis=1))

    # Make figures
    for f in ['Total', 'Filtered']:
        d = lat_bias[lat_bias['FILTER'] == f]
        fig, ax = fp.get_figax(aspect=1.75)
        ax.errorbar(d['LAT'], d['mean'], yerr=d['std'], color=fp.color(4))
        ax.set_xticks(np.arange(10, 70, 10))
        ax = fp.add_labels(ax, 'Latitude', 'Model - Observation')
        ax = fp.add_title(ax, 'Latitudinal Bias in Prior Run')
        fp.save_fig(fig, plot_dir, 'prior_latitudinal_bias_%s' % f.lower())

    ## ----------------------------------------- ##
    ## Monthly bias
    ## ----------------------------------------- ##
    month_bias = pd.read_csv(join(processed_data_dir,
                                  '%04d_month_summary.csv' % year),
                             index_col=0)
    month_bias['month'] = pd.to_datetime(month_bias['MONTH'], format='%m')
    month_bias['month'] = month_bias['month'].dt.month_name().str[:3]

    # Make figures
    for f in ['Total', 'Filtered']:
        d = month_bias[month_bias['FILTER'] == f]
        fig, ax = fp.get_figax(aspect=1.75)
        ax.errorbar(d['month'], d['mean'], yerr=d['std'], color=fp.color(4))
        ax = fp.add_labels(ax, 'Month', 'Model - Observation')
        ax = fp.add_title(ax, 'Seasonal Bias in Prior Run')
        fp.save_fig(fig, plot_dir, 'prior_seasonal_bias_%s' % f.lower())

    ## ----------------------------------------- ##
    ## Albedo bias
    ## ----------------------------------------- ##
    albedo_bias = pd.read_csv(join(processed_data_dir,
                                '%04d_albedo_summary.csv' % year),
                           index_col=0)
    albedo_bias['ALBEDO'] = (albedo_bias['ALBEDO_BIN'].str.split(r'\D',
                                                              expand=True)
                             .iloc[:, [2, 5]]
                             .astype(float).mean(axis=1))/10

    # Make figures
    for f in ['Total', 'Filtered']:
        d = albedo_bias[albedo_bias['FILTER'] == f]
        fig, ax = fp.get_figax(aspect=1.75)
        ax.errorbar(d['ALBEDO'], d['mean'], yerr=d['std'], color=fp.color(4))
        ax.set_xticks(np.arange(0, 1, 0.2))
        ax = fp.add_labels(ax, 'Albedo', 'Model - Observation')
        ax = fp.add_title(ax, 'Albedo Bias in Prior Run')
        fp.save_fig(fig, plot_dir, 'prior_albedo_bias_%s' % f.lower())
