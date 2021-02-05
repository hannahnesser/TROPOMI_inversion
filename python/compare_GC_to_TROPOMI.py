# import glob
# import numpy as np
# import xarray as xr
# import re
import pickle
from os.path import join
from os import listdir
import sys

import numpy as np
import pandas as pd
from itertools import product
# import datetime
# import copy

sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python')
import gcpy as gc

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/ProcessedDir'
years = [2019]
# Excluding December for now because of the bug in the prior run
months = np.arange(1, 12, 1)
days = np.arange(1, 32, 1)
files = 'YYYYMMDD_GCtoTROPOMI.pkl'

# Information on the grid
lat_bins = np.arange(10, 70, 10)
lat_min = 9.75
lat_max = 60
lat_delta = 0.25
lon_min = -130
lon_max = -60
lon_delta = 0.3125
buffers = [3, 3, 3, 3]

## -------------------------------------------------------------------------##
## Define functions
## -------------------------------------------------------------------------##

def save_obj(obj, name):
    with open(name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)

def rmse(diff):
    return np.sqrt(np.mean(diff**2))

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

    return pd.DataFrame({'index' : d1['index'],
                         'count' : count, 'mean' : mean,
                         'std' : std, 'rmse' : rmse})

## -------------------------------------------------------------------------##
## Analysis
## -------------------------------------------------------------------------##
# Get information on the lats and lons (edges of the domain)
lat_edges, lon_edges = gc.adjust_grid_bounds(lat_min, lat_max, lat_delta,
                                             lon_min, lon_max, lon_delta,
                                             buffers)

# # Generate a grid on which to save out average difference
lats, lons = gc.create_gc_grid(*lat_edges, lat_delta, *lon_edges, lon_delta,
                               centers=False, return_xarray=False)
# diff_grid = xr.Dataset({'xch4' : diff_grid, 'count' : diff_grid})
diff_grid = pd.DataFrame(list(product(lats, lons)),
                         columns=['LAT', 'LON'])
diff_grid['DIFF'] = 0
diff_grid['COUNT'] = 0

# Loop through the data
for y in years:
    #  Create a dataframe filled with 0s to store summary yearly data
    lat_intervals = pd.cut(lat_bins, lat_bins)[1:]
    year_summ = pd.DataFrame(index=lat_intervals,
                             columns=['count', 'mean',
                                      'std', 'rmse'])
    year_summ = year_summ.reset_index()
    year_summ = year_summ.append({'index' : 'Total'}, ignore_index=True)
    year_summ = year_summ.replace(np.nan, 0)
    for m in months:
        print('Analyzing data for month %d' % m)
        # Iterate through the days in the month, beginning by initializing
        # on day 1
        data = np.array([]).reshape(0, 11)
        for d in days:
            # Create the file name for that date
            file = files.replace('YYYY', '%04d' % y)
            file = file.replace('MM', '%02d' % m)
            file = file.replace('DD', '%02d' % d)

            # Check if that file is in the data directory
            if file not in listdir(data_dir):
                print('Data for %04d-%02d-%02d is not in the data directory.'
                      % (y, m, d))
                continue

            # Load the data
            # The columns are: 0 OBS, 1 MOD, 2 LON, 3 LAT,
            # 4 iGC, 5 jGC, 6 PRECISION, 7 ALBEDO_SWIR,
            # 8 ALBEDO_NIR, 9 AOD, 10 MOD_COL
            new_data = load_obj(join(data_dir, file))['obs_GC']
            data = np.concatenate((data,
                                   load_obj(join(data_dir, file))['obs_GC']))

        # Create a dataframe
        data = pd.DataFrame(data, columns=['OBS', 'MOD', 'LON', 'LAT',
                                           'iGC', 'jGC', 'PRECISION',
                                           'ALBEDO_SWIR', 'ALBEDO_NIR',
                                           'AOD', 'MOD_COL'])
        # Calculate model - observation
        data['DIFF'] = data['MOD'] - data['OBS']

        # Subset data
        data = data[['LON', 'LAT', 'DIFF']]

        # Take the difference and apply it to a grid.
        data['LAT_EDGES'] = pd.cut(data['LAT'], lat_edges)
        data['LON_EDGES'] = pd.cut(data['LON'], lon_edges)
        data_grid = data.groupby(['LAT_EDGES', 'LON_EDGES'])
        data_grid = data_grid.agg({'DIFF' : ['count', 'sum']})
        data_grid = data_grid['DIFF'].reset_index()
        # Get the latitudes and longitudes


        lat_idx = nearest_loc(data['LAT'].values, diff_grid['LAT'].values)
        lon_idx = nearest_loc(data['LON'].values, diff_grid['LON'].values)
        ##### THIS IS WHERE I LEFT OFF

        # Group the difference between model and observation
        # by latitude bin
        data['LAT_BIN'] = pd.cut(data['LAT'], lat_bins)
        month_summ = data.groupby('LAT_BIN')
        month_summ = month_summ.agg({'DIFF' : ['count', 'mean', 'std', rmse]})
        month_summ = month_summ['DIFF'].reset_index()

        # Add in a total summary for that day
        month_summ = month_summ.append({'LAT_BIN' : 'Total',
                            'count' : data['DIFF'].shape[0],
                            'mean' : data['DIFF'].mean(),
                            'std' : data['DIFF'].std(),
                            'rmse' : rmse(data['DIFF'])},
                             ignore_index=True)

        # Save out monthly summary
        file_name = '%04d%02d_summary.csv' % (y, m)
        month_summ.to_csv(join(data_dir, file_name))

        # Add monthly summary to yearly summary
        year_summ = cum_stats(year_summ, month_summ)

    # Now save out yearly summary
    print('Code Complete.')
    print(year_summ)
    file_name = '%04d_summary.csv' % y
    year_summ.to_csv(join(data_dir, file_name))
