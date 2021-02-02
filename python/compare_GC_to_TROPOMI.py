# import glob
# import numpy as np
# import xarray as xr
# import re
import pickle
from os.path import join
from os import listdir

import numpy as np
import pandas as pd
# import datetime
# import copy

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/ProcessedDir'
years = [2019]
# months = [] # Eventually replace this with sys.argv so that we can
# run it simultaneously for all of the months
months = np.arange(1, 13, 1)
days = np.arange(1, 32, 1)
files = 'YYYYMMDD_GCtoTROPOMI.pkl'

lat_bins = np.arange(0, 100, 10)

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

def cum_mean(old_count, old_mean,
             new_count, new_mean):
    return (old_count*old_mean + new_count*new_mean)/(old_count + new_count)

def cum_std(old_count, old_mean, old_std,
            new_count, new_mean, new_std,
            tot_mean):
    old = old_count*(old_std**2 + (old_mean - tot_mean)**2)
    new = new_count*(new_std**2 + (new_mean - tot_mean)**2)
    return np.sqrt((old + new)/(old_count + new_count))

def cum_rmse(old_count, old_rmse,
             new_count, new_rmse):
    new_rmse = (old_count*old_rmse**2 + new_count*new_rmse**2)
    new_rmse /= (old_count + new_count)
    return np.sqrt(new_rmse)

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
## Read files
## -------------------------------------------------------------------------##
for y in years:
    for m in months:
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
            data = np.concatenate((data,
                                   load_obj(join(data_dir, file))['obs_GC']))
            print(m, d, data[data[:,1] == data[:,1].max()][:,[4,5]])


        # Create a dataframe
        data = pd.DataFrame(data, columns=['OBS', 'MOD', 'LON', 'LAT',
                                           'iGC', 'jGC', 'PRECISION',
                                           'ALBEDO_SWIR', 'ALBEDO_NIR',
                                           'AOD', 'MOD_COL'])
        # Calculate model - observation
        data['DIFF'] = data['MOD'] - data['OBS']

        # Subset data
        data = data[['LAT', 'DIFF']]

        # Group the difference between model and observation
        # by latitude bin
        data['LAT_BIN'] = pd.cut(data['LAT'], lat_bins)
        summ = data.groupby('LAT_BIN').agg({'DIFF' : ['count', 'mean',
                                                      'std', rmse]})
        summ = summ['DIFF'].reset_index()

        # Add in a total summary for that day
        summ = summ.append({'LAT_BIN' : 'Total',
                            'count' : data['DIFF'].shape[0],
                            'mean' : data['DIFF'].mean(),
                            'std' : data['DIFF'].std(),
                            'rmse' : rmse(data['DIFF'])},
                             ignore_index=True)

        # Save out monthly summary
        file_name = '%04d%02d_summary.csv' % (y, m)
        summ.to_csv(join(data_dir, file_name))

    # Now need to consolidate yearly (mainly for the latitudinal bands)




            # Fill nans with 0
            day_tot = day_tot.replace(np.nan, 0)

            # Save to monthly summary
            month_tot = cum_stats(month_tot, day_tot)

        #  Create a dataframe filled with 0s to store summary month data
        lat_intervals = pd.cut(lat_bins, lat_bins)[1:]
        month_tot = pd.DataFrame(index=lat_intervals,
                                 columns=['count', 'mean',
                                          'std', 'rmse'])
        month_tot = month_tot.reset_index()
        month_tot = month_tot.append({'index' : 'Total'}, ignore_index=True)
        month_tot = month_tot.replace(np.nan, 0)
