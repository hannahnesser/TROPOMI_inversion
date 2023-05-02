from os.path import join
from os import listdir
import sys
import copy
import xarray as xr
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 15)

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
base_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_final/'
code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
data_dir = f'{base_dir}ProcessedDir'
output_dir = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/inversion_data'

# Import Custom packages
sys.path.append(code_dir)
import config
import gcpy as gc
import troppy as tp
import inversion_settings as settings

prior_run = [f'{settings.year}{mm:02d}{dd:02d}_GCtoTROPOMI.pkl'
             for mm in settings.months
             for dd in settings.days]
prior_run.sort()

## ------------------------------------------------------------------------ ##
## Get grid information
## ------------------------------------------------------------------------ ##
lats, lons = gc.create_gc_grid(*settings.lats, settings.lat_delta,
                               *settings.lons, settings.lon_delta,
                               centers=False, return_xarray=False)

# GOSAT comparison grid
lats_l, lons_l = gc.create_gc_grid(*settings.lats, 2, *settings.lons, 2,
                                   centers=False, return_xarray=False)

## ------------------------------------------------------------------------ ##
## Load data
## ------------------------------------------------------------------------ ##
data = np.array([]).reshape(0, 14)
for file in prior_run:
    # Check if that file is in the data directory
    if file not in listdir(data_dir):
        print(f'{file} is not in the data directory.')
        continue

    # Get the month
    month = int(file[4:6])
    day = int(file[6:8])
    date = int(file[:8])

    # Load the data. The columns are: 0 OBS, 1 MOD, 2 LON, 3 LAT,
    # 4 iGC, 5 jGC, 6 PRECISION, 7 ALBEDO_SWIR, 8 ALBEDO_NIR, 9 AOD,
    # 10 CLOUDS (15 10 total columns)
    new_data = gc.load_obj(join(data_dir, file))
    new_data = new_data[:, :11]
    nobs = new_data.shape[0]
    new_data = np.append(new_data, month*np.ones((nobs, 1)), axis=1)
    new_data = np.append(new_data, day*np.ones((nobs, 1)), axis=1)
    new_data = np.append(new_data, date*np.ones((nobs, 1)), axis=1)

    data = np.concatenate((data, new_data))

## ----------------------------------------- ##
## Basic data formatting
## ----------------------------------------- ##
# Create a dataframe from the data
columns = ['OBS', 'MOD', 'LON', 'LAT', 'iGC', 'jGC', 'PREC',
           'ALBEDO_SWIR', 'ALBEDO_NIR', 'AOD', 'CLOUDS',
           'MONTH', 'DAY', 'DATE']
data = pd.DataFrame(data, columns=columns)

# Calculate blended albedo
data['BLENDED_ALBEDO'] = tp.blended_albedo(data,
                                           data['ALBEDO_SWIR'],
                                           data['ALBEDO_NIR'])

# Add season
data.loc[:, 'SEASON'] = 'DJF'
data.loc[data['MONTH'].isin([3, 4, 5]), 'SEASON'] = 'MAM'
data.loc[data['MONTH'].isin([6, 7, 8]), 'SEASON'] = 'JJA'
data.loc[data['MONTH'].isin([9, 10, 11]), 'SEASON'] = 'SON'

# Save nearest latitude and longitude centers
data['LAT_CENTER'] = lats[gc.nearest_loc(data['LAT'].values, lats)]
data['LON_CENTER'] = lons[gc.nearest_loc(data['LON'].values, lons)]
data['LAT_CENTER_L'] = lats_l[gc.nearest_loc(data['LAT'].values, lats_l)]
data['LON_CENTER_L'] = lons_l[gc.nearest_loc(data['LON'].values, lons_l)]

# Calculate model - observation
data['DIFF'] = data['MOD'] - data['OBS']

# Subset data
data = data[['iGC', 'jGC', 'SEASON', 'MONTH', 'DAY', 'DATE', 'LON', 'LAT',
             'LON_CENTER', 'LAT_CENTER', 'LON_CENTER_L', 'LAT_CENTER_L',
             'OBS', 'MOD', 'DIFF', 'PREC', 'ALBEDO_SWIR',
             'BLENDED_ALBEDO', 'AOD', 'CLOUDS']]

# Print out some statistics
cols = ['LAT', 'LON', 'MONTH', 'MOD']
print_summary(data)

# Save the data out
print(f'Saving data in {output_dir}/{settings.year}.pkl')
gc.save_obj(data, f'{output_dir}/observations/{settings.year}.pkl')