import glob
import sys
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from copy import deepcopy as dc

sys.path.append('/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/python')
import inversion_settings as s

# ------------------------------------------------------------------------ ##
# Prep files
# ------------------------------------------------------------------------ ##
# Define the variables to keep
data_vars = ['time', 'start_time', 'midpoint_time', 'time_components', 'value',
             'latitude', 'longitude', 'altitude', 'assimilation_concerns',
             'obspack_id']

# Define a filtering function
def filter_obspack(data):
    # Subset variables
    data = data[data_vars]

    # Subset for time and location
    data = data.where(data['time'].dt.year == 2019, drop=True)
    data = data.where((data['latitude'] > s.lat_min) & 
                      (data['latitude'] < s.lat_max),
                      drop=True)
    data = data.where((data['longitude'] > s.lon_min) & 
                      (data['longitude'] < s.lon_max),
                      drop=True)

    # Save out a platform variable
    platform = data.attrs['dataset_project'].split('-')[0]
    data['platform'] = xr.DataArray([platform]*len(data.obs), dims=('obs'))

    # Correct to local timezone if it's an in situ or surface observation
    if (len(data.obs) > 0) and (platform in ['surface', 'tower']):
        utc_conv = data.attrs['site_utc2lst']
        if int(utc_conv) != utc_conv:
            print('UTC CONVERSION FACTOR IS NOT AN INTEGER : ', data.attrs['dataset_name'])
        data['utc_conv'] = xr.DataArray(utc_conv*np.ones(len(data.obs)),
                                        dims=('obs'))
        # data['time_ltc'] = dc(data['time']) + np.timedelta64(int(utc_conv), 'h')
    else:
        data['utc_conv'] = xr.DataArray(np.zeros(len(data.obs)), dims=('obs'))
        # data['time_ltc'] = dc(data['time'])

    return data

# Get a list of the files
obspack_dir = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/obspack'
files = glob.glob(f'{obspack_dir}/*.nc')
files = [f for f in files if f.split('/')[-1][:11] != 'obspack_ch4']
files.sort()

# Iterate through the files and see which are relevant to the domain
conus_files = []
platforms = []
for i, f in enumerate(files):
    op = xr.open_dataset(f)

    # Only use files in the needed time, latitude, and longitude
    # ranges
    try:
        op = filter_obspack(op)
    except ValueError:
        continue
    except KeyError:
        print(f)

    # If the file is empty, continue through the loop
    if len(op.obs) == 0:
        continue

    # If the file still has observations, append it to conus_files
    conus_files.append(f)

    # And get information on the platform
    platforms.append(op.attrs['dataset_project'])

# Sort the files
conus_files.sort()

# Now load all the files
obspack = xr.open_mfdataset(conus_files, concat_dim='obs', combine='nested', 
                            chunks=1e4, mask_and_scale=False, 
                            preprocess=filter_obspack)

# Check for the sampling strategy
## Get the time in hours of each sample
obspack['obs_length'] = (obspack['time'] - obspack['start_time'])
obspack['obs_length'] = obspack['obs_length'].dt.seconds*2/(60*60)

## Convert that to the sampling strategy flag
## ss = place holder for sampling strategy
obspack['ss'] = xr.DataArray(999*np.ones(len(obspack.obs)), dims=('obs'))

## Closest to 4 hours
obspack['ss'] = obspack['ss'].where(obspack['obs_length'] > 5.25, 1)

## Closest to 90 minutes
obspack['ss'] = obspack['ss'].where(obspack['obs_length'] > 2.75, 3)

## Closest to 1 hour
obspack['ss'] = obspack['ss'].where(obspack['obs_length'] > 1.25, 2)

## Closest to instantaneous
obspack['ss'] = obspack['ss'].where(obspack['obs_length'] > 0.5, 4)

## Cast to int
obspack['ss'] = obspack['ss'].astype(int)

# Rename and add attributes
obspack = obspack.rename({'ss' : 'CT_sampling_strategy'})
obspack['CT_sampling_strategy'].attrs = {'_FillValue' : -9,
                                         'long_name' : 'model sampling strategy',
                                         'values' : 'How to sample model. 1=4-hour avg; 2=1-hour avg; 3=90-min avg; 4=instantaneous'}

# Other clean up
obspack.attrs = {}
obspack = obspack.drop(['obs_length', 'start_time', 'midpoint_time'])

# And iterate through the unique days
name_str = 'obspack_ch4.2019'
for mm in range(1, 13):
    for dd in range(1, 32):
        # Subset for that day
        daily = obspack.where((obspack['time'].dt.month == mm) &
                              (obspack['time'].dt.day == dd), drop=True)

        # If there is no data, continue
        if len(daily.obs) == 0:
            continue

        # Data type fix
        daily['obspack_id'] = daily['obspack_id'].astype('S200')
        daily['platform'] = daily['platform'].astype('S50')

        # Time fix
        daily['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00 UTC'
        daily['time'].encoding['calendar'] = 'proleptic_gregorian'

        # daily['time_ltc'].encoding['units'] = 'seconds since 1970-01-01 00:00:00 UTC'
        # daily['time_ltc'].encoding['calendar'] = 'proleptic_gregorian'

        # One last rename
        # daily = daily.rename({'value' : 'obs'})

        # Otherwise, save out
        print(f'Saving 2019-{mm:02d}-{dd:02d}')
        daily.to_netcdf(f'{obspack_dir}/{name_str}{mm:02d}{dd:02d}.nc',
                         unlimited_dims=['obs'])


