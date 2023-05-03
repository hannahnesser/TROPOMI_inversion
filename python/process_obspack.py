import glob
import sys
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from copy import deepcopy as dc

## ------------------------------------------------------------------------ ##
## Process output
## ------------------------------------------------------------------------ ##
# Load prior run output
prior_data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_w37_edf/obspack'
prior_files = glob.glob(f'{prior_data_dir}/*.nc4')
prior_files.sort()

def filter_obspack(data):
    return data[['pressure', 'CH4']]

obspack_prior = xr.open_mfdataset(prior_files, concat_dim='obs', 
                                  combine='nested', chunks=1e4, 
                                  mask_and_scale=False, 
                                  preprocess=filter_obspack)

posterior_data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_w37_edf_posterior/obspack'
posterior_files = glob.glob(f'{posterior_data_dir}/*.nc4')
posterior_files.sort()

# Load posterior run output
obspack_posterior = xr.open_mfdataset(posterior_files, concat_dim='obs', 
                                      combine='nested', chunks=1e4, 
                                      mask_and_scale=False, 
                                      preprocess=filter_obspack)

data_dir =  '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/obspack'
files = glob.glob(f'{data_dir}/obspack_ch4.2019*.nc')
files.sort()

def filter_obspack(data):
    return data[['obspack_id', 'value', 'altitude', 'latitude', 'longitude', 
                 'time', 'utc_conv', 'platform']]

obspack = xr.open_mfdataset(files, concat_dim='obs', 
                            combine='nested', chunks=1e4, 
                            mask_and_scale=False, 
                            preprocess=filter_obspack)

# Combine
obspack_full = pd.DataFrame({'time' : obspack['time'].values,
                             'utc_conv' : obspack['utc_conv'].values,
                             'lat' : obspack['latitude'].values,
                             'lon' : obspack['longitude'].values,
                             'altitude' : obspack['altitude'].values,
                             'pressure' : obspack_prior['pressure'].values,
                             'id' : obspack['obspack_id'].values,
                             'platform' : obspack['platform'].values,
                             'obspack' : obspack['value'].values, 
                             'prior' : obspack_prior['CH4'].values, 
                             'post' : obspack_posterior['CH4'].values})

# Adjust units to ppb
obspack_full[['obspack', 'prior', 'post']] *= 1e9

# # Remove scenes with 0 pressure
# obspack_full = obspack_full[obspack_full['pressure'] != 0]

obspack_full.to_csv(f'{data_dir}/obspack_ch4.2019.csv', header=True)