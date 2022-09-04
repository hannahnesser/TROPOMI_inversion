## ---------------------------------------------------------------------##
## Standard imports
## ---------------------------------------------------------------------##
import sys
import xarray as xr
import numpy as np
import pandas as pd
import glob
code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
sys.path.append(code_dir)
import gcpy as gc
import inversion_settings as s

## ---------------------------------------------------------------------##
## Directories
## ---------------------------------------------------------------------##
data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_w404_edf/ProcessedDir'
output_dir = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/inversion_data'
latitudinal_correction = False
suffix = 'w404_edf_nlc'

## ---------------------------------------------------------------------##
## Load the data
## ---------------------------------------------------------------------##
files = glob.glob(f'{data_dir}/{s.year:04d}????_GCtoTROPOMI.pkl')
files = [p for p in files
         if int(p.split('/')[-1].split('_')[0][4:6]) in s.months]
files.sort()

data = np.array([])
for f in files:
    data = np.concatenate((data, gc.load_obj(f)[:, 1]))

## ---------------------------------------------------------------------##
## Apply the latitudinal correction
## ---------------------------------------------------------------------##
if latitudinal_correction:
    lats = gc.load_obj(f'{output_dir}/2019.pkl')['LAT'].values
    delta = -7.75 + 0.44*lats
    data = data - delta

## ---------------------------------------------------------------------##
## Print
## ---------------------------------------------------------------------##
print('Model maximum       : %.2f' % (data.max()))
print('Model minimum       : %.2f' % (data.min()))

## ---------------------------------------------------------------------##
## Save out
## ---------------------------------------------------------------------##
data = xr.DataArray(data, dims=('nobs'))
data.to_netcdf(f'{output_dir}/ya_{suffix}.nc')
