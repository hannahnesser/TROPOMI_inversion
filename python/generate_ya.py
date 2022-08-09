## ---------------------------------------------------------------------##
## Standard imports
## ---------------------------------------------------------------------##
import sys
import xarray as xr
import numpy as np
import pandas as pd
import glob

## ---------------------------------------------------------------------##
## Directories
## ---------------------------------------------------------------------##
code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_w404/ProcessedDir'
output_dir = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/inversion_data'
suffix = 'w404'

## ---------------------------------------------------------------------##
## Custom imports
## ---------------------------------------------------------------------##
sys.path.append(code_dir)
import gcpy as gc
import inversion_settings as s

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
## Save out
## ---------------------------------------------------------------------##
data = xr.DataArray(data, dims=('nobs'))
data.to_netcdf(f'{output_dir}/ya_{suffix}.nc')
