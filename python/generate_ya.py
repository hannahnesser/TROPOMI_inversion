## ---------------------------------------------------------------------##
## Standard imports
## ---------------------------------------------------------------------##
import os
from copy import deepcopy as dc
import sys
import xarray as xr
import numpy as np
from numpy.polynomial import polynomial as p
import pandas as pd
import glob
code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
sys.path.append(code_dir)
import gcpy as gc
import inversion_settings as s

## ---------------------------------------------------------------------##
## Directories
## ---------------------------------------------------------------------##
data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_w37_edf_posterior/ProcessedDir'
output_dir = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/inversion_data'
inv_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results'
y_file = 'y.nc'
latitudinal_correction = False
err_min = 10
suffix = 'w37_edf'
err_suffix = f'rg2rt_{err_min}t_{suffix}'
print(suffix)

# REMEMBER TO CREATE SYMBOLIC LINKS

## ---------------------------------------------------------------------##
## Load the data
## ---------------------------------------------------------------------##
# Prior run observations
files = glob.glob(f'{data_dir}/{s.year:04d}????_GCtoTROPOMI.pkl')
files = [p for p in files
         if int(p.split('/')[-1].split('_')[0][4:6]) in s.months]
files.sort()

data = np.array([])
for f in files:
    data = np.concatenate((data, gc.load_obj(f)[:, 1]))

# Observational filter
obs_filter = pd.read_csv(f'{output_dir}/obs_filter.csv', header=0)['FILTER'].values
data = data[obs_filter]

# Information for error calculation
data_long = gc.load_obj(f'{output_dir}/2019_corrected.pkl')

# Replace the dataframe with ya values
data_long['MOD'] = dc(data)
data_long['DIFF'] = data_long['MOD'] - data_long['OBS']

## ---------------------------------------------------------------------##
## Apply the latitudinal correction
## ---------------------------------------------------------------------##
if latitudinal_correction:
    # Correct the latitudinal bias
    coef = p.polyfit(data_long['LAT'], data_long['DIFF'], deg=1)
    bias_correction = p.polyval(data_long['LAT'], coef)

    # Print information
    print(f'Data has latitudinal bias removed.')
    print(f'    y = {coef[0]:.2f} + {coef[1]:.2f}x')
    print('-'*70)

    # delta = -7.75 + 0.44*data_long['LAT']
else:
    bias_correction = data_long['DIFF'].mean() 
    print(f'Data has mean bias removed.')
    print(f'    Mean bias of {bias_correction:.2f} ppb removed.')

data_long['MOD'] -= bias_correction
data_long['DIFF'] -= bias_correction
mean_diff = data_long['DIFF'].mean()
print(f'Mean model - observation difference : {mean_diff:.2f}')

## ---------------------------------------------------------------------##
## Solve for observing system errors
## ---------------------------------------------------------------------##
# Averaging groups
groupby = ['LAT_CENTER_L', 'LON_CENTER_L', 'SEASON']

# Delete superfluous columns
data_long = data_long.drop(columns=['PREC_SQ', 'AVG_DIFF', 'AVG_PREC', 
                                    'RES_ERR', 'AVG_RES_ERR', 'VAR',
                                    'AVG_OBS', 'STD', 'SO'])

# Now recalculate the residual error
group_quantities = ['DIFF', 'OBS', 'PREC_SQ']
data_long['PREC_SQ'] = data_long['PREC']**2
res_err = data_long.groupby(groupby).mean()[group_quantities].reset_index()
res_err['PREC_SQ'] **= 0.5

# Rename the columns
res_err = res_err.rename(columns={'DIFF' : 'AVG_DIFF',
                                  'OBS' : 'AVG_OBS',
                                  'PREC_SQ' : 'AVG_PREC'})

# Merge this data_long back into the original data_long frame
data_long = pd.merge(data_long, res_err, on=groupby, how='left')

# Subtract the bias from the difference to calculate the residual error
# This is equivalent to ZQ's eps quantity
data_long['RES_ERR'] = data_long['DIFF'] - data_long['AVG_DIFF']

# Next we calculate the average residual error
avg_err = data_long.groupby(groupby).mean()['RES_ERR'].reset_index()
avg_err = avg_err.rename(columns={'RES_ERR' : 'AVG_RES_ERR'})

# Now calculate the gridded variance and standard deviation of the
# residual error. The standard deviation is weighted by the number
# of observations in a grid cell because this will decrease the
# error in a grid cell.
# (sigma_squared and rrsd, respectively, in ZQ's code)
data_long = pd.merge(data_long, avg_err, on=groupby, how='left')
data_long['VAR'] = (data_long['RES_ERR'] - data_long['AVG_RES_ERR'])**2
var = data_long.groupby(groupby).mean()[['VAR', 'OBS']].reset_index()
var = var.rename(columns={'OBS' : 'AVG_OBS'})
var['STD'] = var['VAR']**0.5/var['AVG_OBS'] # rrsd

# Merge these final variances back into the data_long (first removing
# the initial variance calculation, since this was an intermediary)
data_long = data_long.drop(columns=['VAR', 'AVG_OBS'])
data_long = pd.merge(data_long, var, on=groupby, how='left')

# Scale by the observations
data_long['STD'] *= data_long['OBS']
data_long['VAR'] = data_long['STD']**2

# Where the variance calculated by the residual error method is less
# than the precision squared value calculated above, set the error equal
# to precision squared
cond = data_long['VAR'] < data_long['PREC_SQ']
print(f'We replace {cond.sum()} instances where the residual error is less than the instrumental error.')
data_long.loc[:, 'SO'] = data_long['VAR']
data_long.loc[cond, 'SO'] = data_long.loc[cond, 'PREC_SQ']

# Where the variance is less than 100 ppb^2 (err = 10 pppb, set a threshoold
cond_t = data_long['SO'] < err_min**2
print(f'We replace {cond_t.sum()} instances where the residual error is less than {err_min} ppb.')
data_long.loc[cond_t, 'SO'] = err_min**2

# and then update std
data_long.loc[:, 'STD'] = data_long['SO']**0.5

err_mean = data_long['STD'].mean()
err_std = data_long['STD'].std()
print(f'We find a mean error of {err_mean:.2f} ppb with a standard deviation of {err_std:.2f} ppb.' )

## ---------------------------------------------------------------------##
## Print
## ---------------------------------------------------------------------##
print('Model maximum       : %.2f' % (data_long['MOD'].max()))
print('Model minimum       : %.2f' % (data_long['MOD'].min()))

## ---------------------------------------------------------------------##
## Save out
## ---------------------------------------------------------------------##
ya = xr.DataArray(data_long['MOD'], dims=('nobs'))
ya.to_netcdf(f'{output_dir}/ya_{suffix}.nc')
try:
    os.symlink(f'{output_dir}/ya_{suffix}.nc', f'{inv_dir}/ya_{suffix}.nc')
except:
    print('Symbolic link exists')

so = xr.DataArray(data_long['SO'], dims=('nobs'))
so.to_netcdf(f'{output_dir}/so_{err_suffix}.nc')
try:
    os.symlink(f'{output_dir}/so_{err_suffix}.nc', f'{inv_dir}/so_{err_suffix}.nc')
except:
    print('Symbolic link exists')



