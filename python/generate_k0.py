'''
This python script generates the initial estimate of the Jacobian matrix

Inputs:
    prior_emis      This contains the emissions of the prior run simulation.
                    It is a list of monthly output HEMCO diagnostics files
                    to account for monthly variations in
'''

from os.path import join
import sys

import xarray as xr
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# Cannon
base_dir = '/n/seasasfs02/hnesser/TROPOMI_inversion/'
code_dir = '/n/home04/hnesser/TROPOMI_inversion/python/'
data_dir = f'{base_dir}inversion_data/'
output_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion'

# Import custom packages
sys.path.append(code_dir)
import gcpy as gc
import inversion_settings as settings

# Files
month = int(sys.argv[1])
emis_file = f'{base_dir}prior/total_emissions/HEMCO_diagnostics.{settings.year}.nc'
obs_file = f'{base_dir}observations/{settings.year}_corrected.pkl'
cluster_file = f'{data_dir}clusters.nc'
k_nstate = f'{data_dir}k0_nstate.nc' # None

# Memory constraints
available_memory_GB = int(sys.argv[2])

## ------------------------------------------------------------------------ ##
## Load the clusters
## ------------------------------------------------------------------------ ##
clusters = xr.open_dataset(cluster_file)
nstate = int(clusters['Clusters'].max().values)
print(f'Number of state vector elements : {nstate}')

## ------------------------------------------------------------------------ ##
## Load and process the observations
## ------------------------------------------------------------------------ ##
obs = gc.load_obj(obs_file)[['LON', 'LAT', 'MONTH']]
obs = obs[obs['MONTH'] == month]

nobs = int(obs.shape[0])
print(f'Number of observations : {nobs}')

# Find the indices that correspond to each observation (i.e. the grid
# box in which each observation is found) (Yes, this information should
# be contained in the iGC and jGC columns in obs_file, but stupidly
# I don't have that information for the cluster files)

# First, find the cluster number of the grid box of the obs
lat_idx = gc.nearest_loc(obs['LAT'].values, emis.lat.values)
lon_idx = gc.nearest_loc(obs['LON'].values, emis.lon.values)
obs['CLUSTER'] = emis['Clusters'].values[lat_idx, lon_idx]

# Subset to reduce memory needs
obs = obs[['CLUSTER', 'MONTH']]
obs = obs[obs['MONTH'] == month]
nobs = obs.shape[0]
print(f'In month {month}, there are {nobs} observations.')

# Format
obs[['MONTH', 'CLUSTER']] = obs[['MONTH', 'CLUSTER']].astype(int)

## ------------------------------------------------------------------------ ##
## Load the n x n x 12 base initial estimate
## ------------------------------------------------------------------------ ##
k_nstate = xr.open_dataarray(join(data_dir, 'k0_nstate.nc'),
                             chunks={'nobs' : 1000,
                                     'nstate' : -1,
                                     'month' : 12})
k_nstate = k_nstate.sel(month=month)

print('k0_nstate is loaded.')

# Delete superfluous memory
del(clusters)

# Fancy slicing isn't allowed by dask, so we'll create monthly Jacobians
max_chunk_size = gc.calculate_chunk_size(available_memory_GB)
nobs_clusters = int(max_chunk_size/nstate)
chunks={'nstate' : -1, 'nobs' : nobs_clusters}
print('CHUNK SIZE: ', chunks)


k_m = k_nstate[obs['CLUSTER'].values, :]
k_m = k_m.chunk(chunks)
with ProgressBar():
    k_m.to_netcdf(join(output_dir, f'k0_m{month:02d}.nc'))

# EASY CHECK: USE THIS SCRIPT TO BUILD THE JACOBIAN FOR MY TEST CASE

