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
import numpy as np
import pandas as pd

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# Cannon
base_dir = '/n/seasasfs02/hnesser/TROPOMI_inversion/'
code_dir = '/n/home04/hnesser/TROPOMI_inversion/python/'
data_dir = f'{base_dir}inversion_data/'
output_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion/'

# Import custom packages
sys.path.append(code_dir)
import gcpy as gc
import inversion_settings as settings

# Files
month = int(sys.argv[1])
obs_file = f'{data_dir}/{settings.year}_corrected.pkl'
cluster_file = f'{data_dir}clusters.nc'
k_nstate_file = f'{data_dir}k0_nstate.nc' # None
k_m_file = f'{data_dir}k0_m{month:02d}.nc'

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
## Set up a dask client and cacluate the optimal chunk size
## ------------------------------------------------------------------------ ##
from dask.distributed import Client, LocalCluster, progress
from dask.diagnostics import ProgressBar
import dask.config
dask.config.set({'distributed.comm.timeouts.connect' : 90,
                 'distributed.comm.timeouts.tcp' : 150,
                 'distributed.adaptive.wait-count' : 90})

# Open cluster and client
# We adaptively choose n_workers and threads_per_worker based on nobs size
if nobs > 3e5:
    n_workers = 1
    threads_per_worker = 2
else:
    n_workers = 3
    threads_per_worker = 2
cluster = LocalCluster(local_directory=output_dir,
                       n_workers=n_workers,
                       threads_per_worker=threads_per_worker)
client = Client(cluster)

# We now calcualte the optimal chunk size. Our final matrix will be
# nstate x nstate, so we want our chunks accordingly
n_threads = n_workers*threads_per_worker
max_chunk_size = gc.calculate_chunk_size(available_memory_GB,
                                         n_threads=n_threads)
# We take the squareroot of the max chunk size and scale it down by 5
# to be safe. It's a bit unclear why this works best in tests.
nstate_chunk = int(np.sqrt(max_chunk_size)/5)
nobs_chunk = int(max_chunk_size/nstate_chunk)
print('State vector chunks : ', nstate_chunk)
print('Obs vector chunks   : ', nobs_chunk)

## ------------------------------------------------------------------------ ##
## Generate a monthly K0
## ------------------------------------------------------------------------ ##
k_nstate = xr.open_dataarray(k_nstate_file,
                             chunks={'nobs' : nobs_chunk,
                                     'nstate' : nstate_chunk,
                                     'month' : 1})
k_nstate = k_nstate.sel(month=month)

k_m = k_nstate[obs['CLUSTER'].values, :]
with ProgressBar():
    k_m.to_netcdf(f'{output_dir}k0_m{month:02d}.nc')

# Shutdown the client.
client.shutdown()
