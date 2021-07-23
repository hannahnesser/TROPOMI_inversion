# import copy
import os
from os.path import join
import math
# import itertools
import pickle

import xarray as xr
import dask.array as da
import numpy as np
import pandas as pd
from scipy.linalg import eigh

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
# Cannon
base_dir = '/n/seasasfs02/hnesser/TROPOMI_inversion/'
code_dir = '/n/home04/hnesser/TROPOMI_inversion/python/'
data_dir = f'{base_dir}inversion_data/'

# Import custom packages
import sys
sys.path.append(code_dir)
import inversion as inv
import inversion_settings as s

# Month
month = int(sys.argv[1])

# Files
emis_file = f'{base_dir}prior/total_emissions/HEMCO_diagnostics.{s.year}.nc'
obs_file = f'{base_dir}observations/{s.year}_corrected.pkl'
cluster_file = f'{data_dir}clusters.nc'
k_nstate = f'{data_dir}k0_m{month:02d}'#None

## -------------------------------------------------------------------------##
## Load pertinent data that defines state and observational dimension
## -------------------------------------------------------------------------##
# Observational error
so = gc.read_file(so_file, chunks=nobs_chunk)
so = so.compute()
nobs_tot = so.shape[0]

# Prior error
sa = gc.read_file(sa_file, chunks=nstate_chunk)
sa = sa.compute()
nstate = sa.shape[0]

## ------------------------------------------------------------------------ ##
## Set up a dask client and cacluate the optimal chunk size
## ------------------------------------------------------------------------ ##
from dask.distributed import Client, LocalCluster, progress
from dask.diagnostics import ProgressBar
import dask.config
dask.config.set({'distributed.comm.timeouts.connect' : 90,
                 'distributed.comm.timeouts.tcp' : 150,
                 'distributed.adaptive.wait-count' : 90,
                 'array.slicing.split_large_chunks' : False})

# Open cluster and client
n_workers = 1
threads_per_worker = 2
cluster = LocalCluster(local_directory=output_dir,
                       n_workers=n_workers,
                       threads_per_worker=threads_per_worker)
client = Client(cluster)

# We now calculate chunk size.
n_threads = n_workers*threads_per_worker
max_chunk_size = gc.calculate_chunk_size(available_memory_GB,
                                         n_threads=n_threads)
# We take the squareroot of the max chunk size and scale it down by 5
# to be safe. It's a bit unclear why this works best in tests.
nstate_chunk = int(np.sqrt(max_chunk_size)/5)
nobs_chunk = int(max_chunk_size/nstate_chunk)
chunks = {'nstate' : nstate_chunk, 'nobs' : nobs_chunk}
print('State vector chunks : ', nstate_chunk)
print('Obs vector chunks   : ', nobs_chunk)

## -------------------------------------------------------------------------##
## Generate the prior pre-conditioned Hessian for that month
## -------------------------------------------------------------------------##
# Get the indices for the month using generic chunks
i0 = 0
for m in s.months:
    k_m = gc.read_file(f'{data_dir}k0_m{m:02d}', chunks=chunks)
    i1 = i0 + k_m.shape[0]
    if m != month:
        i0 = i1

# Subset so
so_m = so[i0:i1]

# Calculate the monthly prior pre-conditioned Hessian
sasqrtkt = k_m*(sa**0.5)
pph_m = da.tensordot(sasqrtkt.T/so_m, sasqrtkt, axes=(1, 0))
pph_m = xr.DataArray(pph_m, dims=['nstate_0, nstate_1'],
                     name=f'pph0_m{month:02d}')

# Save out
pph_m.to_netcdf(f'{output_dir}pph0_m{month:02d}.nc')
