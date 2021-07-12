# import copy
import os
from os.path import join
import math
# import itertools
import pickle

import xarray as xr
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
                 'distributed.adaptive.wait-count' : 90})

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
## Open the Jacobian for that month
## -------------------------------------------------------------------------##
# Get the indices for the month using generic chunks
i0 = 0
for m in range(1, month+1):
    k_m = gc.read_file(f'{data_dir}k0_m{m:02d}', chunks=chunks)
    i1 = i0 +k_m.shape[0]

# Vectorized observations and error
y = load_obj(join(data_dir, 'y.pkl')).reshape(-1, 1).astype('float32')
y_base = load_obj(join(data_dir, 'y_base.pkl')).reshape(-1, 1).astype('float32')
so_vec = load_obj(join(data_dir, 'so_vec.pkl')).reshape(-1, 1).astype('float32')

## -------------------------------------------------------------------------##
## Set constants
## -------------------------------------------------------------------------##

RF = 0.001

## -------------------------------------------------------------------------##
## Create inversion object
## -------------------------------------------------------------------------##

# Create a true Reduced Rank Jacobian object
inv_zq = inv.ReducedRankJacobian(k, xa, sa_vec, y, y_base, so_vec)
inv_zq.rf = RF

# Delete the other items from memory
del(k)
del(xa)
del(sa_vec)
del(y)
del(y_base)
del(so_vec)

# Save out the PPH
pph = (inv_zq.sa_vec**0.5)*inv_zq.k.T
pph = np.dot(pph, ((inv_zq.rf/inv_zq.so_vec)*pph.T))
save_obj(pph, join(data_dir, 'pph.pkl'))

# Complete an eigendecomposition of the prior pre-
# conditioned Hessian
assert np.allclose(pph, pph.T, rtol=1e-5), \
       'The prior pre-conditioned Hessian is not symmetric.'

evals, evecs = eigh(pph)
print('Eigendecomposition complete.')

# Sort evals and evecs by eval
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:,idx]

# Force all evals to be non-negative
if (evals < 0).sum() > 0:
    print('Negative eigenvalues. Maximum negative value is %.2e. Setting negative eigenvalues to zero.' \
        % (evals[evals < 0].min()))
    evals[evals < 0] = 0

# Check for imaginary eigenvector components and force all
# eigenvectors to be only the real component.
if np.any(np.iscomplex(evecs)):
    print('Imaginary eigenvectors exist at index %d of %d. Forcing eigenvectors to real component alone.' \
          % ((np.where(np.iscomplex(evecs))[1][0] - 1), len(evecs)))
    evecs = np.real(evecs)

# Saving result to our instance.
print('Saving eigenvalues and eigenvectors.')
# self.evals = evals/(1 + evals)
np.savetxt(join(data_dir, 'evecs.csv'), evecs, delimiter=',')
np.savetxt(join(data_dir, 'evals_h.csv'), evals, delimiter=',')
np.savetxt(join(data_dir, 'evals_q.csv'), evals/(1+evals), delimiter=',')
print('... Complete ...\n')


#
#         # temp = ev.where(ev.perturbation == p, drop=True).drop('perturbation').squeeze('perturbation')
d.lat.attrs = {'long_name': 'latitude', 'units': 'degrees_north'}
d.lon.attrs = {'long_name': 'longitude', 'units': 'degrees_east'}
d.time.attrs = {'long_name' : 'Time'}
d['evec_perturbations'].attrs = {'long_name' : 'Eigenvector perturbations',
                                 'units' : 'kg/m2/s'}
d.to_netcdf('/n/seasasfs02/hnesser/TROPOMI_inversion/evec_perturbations_ZQ/evec_perturbations_0001_scaled_1e-8.nc',
               encoding={'evec_perturbations' : {'_FillValue' : None,
                                                 'dtype' : 'float32'},
                         'time' : {'_FillValue' : None, 'dtype' : 'float32',
                                   'calendar' : 'standard',
                                   'units' : 'hours since 2009-01-01 00:00:00'},
                         'lat' : {'_FillValue' : None, 'dtype' : 'float32'},
                         'lon' : {'_FillValue' : None, 'dtype' : 'float32'}})

# # need to set attribute dictionaries for lat
# # lon, time, and the evec_pertuurbations, complete
# # with units

d.time.attrs = {'long_name' : 'Time', 'units' : 'hours since 2009-01-01 00:00:00', 'calendar' : 'standard'}

