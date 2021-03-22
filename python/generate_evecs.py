# import copy
import os
from os.path import join
import math
# import itertools
import pickle

import sys
sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python')
import inversion as inv

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

# The prior_run can either be a list of files or a single file
# with all of the data for simulation
year = 2019
months = np.arange(1, 12, 1) # excluding December for now
days = np.arange(1, 32, 1)

# Files
emis_file = f'{base_dir}prior/total_emissions/HEMCO_diagnostics.{year}.nc'
obs_file = f'{base_dir}observations/{year}_corrected.pkl'
cluster_file = f'{data_dir}clusters_0.25x0.3125.nc'
k_nstate = f'{data_dir}k0_nstate.pkl'#None

# Which analyses do you wish to perform?

# Information on the grid
lat_bins = np.arange(10, 65, 5)
lat_min = 9.75
lat_max = 60
lat_delta = 0.25
lon_min = -130
lon_max = -60
lon_delta = 0.3125
buffers = [3, 3, 3, 3]

## -------------------------------------------------------------------------##
## Load data
## -------------------------------------------------------------------------##

data_dir='/n/seasasfs02/hnesser/TROPOMI_inversion/evec_perturbations_ZQ'

# Jacobian
k = load_obj(join(data_dir, 'k.pkl')).astype('float32')

# Native-resolution prior and prior error
xa = load_obj(join(data_dir, 'xa.pkl')).reshape(-1, 1).astype('float32')
sa_vec = load_obj(join(data_dir, 'sa_vec.pkl')).reshape(-1, 1).astype('float32')

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

