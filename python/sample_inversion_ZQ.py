## -------------------------------------------------------------------------##
## Load packages and set environment defaults
## -------------------------------------------------------------------------##

from os.path import join
import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import xarray as xr
import pickle
import os
import pandas as pd
import math
import numpy as np

## -------------------------------------------------------------------------##
## Define save and load functions
## -------------------------------------------------------------------------##
def save_obj(obj, name ):
        with open(name , 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
        with open( name, 'rb') as f:
                return pickle.load(f)

## -------------------------------------------------------------------------##
## Read data
## -------------------------------------------------------------------------##

data_dir='/n/seasasfs02/hnesser/TROPOMI_inversion/evec_perturbations_ZQ'

rrsd = load_obj(join(data_dir, 'rrsd.pkl'))
gc = pd.read_csv(join(data_dir, 'gc_output'), delim_whitespace=True, header=0,
                 usecols=['I', 'J', 'GOSAT', 'model', 'S_OBS'])
k = load_obj(join(data_dir, 'kA.pkl')).T

# Now we will filter
# Define filters
k_idx = np.isfinite(k).all(axis=1)
gc_idx = np.isfinite(gc).all(axis=1)

# Apply
k = k[k_idx & gc_idx]
gc = gc[k_idx & gc_idx]

# Reshape error by applying indices from gc
so_vec = (rrsd[gc['J'] - 1, gc['I'] - 1]*gc['GOSAT']*1e9)**2
so_vec[~(so_vec > 0)] = (gc['S_OBS'][~(so_vec > 0)])*1e18
so_vec = so_vec.values

# Now save out some quantities
y_base = gc['model'].values*1e9
y = gc['GOSAT'].values*1e9
xa = np.ones(k.shape[1])
sa_vec = np.ones(k.shape[1])*0.5**2

# save out
save_obj(k, join(data_dir, 'k.pkl'))
save_obj(so_vec, join(data_dir, 'so_vec.pkl'))
save_obj(y_base, join(data_dir, 'y_base.pkl'))
save_obj(y, join(data_dir, 'y.pkl'))
save_obj(xa, join(data_dir, 'xa.pkl'))
save_obj(sa_vec, join(data_dir, 'sa_vec.pkl'))

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

# Calculate the prolongation and reduction operators
prolongation = (evecs * inv_zq.sa_vec**0.5).T
reduction = (1/self.sa_vec**0.5) * evecs.T

# Saving result to our instance.
print('Saving eigenvalues and eigenvectors.')
# self.evals = evals/(1 + evals)
np.savetxt(join(data_dir, 'evecs.csv'), evecs, delimiter=',')
np.savetxt(join(data_dir, 'evals_h.csv'), evals, delimiter=',')
np.savetxt(join(data_dir, 'evals_q.csv'), evals/(1+evals), delimiter=',')
np.savetxt(join(data_dir, 'gamma_star.csv'), prolongation, delimiter=',')
np.savetxt(join(data_dir, 'gamma.csv'), reduction, delimiter=',')
print('... Complete ...\n')


#
#         # temp = ev.where(ev.perturbation == p, drop=True).drop('perturbation').squeeze('perturbation')

# d.to_netcdf('/n/seasasfs02/hnesser/TROPOMI_inversion/evec_perturbations_ZQ/evec_perturbations_0001.nc',
#                encoding={'evec_perturbations' : {'_FillValue' : None,
#                                                  'dtype' : 'float32'},
#                          'time' : {'_FillValue' : None,
#                                    'dtype' : 'float32'},
#                          'lat' : {'_FillValue' : None,
#                                    'dtype' : 'float32'},
#                          'lon' : {'_FillValue' : None,
#                                    'dtype' : 'float32'}})

# # need to set attribute dictionaries for lat
# # lon, time, and the evec_pertuurbations, complete
# # with units

# d.time.attrs = {'long_name' : 'Time', 'units' : 'hours since 2009-01-01 00:00:00', 'calendar' : 'standard'}

## -------------------------------------------------------------------------##
## Check that eigenvector perturbations worked
## -------------------------------------------------------------------------##


## -------------------------------------------------------------------------##
## Calculate Jacobian column from K and eigenvectors
## -------------------------------------------------------------------------##
root_dir = '/n/seasasfs02/hnesser/TROPOMI_inversion/evec_perturbations_ZQ'
k_file = 'kA.pkl'
evec_file = 'evecs.csv'

# Load Jacobian and eigenvectors
k_true = load_obj(join(root_dir, k_file)).astype('float32')
evec = pd.read_csv(join(root_dir, evec_file), header=None, usecols=[0])

# Multiply the two
kw_true = np.dot(k_true, evec.values)


## -------------------------------------------------------------------------##
## Calculate Jacobian column from forward model
## -------------------------------------------------------------------------##

# First, try the scaled summed - prior

# Save the scale factor
beta = 1e-8

# Read in each file individually
root_dir = '/n/holyscratch01/jacob_lab/hnesser/eigenvector_perturbation_test/'
test_prior_dir = join(root_dir, 'test_prior')
test_pert_dir = join(root_dir, 'test_summed')

prior = pd.read_csv(join(test_prior_dir, 'sat_obs.gosat.00.m'),
                    delim_whitespace=True, header=0, usecols=['model'],
                    low_memory=True)
pert = pd.read_csv(join(test_pert_dir, 'sat_obs.gosat.00.m'),
                   delim_whitespace=True,header=0, usecols=['model'],
                   low_memory=True)
kw_gc = (pert - prior)/beta

## -------------------------------------------------------------------------##
## Compare the two
## -------------------------------------------------------------------------##

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(kw_true, kw_gc.values)
ax.set_xlabel('Linear Algebra')
ax.set_ylabel('Forward Model')
ax.set_xlim(-30, 1)
ax.set_ylim(-30, 1)
plt.show()

