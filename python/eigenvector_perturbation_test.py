'''
This script tests eigenvector perturbations using Zhen Qu's 2019 GOSAT
inversion. Because it is a one-time test, a lot of things are hard-coded.
That being said, for the sake of clarity, all of the python scripts needed
to validate the results of eigenvector perturbations are areincluded in
this script. They cannot currently all be run simultaneously--some require
additional model runs. User preferences are set below to minimize unndeed
repetition.
'''

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
import matplotlib.pyplot as plt
from matplotlib import rcParams

import sys
sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python')
import inversion as inv
import gcpy as gc
import format_plots as fp

# os.environ['QT_QPA_PLATFORM'] = 'offscreen'
rcParams['text.usetex'] = False

## -------------------------------------------------------------------------##
## User preferences
## -------------------------------------------------------------------------##
CalculateInversionQuantities = False
CalculateEigenvectors = False
ScaleEigenvectors = False
# ScaleFactor = 5e-8
# ScaleFactor = 100
CompareResults = True

data_dir = '/n/seasasfs02/hnesser/TROPOMI_inversion/evec_perturbations_JDM'
RF = 0.05

## -------------------------------------------------------------------------##
## Read data
## -------------------------------------------------------------------------##
if CalculateInversionQuantities:
    rrsd = load_obj(join(data_dir, 'rrsd.pkl'))
    gc = pd.read_csv(join(data_dir, 'gc_output'),
                     delim_whitespace=True, header=0,
                     usecols=['I', 'J', 'GOSAT', 'model', 'S_OBS'])
    k = load_obj(join(data_dir, 'kA.pkl')).T
    # rat =

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
# This else statement isn't strictly correct--we don't always need to load
# these quantities (i.e. if we have already calculated eigenvectors)
else:
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
## Create inversion object
## -------------------------------------------------------------------------##
if CalculateEigenvectors:
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
    reduction = (1/inv_zq.sa_vec**0.5) * evecs.T

    # Saving result to our instance.
    print('Saving eigenvalues and eigenvectors.')
    # self.evals = evals/(1 + evals)
    np.savetxt(join(data_dir, 'evecs.csv'), evecs, delimiter=',')
    np.savetxt(join(data_dir, 'evals_h.csv'), evals, delimiter=',')
    np.savetxt(join(data_dir, 'evals_q.csv'), evals/(1+evals), delimiter=',')
    np.savetxt(join(data_dir, 'gamma_star.csv'), prolongation, delimiter=',')
    np.savetxt(join(data_dir, 'gamma.csv'), reduction, delimiter=',')
    print('... Complete ...\n')

    # Now match the gamma_star values to the clusters
    clusters = xr.open_dataarray(join(data_dir, 'clusters.nc'),
                                 decode_times=False)

    for i in range(10):
        pert = p.match_data_to_clusters(prolongation[:, i], clusters, 0)
        print(pert)
        pert = pert.to_dataset(name='evec_pert')

        # Define HEMCO attributes
        pert.attrs = {'Title' : 'Eigenvector perturbation %d for the construction of the Jacobian matrix for methane inversions' % i}
        pert.time.attrs = {'long_name' : 'Time',
                           'units' : 'hours since 2009-01-01 00:00:00',
                           'calendar' : 'standard'}
        pert.lat.attrs = {'long_name': 'latitude', 'units': 'degrees_north'}
        pert.lon.attrs = {'long_name': 'longitude', 'units': 'degrees_east'}
        pert['evec_pert'].attrs = {'long_name' : 'Eigenvector perturbations',
                                   'units' : 'kg/m2/s'}

        pert.to_netcdf(join(data_dir, 'evec_pert_%04d.nc' % i),
                       encoding={'evec_pert' : {'_FillValue' : None,
                                                'dtype' : 'float32'},
                                 'time' : {'_FillValue' : None,
                                           'dtype' : 'float32'},
                                 'lat' : {'_FillValue' : None,
                                          'dtype' : 'float32'},
                                 'lon' : {'_FillValue' : None,
                                          'dtype' : 'float32'}})

## -------------------------------------------------------------------------##
## Scale existing eigenvectors
## -------------------------------------------------------------------------##
if ScaleEigenvectors:
    for i in range(10):
        pert = xr.open_dataset(join(data_dir, 'evec_pert_%04d.nc' % i),
                               decode_times=False)
        for j in range(1, 4):
            ScaleFactor = 10**j
            pert['evec_pert'] *= ScaleFactor
            pert.attrs['Scale Factor'] = ScaleFactor
            pert.to_netcdf(join(data_dir,
                                'evec_pert_%04d_scale_%d.nc' % (i, ScaleFactor)),
                           encoding={'evec_pert' : {'_FillValue' : None,
                                                    'dtype' : 'float32'},
                                     'time' : {'_FillValue' : None,
                                               'dtype' : 'float32'},
                                    'lat' : {'_FillValue' : None,
                                             'dtype' : 'float32'},
                                    'lon' : {'_FillValue' : None,
                                             'dtype' : 'float32'}})

## -------------------------------------------------------------------------##
## Check that eigenvector perturbations worked
## -------------------------------------------------------------------------##
if CompareResults:
    ## ---------------------------------------------------------------------##
    ## Calculate Jacobian column from K and eigenvectors
    ## ---------------------------------------------------------------------##
    k_file = 'kA.pkl'
    prolongation_file = 'gamma_star.csv'

    # Load Jacobian and eigenvectors
    k_true = load_obj(join(data_dir, k_file)).astype('float32').T
    evec = pd.read_csv(join(data_dir, prolongation_file),
                       header=None, usecols=[0])

    # Multiply the two
    kw_true = np.dot(k_true, evec.values)

    # Clear up memory
    del(k_true)
    del(evec)

    ## ---------------------------------------------------------------------##
    ## Calculate Jacobian column from forward model
    ## ---------------------------------------------------------------------##

    # First, try the scaled summed - prior

    # Read in each file individually
    run_dir = '/n/holyscratch01/jacob_lab/hnesser/eigenvector_perturbation_test/'
    test_prior_dir = join(run_dir, 'test_prior')
    test_pert_dir = join(run_dir, 'test_summed')
    ScaleFactor = 1

    prior = pd.read_csv(join(test_prior_dir, 'sat_obs.gosat.00.m'),
                        delim_whitespace=True, header=0, usecols=['model'],
                        low_memory=True)
    pert = pd.read_csv(join(test_pert_dir, 'sat_obs.gosat.00.m'),
                       delim_whitespace=True,header=0, usecols=['model'],
                       low_memory=True)
    kw_gc = (pert - prior)/ScaleFactor

    # Subset
    subset = ~np.isnan(kw_true)
    kw_gc = kw_gc[subset]
    kw_true = kw_true[subset]

    ## ---------------------------------------------------------------------##
    ## Compare the two
    ## ---------------------------------------------------------------------##

    # Get square axis limits
    xlim, ylim, _, _, _ = fp.get_square_limits(kw_true, kw_gc.values)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(kw_true, kw_gc.values)
    ax.set_xlabel('Linear Algebra')
    ax.set_ylabel('Forward Model')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()

