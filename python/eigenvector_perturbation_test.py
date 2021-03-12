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
from copy import deepcopy
# os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import xarray as xr
import pickle
import os
import pandas as pd
import math
import numpy as np
from scipy.linalg import eigh

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams

import sys
sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python')
import inversion as inv
import gcpy as gc
import invpy as ip
import format_plots as fp

# os.environ['QT_QPA_PLATFORM'] = 'offscreen'
rcParams['text.usetex'] = False

## -------------------------------------------------------------------------##
## User preferences
## -------------------------------------------------------------------------##
CalculateEigenvectors = False
CompareResults = True

data_dir = '/n/seasasfs02/hnesser/TROPOMI_inversion/evec_perturbations_JDM'
run_dir = '/n/holyscratch01/jacob_lab/hnesser/eigenvector_perturbation_test_JDM/'
RF = 0.05
year = 2015
evec_scaling = ['0', '1']#, '1e-6', '1e-7']
prior_dir = join(run_dir, 'prior_dir')
pert_dir = [join(run_dir, f'pert_dir_{s}') for s in evec_scaling]

## -------------------------------------------------------------------------##
## Calculate eigenvectors
## -------------------------------------------------------------------------##
if CalculateEigenvectors:
    # Read in observing system
    k = xr.open_dataarray(join(data_dir, 'k.nc')).values.T
    y = xr.open_dataarray(join(data_dir, 'y.nc')).values.reshape(-1, 1)
    y_base = xr.open_dataarray(join(data_dir,
                                    'y_base.nc')).values.reshape(-1, 1)
    so_vec = xr.open_dataarray(join(data_dir,
                                    'so_vec.nc')).values.reshape(-1, 1)

    # Subset observing system
    years = xr.open_dataarray(join(data_dir, 'y_year.nc'))
    glint = xr.open_dataarray(join(data_dir, 'y_glint.nc'))
    lats = xr.open_dataarray(join(data_dir, 'y_lat.nc'))
    cond = ((lats < 60) & (years == 2015) & (glint < 0.5)).values
    del(years)
    del(glint)
    del(lats)

    k = k[cond, :]
    y = y[cond]
    y_base = y_base[cond]
    so_vec = so_vec[cond]

    # Force Jacobian to be positive
    k[k < 0] = 0

    # Save
    gc.save_obj(k, join(data_dir, 'k.pkl'))

    # Define dimensions
    nobs, nstate = k.shape

    # Read in prior
    xa = np.ones((nstate, 1))
    xa_abs = xr.open_dataarray(join(data_dir,
                                    'xa_abs.nc')).values.reshape(-1, 1)
    sa_vec = xr.open_dataarray(join(data_dir,
                                    'sa_vec.nc')).values.reshape(-1, 1)

    # Create a true Reduced Rank Jacobian object
    true = inv.ReducedRankJacobian(k, xa, sa_vec, y, y_base, so_vec)
    true.rf = RF

    # Delete the other items from memory
    del(k)
    del(xa)
    del(sa_vec)
    del(y)
    del(y_base)
    del(so_vec)

    # Save out the PPH
    pph = (true.sa_vec**0.5)*true.k.T
    pph = np.dot(pph, ((true.rf/true.so_vec)*pph.T))
    gc.save_obj(pph, join(data_dir, 'pph.pkl'))

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
    prolongation = (evecs * true.sa_vec**0.5).T
    reduction = (1/true.sa_vec**0.5) * evecs.T

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

    for i in range(3):
        pert = ip.match_data_to_clusters(prolongation[:, i], clusters, 0)

        # Define HEMCO attributes
        long_name = 'Eigenvector perturbations'
        pert = gc.define_HEMCO_std_attributes(pert, name='evec_pert')
        # Just for the old version of the model, remove the time dimension
        pert = pert.squeeze()
        pert = gc.define_HEMCO_var_attributes(pert, 'evec_pert',
                                              long_name=long_name,
                                              units='kg/m2/s')
        for s in evec_scaling:
            suffix = f'_{(i+1):04d}_{s}'
            title_str = f'Eigenvector perturbation {i+1} for the construction of the Jacobian matrix for methane inversions.'
            pert.attrs = {'Title' : title_str, 'Scaling' : s}

            # Scale
            p = deepcopy(pert)
            p['evec_pert'] *= float(s)

            gc.save_HEMCO_netcdf(p, data_dir, f'evec_pert{suffix}.nc')

    pert.plot()
    plt.show()

## -------------------------------------------------------------------------##
## Check that eigenvector perturbations worked
## -------------------------------------------------------------------------##
if CompareResults:
    ## ---------------------------------------------------------------------##
    ## Calculate Jacobian column from K and eigenvectors
    ## ---------------------------------------------------------------------##
    k_file = 'k.pkl'
    prolongation_file = 'gamma_star.csv'

    # Load Jacobian and eigenvectors
    k_true = gc.load_obj(join(data_dir, k_file))
    evec = pd.read_csv(join(data_dir, prolongation_file),
                       header=None, usecols=[0])

    # Multiply the two
    kw_true = np.dot(k_true, evec.values)

    # Subset
    subset = ~np.isnan(kw_true).reshape(-1,)
    kw_true = kw_true[subset]

    # Clear up memory
    del(k_true)
    del(evec)

    ## ---------------------------------------------------------------------##
    ## Calculate Jacobian column from forward model and compare
    ## ---------------------------------------------------------------------##
    # Read prior
    prior = pd.read_csv(join(prior_dir, 'sat_obs.gosat.00.m'),
                        delim_whitespace=True, header=0,
                        usecols=['model', 'GLINT', 'LAT'],
                        low_memory=True)
    cond = ((prior['LAT'] < 60) & (prior['GLINT'] < 0.5))
    prior = prior[cond]['model'].values.reshape(-1,)

    # Read perturbations
    for p in pert_dir[1:2]:
        scale_factor = p.split('_')[-1]
        print(scale_factor)
        pert = pd.read_csv(join(p, 'sat_obs.gosat.00.m'),
                           delim_whitespace=True, header=0,
                           usecols=['model', 'GLINT', 'LAT'],
                           low_memory=True)
        cond = ((pert['LAT'] < 60) & (pert['GLINT'] < 0.5))
        pert = pert[cond]['model'].values
        kw_gc = (pert - prior)#/float(scale_factor)
        print(p)
        print(kw_gc)
        # kw_gc = kw_gc[subset]

        ## Plot the comparison
        # Get square axis limits
        xlim, ylim, _, _, _ = fp.get_square_limits(kw_true, kw_gc)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(kw_true, kw_gc)
        ax.set_title(scale_factor)
        ax.set_xlabel('Linear Algebra')
        ax.set_ylabel('Forward Model')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        plt.show()

