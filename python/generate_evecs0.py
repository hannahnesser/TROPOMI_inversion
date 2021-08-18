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

import matplotlib.pyplot as plt

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
local = False

# Cannon
if not local:
    code_dir = '/n/home04/hnesser/TROPOMI_inversion/python/'
    data_dir = f'/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion/'
    plot_dir = None
else:
    base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
    code_dir = f'{base_dir}python/'
    data_dir = f'{base_dir}inversion_data/'
    plot_dir = f'{base_dir}plots/'

# User preferences
calculate_evecs = False
plot_evals = True
format_evecs = False
n_evecs = int(10)
calculate_avker = False
pct_of_info = None
snr = 1.25
rank = None

## ------------------------------------------------------------------------ ##
## Set up working environment
## ------------------------------------------------------------------------ ##
# Import custom packages
import sys
sys.path.append(code_dir)
import inversion as inv
import inversion_settings as s
import gcpy as gc
import invpy as ip
import format_plots as fp
import config as c

if not local:
    # Import dask things
    from dask.distributed import Client, LocalCluster, progress
    from dask.diagnostics import ProgressBar
    import dask.config
    dask.config.set({'distributed.comm.timeouts.connect' : 90,
                     'distributed.comm.timeouts.tcp' : 150,
                     'distributed.adaptive.wait-count' : 90,
                     'temporary_directory' : data_dir})

    # Open cluster and client
    n_workers = 2
    threads_per_worker = 2
    cluster = LocalCluster(n_workers=n_workers,
                           threads_per_worker=threads_per_worker)
    client = Client(cluster)

    # We now calculate chunk size.
    n_threads = n_workers*threads_per_worker
    nstate_chunk = 1e3
    chunks = {'nstate_0' : nstate_chunk, 'nstate_1' : nstate_chunk}
    print('State vector chunks : ', nstate_chunk)

## ------------------------------------------------------------------------ ##
## Load global quantities
## ------------------------------------------------------------------------ ##
# State vector dimension
sa = gc.read_file(f'{data_dir}sa.nc')
sa = sa.values.reshape(-1, 1)
nstate = sa.shape[0]

## -------------------------------------------------------------------------##
## Calculate the eigenvectors
## -------------------------------------------------------------------------##
if calculate_evecs:
    # Sum together the monthly PPHs
    pph = xr.DataArray(np.zeros((nstate, nstate)),
                       dims=['nstate_0', 'nstate_1'], name='pph0')
    for m in s.months:
        print(f'Loading month {m}.')
        temp = xr.open_dataarray(f'{data_dir}pph0_m{m:02d}.nc')
        pph += temp

    # Load pph into memory
    pph = pph.compute()

    # Calculate the eigenvectors (this is the time consuming step)
    evals_h, evecs = eigh(pph)

    # Sort them
    idx = np.argsort(evals_h)[::-1]
    evals_h = evals_h[idx]
    evecs = evecs[:, idx]

    # Calculate evals_q
    evals_q = evals_h/(1 + evals_h)

    # Calculate the prolongation and reduction operators
    prolongation = (evecs * sa**0.5).T
    reduction = (1/sa**0.5) * evecs.T

    # Save out the matrices
    np.save(f'{data_dir}evecs0.npy', evecs)
    np.save(f'{data_dir}evals_h0.npy', evals_h)
    np.save(f'{data_dir}evals_q0.npy', evals_q)
    np.save(f'{data_dir}prolongation0.npy', prolongation)
    np.save(f'{data_dir}reduction0.npy', reduction)

else:
    evals_h = np.load(f'{data_dir}evals_h0.npy')
    if not local:
        evecs = np.load(f'{data_dir}evecs0.npy')
        evals_q = np.load(f'{data_dir}evals_q0.npy')
        prolongation = np.load(f'{data_dir}prolongation0.npy')
        reduction = np.load(f'{data_dir}reduction0.npy')
    else:
        evals_q = evals_h/(1 + evals_h)

## -------------------------------------------------------------------------##
## Plot the eigenvalues
## -------------------------------------------------------------------------##
if plot_evals and (plot_dir is not None):
    evals_h[evals_h < 0] = 0
    snr = evals_h**0.5
    DOFS_frac = np.cumsum(evals_q)/evals_q.sum()

    fig, ax = fp.get_figax(aspect=3)

    # DOFS frac
    ax.plot(DOFS_frac, label='Information content spectrum',
            c=fp.color(3), lw=1)
    for p in [0.9, 0.95, 0.97, 0.98, 0.99]:
        diff = np.abs(DOFS_frac - p)
        rank = np.argwhere(diff == np.min(diff))[0][0]
        ax.scatter(rank, DOFS_frac[rank], marker='*', s=20, c=fp.color(3))
        ax.text(rank, DOFS_frac[rank]-0.05, f'{int(100*p):d}%%',
                ha='center', va='top', c=fp.color(3))
    ax.set_xlabel('Eigenvector index', fontsize=c.LABEL_FONTSIZE*c.SCALE,
                  labelpad=c.LABEL_PAD)
    ax.set_ylabel('Fraction of DOFS', fontsize=c.LABEL_FONTSIZE*c.SCALE,
                  labelpad=c.LABEL_PAD, color=fp.color(3))
    ax.tick_params(axis='both', which='both',
                   labelsize=c.LABEL_FONTSIZE*c.SCALE)
    ax.tick_params(axis='y', labelcolor=fp.color(3))

    # SNR
    ax2 = ax.twinx()
    ax2.plot(snr, label='Signal-to-noise ratio spectrum', c=fp.color(6), lw=1)
    for r in [1, 2]:
        diff = np.abs(snr - r)
        rank = np.argwhere(diff == np.min(diff))[0][0]
        ax2.scatter(rank, snr[rank], marker='.', s=20, c=fp.color(6))
        ax2.text(rank-200, snr[rank], f'{r:d}', ha='right', va='center',
                 c=fp.color(6))
    ax2.set_ylabel('Signal-to-noise ratio', fontsize=c.LABEL_FONTSIZE*c.SCALE,
                  labelpad=c.LABEL_PAD, color=fp.color(6))
    ax2.tick_params(axis='y', which='both', labelsize=c.LABEL_FONTSIZE*c.SCALE,
                    labelcolor=fp.color(6))

    ax = fp.add_title(ax, 'Initial Estimate Information Content Spectrum')

    fp.save_fig(fig, plot_dir, 'eigenvalues')

## -------------------------------------------------------------------------##
## Format the eigenvectors for HEMCO
## -------------------------------------------------------------------------##
if format_evecs:
    # First, print information about the rank/percent of information content/
    # SNR associated with the number of evecs provided
    DOFS_frac = np.cumsum(evals_q)/evals_q.sum()
    print(f'SAVING OUT {n_evecs} EIGENVECTORS')
    print(f'% DOFS : {(100*DOFS_frac[n_evecs])}')
    print(f'SNR    : {(evals_h[n_evecs]**0.5)}')

    # Load clusters
    clusters = xr.open_dataarray(f'{data_dir}clusters.nc')

    # Iterate through columns and save out HEMCO-appropriate files
    for i in range(n_evecs):
        pert = ip.match_data_to_clusters(prolongation[:, i], clusters, 0)

        # Define HEMCO attributes
        long_name = 'Eigenvector perturbations'
        title_str = f'Eigenvector perturbation {i+1} for the construction of the Jacobian matrix for methane inversions.'
        pert = gc.define_HEMCO_std_attributes(pert, name='evec_pert')
        pert = gc.define_HEMCO_var_attributes(pert, 'evec_pert',
                                              long_name=long_name,
                                              units='kg/m2/s')
        pert.attrs = {'Title' : title_str}

        # Scaling (for a later date)
        # for s in evec_scaling:
        #     suffix = f'_{(i+1):04d}_{s}'
        #     pert.attrs = {'Title' : title_str, 'Scaling' : s}
        #     p = deepcopy(pert)
        #     p['evec_pert'] *= float(s)

        gc.save_HEMCO_netcdf(pert, data_dir, f'evec_pert_{(i+1):02d}.nc')

## -------------------------------------------------------------------------##
## Calculate the averaging kernel
## -------------------------------------------------------------------------##
if calculate_avker:
    # Figure out the fraction of information content
    if sum(x is not None for x in [pct_of_info, snr, rank]) > 1:
        raise AttributeError('Conflicting rank arguments provided.')
    elif sum(x is not None for x in [pct_of_info, snr, rank]) == 0:
        raise AttributeError('Insufficient rank arguments provided.')
    elif pct_of_info is not None:
        DOFS_frac = np.cumsum(evals_q)/evals_q.sum()
        diff = np.abs(DOFS_frac - (pct_of_info/100))
        rank = np.argwhere(diff == np.min(diff))[0][0]
        suffix = f'_poi{pct_of_info}'
    elif snr is not None:
        evals_h[evals_h < 0] = 0
        diff = np.abs(evals_h**0.5 - snr)
        rank = np.argwhere(diff == np.min(diff))[0][0]
        suffix = f'_snr{snr}'
    else:
        suffix = f'_rank{rank}'
    print(f'Rank = {rank}')

    # Subset the evals and evecs
    evals_q = evals_q[:rank]
    evecs = evecs[:, :rank]

    # Calculate the averaging kernel (we can leave off Sa when it's
    # constant)
    a = (evecs*evals_q) @ evecs.T

    # Save the result
    np.save(f'{data_dir}a0{suffix}.npy', a)
    np.save(f'{data_dir}dofs0{suffix}.npy', np.diagonal(a))

# ## -------------------------------------------------------------------------##
# ## Create inversion object
# ## -------------------------------------------------------------------------##

# # Create a true Reduced Rank Jacobian object
# inv_zq = inv.ReducedRankJacobian(k, xa, sa_vec, y, y_base, so_vec)
# inv_zq.rf = RF

# # Delete the other items from memory
# del(k)
# del(xa)
# del(sa_vec)
# del(y)
# del(y_base)
# del(so_vec)

# # Save out the PPH
# pph = (inv_zq.sa_vec**0.5)*inv_zq.k.T
# pph = np.dot(pph, ((inv_zq.rf/inv_zq.so_vec)*pph.T))
# save_obj(pph, join(data_dir, 'pph.pkl'))

# # Complete an eigendecomposition of the prior pre-
# # conditioned Hessian
# assert np.allclose(pph, pph.T, rtol=1e-5), \
#        'The prior pre-conditioned Hessian is not symmetric.'

# evals, evecs = eigh(pph)
# print('Eigendecomposition complete.')

# # Sort evals and evecs by eval
# idx = np.argsort(evals)[::-1]
# evals = evals[idx]
# evecs = evecs[:,idx]

# # Force all evals to be non-negative
# if (evals < 0).sum() > 0:
#     print('Negative eigenvalues. Maximum negative value is %.2e. Setting negative eigenvalues to zero.' \
#         % (evals[evals < 0].min()))
#     evals[evals < 0] = 0

# # Check for imaginary eigenvector components and force all
# # eigenvectors to be only the real component.
# if np.any(np.iscomplex(evecs)):
#     print('Imaginary eigenvectors exist at index %d of %d. Forcing eigenvectors to real component alone.' \
#           % ((np.where(np.iscomplex(evecs))[1][0] - 1), len(evecs)))
#     evecs = np.real(evecs)

# # Saving result to our instance.
# print('Saving eigenvalues and eigenvectors.')
# # self.evals = evals/(1 + evals)
# np.savetxt(join(data_dir, 'evecs.csv'), evecs, delimiter=',')
# np.savetxt(join(data_dir, 'evals_h.csv'), evals, delimiter=',')
# np.savetxt(join(data_dir, 'evals_q.csv'), evals/(1+evals), delimiter=',')
# print('... Complete ...\n')


# #
# #         # temp = ev.where(ev.perturbation == p, drop=True).drop('perturbation').squeeze('perturbation')
# d.lat.attrs = {'long_name': 'latitude', 'units': 'degrees_north'}
# d.lon.attrs = {'long_name': 'longitude', 'units': 'degrees_east'}
# d.time.attrs = {'long_name' : 'Time'}
# d['evec_perturbations'].attrs = {'long_name' : 'Eigenvector perturbations',
#                                  'units' : 'kg/m2/s'}
# d.to_netcdf('/n/seasasfs02/hnesser/TROPOMI_inversion/evec_perturbations_ZQ/evec_perturbations_0001_scaled_1e-8.nc',
#                encoding={'evec_perturbations' : {'_FillValue' : None,
#                                                  'dtype' : 'float32'},
#                          'time' : {'_FillValue' : None, 'dtype' : 'float32',
#                                    'calendar' : 'standard',
#                                    'units' : 'hours since 2009-01-01 00:00:00'},
#                          'lat' : {'_FillValue' : None, 'dtype' : 'float32'},
#                          'lon' : {'_FillValue' : None, 'dtype' : 'float32'}})

# # # need to set attribute dictionaries for lat
# # # lon, time, and the evec_pertuurbations, complete
# # # with units

# d.time.attrs = {'long_name' : 'Time', 'units' : 'hours since 2009-01-01 00:00:00', 'calendar' : 'standard'}

