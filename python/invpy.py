'''
Generic inversion functions (this is integrated with the inversion class).
'''

import numpy as np
from numpy.linalg import inv
import pandas as pd
import xarray as xr
import dask.array as da
import pickle
import math
from scipy.stats import linregress
from scipy.linalg import sqrtm

from os.path import join
from os import listdir
import glob
import time
import copy

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams, colorbar, colors
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom
import cartopy.crs as ccrs
import cartopy

# Import information for plotting in a consistent fashion
import sys
sys.path.append('.')
import format_plots as fp
import config

## -------------------------------------------------------------------------##
## Utility functions
## -------------------------------------------------------------------------##
def find_dimension(data):
    if len(data.shape) == 1:
        return 1
    elif (data.shape[1] == 1) or (data.shape[0] == 1):
        return 1
    else:
        return 2

def inv_cov_matrix(data):
    if find_dimension(data) == 1:
        return np.diag(1/data)
    else:
        return inv(data)

def sqrt_cov_matrix(data):
    if find_dimension(data) == 1:
        return np.diag(data**0.5)
    else:
        return sqrtm(data)

def multiply_data_by_inv_cov(data, data_cov):
    if find_dimension(data_cov) == 1:
        return data/data_cov
    else:
        return data @ inv(data_cov)

def multiply_data_by_sqrt_cov(data, data_cov):
    if find_dimension(data_cov) == 1:
        return data*data_cov**0.5
    else:
        return data @ sqrtm(data_cov)

## -------------------------------------------------------------------------##
## Cluster functions
## -------------------------------------------------------------------------##
def match_data_to_clusters(data, clusters, default_value=0):
    '''
    Matches inversion data to a cluster file.
    Parameters:
        data (np.array)        : Inversion data. Must have the same length
                                 as the number of clusters, and must be
                                 sorted in ascending order of cluster number
                                 - i.e. [datapoint for cluster 1, datapoint for
                                 cluster 2, datapoint for cluster 3...]
        clusters (xr.Datarray) : 2d array of cluster values for each gridcell.
                                 You can get this directly from a cluster file
                                 used in an analytical inversion.
                                 Dimensions: ('lat','lon')
        default_value (numeric): The fill value for the array returned.
    Returns:
        result (xr.Datarray)   : A 2d array on the GEOS-Chem grid, with
                                 inversion data assigned to each gridcell based
                                 on the cluster file.
                                 Missing data default to the default_value.
                                 Dimensions: same as clusters ('lat','lon').
    '''
    # check that length of data is the same as number of clusters
    clust_list = np.unique(clusters)[np.unique(clusters)!=0] # unique, nonzero clusters
    assert len(data)==len(clust_list), (f'Data length ({len(data)}) is not the same as '
                                        f'the number of clusters ({len(clust_list)}).')

    # build a lookup table from data.
    #    data_lookup[0] = default_value (value for cluster 0),
    #    data_lookup[1] = value for cluster 1, and so forth
    data_lookup = np.append(default_value, data)

    # use fancy indexing to map data to 2d cluster array
    cluster_index = clusters.squeeze().data.astype(int).tolist()
    result = clusters.copy().squeeze()         # has same shape/dims as clusters
    result.values = data_lookup[cluster_index] # map data to clusters

    return result

def clusters_2d_to_1d(clusters, data, fill_value=0):
    '''
    Flattens data on the GEOS-Chem grid, and ensures the resulting order is
    ascending with respect to cluster number.
    Parameters:
        clusters (xr.Datarray) : 2d array of cluster values for each gridcell.
                                 You can get this directly from a cluster file
                                 used in an analytical inversion.
                                 Dimensions: ('lat','lon')
        data (xr.DataArray)    : Data on a 2d GEOS-Chem grid.
                                 Dimensions: ('lat','lon')
   '''
    # Data must be a dataarray
    assert type(data) == xr.core.dataarray.DataArray, \
           "Input data must be a dataarray."

    # Combine clusters and data into one dataarray
    data = data.to_dataset(name='data')
    data['clusters'] = clusters

    # Convert to a dataframe and reset index to remove lat/lon/time
    # dimensions
    data = data.to_dataframe().reset_index()[['data', 'clusters']]

    # Remove non-cluster datapoints
    data = data[data['clusters'] > 0]

    # Fill nans that may result from data and clusters being different
    # shapes
    data = data.fillna(fill_value)

    # Sort
    data = data.sort_values(by='clusters')

    return data['data'].values

def get_one_statevec_layer(data, category=None, time=None,
                           category_list=None, time_list=None, cluster_list=None):
    '''
    Grabs only the specified category/time combo from the state vector.
    Should result in a list of clusters which can then be mapped onto a cluster file and plotted.
    Parameters:
        data (np.array)      : Inversion attribute.
                               Dimensions: nstate
        category (string)    : The category you would like to extract. Must match with
                               element(s) in in category_list.
        time (string)        : The time you would like to extract. Must match with
                               element(s) in time_list. If there is no time, use None.
        category_list (list) : The category labels for each element of the state vector.
                               Dimensions: nstate
        time_list (list)     : The time labels for each element of the state vector.
                               Dimensions: nstate
        cluster_list (list)  : Cluster numbers for each element of the state vector.
                               If this option is not included, cluster numbers of the
                               data must be in ascending order.
                               Dimensions: nstate.
    Returns:
        data_out (np.array)  : data subset by category & time.
                               Dimensions: # of clusters
    '''
    # check which labels are defined
    # category
    select_by_category = False
    if (category is not None) or (category_list is not None):      # if either are defined
        if (category is not None) and (category_list is not None): # if both are defined
            select_by_category = True
        else:
            raise ValueError('Please define both category and category_list, or neither.')
    # time
    select_by_time = False
    if (time is not None) or (time_list is not None):      # if either are defined
        if (time is not None) and (time_list is not None): # if both are defined
            select_by_time = True
        else:
            raise ValueError('Please define both time and time_list, or neither.')

    # get a boolean mask for the selection
    if select_by_category and select_by_time:
        mask = (category_list==category)&(time_list==time)
    elif select_by_category:
        mask = category_list==category
    elif select_by_time:
        mask = time_list==time

    # select appropriate data
    data_out = data[mask]

    # rearrange data to put in ascending order of clusters
    if cluster_list is not None:
        cluster_idx = cluster_list[mask].astype(int)-1
        assert len(cluster_idx)==len(np.unique(cluster_idx)),\
        ('Duplicate values found in cluster index. '
         'Check that you are selecting only one cluster layer. '
         'Check that time and category are specific enough that '
         'the result corresponds to only one cluster file.')
        data_out = data_out[cluster_idx]

    return data_out

## -------------------------------------------------------------------------##
## Reduced rank inversion functions
## -------------------------------------------------------------------------##
def calculate_Kx(k_dir, x_data, niter, chunks, optimize_bc):
    # List of K files
    k_files = glob.glob(f'{k_dir}/k{niter}_c??.nc')
    k_files.sort()

    # if niter == 2, also load the boundary condition K
    if optimize_bc:
        k_bc = xr.open_dataarray(f'{k_dir}/k{niter}_bc.nc',
                                 chunks=chunks)
        x_data = np.append(x_data, np.ones((4,)))

    # Start time
    start_time = time.time()

    # Iterate
    print('[', end='')
    kx = []
    i0 = 0
    for i, kf in enumerate(k_files):
        print('-', end='')
        k_n = xr.open_dataarray(kf, chunks=chunks)
        # Append the BC K if it's the second iteration
        if optimize_bc:
            i1 = i0 + k_n.shape[0]
            k_bc_n = k_bc[i0:i1, :]
            k_n = xr.concat([k_n, k_bc_n], dim='nstate')
            i0 = copy.deepcopy(i1)
        k_n = da.tensordot(k_n, x_data, axes=(1, 0))
        k_n = k_n.compute()
        kx.append(k_n)
    active_time = (time.time() - start_time)/60
    print(f'] {active_time:02f} minutes')

    return np.concatenate(kx)

def calculate_c(k, ya, xa):
    '''
    Calculate c for the forward model, defined as ybase = Kxa + c.
    Save c as an element of the object.
    '''
    c = ya - k @ xa
    return c

def cost_func(ynew, y, so, xnew, xa, sa):
    '''
    Calculate the value of the Bayesian cost function
        J(x) = (x - xa)T Sa (x-xa) + regularization_factor(y - Kx)T So (y - Kx)
    for a given x. Prints out that value and the contributions from
    the emission and observational terms.

    Parameters:
        x      The state vector at which to evaluate the cost function
        y      The observations vector at which to evaluate the cost function
               Note that by definition y=Kx+c, so: ya=Kxa+c, and yhat=Kxhat+c
    Returns:
        cost   The value of the cost function at x
    '''

    # Calculate the observational component of the cost function
    cost_obs = (ynew - y).T @ inv_cov_matrix(so) @ (ynew - y)

    # Calculate the emissions/prior component of the cost function
    cost_emi = (xnew - xa).T @ inv_cov_matrix(sa) @ (xnew - xa)

    # Calculate the total cost, print out information on the cost, and
    # return the total cost function value
    cost = cost_obs + cost_emi
    print('     Cost function: %.2f (Emissions: %.2f, Observations: %.2f)'
          % (cost, cost_emi, cost_obs))
    return cost

def calculate_xhat(shat, kt_so_ydiff):
    # (this formulation only works with constant errors I think)
    return 1 + np.array(shat @ kt_so_ydiff)

def calculate_xhat_fr(a, sa, kt_so_ydiff):
    shat_kpi = (np.identity(len(sa)) - a)*sa.reshape((1, -1))
    return 1 + np.array(shat_kpi @ kt_so_ydiff)

def calculate_a(evecs, evals_h, sa):
    evals_q = evals_h/(1 + evals_h)
    a = sa**0.5*(evecs*evals_q) @ evecs.T*(1/(sa**0.5))
    return a

def calculate_shat(evecs, evals_h, sa):
    # This formulation only works with diagonal errors
    sa_evecs = evecs*(sa**0.5)
    shat = (sa_evecs*(1/(1 + evals_h))) @ sa_evecs.T
    return shat

def solve_inversion(evecs, evals_h, sa, kt_so_ydiff):
    shat = calculate_shat(evecs, evals_h, sa)
    a = calculate_a(evecs, evals_h, sa)
    xhat = calculate_xhat(shat, kt_so_ydiff)
    xhat_fr = calculate_xhat_fr(a, sa, kt_so_ydiff)
    return xhat, xhat_fr, shat, a

def get_rank(evals_q=None, evals_h=None, pct_of_info=None, rank=None, snr=None):
    # Check whether evals_q or evals_h are provided:
    if sum(x is not None for x in [evals_q, evals_h]) == 0:
        raise AttributeError('Must provide one of evals_q or evals_h.')
    elif evals_h is None:
        evals_h = evals_q/(1 - evals_q)
    elif  evals_q is None:
        evals_q = evals_h/(1 + evals_h)

    # Calculate the cumulative fraction of information content explained
    # by each subsequent eigenvector
    frac = np.cumsum(evals_q/evals_q.sum())

    # Obtain the rank, requiring one and only one of pct_of_info, rank,
    # or snr to be provided
    if sum(x is not None for x in [pct_of_info, rank, snr]) > 1:
        raise AttributeError('Provide only one of pct_of_info, rank, or snr.')
    elif sum(x is not None for x in [pct_of_info, rank, snr]) == 0:
        raise AttributeError('Must provide one of pct_of_info, rank, or snr.')
    elif pct_of_info is not None:
        diff = np.abs(frac - pct_of_info)
        rank = np.argwhere(diff == np.min(diff))[0][0]
        print('Calculated rank from percent of information: %d' % rank)
        print('     Percent of information: %.4f%%' % (100*pct_of_info))
        print('     Signal-to-noise ratio: %.2f'
              % (evals_h[rank])**0.5)
    elif snr is not None:
        diff = np.abs(evals_h**0.5 - snr)
        rank = np.argwhere(diff == np.min(diff))[0][0]
        print('Calculated rank from signal-to-noise ratio : %d' % rank)
        print('     Percent of information: %.4f%%' % (100*frac[rank]))
        print('     Signal-to-noise ratio: %.2f' % snr)
    elif rank is not None:
        print('Using defined rank: %d' % rank)
        print('     Percent of information: %.4f%%' % (100*frac[rank]))
        print('     Signal-to-noise ratio: %.2f'
              % (evals_h[rank])**0.5)
    return rank

def pph(k, so, sa, big_mem=False):
    sasqrt_kt = multiply_data_by_sqrt_cov(k, sa)
    sasqrt_kt_soinv = multiply_data_by_inv_cov(sasqrt_kt.T, so)

    if big_mem:
        pph = da.tensordot(sasqrt_kt_soinv, sasqrt_kt, axes=(1, 0))
        pph = xr.DataArray(pph, dims=['nstate_0', 'nstate_1'],
                           name=f'pph')
    else:
        pph = sasqrt_kt_soinv @ sasqrt_kt

    print('Calculated PPH.')
    return pph

# def edecomp(matrix, eval_threshold=None, number_of_evals=None):
#     print('... Calculating eigendecomposition ...')

#     # Check that the matrix is symmetric
#     assert np.allclose(matrix, matrix.T, rtol=1e-5), \
#            'The provided matrix is not symmetric.'

#     # Perform the eigendecomposition of the prior pre-conditioned Hessian
#     # We return the evals of the projection, not of the
#     # prior pre-conditioned Hessian.
#     if (eval_threshold is None) and (number_of_evals is None):
#          evals, evecs = eigh(pph)
#     elif (eval_threshold is None):
#         n = pph.shape[0]
#         evals, evecs = eigh(pph, subset_by_index=[n - number_of_evals,
#                                                   n - 1])
#     else:
#         evals, evecs = eigh(pph, subset_by_value=[eval_threshold, np.inf])
#     print('Eigendecomposition complete.')

#     # Sort evals and evecs by eval
#     idx = np.argsort(evals)[::-1]
#     evals = evals[idx]
#     evecs = evecs[:,idx]

#     # Force all evals to be non-negative
#     if (evals < 0).sum() > 0:
#         print('Negative eigenvalues. Maximum negative value is %.2e. Setting negative eigenvalues to zero.' \
#             % (evals[evals < 0].min()))
#         evals[evals < 0] = 0

#     # Check for imaginary eigenvector components and force all
#     # eigenvectors to be only the real component.
#     if np.any(np.iscomplex(evecs)):
#         print('Imaginary eigenvectors exist at index %d of %d. Forcing eigenvectors to real component alone.' \
#               % ((np.where(np.iscomplex(evecs))[1][0] - 1), len(evecs)))
#         evecs = np.real(evecs)

#     # Saving result to our instance.
#     print('Saving eigenvalues and eigenvectors to instance.')
#     # self.evals = evals/(1 + evals)
#     self.evals_h = evals
#     self.evals_q = evals/(1 + evals)
#     self.evecs = evecs
#     print('... Complete ...\n')

# def solve_inversion(k, y, ya, so, xa, sa, calculate_cost=True):
#     '''
#     Calculate the solution to an analytic Bayesian inversion for the
#     given Inversion object. The solution includes the posterior state
#     vector (xhat), the posterior error covariance matrix (shat), and
#     the averaging kernel (A). The function prints out progress statements
#     and information about the posterior solution, including the value
#     of the cost function at the prior and posterior, the number of
#     negative state vector elements in the posterior solution, and the
#     DOFS of the posterior solution.
#     '''
#     print('... Solving inversion ...')

#     # Check if the errors are diagonal or not and, if so, modify
#     # calculations involving the error covariance matrices
#     sainv = inv_cov_matrix(sa)
#     kTsoinv = multiply_data_by_inv_cov(k.T, so)

#     # Calculate the cost function at the prior.
#     if calculate_cost:
#         print('Calculating the cost function at the prior mean.')
#         cost_prior = cost_func(ya, y, so, xa, xa, sa)

#     # Calculate the posterior error.
#     print('Calculating the posterior error.')
#     shat = np.array(inv(kTsoinv @ k + sainv))

#     # Calculate the posterior mean
#     print('Calculating the posterior mean.')
#     xhat = np.array(xa + (shat @ kTsoinv @ (y - ya)))
#     print('     Negative cells: %d' % xhat[xhat < 0].sum())

#     # Calculate the averaging kernel.
#     print('Calculating the averaging kernel.')
#     a = np.array(identity(xa.shape[0]) - shat @ sainv)
#     dofs = np.diag(a)
#     print('     DOFS: %.2f' % np.trace(a))

#     # Calculate the new set of modeled observations.
#     print('Calculating updated modeled observations.')
#     yhat = np.array(k @ xhat + c)

#     # Calculate the cost function at the posterior. Also calculate the
#     # number of negative cells as an indicator of inversion success.
#     if calculate_cost:
#         print('Calculating the cost function at the posterior mean.')
#         cost_post = cost_func(yhat, y, so, xhat, xa, sa)

#     print('... Complete ...\n')

#     return xhat, shat, a, yhat

## -------------------------------------------------------------------------##
## Source attribution functions
## -------------------------------------------------------------------------##
def source_attribution(w, xhat, shat=None, a=None, sa=None):
    # xhat red = W xhat
    xhat_red = w @ xhat

    if (shat is not None) and (a is not None) and (sa is not None):
        # Convert error covariance matrices
        # S red = W S W^T
        shat_red = w @ shat @ w.T
        sa_red = w @ sa @ w.T

        # Calculate reduced averaging kernel
        a_red = np.identity(w.shape[0]) - shat_red @ inv(sa_red)

    if shat is None and a is None:
        return xhat_red
    else:
        return xhat_red, shat_red, a_red

## -------------------------------------------------------------------------##
## Plotting functions : state vectors
## -------------------------------------------------------------------------##
def plot_state(data, clusters_plot, default_value=0, cbar=True,
               category=None, time=None, category_list=None,
               time_list=None, cluster_list=None, **kw):
    '''
    Plots a state vector element.
    Parameters:
        data (np.array)         : Inversion data.
                                  Dimensions: nstate
        clusters (xr.Datarray)  : 2d array of cluster values for each gridcell.
                                  You can get this directly from a cluster file
                                  used in an analytical inversion.
                                  Dimensions: ('lat','lon')
    Optional Parameters:
        default_value (numeric) : The fill value for the array returned.
        cbar (bool)             : Should the function a colorbar?
        category (string)       : The category you would like to extract. Must match with
                                  element(s) in in category_list.
        time (string)           : The time you would like to extract. Must match with
                                  element(s) in time_list. If there is no time, use None.
        category_list (list)    : The category labels for each element of the state vector.
                                  Dimensions: nstate
        time_list (list)        : The time labels for each element of the state vector.
                                  Dimensions: nstate
        cluster_list (list)     : Cluster numbers for each element of the state vector.
                                  If this option is not included, the data must be
                                  in ascending order of cluster number.
                                  Dimensions: nstate.

    Returns:
        fig, ax, c: Figure, axis, and colorbar for an mpl plot.
    '''
    # protect inputs from modification
    data_to_plot = np.copy(data)

    # Select one "layer" at a time
    # each "layer" corresponds to one "2d cluster file"
    # if you only have one layer in your dataset, you can skip this.
    if ((category is not None) or (time is not None)
        or (category_list is not None) or (time_list is not None)
        or (cluster_list is not None) ):
        data_to_plot = get_one_statevec_layer(data_to_plot, category=category, time=time,
                                              category_list=category_list,
                                              time_list=time_list,
                                              cluster_list=cluster_list)

    # Put the state vector layer back on the 2d GEOS-Chem grid
    # matches the data to lat/lon data using a cluster file
    data_to_plot = match_data_to_clusters(data_to_plot, clusters_plot, default_value)

    # Plot
    fig, ax, c = plot_state_format(data_to_plot, default_value, cbar, **kw)
    return fig, ax, c

def plot_state_format(data, default_value=0, cbar=True, **kw):
    '''
    Format and plot one layer of the state vector.
    Parameters:
        data (xr.DataArray)     : One layer of the state vector, mapped onto a
                                  2d GEOS-Chem grid using a cluster file. If
                                  your state vector has only one layer,
                                  this may contain your entire state vector.
                                  Dimensions: ('lat','lon')
        default_value (numeric) : The fill value for the array returned.
        cbar (bool)             : Should the function plot a colorbar?
    '''
    # Get kw
    title = kw.pop('title', '')
    kw['cmap'] = kw.get('cmap', 'viridis')
    if 'norm' not in kw:
        kw['vmin'] = kw.get('vmin', data.min())
        kw['vmax'] = kw.get('vmax', data.max())
    kw['add_colorbar'] = False
    cbar_kwargs = kw.pop('cbar_kwargs', {})
    label_kwargs = kw.pop('label_kwargs', {})
    title_kwargs = kw.pop('title_kwargs', {})
    map_kwargs = kw.pop('map_kwargs', {})
    fig_kwargs = kw.pop('fig_kwargs', {})

    # Get figure
    lat_step = np.median(np.diff(data.lat))
    lat_range = [data.lat.min().values - lat_step/2,
                 data.lat.max().values + lat_step/2]
    lon_step = np.median(np.diff(data.lon))
    lon_range = [data.lon.min().values - lon_step/2,
                 data.lon.max().values + lon_step/2]
    fig, ax  = fp.get_figax(maps=True, lats=lat_range, lons=lon_range,
                            **fig_kwargs)

    # Plot data
    c = data.plot(ax=ax, snap=True, **kw)

    # Add title and format map
    ax = fp.add_title(ax, title, **title_kwargs)
    ax = fp.format_map(ax, lat_range, lon_range, **map_kwargs)

    if cbar:
        cbar_title = cbar_kwargs.pop('title', '')
        horiz = cbar_kwargs.pop('horizontal', False)
        cpi = cbar_kwargs.pop('cbar_pad_inches', 0.25)
        if horiz:
            orient = 'horizontal'
            cbar_t_kwargs = {'y' : cbar_kwargs.pop('y', -4)}
        else:
            orient = 'vertical'
            cbar_t_kwargs = {'x' : cbar_kwargs.pop('x', 5)}

        cax = fp.add_cax(fig, ax, horizontal=horiz, cbar_pad_inches=cpi)
        cb = fig.colorbar(c, ax=ax, cax=cax, orientation=orient, **cbar_kwargs)
        cb = fp.format_cbar(cb, cbar_title, horizontal=horiz, **cbar_t_kwargs)
        return fig, ax, cb
    else:
        return fig, ax, c

def plot_state_grid(data, rows, cols, clusters_plot,
                    cbar=True, **kw):
    assert rows*cols == data.shape[1], \
           'Dimension mismatch: Data does not match number of plots.'

    try:
        kw.get('vmin')
        kw.get('vmax')
    except KeyError:
        print('vmin and vmax not supplied. Plots may have inconsistent\
               colorbars.')

    try:
        titles = kw.pop('titles')
        vmins = kw.pop('vmins')
        vmaxs = kw.pop('vmaxs')
    except KeyError:
        pass

    fig_kwargs = kw.pop('fig_kwargs', {})
    cbar_kwargs = kw.pop('cbar_kwargs', {})
    fig, ax = fp.get_figax(rows, cols, maps=True,
                           lats=clusters_plot.lat, lons=clusters_plot.lon,
                            **fig_kwargs)

    for i, axis in enumerate(ax.flatten()):
        kw['fig_kwargs'] = {'figax' : [fig, axis]}
        try:
            kw['title'] = titles[i]
            kw['vmin'] = vmins[i]
            kw['vmax'] = vmaxs[i]
        except NameError:
            pass

        fig, axis, c = plot_state(data[:,i], clusters_plot,
                                  cbar=False, **kw)

    if cbar:
        cax = fp.add_cax(fig, ax)
        cbar_title = cbar_kwargs.pop('title', '')
        c = fig.colorbar(c, cax=cax, **cbar_kwargs)
        c = fp.format_cbar(c, cbar_title)

    return fig, ax, c

def plot_posterior(xhat, dofs, clusters, **kwargs):
    # config.SCALE = 1

    # Get figure
    fig, ax = fp.get_figax(rows=1, cols=2, maps=True,
                           lats=clusters.lat, lons=clusters.lon)
    plt.subplots_adjust(wspace=0.1)
    old_scale = copy.deepcopy(config.SCALE)
    config.SCALE = 1

    # Define kwargs
    sf_cmap_1 = plt.cm.PuOr_r(np.linspace(0.2, 0.5, 256))
    sf_cmap_2 = plt.cm.PuOr_r(np.linspace(0.5, 1, 256))
    sf_cmap = np.vstack((sf_cmap_1, sf_cmap_2))
    sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
    div_norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=1, vmax=3)

    small_map_kwargs = {'draw_labels' : False}
    xhat_cbar_kwargs = {'title' : r'Scale factor', 'horizontal' : True , 
                        'y' : -3}
    xhat_kwargs = {'cmap' : sf_cmap, 'norm' : div_norm,
                   'default_value' : 1, 'cbar_kwargs' : xhat_cbar_kwargs,
                   'title_kwargs' : {'fontsize' : config.TITLE_FONTSIZE},
                   'map_kwargs' : small_map_kwargs,
                   'fig_kwargs' : {'figax' : [fig, ax[0]]},}
    avker_cbar_kwargs = {'title' : r'$\partial\hat{x}_i/\partial x_i$',
                         'horizontal' : True, 'y' : -3}
    avker_kwargs = {'cmap' : fp.cmap_trans('plasma'), 'vmin' : 0, 'vmax' : 1,
                   'title_kwargs' : {'fontsize' : config.TITLE_FONTSIZE},
                    'cbar_kwargs' : avker_cbar_kwargs,
                    'map_kwargs' : small_map_kwargs,
                    'fig_kwargs' : {'figax' : [fig, ax[1]]}}


    # Plot xhat
    fig, ax[0], c = plot_state(xhat, clusters, 
                               title='Posterior emission scale factors',
                               **xhat_kwargs)

    # Plot dofs
    fig, ax[1], c = plot_state(dofs, clusters,
                               title='Averaging kernel sensitivities',
                              **avker_kwargs)
    ax[1].text(0.025, 0.05, 'DOFS = %d' % round(dofs.sum()),
               fontsize=config.LABEL_FONTSIZE,
               transform=ax[1].transAxes)


    # Reset config SCALE
    config.SCALE = old_scale

    return fig, ax


