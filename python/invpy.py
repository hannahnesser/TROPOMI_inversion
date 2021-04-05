'''
Generic inversion functions (this is integrated with the inversion class).
'''

import numpy as np
import pandas as pd
import xarray as xr
import pickle
import math
from scipy.stats import linregress

from os.path import join
from os import listdir

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

def clusters_2d_to_1d(clusters, data):
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
    kw['vmin'] = kw.get('vmin', data.min())
    kw['vmax'] = kw.get('vmax', data.max())
    kw['add_colorbar'] = False
    cbar_kwargs = kw.pop('cbar_kwargs', {})
    label_kwargs = kw.pop('label_kwargs', {})
    title_kwargs = kw.pop('title_kwargs', {})
    map_kwargs = kw.pop('map_kwargs', {})
    fig_kwargs = kw.pop('fig_kwargs', {})

    # Get figure
    lat_range = [data.lat.min(), data.lat.max()]
    lon_range = [data.lon.min(), data.lon.max()]
    fig, ax  = fp.get_figax(maps=True, lats=lat_range, lons=lon_range,
                            **fig_kwargs)

    # Plot data
    c = data.plot(ax=ax, snap=True, **kw)

    # Set limits
    ax.set_xlim(lon_range)
    ax.set_ylim(lat_range)

    # Add title and format map
    ax = fp.add_title(ax, title, **title_kwargs)
    ax = fp.format_map(ax, data.lat, data.lon, **map_kwargs)

    if cbar:
        cbar_title = cbar_kwargs.pop('title', '')
        cax = fp.add_cax(fig, ax)
        cb = fig.colorbar(c, ax=ax, cax=cax, **cbar_kwargs)
        cb = fp.format_cbar(cb, cbar_title)
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
    fig, ax = fp.get_figax(rows, cols, maps=True,
                           lats=clusters_plot.lat, lons=clusters_plot.lon,
                            **fig_kwargs)

    if cbar:
        cax = fp.add_cax(fig, ax)
        cbar_kwargs = kw.pop('cbar_kwargs', {})

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
        cbar_title = cbar_kwargs.pop('title', '')
        c = fig.colorbar(c, cax=cax, **cbar_kwargs)
        c = fp.format_cbar(c, cbar_title)

    return fig, ax, c
