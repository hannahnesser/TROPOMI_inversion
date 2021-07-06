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
    1D to 2D -- eventually rename to clusters_1d_to_2d
    '''
    result = clusters.copy()
    c_array = result.values
    c_idx = np.where(c_array > 0)
    c_val = c_array[c_idx]
    row_idx = [r for _, r, _ in sorted(zip(c_val, c_idx[0], c_idx[1]))]
    col_idx = [c for _, _, c in sorted(zip(c_val, c_idx[0], c_idx[1]))]
    idx = (row_idx, col_idx)

    d_idx = np.where(c_array == 0)

    c_array[c_idx] = data
    c_array[d_idx] = default_value
    result.values = c_array

    return result

def clusters_2d_to_1d(clusters, data):

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

## -------------------------------------------------------------------------##
## Plotting functions : state vectors
## -------------------------------------------------------------------------##
def plot_state(data, clusters_plot, default_value=0, cbar=True, **kw):
    # Match the data to lat/lon data
    data = match_data_to_clusters(data, clusters_plot, default_value)

    # Plot
    fig, ax, c = plot_state_format(data, default_value, cbar, **kw)
    return fig, ax, c

def plot_state_format(data, default_value=0, cbar=True, **kw):
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
