'''
This is a package designed to deal with common GEOS-Chem
processing needs. It also contains several generic functions that
are used in the course of working with GEOS-Chem data.
'''
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import math
from scipy.stats import linregress
import h5py
import dask.array as da

import shapefile
from shapely.geometry import Polygon, MultiPolygon

from os.path import join
from os import listdir
import os
import warnings
import datetime

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
import inversion_settings as s

# Other font details
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'AppleGothic'
rcParams['font.size'] = config.LABEL_FONTSIZE*config.SCALE
rcParams['text.usetex'] = True
# rcParams['mathtext.fontset'] = 'stixsans'
rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'
rcParams['axes.titlepad'] = 0

## -------------------------------------------------------------------------##
## Loading functions
## -------------------------------------------------------------------------##
def file_exists(file_name):
    '''
    Check for the existence of a file
    '''
    data_dir = file_name.rpartition('/')[0]
    if file_name.split('/')[-1] in listdir(data_dir):
        return True
    else:
        print(f'{file_name} is not in the data directory.')
        return False

def save_obj(obj, name):
        with open(name , 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name):
    # Open a generic file using pickle
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def read_file(*file_names, **kwargs):
    file_suffix = file_names[0].split('.')[-1]
    # Require that the file exists
    for f in file_names:
        # Require that the files exist and that all files are the
        # same type
        assert file_exists(f), f'{f} does not exist.'
        assert f.split('.')[-1] == file_suffix, \
               'Variable file types provided.'

        # If multiple files are provided, require that they are netcdfs
        if len(file_names) > 1:
            assert file_suffix[:2] == 'nc', \
                   'Multiple files are provided that are not netcdfs.'

    # If a netcdf, read it using xarray
    if file_suffix[:2] == 'nc':
        file = read_netcdf_file(*file_names, **kwargs)
    # Else, read it using a generic function
    else:
        if 'chunks' in kwargs:
            warnings.warn('NOTE: Chunk sizes were provided, but the file is not a netcdf. Chunk size is ignored.', stacklevel=2)
        file = read_generic_file(*file_names, **kwargs)

    return file

def read_generic_file(file_name):
    # Open a generic file using pickle
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def read_netcdf_file(*file_names, **kwargs):
    # Open a dataset
    if len(file_names) > 1:
        # Currently assumes that we are stacking the files
        # vertically
        if 'concat_dim' in kwargs:
            if len(kwargs['concat_dim']) > 1:
                file_names = [[f] for f in file_names]
        data = xr.open_mfdataset(file_names, **kwargs)
    else:
        if 'dims' in kwargs:
            del kwargs['dims']
        data = xr.open_dataset(file_names[0], **kwargs)

    # If there is only one variable, convert to a dataarray
    variables = list(data.keys())
    if len(variables) == 1:
        data = data[variables[0]]

    # Return the file
    return data

def calculate_chunk_size(available_memory_GB, n_threads=None,
                         dtype='float32'):
    '''
    This function returns a number that gives the total number of
    elements that should be held in a chunk. It does not specify the exact
    chunks for specific dimensions.
    '''

    # Get the number of active threads
    if n_threads is None:
        n_threads = int(os.environ['OMP_NUM_THREADS'])

    # Approximate the number of chunks that are held in memory simultaneously
    # by dask (reference: https://docs.dask.org/en/latest/array-best-practices.html#:~:text=Orient%20your%20chunks,-When%20reading%20data&text=If%20your%20Dask%20array%20chunks,closer%20to%201MB%20than%20100MB.)
    chunks_in_memory = 20*n_threads

    # Calculate the memory that is available per chunk (in GB)
    mem_per_chunk = available_memory_GB/chunks_in_memory

    # Define the number of bytes required for each element
    if dtype == 'float32':
        bytes_per_element = 4
    elif dtype == 'float64':
        bytes_per_element = 8
    else:
        print('Data type is not recognized. Defaulting to reserving 8 bytes')
        print('per element.')
        bytes_per_element = 8

    # Calculate the number of elements that can be held in the available
    # memory for each chunk
    number_of_elements = mem_per_chunk*1e9/bytes_per_element

    # Scale the number of elements down by 10% to allow for wiggle room.
    return int(0.9*number_of_elements)

## -------------------------------------------------------------------------##
## Statistics functions
## -------------------------------------------------------------------------##
def rmse(diff):
    return np.sqrt(np.mean(diff**2))

def add_quad(data):
    return np.sqrt((data**2).sum())

def group_data(data, groupby, quantity='DIFF',
                stats=['count', 'mean', 'std', rmse]):
    return data.groupby(groupby).agg(stats)[quantity].reset_index()

def comparison_stats(xdata, ydata):
    m, b, r, p, err = linregress(xdata.flatten(), ydata.flatten())
    bias = (ydata - xdata).mean()
    std = (ydata - xdata).std()
    return m, b, r, bias, std

## -------------------------------------------------------------------------##
## Grid functions
## -------------------------------------------------------------------------##
def adjust_grid_bounds(lat_min, lat_max, lat_delta,
                       lon_min, lon_max, lon_delta,
                       buffer=[0, 0, 0, 0]):
    '''
    This function adjusts the default GEOS-Chem grid bounds,
    which are given as grid box centers, to grid box edges.
    It also allows for the option to remove buffer grid cells,
    although the default is to assume 0 buffer grid cells.
    (Buffer grid cells should be expressed in the standard
    GEOS-Chem convention of [N S E W].)
    '''
    lat_min = lat_min + lat_delta*buffer[1] - lat_delta/2
    lat_max = lat_max - lat_delta*buffer[0] + lat_delta/2
    lon_min = lon_min + lon_delta*buffer[3] - lon_delta/2
    lon_max = lon_max - lon_delta*buffer[2] + lon_delta/2
    return [lat_min, lat_max], [lon_min, lon_max]

def subset_data_latlon(data, lat_min, lat_max, lon_min, lon_max):
    '''
    This function subsets a given dataset (in xarray form, with
    latitude and longitude variables lat and lon, respectively)
    to a given lat lon grid.
    '''
    data = data.where((data.lat > lat_min) & (data.lat < lat_max) &
                      (data.lon > lon_min) & (data.lon < lon_max),
                      drop=True)
    return data

def create_gc_grid(lat_min, lat_max, lat_delta,
                   lon_min, lon_max, lon_delta,
                   centers=True, return_xarray=True):
    '''
    This function creates a grid with values corresponding to the
    centers of each grid cell. The latitude and longitude limits
    provided correspond to grid cell centers if centers=True and
    edges otherwise.
    '''
    if not centers:
        lat_min += lat_delta/2
        lat_max -= lat_delta/2
        lon_min += lon_delta/2
        lon_max -= lon_delta/2

    lats = np.arange(lat_min, lat_max + lat_delta, lat_delta)
    lons = np.arange(lon_min, lon_max + lon_delta, lon_delta)

    if return_xarray:
        data = xr.DataArray(np.zeros((len(lats), len(lons))),
                            coords=[lats, lons],
                            dims=['lats', 'lons'])
    else:
        data = [lats, lons]

    return data

def nearest_loc(data, compare_data):
    indices = np.abs(compare_data.reshape(-1, 1) -
                     data.reshape(1, -1)).argmin(axis=0)
    return indices

def grid_shape_overlap(clusters, x, y, name=None, plot_dir='../plots'):
    # Initialize mask
    mask = np.zeros(int(clusters.values.max()))

    # # Get edges of the shape
    # x = [i[0] for i in shape.shape.points[:]]
    # y = [i[1] for i in shape.shape.points[:]]

    # Make a polygon
    c_poly = Polygon(np.column_stack((x, y)))
    if not c_poly.is_valid:
        print(f'Buffering {name}')
        c_poly = c_poly.buffer(0)
        # fig, ax = fp.get_figax(cols=2, maps=True, 
        #                        lats=clusters.lat, lons=clusters.lon)
        # ax[0].plot(x, y)
        # try:
        #     ax[1].plot(*c_poly.exterior.xy)
        # except:
        #     for c_geom in c_poly.geoms:
        #         ax[1].plot(*c_geom.exterior.xy)
        # fp.save_fig(fig, plot_dir, f'buffer_{shape.record[0]}')
        # plt.close()

    # Get maximum latitude and longitude limits
    lat_lims = (np.min(y), np.max(y))
    lon_lims = (np.min(x), np.max(x))

    # Convert that to the GC grid (admittedly using grid cell centers 
    # instead of edges, but that should be consesrvative)
    c_lat_lims = (clusters.lat.values[clusters.lat < lat_lims[0]][-1],
                  clusters.lat.values[clusters.lat > lat_lims[1]][0])
    c_lon_lims = (clusters.lon.values[clusters.lon < lon_lims[0]][-1],
                  clusters.lon.values[clusters.lon > lon_lims[1]][0])
    c_clusters = clusters.sel(lat=slice(*c_lat_lims), 
                              lon=slice(*c_lon_lims))
    c_cluster_list = c_clusters.values.flatten()
    c_cluster_list = c_cluster_list[c_cluster_list > 0]

    # Iterate through overlapping grid cells
    for i, gc in enumerate(c_cluster_list):
        # Get center of grid box
        gc_center = c_clusters.where(c_clusters == gc, drop=True)
        gc_center = (gc_center.lon.values[0], gc_center.lat.values[0])
        
        # Get corners
        gc_corners_lon = [gc_center[0] - s.lon_delta/2,
                          gc_center[0] + s.lon_delta/2,
                          gc_center[0] + s.lon_delta/2,
                          gc_center[0] - s.lon_delta/2]
        gc_corners_lat = [gc_center[1] - s.lat_delta/2,
                          gc_center[1] - s.lat_delta/2,
                          gc_center[1] + s.lat_delta/2,
                          gc_center[1] + s.lat_delta/2]

        # Make polygon
        gc_poly = Polygon(np.column_stack((gc_corners_lon, gc_corners_lat)))

        if gc_poly.intersects(c_poly):
            # Get area of overlap area and GC cell and calculate
            # the fractional contribution of the overlap area
            overlap_area = c_poly.intersection(gc_poly).area
            gc_area = gc_poly.area
            mask[int(gc) - 1] = overlap_area/gc_area

    return mask

## -------------------------------------------------------------------------##
## GEOS-Chem correction functions
## -------------------------------------------------------------------------##
def fill_GC_first_hour(data):
    '''
    This function fills in the first hour of GEOS-Chem output data
    because of GEOS-Chem's failure to output data during the first time
    segment of every simulation. What an odd bug.
    '''
    data0 = data.where(data.time == data.time[0], drop=True)

    # Change the time of that data to the 0th hour
    t = pd.Timestamp(data0.time.values[0]).replace(hour=0).to_datetime64()
    data0.time.values[0] = t

    # Merge the datasets
    data = xr.concat([data0, data], dim='time')

    return data

## -------------------------------------------------------------------------##
## HEMCO input functions
## -------------------------------------------------------------------------##
def define_HEMCO_std_attributes(data, name=None):
    '''
    This function defines the attributes for time, lat, and lon,
    the standard GEOS-Chem dimensions. It currently doesn't have the
    capacity to define level attributes.
    '''
    print('Remember to define the following attributes for non-standard')
    print('variables:')
    print(' - title (global)')
    print(' - long_name')
    print(' - units')

    # Check if time is in the dataset and, if not, add it
    if 'time' not in data.coords:
        data = data.assign_coords(time=0)
        data = data.expand_dims('time')

    # Convert to dataset
    if type(data) != xr.core.dataset.Dataset:
        assert name is not None, 'Name is not provided for dataset.'
        data = data.to_dataset(name=name)

    # Set time, lat, and lon attributes
    data.time.attrs = {'long_name' : 'Time',
                       'units' : 'hours since 2009-01-01 00:00:00',
                       'calendar' : 'standard'}
    data.lat.attrs = {'long_name': 'latitude', 'units': 'degrees_north'}
    data.lon.attrs = {'long_name': 'longitude', 'units': 'degrees_east'}
    return data

def define_HEMCO_var_attributes(data, var, long_name, units, **kwargs):
    data[var].attrs = {'long_name' : long_name, 'units' : units,
                       **kwargs}
    return data

def save_HEMCO_netcdf(data, data_dir, file_name, dtype='float32', **kwargs):
    encoding = {'_FillValue' : None, 'dtype' : dtype}
    var = {k : encoding for k in data.keys()}
    coord = {k : encoding for k in data.coords}
    var.update(coord)
    data.to_netcdf(join(data_dir, file_name), encoding=var,
                   unlimited_dims=['time'], **kwargs)

## -------------------------------------------------------------------------##
## Planeflight functions
## -------------------------------------------------------------------------##
STD_PF_COLS = ['POINT', 'TYPE', 'YYYYMMDD', 'HHMM', 'LAT', 'LON', 'PRESS',
               'OBS', 'T-IND', 'P-I', 'I-IND', 'J-IND', 'TRA_001']

def load_pf(file, pf_cols=STD_PF_COLS):
    data = pd.read_csv(file, usecols=pf_cols,
                       sep='[\s]{1,20}',
                       engine='python')
    return data

def load_all_pf(files, data_dir, pf_cols=STD_PF_COLS):
    tot = pd.DataFrame(columns=pf_cols)
    for f in np.sort(files):
        data = load_pf(join(data_dir, f), pf_cols)
        tot = pd.concat([tot, data], sort=False)
    return tot

def process_pf(pf_df):
    # Rename
    pf_df = pf_df.rename(columns={'TRA_001' : 'MOD'})

    # Get rid of non physical points
    pf_df = pf_df[pf_df['MOD'] > 0]

    # Adjust units
    pf_df['MOD'] *= 1e9

    # Set data types
    for col in ['POINT', 'P-I', 'I-IND', 'J-IND']:
        pf_df[col] = pf_df[col].astype(int)

    # Throw out data without pressure measurements
    pf_df = pf_df.loc[pf_df['PRESS'] > 0]

    # Add a difference column
    pf_df['DIFF'] = pf_df['MOD'] - pf_df['OBS']

    return pf_df

def group_by_gridbox(pf_df):
    pf_gr = pf_df.groupby(['P-I', 'I-IND', 'J-IND']).mean()
    pf_gr = pf_gr[['OBS', 'MOD']].reset_index()
    return pf_gr

## -------------------------------------------------------------------------##
## Plotting functions : comparison
## -------------------------------------------------------------------------##
def add_stats_text(ax, r, bias):
    if r**2 <= 0.99:
        ax.text(0.05, 0.9, r'R = %.2f' % r,
                fontsize=config.LABEL_FONTSIZE*config.SCALE,
                transform=ax.transAxes)
    else:
        ax.text(0.05, 0.9, r'R $>$ 0.99',
                fontsize=config.LABEL_FONTSIZE*fp.SCALE,
                transform=ax.transAxes)
    ax.text(0.05, 0.875, 'Bias = %.2f' % bias,
            fontsize=config.LABEL_FONTSIZE*config.SCALE,
            transform=ax.transAxes,
            va='top')
    return ax

def plot_comparison_hexbin(xdata, ydata, cbar, stats, **kw):
    cbar_kwargs = kw.pop('cbar_kwargs', {})
    fig_kwargs = kw.pop('fig_kwargs', {})
    lims = kw.pop('lims', None)
    fig, ax = fp.get_figax(**fig_kwargs)
    ax.set_aspect('equal')

    # Get data limits
    xlim, ylim, xy, dmin, dmax = fp.get_square_limits(xdata, ydata, lims=lims)

    # Set bins and gridsize for hexbin
    if ('vmin' not in kw) or ('vmax' not in kw):
        bin_max = len(xdata)*0.1
        round_by = len(str(len(xdata)/10).split('.')[0]) - 1
        bin_max = 1+int(round(bin_max, -round_by))
        kw['bins'] = np.arange(0, bin_max)
    kw['gridsize'] = math.floor((dmax - dmin)/(xy[1] - xy[0])*40)

    # Plot hexbin
    c = ax.hexbin(xdata, ydata, cmap=fp.cmap_trans('plasma_r'), **kw)

    # Aesthetics
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Print information about R2 on the plot
    if stats:
        _, _, r, bias, _ = comparison_stats(xdata, ydata)
        ax = add_stats_text(ax, r, bias)

    if cbar:
        cbar_title = cbar_kwargs.pop('title', '')
        cax = fp.add_cax(fig, ax)
        cbar = fig.colorbar(c, cax=cax, **cbar_kwargs)
        cbar = fp.format_cbar(cbar, cbar_title)
        return fig, ax, cbar
    else:
        return fig, ax, c

def plot_comparison_scatter(xdata, ydata, stats, **kw):
    fig_kwargs = kw.pop('fig_kwargs', {})
    lims = kw.pop('lims', None)

    fig, ax = fp.get_figax(**fig_kwargs)
    ax.set_aspect('equal')

    # Get data limits
    xlim, ylim, xy, dmin, dmax = fp.get_square_limits(xdata, ydata, lims=lims)

    # Plot hexbin
    kw['color'] = kw.pop('color', fp.color(4))
    kw['s'] = kw.pop('s', 3)
    c = ax.scatter(xdata, ydata, **kw)

    # Aesthetics
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Print information about R2 on the plot
    if stats:
        _, _, r, bias, _ = comparison_stats(xdata, ydata)
        ax = add_stats_text(ax, r, bias)

    return fig, ax, c

def plot_comparison_dict(xdata, ydata, **kw):
    fig_kwargs = kw.pop('fig_kwargs', {})
    fig, ax = fp.get_figax(**fig_kwargs)
    ax.set_aspect('equal')

    # We need to know how many data sets were passed
    n = len(ydata)
    cmap = kw.pop('cmap', 'inferno')

    # Plot data
    count = 0
    for k, ydata in ydata.items():
        ax.scatter(xdata, ydata, alpha=0.5, s=5*fp.SCALE,
                   c=color(count, cmap=cmap, lut=n))
        count += 1

    # Color bar (always True)
    cax = fp.add_cax(fig, ax)
    cbar_ticklabels = kw.pop('cbar_ticklabels', list(ydata.keys()))
    norm = colors.Normalize(vmin=0, vmax=n)
    cbar = colorbar.ColorbarBase(cax, cmap=plt.cm.get_cmap(cmap, lut=n),
                                 norm=norm)
    cbar.set_ticks(0.5 + np.arange(0,n+1))
    cbar.set_ticklabels(cbar_ticklabels)
    cbar = format_cbar(cbar)

    return fig, ax, cbar

def plot_comparison(xdata, ydata, cbar=True, stats=True, hexbin=True, **kw):
    # Get other plot labels
    xlabel = kw.pop('xlabel', '')
    ylabel = kw.pop('ylabel', '')
    label_kwargs = kw.pop('label_kwargs', {})
    title = kw.pop('title', '')
    title_kwargs = kw.pop('title_kwargs', {})

    if type(ydata) == dict:
        fig, ax, c = plot_comparison_dict(xdata, ydata, **kw)
    elif hexbin:
        fig, ax, c = plot_comparison_hexbin(xdata, ydata, cbar, stats, **kw)
    else:
        fig, ax, c = plot_comparison_scatter(xdata, ydata, stats, **kw)

    # Aesthetics
    ax = fp.plot_one_to_one(ax)
    ax = fp.add_labels(ax, xlabel, ylabel, **label_kwargs)
    ax = fp.add_title(ax, title, **title_kwargs)

    # Make sure we have the same ticks
    # ax.set_yticks(ax.get_xticks(minor=False), minor=False)
    ax.set_xticks(ax.get_yticks(minor=False), minor=False)
    ax.set_xlim(ax.get_ylim())

    return fig, ax, c
