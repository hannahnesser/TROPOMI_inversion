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
def save_obj(obj, name, big_mem=False):
    '''
    This is a generic function to save a data object using
    pickle, which reduces the memory requirements.
    '''
    if big_mem:
        h5f = h5py.File(name, 'w')
        h5f.create_dataset('data', data=obj)
        h5f.close()
    else:
        with open(name , 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, big_mem=False):
    '''
    This is a generic function to open a data object using
    pickle, which reduces the memory requirements.
    '''
    if big_mem:
        h5f = h5py.File(name, 'r')
        d = h5f['data']
        x = da.from_array(d)
        return x
    else:
        with open( name, 'rb') as f:
            return pickle.load(f)

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
    return m, b, r, bias

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
## GEOS-Chem output-processing functions
## -------------------------------------------------------------------------##
def load_files(*files, **kwargs):
    '''
    A function that will load one or more files
    '''
    if len(*files) > 1:
        data = xr.open_mfdataset(*files)
    else:
        data = xr.open_dataset(files[0])

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

def define_HEMCO_var_attributes(data, var, long_name, units):
    data[var].attrs = {'long_name' : long_name, 'units' : units}
    return data

def save_HEMCO_netcdf(data, data_dir, file_name):
    encoding = {'_FillValue' : None, 'dtype' : 'float32'}
    var = {k : encoding for k in data.keys()}
    coord = {k : encoding for k in data.coords}
    var.update(coord)
    data.to_netcdf(join(data_dir, file_name), encoding=var)

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
        _, _, r, bias = comparison_stats(xdata, ydata)
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
        _, _, r, bias = comparison_stats(xdata, ydata)
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
