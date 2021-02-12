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
def save_obj(obj, name):
    '''
    This is a generic function to save a data object using
    pickle, which reduces the memory requirements.
    '''
    with open(name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    '''
    This is a generic function to open a data object using
    pickle, which reduces the memory requirements.
    '''
    with open( name, 'rb') as f:
        return pickle.load(f)

## -------------------------------------------------------------------------##
## Statistics functions
## -------------------------------------------------------------------------##
def rmse(diff):
    return np.sqrt(np.mean(diff**2))

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
    return np.abs(compare_data[:, None] - data[None, :]).argmin(axis=0)

## -------------------------------------------------------------------------##
## GEOS-Chem fixing functions
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
## Plotting functions : state vectors
## -------------------------------------------------------------------------##
def plot_state(data, clusters_plot, default_value=0, cbar=True, **kw):
        # Match the data to lat/lon data
    data = match_data_to_clusters(data, clusters_plot, default_value)

    # Plot
    fig, ax, c = plot_state_format(data, default_value, cbar, **kw)
    return fig, ax, c

def match_data_to_clusters(data, clusters, default_value=0):
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
