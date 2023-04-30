import pickle
from os.path import join
from os import listdir
import sys
import datetime
import calendar as cal

import xarray as xr
import numpy as np
from numpy.polynomial import polynomial as p
import pandas as pd
from itertools import product

import matplotlib.pyplot as plt

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
# Time
year = 2018
months = [5]
days = np.arange(1, 32, 1)

base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = f'{base_dir}python/'
data_dir = f'{base_dir}observations/plane_logs/'
raw_data_dir = f'{data_dir}raw/ict_files/'
plot_dir = f'{base_dir}plots/'
pf_files = [f'plane.log.{year}{mm:02d}{dd:02d}'
            for mm in months for dd in days]

# Information on the grid
lat_bins = np.arange(10, 65, 5)
lat_min = 9.75
lat_max = 60
lat_delta = 0.25
lon_min = -130
lon_max = -60
lon_delta = 0.3125
buffers = [3, 3, 3, 3]

# albedo bins
albedo_bins = np.arange(0, 1.1, 0.1)

## ------------------------------------------------------------------------ ##
## Import custom packages
## ------------------------------------------------------------------------ ##
sys.path.append(code_dir)
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
import gcpy as gc
import troppy as tp
import format_plots as fp

## -------------------------------------------------------------------------##
## Planeflight analysis
## -------------------------------------------------------------------------##
## ----------------------------------------- ##
## Load the GEOS-Chem planeflight data
## ----------------------------------------- ##
# If the pf files are a list, compile monthly files
if type(pf_files) == list:
    # data = np.array([]).reshape(0, 13)
    for m in pf_months:
        print('Analyzing data for month %d' % m)
        file_names = []
        for d in days:
            # Check if that file is in the data directory
            if file not in listdir(data_dir):
                print('%s is not in the data directory.' % file)
                continue

            file_names.append(file)

        pf = gc.load_all_pf(file_names, data_dir)
        pf = gc.process_pf(pf)
        pf_gr = gc.group_by_gridbox(pf)

## ----------------------------------------- ##
## Load the raw ATom data (for stratosphere)
## ----------------------------------------- ##
raw_files = [join(raw_data_dir, f) for f in listdir(join(raw_data_dir))
             if isfile(join(join(raw_data_dir), f)) & (f[0] != '.')]
raw_files.sort()

cols = ['UTC_Start', 'UTC_Stop', 'G_LONG', 'G_LAT', 'P',
        'CH4_NOAA', 'CO_NOAA', 'O3_CL']

    # NOTE: this data requires stratosphere analysis

# Read in data with troposphere boolean flag
# pf = pd.read_csv(join(data_dir, 'planeflight_total.csv'), index_col=0,
#                  low_memory=False)

# Subset for troposphere
pf = pf[pf['TROP']]

# Group by grid box
pf_gr = gc.group_by_gridbox(pf)

# figsize
figsize = fp.get_figsize(aspect=1.225, rows=2, cols=1)
fig, ax = plt.subplots(figsize=figsize)
ax.set_facecolor('0.98')

# Plot
fig, ax, c = gc.plot_comparison(pf_gr['OBS'].values,
                                pf_gr['MOD'].values,
                                lims=[1750, 2250],
                                xlabel='Observation', ylabel='Model',
                                hexbin=False, stats=False,
                                fig_kwargs={'figax' : [fig, ax]})
_, _, r, bias = gc.comparison_stats(pf['OBS'].values, pf['MOD'].values)
ax = gc.add_stats_text(ax, r, bias)
fp.save_fig(fig, plot_dir, 'planeflight_bias')

# Lat bias
pf['LAT_BIN'] = pd.cut(pf['LAT'], bins=lat_bins)
pf_lat = gc.group_data(pf, groupby=['LAT_BIN'])
pf_lat['LAT'] = pf_lat['LAT_BIN'].apply(lambda x: x.mid)

fig = plt.figure(figsize=figsize)
gs = fig.add_gridspec(2, 1, height_ratios=(3, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0, hspace=0)

# Latitude plot
ax = fig.add_subplot(gs[1])
ax.errorbar(pf_lat['LAT'], pf_lat['mean'], yerr=pf_lat['std'],
            color=fp.color(4))
ax.set_xticks(np.arange(10, 70, 10))
ax = fp.add_labels(ax, 'Latitude', 'Model - Observation')
ax.set_facecolor('0.98')

# Histogram
ax_hist = fig.add_subplot(gs[0], sharex=ax)
ax_hist.hist(pf['LAT'], bins=np.arange(10, 75, 2.5),
             color=fp.color(4))
ax_hist.tick_params(axis='x', labelbottom=False)
ax_hist.tick_params(axis='y', left=False, labelleft=False)
ax_hist.set_ylabel('Count')
ax_hist.set_facecolor('0.98')

fp.save_fig(fig, plot_dir, 'planeflight_latitudinal_bias')

    # # Albedo bias
    # # figsize
    # figsize = fp.get_figsize(aspect=1.225, rows=2, cols=1)
    # fig, ax = plt.subplots(figsize=figsize)
    # ax.set_facecolor('0.98')

    # # Plot
    # fig, ax, c = gc.plot_comparison(pf_gr['OBS'].values,
    #                                 pf_gr['MOD'].values,
    #                                 lims=[1750, 2250],
    #                                 xlabel='Observation', ylabel='Model',
    #                                 hexbin=False, stats=False,
    #                                 fig_kwargs={'figax' : [fig, ax]})
    # _, _, r, bias = gc.comparison_stats(pf['OBS'].values, pf['MOD'].values)
    # ax = gc.add_stats_text(ax, r, bias)
    # fp.save_fig(fig, plot_dir, 'planeflight_bias')

    # # Lat bias
    # pf['LAT_BIN'] = pd.cut(pf['LAT'], bins=lat_bins)
    # pf_lat = gc.group_data(pf, groupby=['LAT_BIN'])
    # pf_lat['LAT'] = pf_lat['LAT_BIN'].apply(lambda x: x.mid)

    # fig = plt.figure(figsize=figsize)
    # gs = fig.add_gridspec(2, 1, height_ratios=(3, 7),
    #                       left=0.1, right=0.9, bottom=0.1, top=0.9,
    #                       wspace=0, hspace=0)

    # # Latitude plot
    # ax = fig.add_subplot(gs[1])
    # ax.errorbar(pf_lat['LAT'], pf_lat['mean'], yerr=pf_lat['std'],
    #             color=fp.color(4))
    # ax.set_xticks(np.arange(10, 70, 10))
    # ax = fp.add_labels(ax, 'Latitude', 'Model - Observation')
    # ax.set_facecolor('0.98')

    # # Histogram
    # ax_hist = fig.add_subplot(gs[0], sharex=ax)
    # ax_hist.hist(pf['LAT'], bins=np.arange(10, 75, 2.5),
    #              color=fp.color(4))
    # ax_hist.tick_params(axis='x', labelbottom=False)
    # ax_hist.tick_params(axis='y', left=False, labelleft=False)
    # ax_hist.set_facecolor('0.98')

    # fp.save_fig(fig, plot_dir, 'planeflight_latitudinal_bias')
