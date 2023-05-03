from os.path import join
from os import listdir
import sys
import glob
import copy
import math
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Patch as patch
pd.set_option('display.max_columns', 10)

# Custom packages
sys.path.append('.')
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
import gcpy as gc
import troppy as tp
import invpy as ip
import format_plots as fp
import inversion_settings as s

## ------------------------------------------------------------------------ ##
## Directories
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'

## ------------------------------------------------------------------------ ##
## Set plotting preferences
## ------------------------------------------------------------------------ ##
# Small (i.e. non-default) figure settings
small_map_kwargs = {'draw_labels' : False}
small_fig_kwargs = {'max_width' : 4,
                    'max_height' : 3.5}

def heat_map(x, y, data, fig, ax, cmap, n_cmap, vmin, vmax):
    scale = 0.8
    hm_cmap = plt.cm.get_cmap(cmap, n_cmap)
    c = ax.imshow(data, cmap=hm_cmap, vmin=vmin, vmax=vmax)
    for k in range(len(x)):
        for l in range(len(y)):
            text = ax.text(k, l, f'{data[l, k]}', 
                           ha='center', va='center', color='w', 
                           fontsize=config.TICK_FONTSIZE*scale)
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, fontsize=config.TICK_FONTSIZE*scale)
    ax.set_xlabel('Prior error', fontsize=config.TICK_FONTSIZE*scale)
    ax.set_yticks(np.arange(len(y)))
    ax.set_yticklabels(y, fontsize=config.TICK_FONTSIZE*scale)
    ax.set_ylabel('Regularization factor', fontsize=config.TICK_FONTSIZE*scale)
    cax = fp.add_cax(fig, ax, cbar_pad_inches=0.1)
    cb = fig.colorbar(c, cax=cax)
    cb.ax.tick_params(labelsize=config.TICK_FONTSIZE*scale)
    return fig, ax

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# DOFS_filter
DOFS_filter = 0.05

# Define file names
f = '_bc_mb'
# f = '_bc_lb'
# f = '_mb'
# f = '_lb'

# Define rfs, sa values, and DOFS thresholds
# rfs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0]
rfs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1.0]
sas = [0.5, 0.75, 1.0]
dts = [0.05, 0.1]

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
jas = np.load(f'{data_dir}regularization_factor/ja_fr2{f}.npy')
negs = np.load(f'{data_dir}regularization_factor/negs2{f}.npy')
avgs = np.load(f'{data_dir}regularization_factor/avg2{f}.npy')
nfs = np.load(f'{data_dir}regularization_factor/n_func2{f}.npy')

## ------------------------------------------------------------------------ ##
## Regularization factor
## ------------------------------------------------------------------------ ##
for i, dt in enumerate(dts):
    fig, ax = fp.get_figax(rows=1, cols=3, aspect=len(sas)/len(rfs)*2)
    plt.subplots_adjust(hspace=0.5)

    # Plot ja
    ja = jas[:, :, i]
    ja = np.around(ja, 1)
    fig, ax[0] = heat_map(sas, rfs, ja, fig, ax[0], 'viridis', 8, 0, 2)
    fp.add_title(ax[0], 'Ja/n', fontsize=config.SUBTITLE_FONTSIZE*0.8)

    # Plot negs
    neg = negs[:, :, i]
    neg = neg.astype(int)
    nf = nfs[:, :, i]
    nf = nf.astype(int)
    neg_frac = np.around(neg/nf*100, 1)
    fig, ax[1] = heat_map(sas, rfs, neg_frac, fig, ax[1], 
                          'plasma', 20, -0.5, 100)
    fp.add_title(ax[1], '%% functional\nnegative values', 
                 fontsize=config.SUBTITLE_FONTSIZE*0.8)

    # Plot functional ns
    fig, ax[2] = heat_map(sas, rfs, nf, fig, ax[2], 
                          'cividis', 100, -0.5, 6e3)
    fp.add_title(ax[2], r'Functional n', 
                 fontsize=config.SUBTITLE_FONTSIZE*0.8)

    # Save
    fp.save_fig(fig, plot_dir, f'fig_rfs_sas_dt{dt}{f}')
