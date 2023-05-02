import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 15)

base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python'
data_dir = base_dir + 'inversion_data'
output_dir = base_dir + 'inversion_data'
plot_dir = base_dir + 'plots'

# Import Custom packages
sys.path.append(code_dir)
import gcpy as gc
import format_plots as fp

# Load data
data = gc.load_obj(f'{output_dir}/observations/2019_corrected.pkl')
so_lb = xr.open_dataarray(f'{data_dir}/observations/so_lb.nc')
so_mb = xr.open_dataarray(f'{data_dir}/observations/so_mb.nc')

# Append error to data dataframe
data['SO_LB'] = so_lb
data['SO_MB'] = so_mb
data['STD_LB'] = so_lb**0.5
data['STD_MB'] = so_mb**0.5

# Subset
data = data[['iGC', 'jGC', 'MONTH', 'SEASON',
             'LON', 'LAT', 'LON_CENTER', 'LAT_CENTER',
             'OBS', 'MOD_MB', 'MOD_LB', 'DIFF_MB', 'DIFF_LB',
             'ALBEDO_SWIR', 'BLENDED_ALBEDO', 'SO_MB', 'SO_LB',
             'STD_MB', 'STD_LB']]

# Now plot the histograms
hist_bins = np.arange(0, 26, 0.5)

for mod_suffix in ['_MB', '_LB']:
    # Standard
    fig, ax = fp.get_figax(aspect=1.75)
    data[f'STD{mod_suffix}'].plot(ax=ax, kind='density', ind=100, 
                                  color=fp.color(4))
    ax.set_xlim(0, 25)
    ax = fp.add_labels(ax, 'Observational Error (ppb)', 'Count')
    ax = fp.add_title(ax, 'Observational Error')
    fp.save_fig(fig, plot_dir, f'observational_error{mod_suffix}')

    # SEASONAL
    fig, ax = fp.get_figax(aspect=1.75)
    ax.set_xlim(0, 25)
    ax = fp.add_labels(ax, 'Observational Error (ppb)', 'Count')
    ax = fp.add_title(ax, 'Observational Error')
    for i, season in enumerate(np.unique(data['SEASON'])):
        hist_data = data[data['SEASON'] == season][f'STD{mod_suffix}']
        # ax.hist(hist_data, histtype='step', bins=hist_bins, label=season,
        #         color=fp.color(2+2*i), lw=1)
        hist_data.plot(ax=ax, kind='density', ind=100,
                       color=fp.color(2+2*i), lw=1, label=season)
        ax.axvline(hist_data.mean(), color=fp.color(2+2*i), lw=1, ls=':')
    ax = fp.add_legend(ax)
    fp.save_fig(fig, plot_dir, f'observational_error_seasonal_hist{mod_suffix}')

    # LATITUDE
    fig, ax = fp.get_figax(aspect=1.75)
    ax.set_xlim(0, 25)
    ax = fp.add_labels(ax, 'Observational Error (ppb)', 'Count')
    ax = fp.add_title(ax, 'Observational Error')
    data['LAT_BIN_HIST'] = pd.cut(data['LAT'], np.arange(10, 70, 10))
    for i, lat_bin in enumerate(np.unique(data['LAT_BIN_HIST'])):
        hist_data = data[data['LAT_BIN_HIST'] == lat_bin][f'STD{mod_suffix}']
        hist_data.plot(ax=ax, kind='density', ind=100,
                       color=fp.color(2*i), lw=1, label=lat_bin)
        ax.axvline(hist_data.mean(), color=fp.color(2*i), lw=1, ls=':')

    ax = fp.add_legend(ax)
    fp.save_fig(fig, plot_dir, f'observational_error_latitude_hist{mod_suffix}')

    # ALBEDO
    fig, ax = fp.get_figax(aspect=1.75)
    ax.set_xlim(0, 25)
    ax = fp.add_labels(ax, 'Observational Error (ppb)', 'Count')
    ax = fp.add_title(ax, 'Observational Error')
    data['ALBEDO_BIN_HIST'] = pd.cut(data['ALBEDO_SWIR'],
                                     np.arange(0, 1.25, 0.25))
    for i, alb_bin in enumerate(np.unique(data['ALBEDO_BIN_HIST'])):
        hist_data = data[data['ALBEDO_BIN_HIST'] == alb_bin][f'STD{mod_suffix}']
        hist_data.plot(ax=ax, kind='density', ind=100,
                       color=fp.color(2*i), lw=1, label=alb_bin)
        ax.axvline(hist_data.mean(), color=fp.color(2*i), lw=1, ls=':')

    ax = fp.add_legend(ax)
    fp.save_fig(fig, plot_dir, f'observational_error_albedo_hist{mod_suffix}')