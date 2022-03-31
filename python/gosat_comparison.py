from os.path import join
from os import listdir
import sys
import copy
import calendar as cal
import xarray as xr
import numpy as np
from numpy.polynomial import polynomial as p
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# from matplotlib import cm
pd.set_option('display.max_columns', 15)

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
local = True

if local:
    # # Local preferences
    base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
    code_dir = base_dir + 'python'
    data_dir = base_dir + 'inversion_data'
    output_dir = base_dir + 'inversion_data'
    plot_dir = base_dir + 'plots'
else:
    # Cannon long-path preferences
    base_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_final/'
    code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
    data_dir = f'{base_dir}ProcessedDir'
    output_dir = '/n/seasasfs02/hnesser/TROPOMI_inversion/inversion_data'

# Cannon preferences
# code_dir = sys.argv[1]
# base_dir = sys.argv[2]
# data_dir = f'{base_dir}ProcessedDir'
# output_dir = sys.argv[3]
# plot_dir = None

# Import Custom packages
sys.path.append(code_dir)
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
import gcpy as gc
import troppy as tp
import format_plots as fp
import inversion_settings as settings

# Compare to GOSAT?
compare_gosat = True
# gosat_dir = '/n/seasasfs02/CH4_inversion/InputData/Obs/ULGOSAT_v9/2019'
# gosat = [f'{gosat_dir}/UoL-GHG-L2-CH4-GOSAT-OCPR-{settings.year}{mm:02d}{dd:02d}-fv9.0.nc'
#          for mm in settings.months for dd in settings.days]
gosat = None



## ------------------------------------------------------------------------ ##
## Get grid information
## ------------------------------------------------------------------------ ##
lats, lons = gc.create_gc_grid(*settings.lats, settings.lat_delta,
                               *settings.lons, settings.lon_delta,
                               centers=False, return_xarray=False)

# lats_l, lons_l = gc.create_gc_grid(*settings.lats, 2, *settings.lons, 2,
                                   # centers=False, return_xarray=False)

## ------------------------------------------------------------------------ ##
## Load TROPOMI and GOSAT data
## ------------------------------------------------------------------------ ##
TROPOMI_RAW = gc.read_file(f'{output_dir}/2019_corrected.pkl')
GOSAT_RAW = gc.read_file(f'{output_dir}/2019_gosat.pkl')

## ------------------------------------------------------------------------ ##
## Try a different blended albedo definition
## ------------------------------------------------------------------------ ##
# TROPOMI_RAW['ALBEDO_NIR'] = (TROPOMI_RAW['BLENDED_ALBEDO'] + 1.13*TROPOMI_RAW['ALBEDO_SWIR'])/2.4
# TROPOMI_RAW['BLENDED_ALBEDO'] = 4.8*TROPOMI_RAW['ALBEDO_NIR'] - 2.26*TROPOMI_RAW['ALBEDO_SWIR']

## ------------------------------------------------------------------------ ##
## Grid the GOSAT and TROPOMI data and take the difference
## ------------------------------------------------------------------------ ##
def to_gridded_array(data, fields=['OBS']):
    data = data.groupby(['LAT_CENTER', 'LON_CENTER', 'DATE']).mean()[fields]
    data = data.to_xarray().rename({'LAT_CENTER' : 'LATS',
                                    'LON_CENTER' : 'LONS'})
    return data

GOSAT_RAW['LAT_CENTER'] = lats[gc.nearest_loc(GOSAT_RAW['LAT'].values, lats)]
GOSAT_RAW['LON_CENTER'] = lons[gc.nearest_loc(GOSAT_RAW['LON'].values, lons)]

# Seasonal GOSAT
GOSAT_S = gc.group_data(GOSAT_RAW, groupby=['MONTH'], quantity=['OBS'])
GOSAT_S['MONTH'] = GOSAT_S['MONTH'].apply(lambda x: f'2019{x:02d}15')
GOSAT_S['MONTH'] = pd.to_datetime(GOSAT_S['MONTH'].values, format='%Y%m%d')
fig, ax = fp.get_figax(aspect=1.75)
ax.scatter(pd.to_datetime(GOSAT_RAW['DATE'].values, format='%Y%m%d'),
           GOSAT_RAW['OBS'], c=fp.color(3), alpha=0.1, s=5)
ax.errorbar(GOSAT_S['MONTH'], GOSAT_S['OBS']['mean'],
            yerr=GOSAT_S['OBS']['std'], c=fp.color(2))
fp.add_labels(ax, 'Day of year', 'XCH4')
fp.add_title(ax, 'Seasonal variability in GOSAT')
fp.save_fig(fig, plot_dir, 'gosat_seasonal')

# Remove seasonal trend
GOSAT_RAW['OBS_DS'] = (GOSAT_RAW['OBS'] -
                       GOSAT_RAW.groupby(GOSAT_RAW['MONTH']).OBS.transform('mean') +
                       GOSAT_RAW['OBS'].mean())

# Grid
GOSAT = to_gridded_array(GOSAT_RAW, fields=['OBS', 'OBS_DS'])

DIFF = to_gridded_array(TROPOMI_RAW, fields=['OBS', 'BLENDED_ALBEDO'])
DIFF = DIFF.rename({'OBS' : 'TROPOMI'})
DIFF['DIFF'] = DIFF['TROPOMI'] - GOSAT['OBS']
DIFF['GOSAT'] = GOSAT['OBS']
DIFF['GOSAT_DS'] = GOSAT['OBS_DS']

## ------------------------------------------------------------------------ ##
## Get seasonal information
## ------------------------------------------------------------------------ ##
DIFF['DATE'] = pd.to_datetime(DIFF['DATE'].values, format='%Y%m%d')
DIFF['MONTH'] = DIFF['DATE'].dt.month
DIFF['SEASON'] = DIFF['DATE'].dt.season

## ------------------------------------------------------------------------ ##
## Convert to a dataframe
## ------------------------------------------------------------------------ ##
DIFF = DIFF.to_dataframe()[['GOSAT', 'GOSAT_DS', 'TROPOMI', 'DIFF',
                            'MONTH', 'SEASON', 'BLENDED_ALBEDO']]
DIFF = DIFF.dropna().reset_index()

## ------------------------------------------------------------------------ ##
## Group by blended albedo bins
## ------------------------------------------------------------------------ ##
ba_bins = np.arange(0, 2.5, 0.1)
DIFF['BLENDED_ALBEDO_BIN'] = pd.cut(DIFF['BLENDED_ALBEDO'], ba_bins)

# Blended albedo
DIFF_G = gc.group_data(DIFF, groupby=['BLENDED_ALBEDO_BIN'],
                       quantity=['TROPOMI', 'GOSAT', 'GOSAT_DS', 'DIFF'])
DIFF_G['BLENDED_ALBEDO'] = DIFF_G['BLENDED_ALBEDO_BIN'].apply(lambda x: x.mid).astype(float)

# Blended albedo and season
DIFF_G_S = gc.group_data(DIFF, groupby=['BLENDED_ALBEDO_BIN', 'SEASON'],
                       quantity=['TROPOMI', 'GOSAT', 'GOSAT_DS', 'DIFF'])
DIFF_G_S['BLENDED_ALBEDO'] = DIFF_G_S['BLENDED_ALBEDO_BIN'].apply(lambda x: x.mid).astype(float)

# Season
DIFF_S = gc.group_data(DIFF, groupby=['MONTH'],
                       quantity=['TROPOMI', 'GOSAT', 'GOSAT_DS', 'DIFF'])

## ------------------------------------------------------------------------ ##
## Plot
## ------------------------------------------------------------------------ ##
# # Seasonal difference
# fig, ax = fp.get_figax(aspect=1.75)
# ax.errorbar(DIFF_S['MONTH'], DIFF_S['DIFF']['mean'],
#             yerr=DIFF_S['DIFF']['std'], c=fp.color(3))
# # fp.add_legend(ax, bbox_to_anchor=(1, 0.5), loc='center left')
# fp.add_labels(ax, 'Month', 'XCH4')
# fp.add_title(ax, 'Seasonal bias in TROPOMI - GOSAT')
# fp.save_fig(fig, plot_dir, 'gosat_tropomi_seasonal')


# # Deseasonalized GOSAT blended albedo
# fig, ax = fp.get_figax(aspect=1.75)
# ll = ['--', ':', '-.', '-']
# ax.errorbar(DIFF_G['BLENDED_ALBEDO']-0.005, DIFF_G['GOSAT_DS']['mean'],
#             yerr=DIFF_G['GOSAT_DS']['std'], c='black', ls='--', lw=2,
#             label='Annual Average')
# for i, s in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
#     d_g = DIFF_G_S[DIFF_G_S['SEASON'] == s]
#     # ax.plot(d_g['BLENDED_ALBEDO'], d_g['DIFF']['mean'],
#     #         c=fp.color(3), ls=ll[i], lw=0.5, label=s)
#     ax.errorbar(d_g['BLENDED_ALBEDO']+0.005*i, d_g['GOSAT_DS']['mean'],
#                 yerr=d_g['GOSAT']['std'], c=fp.color(2*i), ls='-', lw=1,
#                 label=s)
#     # ax.plot(d_g['BLENDED_ALBEDO'], d_g['TROPOMI']['mean'],
#     #         c=fp.color(5), ls=ll[i], lw=0.5)

# fp.add_legend(ax, bbox_to_anchor=(1, 0.5), loc='center left')
# fp.add_labels(ax, 'Blended albedo', 'XCH4')
# fp.add_title(ax, 'Deseasonalized GOSAT vs. blended albedo')

# fp.save_fig(fig, plot_dir, 'gosat_deseasonalized_blended_albedo')

# # GOSAT blended albedo
# fig, ax = fp.get_figax(aspect=1.75)
# ll = ['--', ':', '-.', '-']
# ax.errorbar(DIFF_G['BLENDED_ALBEDO']-0.005, DIFF_G['GOSAT']['mean'],
#             yerr=DIFF_G['GOSAT']['std'], c='black', ls='--', lw=2,
#             label='Annual Average')
# for i, s in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
#     d_g = DIFF_G_S[DIFF_G_S['SEASON'] == s]
#     # ax.plot(d_g['BLENDED_ALBEDO'], d_g['DIFF']['mean'],
#     #         c=fp.color(3), ls=ll[i], lw=0.5, label=s)
#     ax.errorbar(d_g['BLENDED_ALBEDO']+0.005*i, d_g['GOSAT']['mean'],
#                 yerr=d_g['GOSAT']['std'], c=fp.color(2*i), ls='-', lw=1,
#                 label=s)
#     # ax.plot(d_g['BLENDED_ALBEDO'], d_g['TROPOMI']['mean'],
#     #         c=fp.color(5), ls=ll[i], lw=0.5)

# fp.add_legend(ax, bbox_to_anchor=(1, 0.5), loc='center left')
# fp.add_labels(ax, 'Blended albedo', 'XCH4')
# fp.add_title(ax, 'GOSAT vs. blended albedo')
# fp.save_fig(fig, plot_dir, 'gosat_blended_albedo')

# Difference
fig, axis = fp.get_figax(aspect=1.5, rows=2, cols=2, sharex=True, sharey=True)
ll = ['--', ':', '-.', '-']
# corr_dict = {}
TROPOMI_RAW['OBS_CORR'] = 0
for i, s in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
    ax = axis.flatten()[i]
    ax.text(0.98, 0.02, s, fontsize=config.LABEL_FONTSIZE*config.SCALE,
            ha='right', va='bottom', transform=ax.transAxes)
    if i == 3:
        labels = ['Seasonal average', 'Best fit']
    else:
        labels = ['', '']

    d_g = DIFF_G_S[DIFF_G_S['SEASON'] == s]
    # ax.plot(d_g['BLENDED_ALBEDO'], d_g['DIFF']['mean'],
    #         c=fp.color(3), ls=ll[i], lw=0.5, label=s)
    ax.errorbar(d_g['BLENDED_ALBEDO'], d_g['DIFF']['mean'],
                yerr=d_g['DIFF']['std'], c='0.7', ls='-', lw=2,
                label=labels[0])
    # ax.plot(d_g['BLENDED_ALBEDO'], d_g['TROPOMI']['mean'],
    #         c=fp.color(5), ls=ll[i], lw=0.5)

    d_g = DIFF[DIFF['SEASON'] == s]
    coef = p.polyfit(d_g['BLENDED_ALBEDO'], d_g['DIFF'], deg=1)
    bias_correction = p.polyval(d_g['BLENDED_ALBEDO'], coef)
    fit = p.polyval(np.arange(0.05, 1.21, 0.01), coef)
    # corr_dict[s] = coef
    TROPOMI_RAW.loc[TROPOMI_RAW['SEASON'] == s, 'OBS_CORR'] = d_g['TROPOMI'] - bias_correction
    ax.plot(np.arange(0.05, 1.21, 0.01), fit, c='0.4', lw=2, ls=':',
            label=labels[1])

    # Actual data
    # ax.scatter(d_g['BLENDED_ALBEDO'], d_g['DIFF'], c=fp.color(2*i),
    #            alpha=0.2, s=5)
    ax.hist2d(d_g['BLENDED_ALBEDO'], d_g['DIFF'],
              vmin=0, vmax=20, cmap=fp.cmap_trans('plasma'),
              bins=[np.arange(0, 1.3, 0.01), np.arange(-30, 31, 10)])

for ax in axis.flatten():
    ax.errorbar(DIFF_G['BLENDED_ALBEDO']-0.005, DIFF_G['DIFF']['mean'],
                yerr=DIFF_G['DIFF']['std'], c='black', ls='--', lw=2,
                label='Annual Average')
    ax.set_ylim(-40, 40)
    ax.set_xlim(0, 1.3)

fp.add_labels(axis[0, 0], '', r'$\Delta$XCH4')
fp.add_labels(axis[1, 0], 'Blended albedo', r'$\Delta$XCH4')
fp.add_labels(axis[1, 1], 'Blended albedo', '')
handles_labels = [ax.get_legend_handles_labels() for ax in axis.flatten()]
handles, labels = [sum(a, []) for a in zip(*handles_labels)]
fp.add_legend(axis[0, 0], handles=handles, labels=labels,
              bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2,
              bbox_transform=fig.transFigure)
fig.suptitle('TROPOMI - GOSAT vs. blended albedo',
             fontsize=config.TITLE_FONTSIZE*config.SCALE)
fp.save_fig(fig, plot_dir, 'gosat_tropomi_blended_albedo_0.1')

# Plot corrected data
print('Data is corrected.')
TROPOMI_RAW['DIFF_CORR'] = TROPOMI_RAW['MOD'] - TROPOMI_RAW['OBS_CORR']
print(f'm = {TROPOMI_RAW.shape[0]}')
print('MODEL MAXIMUM : ', TROPOMI_RAW['MOD'].max())
print('MODEL MINIMUM : ', TROPOMI_RAW['MOD'].min())
print('TROPOMI MAXIMUM : ', TROPOMI_RAW['OBS_CORR'].max())
print('TROPOMI MINIMUM : ', TROPOMI_RAW['OBS_CORR'].min())
print('DIFFERENCE MAXIMUM : ', np.abs(TROPOMI_RAW['DIFF_CORR']).max())
print('DIFFERENCE MEAN : ', np.mean(TROPOMI_RAW['DIFF_CORR']))
print('DIFFERENCE STD : ', np.std(TROPOMI_RAW['DIFF_CORR']))

# cax = fp.add_cax(fig, ax)

# c_g, x_g, y_g = np.histogram2d(DIFF['BLENDED_ALBEDO'], DIFF['GOSAT'], bins=20)
# c_t, x_t, y_t = np.histogram2d(DIFF['BLENDED_ALBEDO'], DIFF['TROPOMI'], bins=20)
# ax.contour(c_g.T, extent=[x_g.min(), x_g.max(), y_g.min(), y_g.max()],
#            colors=fp.color(3), linestyles='solid')
# ax.contour(c_t.T, extent=[x_t.min(), x_t.max(), y_t.min(), y_t.max()],
#            colors=fp.color(7), linestyles='solid')

# c = ax.scatter(DIFF_G['BLENDED_ALBEDO_BIN'], DIFF['GOSAT'], c=DIFF['MONTH'],
#                cmap='plasma', alpha=0.1)
# cb = fig.colorbar(c, ax=ax, cax=cax)
# cb = fp.format_cbar(cb, 'XCH4')

# ax.errorbar(DIFF_G['BLENDED_ALBEDO'], DIFF_G['DIFF']['mean'],
#             yerr=DIFF_G['DIFF']['std'], c=fp.color(3))
# ax.errorbar(DIFF_G['BLENDED_ALBEDO'], DIFF_G['GOSAT']['mean'],
#             yerr=DIFF_G['GOSAT']['std'], c=fp.color(3))
# ax.errorbar(DIFF_G['BLENDED_ALBEDO'] + 0.01, DIFF_G['TROPOMI']['mean'],
#             yerr=DIFF_G['TROPOMI']['std'], c=fp.color(5))

