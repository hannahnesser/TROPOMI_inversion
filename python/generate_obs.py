'''
This script generates the observation vector, the prior observation vector (i.e. F(xa)), and the observational error variances. It applies filters on albedo, latitude, and seaason to remove problematic TROPOMI observations. The variances are calcualted using the residual error method.

   **Inputs**

   | ----------------- | -------------------------------------------------- |
   | Input             | Description                                        |
   | ----------------- | -------------------------------------------------- |
   | prior_run         | A file or files containing the processed output of |
   |                   | the prior run of the forward model (after applying |
   |                   | the satellite averaging kernel. The input here can |
   |                   | be either a list of daily files or a single file   |
   |                   | containing all observations for the year.          |
   | ----------------- | -------------------------------------------------- |
   | filter_on_\       | A boolean flag indicating whether or not to filter |
   | blended_albedo    | on blended albedo, which should remove snow and    |
   |                   | ice covered scenes, as recommended by Lorente et   |
   |                   | al. 2021 and described in Wunch et al. 2011.       |
   | ----------------- | -------------------------------------------------- |
   | blended_albedo_ \ | The blended albedo threshold above which           |
   | threshold         | are removed from the observation vector. Lorente   |
   |                   | et al. find a value of 0.85 and Wunch et al. find  |
   |                   | a value of about 1. We use 1.1.                    |
   | ----------------- | -------------------------------------------------- |
   | filter_on_albedo  | A boolean flag indicating whether or not to filter |
   |                   | out scenes below the albedo_threshold, following   |
   |                   | the recommendation of de Gouw et al. 2020.         |
   | ----------------- | -------------------------------------------------- |
   | albedo_threshold  | The albedo threshold below which observations are  |
   |                   | removed. De Gouw et al. use 0.05. We do too.       |
   | ----------------- | -------------------------------------------------- |
   | filter_on_ \      | A boolean flag indicating whether or not to remove |
   | seaasonal_ \      | observations north of 50 degrees N during winter   |
   | latitude          | (DJF) months to further remove snow- and ice-      |
   |                   | covered scenes.                                    |
   | ----------------- | -------------------------------------------------- |
   | remove_ \         | A boolean flag indicating whether or not to remove |
   | latitudinal_bias  | the latitudinal bias in the model - observation    |
   |                   | difference with a first order polynomial.          |
   | ----------------- | -------------------------------------------------- |
   | albedo_bins       | The albedo increments in which to bin the model -  |
   |                   | observation difference for statistical analysis.   |
   | ----------------- | -------------------------------------------------- |
   | lat_bins          | The latitude increments in which to bin the model  |
   |                   | - observation difference for statistical analysis. |
   | ----------------- | -------------------------------------------------- |

   **Outputs**

   | ----------------- | -------------------------------------------------- |
   | Output            | Description                                        |
   | ----------------- | -------------------------------------------------- |
   | y.nc              | The observation vector containing bias-corrected   |
   |                   | TROPOMI observations.                              |
   | ----------------- | -------------------------------------------------- |
   | ya.nc             | The prior observation vector containing the output |
   |                   | of the prior simulation, i.e. F(xa).               |
   | ----------------- | -------------------------------------------------- |
   | so.nc             | The observational error variances calculated using |
   |                   | the residual error method.                         |
   | ----------------- | -------------------------------------------------- |
   | blended_albedo_ \ | Monthly plots of observed XCH4 vs. blended albedo. |
   | filter_mm*.png    |                                                    |
   | ----------------- | -------------------------------------------------- |
   | prior_bias*.png   | Scatter plot of model vs. observation.             |
   | ----------------- | -------------------------------------------------- |
   | prior_spatial_ \  | Map of the model - observation bias on the inverse |
   | bias*.png         | grid.                                              |
   | ----------------- | -------------------------------------------------- |
   | prior_ \          | Plot of the mean and standard deviation of the     |
   | latitudinal_ \    | model - observation bias binned by lat_bin.        |
   | bias*.png         |                                                    |
   | ----------------- | -------------------------------------------------- |
   | prior_seasonal \  | Plot of the mean and standard deviation of the     |
   | bias*.png         | model - observation bias binned by month.          |
   | ----------------- | -------------------------------------------------- |
   | prior_seasonal \  | Plot of the mean and standard deviation of the     |
   | latitudinal_ \    | model - observation bias binned by lat_bin and     |
   | bias*.png         | month.                                             |
   | ----------------- | -------------------------------------------------- |
   | prior_albedo_ \   | Plot of the mean and standard deviation of the     |
   | bias*.png         | model - observation bias binned by albedo_bin.     |
   | ----------------- | -------------------------------------------------- |
   | observations*.png | Seasonal plots of the observations averaged over   |
   |                   | the inversion grid.                                |
   | ----------------- | -------------------------------------------------- |
'''

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
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python'
data_dir = base_dir + 'inversion_data'
output_dir = base_dir + 'inversion_data'
plot_dir = base_dir + 'plots'

# Import Custom packages
sys.path.append(code_dir)
import config
import gcpy as gc
import troppy as tp
import format_plots as fp
import inversion_settings as settings

# Define prior run file
prior_run = f'{settings.year}_full.pkl'

# Define the blended albedo threshold
filter_on_blended_albedo = True
blended_albedo_threshold = 0.75
blended_albedo_bins = np.arange(0, 2.5, 0.1)

# Define a plain old albedo threshold
filter_on_albedo = True
albedo_threshold = 0.05
albedo_bins = np.arange(0, 1.1, 0.05)

# Define a seasonal latitudinal filter
filter_on_seasonal_latitude = True

# Remove latitudinal bias
remove_latitudinal_bias = False
remove_mean_bias = True
lat_bins = np.arange(10, 65, 5)

## ------------------------------------------------------------------------ ##
## Define functions
## ------------------------------------------------------------------------ ##
def print_summary(data, mod_suffix=''):
    print('Model maximum       : %.2f' % (data[f'MOD{mod_suffix}'].max()))
    print('Model minimum       : %.2f' % (data[f'MOD{mod_suffix}'].min()))
    print('TROPOMI maximum     : %.2f' % (data['OBS'].max()))
    print('TROPOMI minimum     : %.2f' % (data['OBS'].min()))
    print('Difference maximum  : %.2f' % (np.abs(data[f'DIFF{mod_suffix}']).max()))
    print('Difference mean     : %.2f' % (np.mean(data[f'DIFF{mod_suffix}'])))
    print('Difference STD      : %.2f' % (np.std(data[f'DIFF{mod_suffix}'])))

def print_filter_summary(obs_filter, seasons, filter_string):
    tot_nobs = len(obs_filter)
    nobs = obs_filter.sum()
    season_nobs = {}
    for seas in ['DJF', 'MAM', 'JJA', 'SON']:
        d = obs_filter[seasons == seas]
        season_nobs[seas] = '%.2f %%' % ((d.sum()/len(d))*100)
    filter_string = f'{filter_string} filter applied'
    print(f'{filter_string:<40}: {nobs} observations remain ({(nobs/tot_nobs*100):.1f}%)', season_nobs)

## ------------------------------------------------------------------------ ##
## Load the data for the year
## ------------------------------------------------------------------------ ##
print(f'Opening data in {output_dir}/{prior_run}')
data = gc.load_obj(f'{output_dir}/observations/{prior_run}')

# Print summary
print('-'*70)
print('Original data (pre-filtering) is loaded.')
print(f'm = {data.shape[0]}')
print_summary(data)

## ----------------------------------------- ##
## Make and apply observational mask
## ----------------------------------------- ##
print('-'*70)
suffix = ''

# Create a vector for storing the observational filter
obs_filter = np.ones(data.shape[0], dtype=bool)

# We always filter on clouds (this variable is a count of the number
# of nan values in the cloud_fraction variable, so where it is greater
# than zero, we wish to filter out the data)
cloud_filter = (data['CLOUDS'] == 0)
obs_filter = (obs_filter & cloud_filter)
m = obs_filter.sum()
print_filter_summary(obs_filter, data['SEASON'].values, 'No')

# Incorporate other filters as specified
if filter_on_blended_albedo:
    suffix += '_BAF'
    BAF_filter = ((data['MONTH'].isin(np.arange(6, 9, 1))) |
                  (data['BLENDED_ALBEDO'] < blended_albedo_threshold))
    obs_filter = (obs_filter & BAF_filter)
    print_filter_summary(obs_filter, data['SEASON'].values, 'Blended albedo')

if filter_on_albedo:
    suffix += '_AF'
    albedo_filter = (data['ALBEDO_SWIR'] > albedo_threshold)
    obs_filter = (obs_filter & albedo_filter)
    print_filter_summary(obs_filter, data['SEASON'].values, 'Albedo')

if filter_on_seasonal_latitude:
    suffix += '_LF'
    latitude_filter = ((data['MONTH'].isin(np.arange(3, 12, 1))) |
                       (data['LAT'] <= 50))
    obs_filter = (obs_filter & latitude_filter)
    print_filter_summary(obs_filter, data['SEASON'].values, 'Seasonal latitude')

# Apply filter
data = data[obs_filter]

# Print summary
print('-'*70)
print('Data is filtered.')
print(f'm = {data.shape[0]}')
print_summary(data)
print('-'*70)

## ----------------------------------------- ##
## Apply latitudinal bias correction
## ----------------------------------------- ##
if remove_latitudinal_bias:
    suffix += '_BC'

# Correct the latitudinal bias
coef = p.polyfit(data['LAT'], data['DIFF'], deg=1)
print(coef)
bias_correction = p.polyval(data['LAT'], coef)
data['MOD_LB'] = data['MOD'] - bias_correction # latitudinal bias correction
data['DIFF_LB'] = data['DIFF'] - bias_correction

# Print information
print(f'Data has latitudinal bias removed.')
print(f'    y = {coef[0]:.2f} + {coef[1]:.2f}x')
print_summary(data, '_LB')
print('-'*70)

bias_correction = data['DIFF'].mean()
data['MOD_MB'] = data['MOD'] - bias_correction
data['DIFF_MB'] = data['DIFF'] - bias_correction
print(f'Data has mean bias removed.')
print(f'    Mean bias of {bias_correction:.2f} ppb removed.')
print_summary(data, '_MB')

## ----------------------------------------- ##
## Save out resulting data and observational mask
## ----------------------------------------- ##
obs_filter = pd.DataFrame({'MONTH' : data['MONTH'], 'FILTER' :  obs_filter})
obs_filter.to_csv(join(output_dir, 'observations/obs_filter.csv'))

print(f'Saving data in {output_dir}/observations/{settings.year}_corrected.pkl')
gc.save_obj(data, f'{output_dir}/observations/{settings.year}_corrected.pkl')

# Calculate the number of observations
nobs = data.shape[0]

## ------------------------------------------------------------------------ ##
## Plot data
## ------------------------------------------------------------------------ ##
d_p = data.groupby(['LAT_CENTER', 'LON_CENTER']).agg({'OBS' : 'mean', 
                                                      'MOD' : 'count'})
d_p = d_p.rename(columns={'OBS' : 'AVG_OBS', 'MOD' : 'COUNT'})
d_p = d_p[['AVG_OBS', 'COUNT']].to_xarray()
d_p = d_p.rename({'LAT_CENTER' : 'lats', 'LON_CENTER' : 'lons'})

fig, ax = fp.get_figax(rows=1, cols=2, maps=True,
                       lats=d_p.lats, lons=d_p.lons,
                       max_width=config.BASE_WIDTH,
                       max_height=config.BASE_HEIGHT)
plt.subplots_adjust(hspace=-0.25, wspace=0.1) # wspace=0.05)

c = d_p['AVG_OBS'].plot(ax=ax[0], cmap='plasma', vmin=1820, vmax=1880,
                        add_colorbar=False)
ax[0] = fp.add_title(ax[0], '2019 TROPOMI methane observations')
c_c = d_p['COUNT'].plot(ax=ax[1], cmap='inferno', vmin=0, vmax=1e3,
                        add_colorbar=False)
ax[1] = fp.add_title(ax[1], 'Observational density')

cax = []
for axis in ax:
    axis = fp.format_map(axis, d_p.lats, d_p.lons)
    caxis = fp.add_cax(fig, axis, horizontal=True, cbar_pad_inches=0.25)
    cax.append(caxis)

cb = fig.colorbar(c, ax=ax[0], cax=cax[0], orientation='horizontal')
cb = fp.format_cbar(cb, 'Methane mixing ratio (ppb)', y=-3, horizontal='horizontal')

cb_c = fig.colorbar(c_c, ax=ax[1], cax=cax[1], orientation='horizontal')
cb_c = fp.format_cbar(cb_c, 'Count', y=-3, horizontal='horizontal')

fp.save_fig(fig, plot_dir, f'observations{suffix}')

## ------------------------------------------------------------------------ ##
## Analyze data
## ------------------------------------------------------------------------ ##
for mod_suffix in ['_LB', '_MB']:
    groupby = ['LAT_CENTER', 'LON_CENTER', 'SEASON']

    ## Spatial bias
    s_b = data.groupby(groupby).mean()[[f'DIFF{mod_suffix}', 'ALBEDO_SWIR', 'AOD']]
    s_b = s_b.to_xarray().rename({'LAT_CENTER' : 'lats', 'LON_CENTER' : 'lons',
                                  'SEASON' : 'season'})
    print('Spatial bias analyzed.')

    ## Spatial distribution of blended albedo
    s_ba = data.groupby(groupby).mean()['BLENDED_ALBEDO']
    s_ba = s_ba.to_xarray().rename({'LAT_CENTER' : 'lats', 'LON_CENTER' : 'lons',
                                    'SEASON' : 'season'})
    print('Spatial distribution of blended albedo analyzed.')

    ## Latitudinal bias
    data['LAT_BIN'] = pd.cut(data['LAT'], lat_bins)
    l_b = gc.group_data(data, groupby=['LAT_BIN'], quantity=f'DIFF{mod_suffix}')
    l_b['LAT'] = l_b['LAT_BIN'].apply(lambda x: x.mid)
    print('Latitudinal bias analyzed.')

    ## Seasonality
    m_b = gc.group_data(data, groupby=['MONTH'], quantity=f'DIFF{mod_suffix}')
    m_b['month'] = pd.to_datetime(m_b['MONTH'], format='%m')
    m_b['month'] = m_b['month'].dt.month_name().str[:3]
    print('Monthly bias analyzed.')

    ## Latitudinal bias and seasonality
    lm_b = gc.group_data(data, groupby=['LAT_BIN', 'SEASON'], 
                         quantity=f'DIFF{mod_suffix}')
    lm_b['LAT'] = lm_b['LAT_BIN'].apply(lambda x: x.mid)
    print('Seasonal latitudinal bias analyzed.')

    ## Albedo
    data['ALBEDO_BIN'] = pd.cut(data['ALBEDO_SWIR'], albedo_bins)
    a_b = gc.group_data(data, groupby=['ALBEDO_BIN'], quantity=f'DIFF{mod_suffix}')
    a_b['ALBEDO'] = a_b['ALBEDO_BIN'].apply(lambda x: x.mid)
    print('Albedo bias analyzed.')

    ## Blended albedo
    data['BLENDED_ALBEDO_BIN'] = pd.cut(data['BLENDED_ALBEDO'],
                                        blended_albedo_bins)
    ba_b = gc.group_data(data, groupby=['BLENDED_ALBEDO_BIN'], 
                         quantity=f'DIFF{mod_suffix}')
    bam_b = gc.group_data(data, groupby=['BLENDED_ALBEDO_BIN', 'MONTH'], 
                          quantity=f'DIFF{mod_suffix}')
    ba_b['BLENDED_ALBEDO'] = ba_b['BLENDED_ALBEDO_BIN'].apply(lambda x: x.mid)
    bam_b['BLENDED_ALBEDO'] = bam_b['BLENDED_ALBEDO_BIN'].apply(lambda x: x.mid)
    print('Blended albedo bias analyzed.')

    ## -------------------------------------------------------------------- ##
    ## Plot data
    ## -------------------------------------------------------------------- ##
    ## ----------------------------------------- ##
    ## Scatter plot
    ## ----------------------------------------- ##
    fig, ax, c = gc.plot_comparison(data['OBS'].values, 
                                    data[f'MOD{mod_suffix}'].values,
                                    lims=[1750, 1950], vmin=0, vmax=3e4,
                                    xlabel='Observation', ylabel='Model')
    ax.set_xticks(np.arange(1750, 2000, 100))
    ax.set_yticks(np.arange(1750, 2000, 100))
    fp.save_fig(fig, plot_dir, f'prior_bias{suffix}{mod_suffix}')
    plt.close()

    ## ----------------------------------------- ##
    ## Spatial bias
    ## ----------------------------------------- ##
    # Satellite observation difference
    fig, ax = fp.get_figax(rows=2, cols=2, maps=True,
                           lats=s_b.lats, lons=s_b.lons)
    for i, axis in enumerate(ax.flatten()):
        s = s_b.season.values[i]
        d = s_b.sel(season=s)[f'DIFF{mod_suffix}']
        c = d.plot(ax=axis, cmap='PuOr_r', vmin=-30, vmax=30, 
                   add_colorbar=False)
        axis = fp.format_map(axis, d.lats, d.lons)
        axis = fp.add_title(axis, f'{s}')
    cax = fp.add_cax(fig, ax)
    cb = fig.colorbar(c, ax=ax, cax=cax)
    cb = fp.format_cbar(cb, r'Model - Observation')
    fp.save_fig(fig, plot_dir, f'prior_spatial_bias{suffix}{mod_suffix}')
    plt.close()

    # Blended albedo
    ba_cmap_1 = plt.cm.RdYlGn_r(np.linspace(0, 0.5, 256))
    ba_cmap_2 = plt.cm.RdYlGn_r(np.linspace(0.5, 1, 256))
    ba_cmap = np.vstack((ba_cmap_1, ba_cmap_2))
    ba_cmap = colors.LinearSegmentedColormap.from_list('ba_cmap', ba_cmap)
    div_norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.75, vmax=3)
    fig, ax = fp.get_figax(rows=2, cols=2, maps=True,
                           lats=s_ba.lats, lons=s_ba.lons)
    for i, axis in enumerate(ax.flatten()):
        s = s_ba.season.values[i]
        d = s_ba.sel(season=s)
        # d = d.where(d >= 0.75, 0)
        c = d.plot(ax=axis, norm=div_norm, cmap=ba_cmap,
                   add_colorbar=False)
        axis = fp.format_map(axis, d.lats, d.lons)
        axis = fp.add_title(axis, f'{s}')
    cax = fp.add_cax(fig, ax)
    cb = fig.colorbar(c, ax=ax, cax=cax)
    cb = fp.format_cbar(cb, r'Blended albedo')
    fp.save_fig(fig, plot_dir, f'blended_albedo_spatial{suffix}{mod_suffix}')
    plt.close()

    # Albedo
    fig, ax = fp.get_figax(rows=2, cols=2, maps=True, lats=s_b.lats, lons=s_b.lons)
    for i, axis in enumerate(ax.flatten()):
        s = s_b.season.values[i]
        d = s_b.sel(season=s)['ALBEDO_SWIR']
        c = d.plot(ax=axis, cmap='plasma', vmin=0, vmax=0.3, add_colorbar=False)
        axis = fp.format_map(axis, d.lats, d.lons)
        axis = fp.add_title(axis, f'{s}')
    cax = fp.add_cax(fig, ax)
    cb = fig.colorbar(c, ax=ax, cax=cax)
    cb = fp.format_cbar(cb, r'Shortwave infrared albedo')
    fp.save_fig(fig, plot_dir, f'albedo_spatial{suffix}{mod_suffix}')
    plt.close()

    # AOD
    fig, ax = fp.get_figax(rows=2, cols=2, maps=True,
                           lats=s_b.lats, lons=s_b.lons)
    for i, axis in enumerate(ax.flatten()):
        s = s_b.season.values[i]
        d = s_b.sel(season=s)['AOD']
        c = d.plot(ax=axis, cmap='plasma', vmin=0.005, vmax=0.05,
                   add_colorbar=False)
        axis = fp.format_map(axis, d.lats, d.lons)
        axis = fp.add_title(axis, f'{s}')
    cax = fp.add_cax(fig, ax)
    cb = fig.colorbar(c, ax=ax, cax=cax)
    cb = fp.format_cbar(cb, r'Aerosol optical depth')
    fp.save_fig(fig, plot_dir, f'aod_spatial{suffix}{mod_suffix}')
    plt.close()

    ## ----------------------------------------- ##
    ## Monthly bias
    ## ----------------------------------------- ##
    fig, ax = fp.get_figax(aspect=1.75)
    ax.errorbar(m_b['month'], m_b['mean'], yerr=m_b['std'], color=fp.color(4))
    ax = fp.add_labels(ax, 'Month', 'Model - Observation')
    ax = fp.add_title(ax, 'Seasonal bias in prior run')
    fp.save_fig(fig, plot_dir, f'prior_seasonal_bias{suffix}{mod_suffix}')
    plt.close()

    if mod_suffix == '_MB':
        ## ----------------------------------------- ##
        ## Seasonal latitudinal bias
        ## ----------------------------------------- ##
        fig, ax = fp.get_figax(aspect=1.75, max_width=config.BASE_WIDTH/2,
                               max_height=config.BASE_HEIGHT/2)
        ax.errorbar(l_b['LAT'], l_b['mean'] + 9.11, yerr=l_b['std'],
                    color=fp.color(4), label='Annual average', zorder=10,
                    ecolor=fp.color(4), elinewidth=0.75, capsize=1, 
                    capthick=0.5)
        ax.set_xticks(np.arange(10, 70, 10))
        ax.set_xlim(10, 60)
        ax = fp.add_labels(ax, 'Latitude (degrees)', 
                           'Model - observation (ppb)',
                           labelpad=10)
        ax = fp.add_title(ax, f'Latitudinal and seasonal bias in prior simulation')
        linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
        for i, season in enumerate(np.unique(lm_b['SEASON'])):
            d = lm_b[lm_b['SEASON'] == season]
            ax.plot(d['LAT'].values, d['mean'].values + 9.11, 
                    color=fp.color(4), alpha=0.5, label=f'{season} average', 
                    ls=linestyles[i], lw=0.5, zorder=0)
        ax.axhline(9.11, color='0.6', ls='-', label='Mean bias', zorder=20,
                   lw=0.75)
        ax.plot(np.arange(10, 70, 10), -5.40 + 0.39*np.arange(10, 70, 10),
                color='0.6', ls='--', label='Latitudinal bias fit', zorder=20,
                lw=0.75)
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[i] for i in [6, 0, 1, 2, 3, 4, 5]]
        labels = [labels[i] for i in [6, 0, 1, 2, 3, 4, 5]]
        fp.add_legend(ax, handles=handles, labels=labels, 
                      bbox_to_anchor=(1, 0.5), loc='center left', ncol=1)
        fp.save_fig(fig, plot_dir,
                    f'prior_seasonal_latitudinal_bias{suffix}{mod_suffix}')
        plt.close()

    ## ----------------------------------------- ##
    ## Albedo bias
    ## ----------------------------------------- ##
    fig, ax = fp.get_figax(aspect=1.75)
    ax.errorbar(a_b['ALBEDO'], a_b['mean'], yerr=a_b['std'], color=fp.color(4))
    ax.set_xticks(np.arange(0, 1, 0.2))
    ax = fp.add_labels(ax, 'Albedo', 'Model - observation')
    ax = fp.add_title(ax, 'Albedo bias in prior run')
    fp.save_fig(fig, plot_dir, f'prior_albedo_bias{suffix}{mod_suffix}')
    plt.close()

    ## ----------------------------------------- ##
    ## Seasonal blended albedo bias
    ## ----------------------------------------- ##
    fig, ax = fp.get_figax(aspect=1.75)
    ax.errorbar(ba_b['BLENDED_ALBEDO'], ba_b['mean'], yerr=ba_b['std'],
                color=fp.color(4))
    ax.set_xticks(np.arange(0, 3.1, 0.25))
    ax.set_xlim(0, 2.25)
    ax.set_ylim(-30, 30)
    ax = fp.add_labels(ax, 'Blended albedo', 'Model - observation')
    ax = fp.add_title(ax, 'Blended albedo bias in prior run')
    fp.save_fig(fig, plot_dir, f'prior_blended_albedo_bias{suffix}{mod_suffix}')
    plt.close()
