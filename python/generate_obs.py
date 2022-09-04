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
   | analyze_biases    | A boolean flag indicating whether or not to        |
   |                   | analyze and plot spatial, latitudinal, seasonal,   |
   |                   | and albedinal, biases in the model - observation   |
   |                   | difference.                                        |
   | ----------------- | -------------------------------------------------- |
   | albedo_bins       | The albedo increments in which to bin the model -  |
   |                   | observation difference for statistical analysis.   |
   | ----------------- | -------------------------------------------------- |
   | lat_bins          | The latitude increments in which to bin the model  |
   |                   | - observation difference for statistical analysis. |
   | ----------------- | -------------------------------------------------- |
   | calculate_so      | A boolean flag indicating whether or not to        |
   |                   | calculate the observational error variances using  |
   |                   | the residual error method (Heald et al. 2004).     |
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
   | errors*.png       | Seaasonal plots of the standard deviation averaged |
   |                   | over the inversion grid.                           |
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
    output_dir = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/inversion_data'

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

# The prior_run can either be a list of files or a single file
# with all of the data for simulation
prior_run = f'{settings.year}.pkl'
# prior_run = [f'{settings.year}{mm:02d}{dd:02d}_GCtoTROPOMI.pkl'
#              for mm in settings.months
#              for dd in settings.days]
# prior_run.sort()

# Compare to GOSAT?
compare_gosat = True
# gosat_dir = '/n/jacob_lab/Lab/seasasfs02/CH4_inversion/InputData/Obs/ULGOSAT_v9/2019'
# gosat = [f'{gosat_dir}/UoL-GHG-L2-CH4-GOSAT-OCPR-{settings.year}{mm:02d}{dd:02d}-fv9.0.nc'
#          for mm in settings.months for dd in settings.days]
gosat = None

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
remove_latitudinal_bias = True
lat_bins = np.arange(10, 65, 5)

# Which analyses do you wish to perform?
analyze_biases = True

# Calculate the error variances?
calculate_so = True
err_min = 10

## ------------------------------------------------------------------------ ##
## Get grid information
## ------------------------------------------------------------------------ ##
lats, lons = gc.create_gc_grid(*settings.lats, settings.lat_delta,
                               *settings.lons, settings.lon_delta,
                               centers=False, return_xarray=False)

# GOSAT comparison grid
lats_l, lons_l = gc.create_gc_grid(*settings.lats, 2, *settings.lons, 2,
                                   centers=False, return_xarray=False)

# Error evaluation grid
lats_so, lons_so = gc.create_gc_grid(*settings.lats, 2, *settings.lons, 2,
                                     centers=False, return_xarray=False)
groupby = ['LAT_CENTER_L', 'LON_CENTER_L', 'SEASON']
err_suffix = '_rg2rt'

## ------------------------------------------------------------------------ ##
## Define functions
## ------------------------------------------------------------------------ ##
def print_summary(data):
    print('Model maximum       : %.2f' % (data['MOD'].max()))
    print('Model minimum       : %.2f' % (data['MOD'].min()))
    print('TROPOMI maximum     : %.2f' % (data['OBS'].max()))
    print('TROPOMI minimum     : %.2f' % (data['OBS'].min()))
    print('Difference maximum  : %.2f' % (np.abs(data['DIFF']).max()))
    print('Difference mean     : %.2f' % (np.mean(data['DIFF'])))
    print('Difference STD      : %.2f' % (np.std(data['DIFF'])))

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
if type(prior_run) == list:
    data = np.array([]).reshape(0, 14)
    for file in prior_run:
        # Check if that file is in the data directory
        if file not in listdir(data_dir):
            print(f'{file} is not in the data directory.')
            continue

        # Get the month
        month = int(file[4:6])
        day = int(file[6:8])
        date = int(file[:8])

        # Load the data. The columns are: 0 OBS, 1 MOD, 2 LON, 3 LAT,
        # 4 iGC, 5 jGC, 6 PRECISION, 7 ALBEDO_SWIR, 8 ALBEDO_NIR, 9 AOD,
        # 10 CLOUDS (15 10 total columns)
        new_data = gc.load_obj(join(data_dir, file))
        new_data = new_data[:, :11]
        nobs = new_data.shape[0]
        new_data = np.append(new_data, month*np.ones((nobs, 1)), axis=1)
        new_data = np.append(new_data, day*np.ones((nobs, 1)), axis=1)
        new_data = np.append(new_data, date*np.ones((nobs, 1)), axis=1)

        data = np.concatenate((data, new_data))

    ## ----------------------------------------- ##
    ## Basic data formatting
    ## ----------------------------------------- ##
    # Create a dataframe from the data
    columns = ['OBS', 'MOD', 'LON', 'LAT', 'iGC', 'jGC', 'PREC',
               'ALBEDO_SWIR', 'ALBEDO_NIR', 'AOD', 'CLOUDS',
               'MONTH', 'DAY', 'DATE']
    data = pd.DataFrame(data, columns=columns)

    # Calculate blended albedo
    data['BLENDED_ALBEDO'] = tp.blended_albedo(data,
                                               data['ALBEDO_SWIR'],
                                               data['ALBEDO_NIR'])

    # Add season
    data.loc[:, 'SEASON'] = 'DJF'
    data.loc[data['MONTH'].isin([3, 4, 5]), 'SEASON'] = 'MAM'
    data.loc[data['MONTH'].isin([6, 7, 8]), 'SEASON'] = 'JJA'
    data.loc[data['MONTH'].isin([9, 10, 11]), 'SEASON'] = 'SON'

    # Save nearest latitude and longitude centers
    data['LAT_CENTER'] = lats[gc.nearest_loc(data['LAT'].values, lats)]
    data['LON_CENTER'] = lons[gc.nearest_loc(data['LON'].values, lons)]
    data['LAT_CENTER_L'] = lats_l[gc.nearest_loc(data['LAT'].values, lats_l)]
    data['LON_CENTER_L'] = lons_l[gc.nearest_loc(data['LON'].values, lons_l)]

    # Calculate model - observation
    data['DIFF'] = data['MOD'] - data['OBS']

    # Subset data
    data = data[['iGC', 'jGC', 'SEASON', 'MONTH', 'DAY', 'DATE', 'LON', 'LAT',
                 'LON_CENTER', 'LAT_CENTER', 'LON_CENTER_L', 'LAT_CENTER_L',
                 'OBS', 'MOD', 'DIFF', 'PREC', 'ALBEDO_SWIR',
                 'BLENDED_ALBEDO', 'AOD', 'CLOUDS']]

    # Print out some statistics
    cols = ['LAT', 'LON', 'MONTH', 'MOD']
    print_summary(data)

    # Save the data out
    print(f'Saving data in {output_dir}/{settings.year}.pkl')
    gc.save_obj(data, join(output_dir, f'{settings.year}.pkl'))

else:
    print(f'Opening data in {output_dir}/{prior_run}')
    data = gc.load_obj(join(output_dir, prior_run))

# Prnt summary
print('-'*70)
print('Original data (pre-filtering) is loaded.')
print(f'm = {data.shape[0]}')
print_summary(data)

## ----------------------------------------- ##
## Make and apply observational mask
## ----------------------------------------- ##
print('-'*70)

# Define an empty suffix for file names
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
    bias_correction = p.polyval(data['LAT'], coef)
    data['MOD'] -= bias_correction
    data['DIFF'] -= bias_correction

    # Print information
    print(f'Data has latitudinal bias removed.')
    print(f'    y = {coef[0]:.2f} + {coef[1]:.2f}x')
    print_summary(data)
    print('-'*70)

## ----------------------------------------- ##
## Save out resulting data and observational mask
## ----------------------------------------- ##
obs_filter = pd.DataFrame({'MONTH' : data['MONTH'], 'FILTER' :  obs_filter})
obs_filter.to_csv(join(output_dir, 'obs_filter.csv'))

print(f'Saving data in {output_dir}/{settings.year}_corrected.pkl')
gc.save_obj(data, join(output_dir, f'{settings.year}_corrected.pkl'))

# Calculate the number of observations
nobs = data.shape[0]

## ----------------------------------------- ##
## Compare TROPOMI to GOSAT
## ----------------------------------------- ##
# Load data
if compare_gosat and type(gosat) == list:
    gosat_data = pd.DataFrame(columns=['DATE', 'LAT', 'LON', 'OBS'])
    for file in gosat:
        # Check if that file is in the data directory
        if file.split('/')[-1] not in listdir(gosat_dir):
            print(f'{file} is not in the data directory.')
            continue

        # Load the data.
        gosat_fields = ['latitude', 'longitude', 'xch4', 'xch4_quality_flag']
        new_data = gc.read_file(join(gosat_dir, file))[gosat_fields]
        new_data = new_data.where((new_data['xch4_quality_flag'] == 0) &
                                  (new_data['latitude'] > settings.lat_min) &
                                  (new_data['latitude'] < settings.lat_max) &
                                  (new_data['longitude'] > settings.lon_min) &
                                  (new_data['longitude'] < settings.lon_max),
                                  drop=True)
        new_data = new_data.rename({'latitude' : 'LAT', 'longitude' : 'LON',
                                    'xch4' : 'OBS'})
        new_data = new_data[['LAT', 'LON', 'OBS']].to_dataframe()
        new_data['DATE'] = int(file.split('-')[-2])

        # Concatenate
        gosat_data = pd.concat([gosat_data, new_data]).reset_index(drop=True)

    # Save nearest latitude and longitude centers
    gosat_data['LAT_CENTER'] = lats_l[gc.nearest_loc(gosat_data['LAT'].values,
                                                   lats_l)]
    gosat_data['LON_CENTER'] = lons_l[gc.nearest_loc(gosat_data['LON'].values,
                                                   lons_l)]


    # Add month and season
    gosat_data.loc[:, 'MONTH'] = pd.to_datetime(gosat_data['DATE'].values,
                                       format='%Y%m%d').month
    gosat_data.loc[:, 'SEASON'] = 'DJF'
    gosat_data.loc[gosat_data['MONTH'].isin([3, 4, 5]), 'SEASON'] = 'MAM'
    gosat_data.loc[gosat_data['MONTH'].isin([6, 7, 8]), 'SEASON'] = 'JJA'
    gosat_data.loc[gosat_data['MONTH'].isin([9, 10, 11]), 'SEASON'] = 'SON'

    # Save the data out
    print(f'Saving data in {output_dir}/{settings.year}_gosat.pkl')
    gc.save_obj(gosat_data, join(output_dir, f'{settings.year}_gosat.pkl'))

    # Grid the GOSAT data
    gosat_grid = gosat_data.groupby(['LAT_CENTER', 'LON_CENTER',
                                     'DATE']).mean()['OBS']
    gosat_grid = gosat_grid.to_xarray().rename({'LAT_CENTER' : 'lats',
                                                'LON_CENTER' : 'lons',
                                                'DATE' : 'date'})

    # Save out
    gosat_grid.to_netcdf(join(output_dir, f'{settings.year}_gosat_gridded.nc'))

elif compare_gosat and gosat is None:
    print('-'*70)
    print('Conducting GOSAT comparison')
    print(f'Opening data in {output_dir}')
    gosat_data = gc.read_file(join(output_dir, f'{settings.year}_gosat.pkl'))
    gosat_grid = gc.read_file(join(output_dir,
                                   f'{settings.year}_gosat_gridded.nc'))

    # Save the data out
    # print(f'Saving data in {output_dir}/{settings.year}_gosat_diff.pkl')
    # gc.save_obj(diff_grid, join(output_dir, f'{settings.year}_gosat_diff.pkl'))

if compare_gosat:
    # Grid the TROPOMI data
    trop_grid = data.groupby(['LAT_CENTER_L', 'LON_CENTER_L',
                              'DATE']).mean()[['OBS', 'BLENDED_ALBEDO']]
    trop_grid = trop_grid.to_xarray().rename({'LAT_CENTER_L' : 'lats',
                                              'LON_CENTER_L' : 'lons',
                                              'DATE' : 'date'})

    # Take the difference and save out some time information
    trop_grid['DIFF'] = (trop_grid['OBS'] - gosat_grid)
    trop_grid['GOSAT'] = gosat_grid
    trop_grid = trop_grid.rename({'OBS' : 'TROPOMI'})
    trop_grid['date'] = pd.to_datetime(trop_grid['date'].values,
                                       format='%Y%m%d')
    trop_grid['MONTH'] = trop_grid['date'].dt.month
    trop_grid['SEASON'] = trop_grid['date'].dt.season

    # Create a dataframe
    diff_grid = trop_grid.to_dataframe()[['GOSAT', 'TROPOMI', 'DIFF',
                                          'MONTH', 'SEASON', 'BLENDED_ALBEDO']]
    diff_grid = diff_grid.dropna().reset_index()

    # Cut into blended albedo bins and latitude bins
    diff_grid['BLENDED_ALBEDO_BIN'] = pd.cut(diff_grid['BLENDED_ALBEDO'],
                                             blended_albedo_bins)
    diff_grid['LAT_BIN'] = pd.cut(diff_grid['lats'], lat_bins)

    # Group GOSAT-TROPOMI analysis
    ba_b_gos = gc.group_data(diff_grid, groupby=['BLENDED_ALBEDO_BIN'])
    bam_b_gos = gc.group_data(diff_grid, groupby=['BLENDED_ALBEDO_BIN',
                                                  'MONTH'])
    bal_b_gos = gc.group_data(diff_grid, groupby=['BLENDED_ALBEDO_BIN',
                                                  'LAT_BIN'])

    # Also do a spatial analysis, grouped by month
    trop_grid = trop_grid.groupby('date.season').mean()

    # Get midpoints
    ba_b_gos['BLENDED_ALBEDO'] = ba_b_gos['BLENDED_ALBEDO_BIN'].apply(lambda x: x.mid)
    bam_b_gos['BLENDED_ALBEDO'] = bam_b_gos['BLENDED_ALBEDO_BIN'].apply(lambda x: x.mid)
    bal_b_gos['BLENDED_ALBEDO'] = bal_b_gos['BLENDED_ALBEDO_BIN'].apply(lambda x: x.mid)

    ## ----------------------------------------- ##
    ## Plot
    ## ----------------------------------------- ##
    # Distributions
    fig, ax = fp.get_figax(aspect=1.75)
    for i, s in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
        print(s)
        d = data[data['SEASON'] == s]['OBS']
        TROPOMI_mean = d.mean()
        TROPOMI_std = d.std()
        ax.hist(d, histtype='step', density=True, bins=200,
                color=fp.color(i, lut=4), label=f'TROPOMI {s}')
        d = gosat_data[gosat_data['SEASON'] == s]['OBS']
        GOSAT_mean = d.mean()
        GOSAT_std = d.std()
        print(f'  Difference of means: {(TROPOMI_mean - GOSAT_mean):.2f}  TROPOMI std: {TROPOMI_std:.2f}  GOSAT std: {GOSAT_std:.2f}')
        ax.hist(d, histtype='step', density=True, bins=50,
                color=fp.color(i, lut=4), ls='--', label=f'GOSAT {s}')
    ax.set_xlim(1750, 1950)
    fp.add_legend(ax, bbox_to_anchor=(1, 0.5), loc='center left', ncol=1)
    ax = fp.add_labels(ax, 'XCH4', 'Density')
    ax = fp.add_title(ax, 'Blended albedo bias in TROPOMI - GOSAT')
    fp.save_fig(fig, plot_dir, f'gosat_tropomi_distributions{suffix}')
    plt.close()

    # Blended albedo
    fig, ax = fp.get_figax(aspect=1.75)
    ax.errorbar(ba_b_gos['BLENDED_ALBEDO'], ba_b_gos['mean'],
                yerr=ba_b_gos['std'], color=fp.color(4))
    ax.set_xticks(np.arange(0, 3.1, 0.25))
    ax.set_xlim(0, 2.25)
    ax.set_ylim(-30, 30)
    ax = fp.add_labels(ax, 'Blended albedo', 'TROPOMI - GOSAT')
    ax = fp.add_title(ax, 'Blended albedo bias in TROPOMI - GOSAT')
    fp.add_legend(ax, bbox_to_anchor=(1, 0.5),
                  loc='center left', ncol=1)
    fp.save_fig(fig, plot_dir, f'gosat_blended_albedo_bias{suffix}')
    plt.close()

    # Blended albedo and monthly bias
    fig, ax = fp.get_figax(aspect=1.75)
    ax.errorbar(ba_b_gos['BLENDED_ALBEDO'], ba_b_gos['mean'],
                yerr=ba_b_gos['std'], color=fp.color(4))
    ax.set_xticks(np.arange(0, 3.1, 0.25))
    ax.set_xlim(0, 2.25)
    ax.set_ylim(-30, 30)
    ax = fp.add_labels(ax, 'Blended albedo', 'TROPOMI - GOSAT')
    ax = fp.add_title(ax, 'Blended albedo bias in TROPOMI - GOSAT')
    for i, m in enumerate(np.unique(bam_b_gos['MONTH'])):
        d = bam_b_gos[bam_b_gos['MONTH'] == m]
        ax.plot(d['BLENDED_ALBEDO'].values, d['mean'].values,
                color=fp.color(i, lut=12), label=m, lw=0.5)
    fp.add_legend(ax, bbox_to_anchor=(1, 0.5),
                  loc='center left', ncol=1)
    fp.save_fig(fig, plot_dir, f'gosat_blended_albedo_bias_month{suffix}')
    plt.close()

    # Blended albedo and latitudinal bias
    fig, ax = fp.get_figax(aspect=1.75)
    ax.errorbar(ba_b_gos['BLENDED_ALBEDO'], ba_b_gos['mean'],
                yerr=ba_b_gos['std'], color=fp.color(4))
    ax.set_xticks(np.arange(0, 3.1, 0.25))
    ax.set_xlim(0, 2.25)
    ax.set_ylim(-30, 30)
    ax = fp.add_labels(ax, 'Blended albedo', 'TROPOMI - GOSAT')
    ax = fp.add_title(ax, 'Blended albedo bias in TROPOMI - GOSAT')
    for i, lb in enumerate(np.unique(bal_b_gos['LAT_BIN'])):
        d = bal_b_gos[bal_b_gos['LAT_BIN'] == lb]
        ax.plot(d['BLENDED_ALBEDO'].values, d['mean'].values,
                color=fp.color(i, lut=10), label=lb, lw=0.5)
    fp.add_legend(ax, bbox_to_anchor=(1, 0.5), loc='center left', ncol=1)
    fp.save_fig(fig, plot_dir, f'gosat_blended_albedo_bias_lat{suffix}')
    plt.close()

    # Spatial difference
    fig, ax = fp.get_figax(rows=2, cols=2, maps=True,
                           lats=trop_grid.lats, lons=trop_grid.lons)
    for i, axis in enumerate(ax.flatten()):
        s = trop_grid.season.values[i]
        d = trop_grid.sel(season=s)['DIFF']
        c = d.plot(ax=axis, cmap='PuOr_r', vmin=-20, vmax=20,
                   add_colorbar=False)
        axis = fp.format_map(axis, settings.lats, settings.lons)
        axis = fp.add_title(axis, f'{s}')
    cax = fp.add_cax(fig, ax)
    cb = fig.colorbar(c, ax=ax, cax=cax)
    cb = fp.format_cbar(cb, r'TROPOMI - GOSAT')
    fp.save_fig(fig, plot_dir, f'gosat_spatial_bias{suffix}')
    plt.close()

    # Scatter plots
    fig, ax = fp.get_figax(aspect=1)
    for i, s in enumerate(np.unique(diff_grid['SEASON'])):
        d = diff_grid[diff_grid['SEASON'] == s]
        m, b, r, bias, std = gc.comparison_stats(d['GOSAT'].values,
                                                 d['TROPOMI'].values)
        systematic_bias = d['DIFF'].std()
        print(s)
        # print(f'  y = {m}x + {b}')
        print(f'  R2            : {(r**2):.2f}')
        print(f'  bias          : {bias:.2f}')
        print(f'  std           : {std:.2f}')
        print(f'  regional bias : {systematic_bias:.2f}')
        ax.scatter(d['GOSAT'], d['TROPOMI'], c=fp.color(i, lut=4), alpha=0.5,
                   s=5, label=s)
    ax = fp.plot_one_to_one(ax)
    fp.add_legend(ax, bbox_to_anchor=(1, 0.5), loc='center left', ncol=1)
    fp.save_fig(fig, plot_dir, 'gosat_tropomi_comparison')

## ------------------------------------------------------------------------ ##
## Analyze data
## ------------------------------------------------------------------------ ##
if analyze_biases:
    ## Spatial bias
    s_b = data.groupby(['LAT_CENTER', 'LON_CENTER',
                        'SEASON']).mean()[['DIFF', 'ALBEDO_SWIR', 'AOD']]
    s_b = s_b.to_xarray().rename({'LAT_CENTER' : 'lats',
                                 'LON_CENTER' : 'lons',
                                 'SEASON' : 'season'})
    print('Spatial bias analyzed.')

    ## Spatial distribution of blended albedo
    s_ba = data.groupby(['LAT_CENTER', 'LON_CENTER',
                         'SEASON']).mean()['BLENDED_ALBEDO']
    s_ba = s_ba.to_xarray().rename({'LAT_CENTER' : 'lats',
                                   'LON_CENTER' : 'lons',
                                   'SEASON' : 'season'})
    print('Spatial distribution of blended albedo analyzed.')

    ## Latitudinal bias
    data['LAT_BIN'] = pd.cut(data['LAT'], lat_bins)
    l_b = gc.group_data(data, groupby=['LAT_BIN'])
    print('Latitudinal bias analyzed.')

    ## Seasonality
    m_b = gc.group_data(data, groupby=['MONTH'])
    print('Monthly bias analyzed.')

    ## Latitudinal bias and seasonality
    lm_b = gc.group_data(data, groupby=['LAT_BIN', 'SEASON'])
    print('Seasonal latitudinal bias analyzed.')

    ## Albedo
    data['ALBEDO_BIN'] = pd.cut(data['ALBEDO_SWIR'], albedo_bins)
    a_b = gc.group_data(data, groupby=['ALBEDO_BIN'])
    print('Albedo bias analyzed.')

    ## Blended albedo
    data['BLENDED_ALBEDO_BIN'] = pd.cut(data['BLENDED_ALBEDO'],
                                        blended_albedo_bins)
    ba_b = gc.group_data(data, groupby=['BLENDED_ALBEDO_BIN'])
    bam_b = gc.group_data(data, groupby=['BLENDED_ALBEDO_BIN', 'MONTH'])
    print('Blended albedo bias analyzed.')

    ## -------------------------------------------------------------------- ##
    ## Plot data
    ## -------------------------------------------------------------------- ##
    if plot_dir is not None:
        ## ----------------------------------------- ##
        ## Scatter plot
        ## ----------------------------------------- ##
        fig, ax, c = gc.plot_comparison(data['OBS'].values, data['MOD'].values,
                                        lims=[1750, 1950], vmin=0, vmax=3e4,
                                        xlabel='Observation', ylabel='Model')
        ax.set_xticks(np.arange(1750, 2000, 100))
        ax.set_yticks(np.arange(1750, 2000, 100))
        fp.save_fig(fig, plot_dir, f'prior_bias{suffix}')
        plt.close()

        ## ----------------------------------------- ##
        ## Spatial bias
        ## ----------------------------------------- ##
        # Satellite observation difference
        fig, ax = fp.get_figax(rows=2, cols=2, maps=True,
                               lats=s_b.lats, lons=s_b.lons)
        for i, axis in enumerate(ax.flatten()):
            s = s_b.season.values[i]
            d = s_b.sel(season=s)['DIFF']
            c = d.plot(ax=axis, cmap='PuOr_r', vmin=-30, vmax=30,
                         add_colorbar=False)
            axis = fp.format_map(axis, d.lats, d.lons)
            axis = fp.add_title(axis, f'{s}')
        cax = fp.add_cax(fig, ax)
        cb = fig.colorbar(c, ax=ax, cax=cax)
        cb = fp.format_cbar(cb, r'Model - Observation')
        fp.save_fig(fig, plot_dir, f'prior_spatial_bias{suffix}')
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
        fp.save_fig(fig, plot_dir, f'blended_albedo_spatial{suffix}')
        plt.close()

        # Albedo
        fig, ax = fp.get_figax(rows=2, cols=2, maps=True,
                               lats=s_b.lats, lons=s_b.lons)
        for i, axis in enumerate(ax.flatten()):
            s = s_b.season.values[i]
            d = s_b.sel(season=s)['ALBEDO_SWIR']
            # d = d.where(d >= 0.75, 0)
            c = d.plot(ax=axis, cmap='plasma', vmin=0, vmax=0.3,
                       add_colorbar=False)
            axis = fp.format_map(axis, d.lats, d.lons)
            axis = fp.add_title(axis, f'{s}')
        cax = fp.add_cax(fig, ax)
        cb = fig.colorbar(c, ax=ax, cax=cax)
        cb = fp.format_cbar(cb, r'Shortwave infrared albedo')
        fp.save_fig(fig, plot_dir, f'albedo_spatial{suffix}')
        plt.close()

        # AOD
        fig, ax = fp.get_figax(rows=2, cols=2, maps=True,
                               lats=s_b.lats, lons=s_b.lons)
        for i, axis in enumerate(ax.flatten()):
            s = s_b.season.values[i]
            d = s_b.sel(season=s)['AOD']
            # d = d.where(d >= 0.75, 0)
            c = d.plot(ax=axis, cmap='plasma', vmin=0.005, vmax=0.05,
                       add_colorbar=False)
            axis = fp.format_map(axis, d.lats, d.lons)
            axis = fp.add_title(axis, f'{s}')
        cax = fp.add_cax(fig, ax)
        cb = fig.colorbar(c, ax=ax, cax=cax)
        cb = fp.format_cbar(cb, r'Aerosol optical depth')
        fp.save_fig(fig, plot_dir, f'aod_spatial{suffix}')
        plt.close()

        ## ----------------------------------------- ##
        ## Latitudinal bias
        ## ----------------------------------------- ##
        l_b['LAT'] = l_b['LAT_BIN'].apply(lambda x: x.mid)
        fig, ax = fp.get_figax(aspect=1.75)
        ax.errorbar(l_b['LAT'], l_b['mean'], yerr=l_b['std'],
                    color=fp.color(4))
        ax.set_xticks(np.arange(10, 70, 10))
        ax.set_xlim(10, 60)
        ax = fp.add_labels(ax, 'Latitude', 'Model - observation')
        ax = fp.add_title(ax, 'Latitudinal bias in prior run')
        fp.save_fig(fig, plot_dir, f'prior_latitudinal_bias{suffix}')

        ## ----------------------------------------- ##
        ## Monthly bias
        ## ----------------------------------------- ##
        m_b['month'] = pd.to_datetime(m_b['MONTH'], format='%m')
        m_b['month'] = m_b['month'].dt.month_name().str[:3]
        fig, ax = fp.get_figax(aspect=1.75)
        ax.errorbar(m_b['month'], m_b['mean'], yerr=m_b['std'],
                    color=fp.color(4))
        ax = fp.add_labels(ax, 'Month', 'Model - Observation')
        ax = fp.add_title(ax, 'Seasonal bias in prior run')
        fp.save_fig(fig, plot_dir, f'prior_seasonal_bias{suffix}')
        plt.close()

        ## ----------------------------------------- ##
        ## Seasonal latitudinal bias
        ## ----------------------------------------- ##
        lm_b['LAT'] = lm_b['LAT_BIN'].apply(lambda x: x.mid)
        fig, ax = fp.get_figax(aspect=1.75)
        ax.errorbar(l_b['LAT'], l_b['mean'], yerr=l_b['std'],
                    color=fp.color(4))
        ax.set_xticks(np.arange(10, 70, 10))
        ax.set_xlim(10, 60)
        ax = fp.add_labels(ax, 'Latitude', 'Model - observation')
        ax = fp.add_title(ax, f'Latitudinal bias in prior run')
        linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
        for i, season in enumerate(np.unique(lm_b['SEASON'])):
            d = lm_b[lm_b['SEASON'] == season]
            ax.plot(d['LAT'].values, d['mean'].values, color=fp.color(4),
                    label=season, ls=linestyles[i], lw=0.5)
        fp.add_legend(ax)
        fp.save_fig(fig, plot_dir,
                    f'prior_seasonal_latitudinal_bias{suffix}')
        plt.close()

        ## ----------------------------------------- ##
        ## Albedo bias
        ## ----------------------------------------- ##
        a_b['ALBEDO'] = a_b['ALBEDO_BIN'].apply(lambda x: x.mid)
        fig, ax = fp.get_figax(aspect=1.75)
        ax.errorbar(a_b['ALBEDO'], a_b['mean'], yerr=a_b['std'],
                    color=fp.color(4))
        ax.set_xticks(np.arange(0, 1, 0.2))
        ax = fp.add_labels(ax, 'Albedo', 'Model - observation')
        ax = fp.add_title(ax, 'Albedo bias in prior run')
        fp.save_fig(fig, plot_dir, f'prior_albedo_bias{suffix}')
        plt.close()

        ## ----------------------------------------- ##
        ## Seasonal blended albedo bias
        ## ----------------------------------------- ##
        ba_b['BLENDED_ALBEDO'] = ba_b['BLENDED_ALBEDO_BIN'].apply(lambda x: x.mid)
        bam_b['BLENDED_ALBEDO'] = bam_b['BLENDED_ALBEDO_BIN'].apply(lambda x: x.mid)
        fig, ax = fp.get_figax(aspect=1.75)
        ax.errorbar(ba_b['BLENDED_ALBEDO'], ba_b['mean'], yerr=ba_b['std'],
                    color=fp.color(4))
        ax.set_xticks(np.arange(0, 3.1, 0.25))
        ax.set_xlim(0, 2.25)
        ax.set_ylim(-30, 30)
        ax = fp.add_labels(ax, 'Blended albedo', 'Model - observation')
        ax = fp.add_title(ax, 'Blended albedo bias in prior run')
        # for i, m in enumerate(np.unique(bam_b['MONTH'])):
        #     d = bam_b[bam_b['MONTH'] == m]
        #     ax.plot(d['BLENDED_ALBEDO'].values, d['mean'].values,
        #             color=fp.color(i, lut=12), label=m, lw=0.5)
        # fp.add_legend(ax, bbox_to_anchor=(1, 0.5),
        #               loc='center left', ncol=1)
        fp.save_fig(fig, plot_dir, f'prior_blended_albedo_bias{suffix}')
        plt.close()

## ------------------------------------------------------------------------ ##
## Calculate So
## ------------------------------------------------------------------------ ##
if calculate_so:
    # We calculate the mean bias, observation, and precision on the GEOS-Chem
    # grid, accounting for the squaring of the precision
    suffix += err_suffix
    data['LAT_CENTER_L'] = lats_so[gc.nearest_loc(data['LAT'].values, lats_so)]
    data['LON_CENTER_L'] = lons_so[gc.nearest_loc(data['LON'].values, lons_so)]
    group_quantities = ['DIFF', 'OBS', 'PREC_SQ']
    data['PREC_SQ'] = data['PREC']**2
    res_err = data.groupby(groupby).mean()[group_quantities].reset_index()
    res_err['PREC_SQ'] **= 0.5

    # Rename the columns
    res_err = res_err.rename(columns={'DIFF' : 'AVG_DIFF',
                                      'OBS' : 'AVG_OBS',
                                      'PREC_SQ' : 'AVG_PREC'})

    # Merge this data back into the original data frame
    data = pd.merge(data, res_err, on=groupby, how='left')

    # Subtract the bias from the difference to calculate the residual error
    # This is equivalent to ZQ's eps quantity
    data['RES_ERR'] = data['DIFF'] - data['AVG_DIFF']

    # Next we calculate the average residual error
    avg_err = data.groupby(groupby).mean()['RES_ERR'].reset_index()
    avg_err = avg_err.rename(columns={'RES_ERR' : 'AVG_RES_ERR'})
    # ZQ saves out:
    # average residual error as err_month.pkl,
    # average observations as obs_month.pkl
    # average precision as prec_month.pkl

    # Now calculate the gridded variance and standard deviation of the
    # residual error. The standard deviation is weighted by the number
    # of observations in a grid cell because this will decrease the
    # error in a grid cell.
    # (sigma_squared and rrsd, respectively, in ZQ's code)
    data = pd.merge(data, avg_err, on=groupby, how='left')
    data['VAR'] = (data['RES_ERR'] - data['AVG_RES_ERR'])**2
    var = data.groupby(groupby).mean()[['VAR', 'OBS']].reset_index()
    var = var.rename(columns={'OBS' : 'AVG_OBS'})
    var['STD'] = var['VAR']**0.5/var['AVG_OBS'] # rrsd
    # # var['VAR'] is sigmasq

    # Merge these final variances back into the data (first removing
    # the initial variance calculation, since this was an intermediary)
    data = data.drop(columns=['VAR', 'AVG_OBS'])
    data = pd.merge(data, var, on=groupby, how='left')

    # Scale by the observations
    data['STD'] *= data['OBS']
    data['VAR'] = data['STD']**2

    # Where the variance calculated by the residual error method is less
    # than the precision squared value calculated above, set the error equal
    # to precision squared
    cond = data['VAR'] < data['PREC_SQ']
    print(f'We replace {cond.sum()} instances where the residual error is less than the instrumental error.')
    data.loc[:, 'SO'] = data['VAR']
    data.loc[cond, 'SO'] = data.loc[cond, 'PREC_SQ']

    # Where the variance is less than 100 ppb^2 (err = 10 pppb, set a threshoold
    cond_t = data['SO'] < err_min**2
    print(f'We replace {cond_t.sum()} instances where the residual error is less than {err_min} ppb.')
    data.loc[cond_t, 'SO'] = err_min**2

    # and then update std
    data.loc[:, 'STD'] = data['SO']**0.5

    err_mean = data['STD'].mean()
    print(f'We find a mean error of {err_mean:.2f} ppb.' )

    # Save out the data
    print(f'Saving data with So in {output_dir}/{settings.year}_corrected.pkl')
    gc.save_obj(data, join(output_dir, f'{settings.year}_corrected.pkl'))

## ------------------------------------------------------------------------ ##
## Plots
## ------------------------------------------------------------------------ ##
if (plot_dir is not None) and calculate_so:
    plot_data = copy.deepcopy(data[['iGC', 'jGC', 'MONTH', 'LON', 'LAT',
                                    'OBS', 'MOD', 'DIFF', 'PREC',
                                    'ALBEDO_SWIR', 'BLENDED_ALBEDO', 'SO']])

    # Save nearest latitude and longitude centers
    lat_idx = gc.nearest_loc(plot_data['LAT'].values, lats)
    plot_data.loc[:, 'LAT_CENTER'] = lats[lat_idx]
    lon_idx = gc.nearest_loc(plot_data['LON'].values, lons)
    plot_data.loc[:, 'LON_CENTER'] = lons[lon_idx]

    # (and seasonally, just to show the variability in coverage)
    plot_data.loc[:, 'SEASON'] = 'DJF'
    plot_data.loc[plot_data['MONTH'].isin([3, 4, 5]), 'SEASON'] = 'MAM'
    plot_data.loc[plot_data['MONTH'].isin([6, 7, 8]), 'SEASON'] = 'JJA'
    plot_data.loc[plot_data['MONTH'].isin([9, 10, 11]), 'SEASON'] = 'SON'

    # Also calculate seasonal errors
    # We calculate the mean bias, observation, and precision on the GEOS-Chem
    # grid, accounting for the squaring of the precision
    groupby = ['LAT_CENTER', 'LON_CENTER', 'SEASON']
    group_quantities = ['DIFF', 'OBS', 'PREC_SQ']
    plot_data['PREC_SQ'] = plot_data['PREC']**2
    res_err = plot_data.groupby(groupby).mean()[group_quantities].reset_index()
    res_err['PREC_SQ'] **= 0.5

    # Rename the columns
    res_err = res_err.rename(columns={'DIFF' : 'AVG_DIFF',
                                      'OBS' : 'AVG_OBS',
                                      'PREC_SQ' : 'AVG_PREC'})

    # Merge this plot_data back into the original plot_data frame
    plot_data = pd.merge(plot_data, res_err, on=groupby, how='left')

    # Subtract the bias from the difference to calculate the residual error
    # This is equivalent to ZQ's eps quantity
    plot_data['RES_ERR'] = plot_data['DIFF'] - plot_data['AVG_DIFF']

    # Next we calculate the average residual error
    avg_err = plot_data.groupby(groupby).mean()['RES_ERR'].reset_index()
    avg_err = avg_err.rename(columns={'RES_ERR' : 'AVG_RES_ERR'})

    # Now calculate the gridded variance and standard deviation of the
    # residual error. The standard deviation is weighted by the number
    # of observations in a grid cell because this will decrease the
    # error in a grid cell.
    # (sigma_squared and rrsd, respectively, in ZQ's code)
    plot_data = pd.merge(plot_data, avg_err, on=groupby, how='left')
    plot_data['VAR'] = (plot_data['RES_ERR'] - plot_data['AVG_RES_ERR'])**2
    d_p = plot_data.groupby(groupby).agg({'VAR' : 'mean',
                                          'OBS' : 'mean',
                                          'DIFF' : 'mean',
                                          'RES_ERR' : 'count'})

    d_p = d_p.rename(columns={'OBS' : 'AVG_OBS', 'DIFF' : 'AVG_DIFF',
                              'RES_ERR' : 'COUNT'})
    # d_p = copy.deepcopy(plot_data)
    d_p['STD'] = d_p['VAR']**0.5#/d_p['AVG_OBS']
    d_p = d_p[['STD', 'AVG_OBS', 'AVG_DIFF', 'COUNT']].to_xarray()
    d_p = d_p.rename({'LAT_CENTER' : 'lats', 'LON_CENTER' : 'lons'})

    fig, ax = fp.get_figax(rows=2, cols=4, maps=True,
                           lats=d_p.lats, lons=d_p.lons,
                           max_width=config.PRES_WIDTH*config.SCALE*1.5,
                           max_height=config.PRES_HEIGHT*config.SCALE*1.5)
    plt.subplots_adjust(hspace=-0.25, wspace=0.05)

    # fig_c, ax_c = fp.get_figax(rows=1, cols=4, maps=True,
    #                            lats=d_p.lats, lons=d_p.lons,
    #                            max_width=config.PRES_WIDTH*config.SCALE*1.5,
    #                            max_height=config.PRES_HEIGHT*config.SCALE*1.5)
    fig_e, ax_e = fp.get_figax(rows=1, cols=4, maps=True,
                               lats=d_p.lats, lons=d_p.lons,
                               max_width=config.PRES_WIDTH*config.SCALE*1.5,
                               max_height=config.PRES_HEIGHT*config.SCALE*1.5)
    fig_cb, ax_cb = fp.get_figax(rows=1, cols=4, maps=True,
                                 lats=d_p.lats, lons=d_p.lons,
                                 max_width=config.PRES_WIDTH*config.SCALE*1.5,
                                 max_height=config.PRES_HEIGHT*config.SCALE*1.5)
    for i, s in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
        d = d_p.where(d_p.SEASON == s, drop=True)

        c = d['AVG_OBS'].plot(ax=ax[0, i], cmap='plasma', vmin=1800, vmax=1900,
                              add_colorbar=False)
        c_c = d['COUNT'].plot(ax=ax[1, i], cmap='afmhot', vmin=0, vmax=300,
                              add_colorbar=False)
        c_e = d['STD'].plot(ax=ax_e[i], cmap='plasma', vmin=0, vmax=25,
                            add_colorbar=False)
        # c_c = d['COUNT'].plot(ax=ax_c[i], cmap='afmhot', vmin=0, vmax=300,
        #                       add_colorbar=False)
        c_cb = d['AVG_DIFF'].plot(ax=ax_cb[i], cmap='PuOr_r',
                                  vmin=-30, vmax=30, add_colorbar=False)
        for j, axis in enumerate([ax[0, i], ax_e[i], ax_cb[i]]):
            axis = fp.format_map(axis, d.lats, d.lons)
            axis = fp.add_title(axis, s)
        ax[1, i] = fp.format_map(ax[1, i], d.lats, d.lons)
        ax[1, i] = fp.add_title(ax[1, i], '')

    cax = fp.add_cax(fig, ax[0, :], cbar_pad_inches=0.075)
    cb = fig.colorbar(c, ax=ax[0, :], cax=cax)
    cb = fp.format_cbar(cb, 'XCH4 (ppb)')

    cax_c = fp.add_cax(fig, ax[1, :], cbar_pad_inches=0.075)
    cb_c = fig.colorbar(c_c, ax=ax[1, :], cax=cax_c, ticks=[50, 150, 250])
    cb_c = fp.format_cbar(cb_c, 'Count')
    # fp.save_fig(fig_c, plot_dir, f'counts{suffix}')

    fp.save_fig(fig, plot_dir, f'observations{suffix}')

    cax_e = fp.add_cax(fig_e, ax_e)
    cb_e = fig.colorbar(c_e, ax=ax_e, cax=cax_e)
    cb_e = fp.format_cbar(cb_e, 'St. Dev.\n(ppb)')
    fp.save_fig(fig_e, plot_dir, f'errors{suffix}')

    cax_cb = fp.add_cax(fig_cb, ax_cb)
    cb_cb = fig.colorbar(c_cb, ax=ax_cb, cax=cax_cb)
    cb_cb = fp.format_cbar(cb_cb, 'GC-TROPOMI\n(ppb)')
    fp.save_fig(fig_cb, plot_dir, f'diff{suffix}')

    # Now plot the histograms
    hist_bins = np.arange(0, 26, 0.5)

    # Standard
    fig, ax = fp.get_figax(aspect=1.75)
    # ax.hist(data['STD'], bins=hist_bins, density=True, color=fp.color(4))
    data['STD'].plot(ax=ax, kind='density', ind=100, color=fp.color(4))
    ax.set_xlim(0, 25)
    ax = fp.add_labels(ax, 'Observational Error (ppb)', 'Count')
    ax = fp.add_title(ax, 'Observational Error')
    fp.save_fig(fig, plot_dir, f'observational_error{err_suffix}')

    # SEASONAL
    fig, ax = fp.get_figax(aspect=1.75)
    ax.set_xlim(0, 25)
    ax = fp.add_labels(ax, 'Observational Error (ppb)', 'Count')
    ax = fp.add_title(ax, 'Observational Error')
    for i, season in enumerate(np.unique(data['SEASON'])):
        hist_data = data[data['SEASON'] == season]['STD']
        # ax.hist(hist_data, histtype='step', bins=hist_bins, label=season,
        #         color=fp.color(2+2*i), lw=1)
        hist_data.plot(ax=ax, kind='density', ind=100,
                       color=fp.color(2+2*i), lw=1, label=season)
        ax.axvline(hist_data.mean(), color=fp.color(2+2*i), lw=1, ls=':')
    ax = fp.add_legend(ax)
    fp.save_fig(fig, plot_dir, f'observational_error_seasonal_hist{err_suffix}')

    # LATITUDE
    fig, ax = fp.get_figax(aspect=1.75)
    ax.set_xlim(0, 25)
    ax = fp.add_labels(ax, 'Observational Error (ppb)', 'Count')
    ax = fp.add_title(ax, 'Observational Error')
    data['LAT_BIN_HIST'] = pd.cut(data['LAT'], np.arange(10, 70, 10))
    for i, lat_bin in enumerate(np.unique(data['LAT_BIN_HIST'])):
        hist_data = data[data['LAT_BIN_HIST'] == lat_bin]['STD']
        # ax.hist(hist_data, histtype='step', bins=hist_bins, label=lat_bin,
        #         color=fp.color(2*i), lw=1)
        hist_data.plot(ax=ax, kind='density', ind=100,
                       color=fp.color(2*i), lw=1, label=lat_bin)
        ax.axvline(hist_data.mean(), color=fp.color(2*i), lw=1, ls=':')

    ax = fp.add_legend(ax)
    fp.save_fig(fig, plot_dir, f'observational_error_latitude_hist{err_suffix}')

    # fig, ax = fp.get_figax(aspect=1.75)
    # ax.scatter(data['LAT'], data['STD'], c=fp.color(4), s=2, alpha=0.1)
    # ax = fp.add_labels(ax, 'Latitude', 'Observational Error (ppb)')
    # ax = fp.add_title(ax, 'Observational Error')
    # fp.save_fig(fig, plot_dir, f'observational_error_latitude_scatter{err_suffix}')

    # ALBEDO
    fig, ax = fp.get_figax(aspect=1.75)
    ax.set_xlim(0, 25)
    ax = fp.add_labels(ax, 'Observational Error (ppb)', 'Count')
    ax = fp.add_title(ax, 'Observational Error')
    data['ALBEDO_BIN_HIST'] = pd.cut(data['ALBEDO_SWIR'],
                                     np.arange(0, 1.25, 0.25))
    for i, alb_bin in enumerate(np.unique(data['ALBEDO_BIN_HIST'])):
        hist_data = data[data['ALBEDO_BIN_HIST'] == alb_bin]['STD']
        # ax.hist(hist_data, histtype='step', bins=hist_bins, label=alb_bin,
        #         color=fp.color(2*i), lw=1)
        hist_data.plot(ax=ax, kind='density', ind=100,
                       color=fp.color(2*i), lw=1, label=alb_bin)
        ax.axvline(hist_data.mean(), color=fp.color(2*i), lw=1, ls=':')

    ax = fp.add_legend(ax)
    fp.save_fig(fig, plot_dir, f'observational_error_albedo_hist{err_suffix}')

    # fig, ax = fp.get_figax(aspect=1.75)
    # ax.scatter(data['ALBEDO_SWIR'], data['STD'], c=fp.color(4), s=2, alpha=0.1)
    # ax = fp.add_labels(ax, 'Albedo', 'Observational Error (ppb)')
    # ax = fp.add_title(ax, 'Observational Error')
    # fp.save_fig(fig, plot_dir, f'observational_error_albedo_scatter{err_suffix}')

## ------------------------------------------------------------------------ ##
## Save out inversion quantities
## ------------------------------------------------------------------------ ##
print(f'Saving data in {output_dir}')

y = xr.DataArray(data['OBS'], dims=('nobs'))
y.to_netcdf(join(output_dir, 'y.nc'))

ya = xr.DataArray(data['MOD'], dims=('nobs'))
ya.to_netcdf(join(output_dir, 'ya.nc'))

if calculate_so:
    so = xr.DataArray(data['SO'], dims=('nobs'))
    so.to_netcdf(join(output_dir, f'so{err_suffix}_{err_min}t.nc'))

print('=== CODE COMPLETE ====')
