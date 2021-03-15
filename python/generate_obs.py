'''
This python script generates the observation vector and the observational
error covariance matrix.

Inputs:
    prior_run       This contains the output of the prior run simulation
                    after processing to apply the TROPOMI averaging kernel.
                    The data has not yet been filtered or bias corrected
                    other than removing the qa < 0.5 data.

                    The data can either be a list of files or an individual,
                    concatenated file
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
pd.set_option('display.max_columns', None)

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# Local preferences
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python'
data_dir = base_dir + 'observations'
output_dir = base_dir + 'inversion_data'
plot_dir = base_dir + 'plots'

# # Cannon preferences
# base_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/'
# code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'

# base_dir = sys.argv[1]
# code_dir = sys.argv[2]
# data_dir = f'{base_dir}ProcessedDir'
# output_dir = f'{base_dir}SummaryDir'
# plot_dir = None

# The prior_run can either be a list of files or a single file
# with all of the data for simulation
year = 2019
months = np.arange(1, 13, 1) # excluding December for now
days = np.arange(1, 32, 1)
prior_run = f'{year}.pkl'
# prior_run = [f'{year}{mm:02d}{dd:02d}_GCtoTROPOMI.pkl'
#              for mm in months for dd in days]
# prior_run.sort()

# Define the blended albedo threshold
filter_on_blended_albedo = True
blended_albedo_threshold = 1
albedo_bins = np.arange(0, 1.1, 0.1)

# Remove latitudinal bias
remove_latitudinal_bias = True

# Which analyses do you wish to perform?
analyze_biases = False
calculate_so = True

# Information on the grid
lat_bins = np.arange(10, 65, 5)
lat_min = 9.75
lat_max = 60
lat_delta = 0.25
lon_min = -130
lon_max = -60
lon_delta = 0.3125
buffers = [3, 3, 3, 3]

## ------------------------------------------------------------------------ ##
## Import custom packages
## ------------------------------------------------------------------------ ##
# Custom packages
sys.path.append(code_dir)
import config
# config.SCALE = config.PRES_SCALE
# config.BASE_WIDTH = config.PRES_WIDTH
# config.BASE_HEIGHT = config.PRES_HEIGHT
import gcpy as gc
import troppy as tp
import format_plots as fp

# Define an empty suffix for file names
suffix = ''
if filter_on_blended_albedo:
    suffix += '_BAF'
if remove_latitudinal_bias:
    suffix += '_BC'

## ------------------------------------------------------------------------ ##
## Calculate y
## ------------------------------------------------------------------------ ##
if type(prior_run) == list:
    ## ----------------------------------------- ##
    ## Load data for the year
    ## ----------------------------------------- ##
    data = np.array([]).reshape(0, 12)
    for file in prior_run:
        # Check if that file is in the data directory
        if file not in listdir(data_dir):
            print(f'{file} is not in the data directory.')
            continue

        # Get the month
        month = int(file[4:6])

        # Load the data. The columns are: 0 OBS, 1 MOD, 2 LON, 3 LAT,
        # 4 iGC, 5 jGC, 6 PRECISION, 7 ALBEDO_SWIR, 8 ALBEDO_NIR, 9 AOD,
        # 10 MOD_COL
        new_data = gc.load_obj(join(data_dir, file))['obs_GC']
        new_data = np.insert(new_data, 11, month, axis=1) # add month

        data = np.concatenate((data, new_data))

    ## ----------------------------------------- ##
    ## Basic data formatting
    ## ----------------------------------------- ##
    # Create a dataframe from the data
    columns = ['OBS', 'MOD', 'LON', 'LAT', 'iGC', 'jGC', 'PREC',
               'ALBEDO_SWIR', 'ALBEDO_NIR', 'AOD',
               'MOD_COL', 'MONTH']
    data = pd.DataFrame(data, columns=columns)

    # Calculate blended albedo
    data['BLENDED_ALBEDO'] = tp.blended_albedo(data,
                                               data['ALBEDO_SWIR'],
                                               data['ALBEDO_NIR'])

    # Subset data
    data = data[['iGC', 'jGC', 'MONTH', 'LON', 'LAT', 'OBS', 'MOD',
                 'PREC', 'ALBEDO_SWIR', 'BLENDED_ALBEDO']]

    # Calculate model - observation
    data['DIFF'] = data['MOD'] - data['OBS']

    # Print out some statistics
    cols = ['LAT', 'LON', 'MONTH', 'MOD']
    print('MODEL MAXIMUM : ', data['MOD'].max())
    print('MODEL MINIMUM : ', data['MOD'].min())
    print('TROPOMI MAXIMUM : ', data['OBS'].max())
    print('TROPOMI MINIMUM : ', data['OBS'].min())
    print('DIFFERENCE MAXIMUM : ', data['DIFF'].max())

    # Save the data out
    gc.save_obj(data, join(output_dir, f'{year}.pkl'))

else:
    ## ----------------------------------------- ##
    ## Load data for the year
    ## ----------------------------------------- ##
    data = gc.load_obj(join(data_dir, prior_run))

print('Data is loaded.')

## ----------------------------------------- ##
## Additional data corrections
## ----------------------------------------- ##
if filter_on_blended_albedo:
    # Plot values
    if plot_dir is not None:
        for m in months:
            d = data[data['MONTH'] == m]
            fig, ax = fp.get_figax(aspect=1.75)
            ax.scatter(d['BLENDED_ALBEDO'], d['OBS'],
                       c=fp.color(4), s=3, alpha=0.5)
            ax.axvline(blended_albedo_threshold, c=fp.color(7), ls='--')
            ax.set_xlim(0, 2)
            ax = fp.add_title(ax, cal.month_name[m])
            ax = fp.add_labels(ax, 'Blended Albedo', 'XCH4 (ppb)')
            fp.save_fig(fig, plot_dir,
                        f'blended_albedo_filter_{m:02d}{suffix}')

    # Apply the blended albedo filter
    old_nobs = data.shape[0]
    data = data[data['BLENDED_ALBEDO'] < blended_albedo_threshold]
    nobs = data.shape[0]

    # Print information
    print(f'\nData is filtered on blended albedo < {blended_albedo_threshold}.')
    print(f'    {100*nobs/old_nobs:.1f}% of data is preserved.')
    print(f'    TROPOMI statistics:')
    summ = data.groupby('MONTH')[['OBS', 'DIFF']]
    summ = pd.concat([summ.min(), summ.max()], axis=1)
    print(summ)

if remove_latitudinal_bias:
    # Correct the latitudinal bias
    coef = p.polyfit(data['LAT'], data['DIFF'], deg=1)
    bias_correction = p.polyval(data['LAT'], coef)
    data['MOD'] -= bias_correction
    data['DIFF'] -= bias_correction

    # Print information
    print(f'\nData has latitudinal bias removed.')
    print(f'    y = {coef[0]:.2f} + {coef[1]:.2f}x')
    print(f'    Model statistics:')
    summ = data.groupby('MONTH')[['MOD', 'DIFF']]
    summ = pd.concat([summ.min(), summ.max()], axis=1)
    print(summ)

# Save out result
if filter_on_blended_albedo or remove_latitudinal_bias:
    gc.save_obj(data, join(output_dir, f'{year}_corrected.pkl'))

## ------------------------------------------------------------------------ ##
## Analyze data
## ------------------------------------------------------------------------ ##
if analyze_biases:
    ## Spatial bias
    # Get information on the lats and lons (edges of the domain)
    lat_e, lon_e = gc.adjust_grid_bounds(lat_min, lat_max, lat_delta,
                                         lon_min, lon_max, lon_delta, buffers)

    # Generate a grid on which to save out average difference
    lats, lons = gc.create_gc_grid(*lat_e, lat_delta, *lon_e, lon_delta,
                                   centers=False, return_xarray=False)

    # Save nearest latitude and longitude centers
    data['LAT_CENTER'] = lats[gc.nearest_loc(data['LAT'].values, lats)]
    data['LON_CENTER'] = lons[gc.nearest_loc(data['LON'].values, lons)]

    # Group on that grid
    s_b = data.groupby(['LAT_CENTER', 'LON_CENTER']).mean()['DIFF']
    s_b = s_b.to_xarray().rename({'LAT_CENTER' : 'lats',
                                 'LON_CENTER' : 'lons'})
    print('Spatial bias analyzed.')

    ## Latitudinal bias
    data['LAT_BIN'] = pd.cut(data['LAT'], lat_bins)
    l_b = gc.group_data(data, groupby=['LAT_BIN'])
    print('Latitudinal bias analyzed.')

    ## Seasonality
    m_b = gc.group_data(data, groupby=['MONTH'])
    print('Monthly bias analyzed.')

    ## Albedo
    data['ALBEDO_BIN'] = pd.cut(data['ALBEDO_SWIR'], albedo_bins)
    a_b = gc.group_data(data, groupby=['ALBEDO_BIN'])
    print('Albedo bias analyzed.')

    ## -------------------------------------------------------------------- ##
    ## Plot data
    ## -------------------------------------------------------------------- ##

    ## ----------------------------------------- ##
    ## Scatter plot
    ## ----------------------------------------- ##
    fig, ax, c = gc.plot_comparison(data['OBS'].values, data['MOD'].values,
                                    lims=[1750, 1950], vmin=0, vmax=3e4,
                                    xlabel='Observation', ylabel='Model')
    ax.set_xticks(np.arange(1750, 2000, 100))
    ax.set_yticks(np.arange(1750, 2000, 100))
    fp.save_fig(fig, plot_dir, f'prior_bias{suffix}')

    ## ----------------------------------------- ##
    ## Spatial bias
    ## ----------------------------------------- ##
    fig, ax = fp.get_figax(maps=True, lats=s_b.lats, lons=s_b.lons)
    c = s_b.plot(ax=ax, cmap='RdBu_r', vmin=-30, vmax=30, add_colorbar=False)
    cax = fp.add_cax(fig, ax)
    cb = fig.colorbar(c, ax=ax, cax=cax)
    cb = fp.format_cbar(cb, 'Model - Observation')
    ax = fp.format_map(ax, s_b.lats, s_b.lons)
    ax = fp.add_title(ax, 'Spatial Bias in Prior Run')
    fp.save_fig(fig, plot_dir, f'prior_spatial_bias{suffix}')

    ## ----------------------------------------- ##
    ## Latitudinal bias
    ## ----------------------------------------- ##
    l_b['LAT'] = l_b['LAT_BIN'].apply(lambda x: x.mid)
    fig, ax = fp.get_figax(aspect=1.75)
    ax.errorbar(l_b['LAT'], l_b['mean'], yerr=l_b['std'], color=fp.color(4))
    ax.set_xticks(np.arange(10, 70, 10))
    ax = fp.add_labels(ax, 'Latitude', 'Model - Observation')
    ax = fp.add_title(ax, 'Latitudinal Bias in Prior Run')
    fp.save_fig(fig, plot_dir, f'prior_latitudinal_bias{suffix}')

    ## ----------------------------------------- ##
    ## Monthly bias
    ## ----------------------------------------- ##
    m_b['month'] = pd.to_datetime(m_b['MONTH'], format='%m')
    m_b['month'] = m_b['month'].dt.month_name().str[:3]
    fig, ax = fp.get_figax(aspect=1.75)
    ax.errorbar(m_b['month'], m_b['mean'], yerr=m_b['std'], color=fp.color(4))
    ax = fp.add_labels(ax, 'Month', 'Model - Observation')
    ax = fp.add_title(ax, 'Seasonal Bias in Prior Run')
    fp.save_fig(fig, plot_dir, f'prior_seasonal_bias{suffix}')

    ## ----------------------------------------- ##
    ## Albedo bias
    ## ----------------------------------------- ##
    a_b['ALBEDO'] = a_b['ALBEDO_BIN'].apply(lambda x: x.mid)
    fig, ax = fp.get_figax(aspect=1.75)
    ax.errorbar(a_b['ALBEDO'], a_b['mean'], yerr=a_b['std'], color=fp.color(4))
    ax.set_xticks(np.arange(0, 1, 0.2))
    ax = fp.add_labels(ax, 'Albedo', 'Model - Observation')
    ax = fp.add_title(ax, 'Albedo Bias in Prior Run')
    fp.save_fig(fig, plot_dir, f'prior_albedo_bias{suffix}')

## ------------------------------------------------------------------------ ##
## Calculate So
## ------------------------------------------------------------------------ ##
if calculate_so:
    # We calculate the mean bias, observation, and precision on the GEOS-Chem
    # grid, accounting for the squaring of the precision
    groupby = ['iGC', 'jGC', 'MONTH']
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
    var['STD'] = var['VAR']**0.5/var['AVG_OBS']

    # Merge these final variances back into the data (first removing
    # the initial variance calculation, since this was an intermediary)
    data = data.drop(columns=['VAR', 'AVG_OBS'])
    data = pd.merge(data, var, on=groupby, how='left')

    # Scale by the observations
    data['VAR'] = data['VAR']*data['OBS']

    # Where the variance calculated by the residual error method is less
    # than the precision squared value calculated above, set the error equal
    # to precision squared
    cond = data['VAR'] < data['PREC_SQ']
    data.loc[:, 'SO'] = data['VAR']
    data.loc[cond, 'SO'] = data.loc[cond, 'PREC_SQ']

## ------------------------------------------------------------------------ ##
## Plots
## ------------------------------------------------------------------------ ##
if plot_dir is not None:
    plot_data = copy.deepcopy(data[['iGC', 'jGC', 'MONTH', 'LON', 'LAT',
                                    'OBS', 'MOD', 'DIFF', 'PREC',
                                    'ALBEDO_SWIR', 'BLENDED_ALBEDO', 'SO']])

    # Observations grouped on the grid
    # Get information on the lats and lons (edges of the domain)
    lat_e, lon_e = gc.adjust_grid_bounds(lat_min, lat_max, lat_delta,
                                         lon_min, lon_max, lon_delta, buffers)

    # Generate a grid on which to save out average difference
    lats, lons = gc.create_gc_grid(*lat_e, lat_delta, *lon_e, lon_delta,
                                   centers=False, return_xarray=False)

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
    d_p = plot_data.groupby(groupby).mean()[['VAR', 'OBS']]#.reset_index()
    d_p = d_p.rename(columns={'OBS' : 'AVG_OBS'})
    d_p['STD'] = d_p['VAR']**0.5#/var['AVG_OBS']
    d_p = d_p[['STD', 'AVG_OBS']].to_xarray().rename({'LAT_CENTER' : 'lats',
                                                      'LON_CENTER' : 'lons'})

    fig, ax = fp.get_figax(rows=1, cols=4, maps=True,
                           lats=d_p.lats, lons=d_p.lons)
    fig_e, ax_e = fp.get_figax(rows=1, cols=4, maps=True,
                               lats=d_p.lats, lons=d_p.lons)
    for i, s in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
        d = d_p.where(d_p.SEASON == s, drop=True)

        c = d['AVG_OBS'].plot(ax=ax[i], cmap='plasma', vmin=1800, vmax=1900,
                              add_colorbar=False)
        c_e = d['STD'].plot(ax=ax_e[i], cmap='plasma', vmin=0, vmax=25,
                            add_colorbar=False)
        for j, axis in enumerate([ax[i], ax_e[i]]):
            axis = fp.format_map(axis, d.lats, d.lons)
            axis = fp.add_title(axis, s)
    cax = fp.add_cax(fig, ax)
    cb = fig.colorbar(c, ax=ax, cax=cax)
    cb = fp.format_cbar(cb, 'XCH4 (ppb)')
    fp.save_fig(fig, plot_dir, f'observations{suffix}')

    cax_e = fp.add_cax(fig_e, ax_e)
    cb_e = fig.colorbar(c_e, ax=ax_e, cax=cax_e)
    cb_e = fp.format_cbar(cb_e, 'St. Dev. (ppb)')
    fp.save_fig(fig_e, plot_dir, f'errors{suffix}')

## ------------------------------------------------------------------------ ##
## Save out inversion quantities
## ------------------------------------------------------------------------ ##
gc.save_obj(data['OBS'], join(output_dir, 'y.pkl'))
gc.save_obj(data['MOD'], join(output_dir, 'kxa.pkl'))
if calculate_so:
    gc.save_obj(data['SO'], join(output_dir, 'so.pkl'))

# print(data['OBS'])
# print(data['SO'])

# # print(data)
nobs = data['OBS'].shape[0]
print(f'Number of observations : {nobs}')
print('=== CODE COMPLETE ====')
