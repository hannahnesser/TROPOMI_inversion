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
# Perform the raw data analysis?
Analysis = True

# Do bias correction?
BiasCorrection = False

# Does a bias corrected file exist?
BiasCorrected = True
bias_corrected_data = '2019_bias_corrected.pkl'

# Make plots?
Plots = True
ScatterPlot = True
SpatialBiasPlot = True
LatBiasPlot = True
MonthlyBiasPlot = True
AlbedoBiasPlot = True

# Perform planeflight analysis?
PlaneFlight = False

# Set directories (this needs to be amended)
if Analysis:
    # analysis_code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
    # analysis_data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_pf_check/ProcessedDir'
    # processed_data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/SummaryDir'
    # analysis_processed_data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_pf_check/SummaryDir'
    analysis_code_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/python'
    analysis_data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/prior/prior_run'
    analysis_processed_data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/prior/prior_run'

if BiasCorrection:
    bias_code_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/python'
    bias_data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/prior/prior_run'
    bias_plot_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/plots'

if Plots:
    plot_code_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/python'
    plot_processed_data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/prior/prior_run'
    plot_plot_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/plots'

if PlaneFlight:
    pf_code_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/python'
    pf_data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/prior/plane_logs'
    pf_plot_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/plots'

# Define file name format
files = 'YYYYMMDD_GCtoTROPOMI.pkl'
pf_files = 'plane.log.YYYYMMDD'

# Set years, months, and dates
year = 2019
months = np.arange(1, 12, 1) # Excluding December for now
days = np.arange(1, 32, 1)

pf_year = 2018
pf_months = [5]

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

## -------------------------------------------------------------------------##
## Analysis
## -------------------------------------------------------------------------##
if Analysis:
    ## ----------------------------------------- ##
    ## Set code directories
    ## ----------------------------------------- ##
    code_dir = analysis_code_dir
    data_dir = analysis_data_dir
    processed_data_dir = analysis_processed_data_dir

    ## ----------------------------------------- ##
    ## Import additional packages
    ## ----------------------------------------- ##
    sys.path.append(code_dir)
    import config
    config.SCALE = config.PRES_SCALE
    config.BASE_WIDTH = config.PRES_WIDTH
    config.BASE_HEIGHT = config.PRES_HEIGHT
    import gcpy as gc
    import troppy as tp
    import format_plots as fp

    ## ----------------------------------------- ##
    ## Get grid information
    ## ----------------------------------------- ##
    # Get information on the lats and lons (edges of the domain)
    lat_edges, lon_edges = gc.adjust_grid_bounds(lat_min, lat_max, lat_delta,
                                                 lon_min, lon_max, lon_delta,
                                                 buffers)

    # Generate a grid on which to save out average difference
    lats, lons = gc.create_gc_grid(*lat_edges, lat_delta,
                                   *lon_edges, lon_delta,
                                   centers=False, return_xarray=False)

    ## ----------------------------------------- ##
    ## Load data for the year
    ## ----------------------------------------- ##
    if bias_corrected_data is None:
        data = np.array([]).reshape(0, 13)
        for m in months:
            print('Analyzing data for month %d' % m)
            for d in days:
                # Create the file name for that date
                file = files.replace('YYYY', '%04d' % year)
                file = file.replace('MM', '%02d' % m)
                file = file.replace('DD', '%02d' % d)

                # Check if that file is in the data directory
                if file not in listdir(data_dir):
                    print('Data for %04d-%02d-%02d is not in the data directory.'
                          % (year, m, d))
                    continue

                # Load the data
                # The columns are: 0 OBS, 1 MOD, 2 LON, 3 LAT,
                # 4 iGC, 5 jGC, 6 PRECISION, 7 ALBEDO_SWIR,
                # 8 ALBEDO_NIR, 9 AOD, 10 MOD_COL
                new_data = gc.load_obj(join(data_dir, file))['obs_GC']
                new_data = np.insert(new_data, 12, m, axis=1) # add month

                data = np.concatenate((data, new_data))

        ## ----------------------------------------- ##
        ## Basic data formatting
        ## ----------------------------------------- ##
        # Create a dataframe from the data
        data = pd.DataFrame(data, columns=['OBS', 'MOD', 'LON', 'LAT',
                                           'iGC', 'jGC', 'PRECISION',
                                           'ALBEDO_SWIR', 'ALBEDO_NIR',
                                           'AOD', 'MOD_COL', 'MOD_STRAT',
                                           'MONTH'])

        # Create a column for the blended albedo filter
        data['BLENDED_ALBEDO'] = tp.blended_albedo(data,
                                                   data['ALBEDO_SWIR'],
                                                   data['ALBEDO_NIR'])
        data['FILTER'] = (data['BLENDED_ALBEDO'] < 1)

        # Subset data
        data = data[['MONTH', 'LON', 'LAT', 'OBS', 'MOD',
                     'ALBEDO_SWIR', 'BLENDED_ALBEDO', 'FILTER']]

        # Calculate model - observation
        data['DIFF'] = data['MOD'] - data['OBS']

        # Calculate the nearest latitude & longitude center
        data['LAT_CENTER'] = lats[gc.nearest_loc(data['LAT'].values, lats)]
        data['LON_CENTER'] = lons[gc.nearest_loc(data['LON'].values, lons)]

        # Group the difference between model and observation by latitude bin
        data['LAT_BIN'] = pd.cut(data['LAT'], lat_bins)

        # Group by albedo bin
        data['ALBEDO_BIN'] = pd.cut(data['ALBEDO_SWIR'], albedo_bins)

        # Save the data
        gc.save_obj(data, join(processed_data_dir, '%d.pkl' % year))

        # Create a dictionary of total and filtered data
        data_dict = {'Total' : data, 'Filtered' : data[data['FILTER']]}
        suffix = ''

    else:
        data = gc.load_obj(join(data_dir, bias_corrected_data))
        data_dict = {'Filtered' : data}
        suffix = '_BC'

    ## ----------------------------------------- ##
    ## Calculate spatial bias
    ## ----------------------------------------- ##
    # Apply DIFF to a grid so that we can track the spatial
    # distribution of the average difference annually.

    # We group by the latitude and longitude centers and convert it
    # to an xarray. We do this for filtered and unfiltered data.
    data_grid = {}
    for k, d in data_dict.items():
        d_g = d.groupby(['LAT_CENTER', 'LON_CENTER'])
        d_g = d_g.mean()['DIFF']
        d_g = d_g.to_xarray().rename({'LAT_CENTER' : 'lats',
                                      'LON_CENTER' : 'lons'})
        d_g = d_g.rename(k)
        data_grid[k] = d_g
    data_grid = xr.Dataset(data_grid)

    file_name = '%04d_spatial_summary%s.nc' % (year, suffix)
    data_grid.to_netcdf(join(processed_data_dir, file_name))

    ## ----------------------------------------- ##
    ## Calculate latitudinal bias
    ## ----------------------------------------- ##
    lat_bias = pd.DataFrame(columns={'FILTER', 'LAT_BIN', 'count', 'mean',
                                     'std', 'rmse'})
    for k, d in data_dict.items():
        l_b = gc.group_data(d, groupby=['LAT_BIN'])
        l_b['FILTER'] = k
        lat_bias = lat_bias.append(l_b)

    # Save out latitudinal bias summary
    file_name = '%04d_lat_summary%s.csv' % (year, suffix)
    lat_bias.to_csv(join(processed_data_dir, file_name))

    ## ----------------------------------------- ##
    ## Calculate monthly bias
    ## ----------------------------------------- ##
    month_bias = pd.DataFrame(columns={'FILTER', 'MONTH', 'count', 'mean',
                                       'std', 'rmse'})
    for k, d in data_dict.items():
        m_b = gc.group_data(d, groupby=['MONTH'])
        m_b['FILTER'] = k
        month_bias = month_bias.append(m_b)

    # Save out seasonal bias summary
    file_name = '%04d_month_summary%s.csv' % (year, suffix)
    month_bias.to_csv(join(processed_data_dir, file_name))

    ## ----------------------------------------- ##
    ## Calculate albedo bias
    ## ----------------------------------------- ##
    albedo_bias = pd.DataFrame(columns={'FILTER', 'ALBEDO_BIN', 'count',
                                        'mean', 'std', 'rmse'})
    for k, d in data_dict.items():
        a_b = gc.group_data(d, groupby=['ALBEDO_BIN'])
        a_b['FILTER'] = k
        albedo_bias = albedo_bias.append(a_b)

    # Save out albedo bias summary
    file_name = '%04d_albedo_summary%s.csv' % (year, suffix)
    albedo_bias.to_csv(join(processed_data_dir, file_name))

    ## ----------------------------------------- ##
    ## Complete
    ## ----------------------------------------- ##
    print('Code Complete.')

## -------------------------------------------------------------------------##
## Bias Correction
## -------------------------------------------------------------------------##
if BiasCorrection:
    code_dir = bias_code_dir
    data_dir = bias_data_dir
    plot_dir = bias_plot_dir

    ## ----------------------------------------- ##
    ## Import additional packages
    ## ----------------------------------------- ##
    sys.path.append(code_dir)
    import config
    config.SCALE = config.PRES_SCALE
    config.BASE_WIDTH = config.PRES_WIDTH
    config.BASE_HEIGHT = config.PRES_HEIGHT
    import gcpy as gc
    import troppy as tp
    import format_plots as fp

    ## ----------------------------------------- ##
    ## Latitudinal bias correction
    ## ----------------------------------------- ##
    # Load raw data
    data = gc.load_obj(join(data_dir, '%04d.pkl' % year))
    data = data[data['FILTER']]

    # Plot
    lat_bias = pd.read_csv(join(data_dir,
                                '%04d_lat_summary.csv' % year),
                                index_col=0)
    lat_bias['LAT'] = (lat_bias['LAT_BIN'].str.split(r'\D', expand=True)
                       .iloc[:, [1, 3]]
                       .astype(float).mean(axis=1))

    # Isolate standard difference
    diff_cols = ['DIFF' + s for s in ['', '.1', '.2', '.3']]
    lat_bias_diff = lat_bias[diff_cols].reset_index(drop=True)
    lat_bias_diff.columns = lat_bias_diff.iloc[0]
    lat_bias_diff = lat_bias_diff[1:].reset_index(drop=True)
    lat_bias_diff = pd.concat([lat_bias_diff,
                               lat_bias[['FILTER', 'LAT']].iloc[1:].reset_index(drop=True)],
                              axis=1)

    # Amend data types
    for col in ['count', 'mean', 'std', 'rmse']:
        lat_bias_diff[col] = lat_bias_diff[col].astype(float)

    d = lat_bias_diff[lat_bias_diff['FILTER'] == 'Filtered']
    fig, ax = fp.get_figax(aspect=1.75)
    ax.errorbar(d['LAT'], d['mean'], yerr=d['std'], color=fp.color(4))
    ax.set_xticks(np.arange(10, 70, 10))
    ax.set_xlim(10, 60)
    ax = fp.add_labels(ax, 'Latitude', 'Model - Observation')
    ax = fp.add_title(ax, 'Latitudinal Bias in Prior Run')
    fp.save_fig(fig, plot_dir, 'prior_latitudinal_bias_filtered')

    # Best fit
    def poly(coef, x):
        y = 0
        for i, c in enumerate(coef):
            y += c*x**i
        return y

    coef, stats = p.polyfit(data['LAT'], data['DIFF'], 1, full=True)
    x = np.arange(lat_min, lat_max, lat_delta)
    y = poly(coef, x)

    ax.plot(x, y, color=fp.color(7), ls='--')
    ax.text(0.05, 0.9, 'y = %.2f x + %.2f' % (coef[1], coef[0]),
            fontsize=config.LABEL_FONTSIZE*config.SCALE,
            transform=ax.transAxes)

    fp.save_fig(fig, plot_dir, 'prior_latitudinal_bias_bestfit')

    # Correct data
    data['BIAS_CORRECTION'] = poly(coef, data['LAT'])
    data['MOD'] -= data['BIAS_CORRECTION']
    data['DIFF'] = data['MOD'] - data['OBS']

    # Save out
    gc.save_obj(data,
                join(data_dir, '%04d_bias_corrected.pkl' % year))


## -------------------------------------------------------------------------##
## Plots
## -------------------------------------------------------------------------##
if Plots:
    code_dir = plot_code_dir
    processed_data_dir = plot_processed_data_dir
    plot_dir = plot_plot_dir

    if BiasCorrected:
        data_file = '%04d_bias_corrected.pkl' % year
        options = ['Filtered']
        suffix = '_BC'
    else:
        data_file = '%04d.pkl' % year
        options = ['Total', 'Filtered']
        suffix = ''

    ## ----------------------------------------- ##
    ## Import additional packages
    ## ----------------------------------------- ##
    sys.path.append(code_dir)
    import config
    config.SCALE = config.PRES_SCALE
    config.BASE_WIDTH = config.PRES_WIDTH
    config.BASE_HEIGHT = config.PRES_HEIGHT
    import gcpy as gc
    import troppy as tp
    import format_plots as fp

    ## ----------------------------------------- ##
    ## Total summary
    ## ----------------------------------------- ##
    if ScatterPlot:
        ## ----------------------------------------- ##
        ## Determine blended albedo filter value
        ## ----------------------------------------- ##
        data = gc.load_obj(join(processed_data_dir, data_file))
        for m in months:
            d = data[data['MONTH'] == m]
            fig, ax = fp.get_figax(aspect=1.75)
            ax.scatter(d['BLENDED_ALBEDO'], d['OBS'], c=fp.color(4),
                       s=3, alpha=0.5)
            ax.axvline(1, c=fp.color(7), ls='--')
            ax = fp.add_title(ax, cal.month_name[m])
            ax = fp.add_labels(ax, 'Blended Albedo', 'XCH4 (ppb)')
            fp.save_fig(fig, plot_dir,
                        'blended_albedo_filter_%d%s' % (m, suffix))

        ## ----------------------------------------- ##
        ## And then plot scatter plot
        ## ----------------------------------------- ##
        for f in options:
            if f == 'Filtered':
                d = data[data['FILTER']]
            else:
                d = data

            fig, ax, c = gc.plot_comparison(d['OBS'].values, d['MOD'].values,
                                            lims=[1750, 1950],
                                            xlabel='Observation',
                                            ylabel='Model',
                                            vmin=0, vmax=3e4)
            ax.set_xticks(np.arange(1750, 2000, 100))
            ax.set_yticks(np.arange(1750, 2000, 100))
            fp.save_fig(fig, plot_dir,
                        'prior_bias_%s%s' % (f.lower(), suffix))

    ## ----------------------------------------- ##
    ## Spatial bias
    ## ----------------------------------------- ##
    if SpatialBiasPlot:
        spat_bias = xr.open_dataset(join(processed_data_dir,
                                         ('%04d_spatial_summary%s.nc'
                                         % (year, suffix))))

        # Make figures
        for f in options:
            d = spat_bias[f]
            fig, ax = fp.get_figax(maps=True,
                                   lats=spat_bias.lats, lons=spat_bias.lons)
            c = d.plot(ax=ax, cmap='RdBu_r', vmin=-30, vmax=30, add_colorbar=False)
            cax = fp.add_cax(fig, ax)
            cb = fig.colorbar(c, ax=ax, cax=cax)
            cb = fp.format_cbar(cb, 'Model - Observation')
            ax = fp.format_map(ax, spat_bias.lats, spat_bias.lons)
            ax = fp.add_title(ax, 'Spatial Bias in Prior Run')
            fp.save_fig(fig, plot_dir,
                        'prior_spatial_bias_%s%s' % (f.lower(), suffix))

    ## ----------------------------------------- ##
    ## Latitudinal bias
    ## ----------------------------------------- ##
    if LatBiasPlot:
        lat_bias = pd.read_csv(join(processed_data_dir,
                                    '%04d_lat_summary%s.csv' % (year, suffix)),
                               index_col=0)

        lat_bias['LAT'] = (lat_bias['LAT_BIN'].str.split(r'\D', expand=True)
                           .iloc[:, [1, 3]]
                           .astype(float).mean(axis=1))

        print(lat_bias)

        # Isolate standard difference
        if not BiasCorrected:
            diff_cols = ['DIFF' + s for s in ['', '.1', '.2', '.3']]
            lat_bias_diff = lat_bias[diff_cols].reset_index(drop=True)
            lat_bias_diff.columns = lat_bias_diff.iloc[0]
            lat_bias_diff = lat_bias_diff[1:].reset_index(drop=True)
            lat_bias_diff = pd.concat([lat_bias_diff,
                                       lat_bias[['FILTER', 'LAT']].iloc[1:].reset_index(drop=True)],
                                      axis=1)

            # Amend data types
            for col in ['count', 'mean', 'std', 'rmse']:
                lat_bias_diff[col] = lat_bias_diff[col].astype(float)
        else:
            lat_bias_diff = lat_bias

        # Make figures: standard lat bias difference
        for f in options:
            d = lat_bias_diff[lat_bias_diff['FILTER'] == f]
            fig, ax = fp.get_figax(aspect=1.75)
            ax.errorbar(d['LAT'], d['mean'], yerr=d['std'], color=fp.color(4))
            ax.set_xticks(np.arange(10, 70, 10))
            ax = fp.add_labels(ax, 'Latitude', 'Model - Observation')
            ax = fp.add_title(ax, 'Latitudinal Bias in Prior Run')
            fp.save_fig(fig, plot_dir, 'prior_latitudinal_bias_%s%s'
                        % (f.lower(), suffix))

    ## ----------------------------------------- ##
    ## Monthly bias
    ## ----------------------------------------- ##
    if MonthlyBiasPlot:
        month_bias = pd.read_csv(join(processed_data_dir,
                                      '%04d_month_summary%s.csv'
                                      % (year, suffix)),
                                 index_col=0)
        month_bias['month'] = pd.to_datetime(month_bias['MONTH'], format='%m')
        month_bias['month'] = month_bias['month'].dt.month_name().str[:3]

        # Make figures
        for f in options:
            d = month_bias[month_bias['FILTER'] == f]
            fig, ax = fp.get_figax(aspect=1.75)
            ax.errorbar(d['month'], d['mean'], yerr=d['std'],
                        color=fp.color(4))
            ax = fp.add_labels(ax, 'Month', 'Model - Observation')
            ax = fp.add_title(ax, 'Seasonal Bias in Prior Run')
            fp.save_fig(fig, plot_dir,
                        'prior_seasonal_bias_%s%s' % (f.lower(), suffix))

    ## ----------------------------------------- ##
    ## Albedo bias
    ## ----------------------------------------- ##
    if AlbedoBiasPlot:
        albedo_bias = pd.read_csv(join(processed_data_dir,
                                       '%04d_albedo_summary%s.csv'
                                       % (year, suffix)),
                                  index_col=0)
        albedo_bias['ALBEDO'] = (albedo_bias['ALBEDO_BIN'].str.split(r'\D',
                                                                  expand=True)
                                 .iloc[:, [2, 5]]
                                 .astype(float).mean(axis=1))/10

        # Make figures
        for f in ['Filtered']:#['Total', 'Filtered']:
            d = albedo_bias[albedo_bias['FILTER'] == f]
            fig, ax = fp.get_figax(aspect=1.75)
            ax.errorbar(d['ALBEDO'], d['mean'], yerr=d['std'],
                        color=fp.color(4))
            ax.set_xticks(np.arange(0, 1, 0.2))
            ax = fp.add_labels(ax, 'Albedo', 'Model - Observation')
            ax = fp.add_title(ax, 'Albedo Bias in Prior Run')
            fp.save_fig(fig, plot_dir,
                        'prior_albedo_bias_%s%s' % (f.lower(), suffix))

## -------------------------------------------------------------------------##
## Planeflight analysis
## -------------------------------------------------------------------------##
if PlaneFlight:
    code_dir = pf_code_dir
    data_dir = pf_data_dir
    plot_dir = pf_plot_dir

    ## ----------------------------------------- ##
    ## Import additional packages
    ## ----------------------------------------- ##
    sys.path.append(code_dir)
    import config
    config.SCALE = config.PRES_SCALE
    config.BASE_WIDTH = config.PRES_WIDTH
    config.BASE_HEIGHT = config.PRES_HEIGHT
    import gcpy as gc
    import troppy as tp
    import format_plots as fp

    ## ----------------------------------------- ##
    ## Load data
    ## ----------------------------------------- ##
    # # data = np.array([]).reshape(0, 13)
    # for m in pf_months:
    #     print('Analyzing data for month %d' % m)
    #     file_names = []
    #     for d in days:
    #         # Create the file name for that date
    #         file = pf_files.replace('YYYY', '%04d' % pf_year)
    #         file = file.replace('MM', '%02d' % m)
    #         file = file.replace('DD', '%02d' % d)

    #         # Check if that file is in the data directory
    #         if file not in listdir(data_dir):
    #             print('%s is not in the data directory.' % file)
    #             continue

    #         file_names.append(file)

    #     pf = gc.load_all_pf(file_names, data_dir)
    #     pf = gc.process_pf(pf)
    #     pf_gr = gc.group_by_gridbox(pf)
    #     print(pf.columns)

    # Read in data with troposphere boolean flag
    pf = pd.read_csv(join(data_dir, 'planeflight_total.csv'), index_col=0,
                     low_memory=False)

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
    ax_hist.set_facecolor('0.98')

    fp.save_fig(fig, plot_dir, 'planeflight_latitudinal_bias')
