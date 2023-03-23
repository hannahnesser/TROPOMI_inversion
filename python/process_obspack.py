# import glob
# import sys
# import xarray as xr
# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# from copy import deepcopy as dc

# sys.path.append('/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/python')
# import inversion_settings as s

## ------------------------------------------------------------------------ ##
## Prep files
## ------------------------------------------------------------------------ ##
# # Define the variables to keep
# data_vars = ['time', 'start_time', 'midpoint_time', 'time_components', 'value',
#              'latitude', 'longitude', 'altitude', 'assimilation_concerns',
#              'obspack_id']

# # Define a filtering function
# def filter_obspack(data):
#     # Subset variables
#     data = data[data_vars]

#     # Subset for time and location
#     data = data.where(data['time'].dt.year == 2019, drop=True)
#     data = data.where((data['latitude'] > s.lat_min) & 
#                       (data['latitude'] < s.lat_max),
#                       drop=True)
#     data = data.where((data['longitude'] > s.lon_min) & 
#                       (data['longitude'] < s.lon_max),
#                       drop=True)

#     # Save out a platform variable
#     platform = data.attrs['dataset_project'].split('-')[0]
#     data['platform'] = xr.DataArray([platform]*len(data.obs), dims=('obs'))

#     # Correct to local timezone if it's an in situ or surface observation
#     if (len(data.obs) > 0) and (platform in ['surface', 'tower']):
#         utc_conv = data.attrs['site_utc2lst']
#         if int(utc_conv) != utc_conv:
#             print('UTC CONVERSION FACTOR IS NOT AN INTEGER : ', data.attrs['dataset_name'])
#         data['utc_conv'] = xr.DataArray(utc_conv*np.ones(len(data.obs)),
#                                         dims=('obs'))
#         # data['time_ltc'] = dc(data['time']) + np.timedelta64(int(utc_conv), 'h')
#     else:
#         data['utc_conv'] = xr.DataArray(np.zeros(len(data.obs)), dims=('obs'))
#         # data['time_ltc'] = dc(data['time'])

#     return data

# # Get a list of the files
# obspack_dir = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/obspack'
# files = glob.glob(f'{obspack_dir}/*.nc')
# files = [f for f in files if f.split('/')[-1][:11] != 'obspack_ch4']
# files.sort()

# # Iterate through the files and see which are relevant to the domain
# conus_files = []
# platforms = []
# for i, f in enumerate(files):
#     op = xr.open_dataset(f)

#     # Only use files in the needed time, latitude, and longitude
#     # ranges
#     try:
#         op = filter_obspack(op)
#     except ValueError:
#         continue
#     except KeyError:
#         print(f)

#     # If the file is empty, continue through the loop
#     if len(op.obs) == 0:
#         continue

#     # If the file still has observations, append it to conus_files
#     conus_files.append(f)

#     # And get information on the platform
#     platforms.append(op.attrs['dataset_project'])

# # Sort the files
# conus_files.sort()

# # Now load all the files
# obspack = xr.open_mfdataset(conus_files, concat_dim='obs', combine='nested', 
#                             chunks=1e4, mask_and_scale=False, 
#                             preprocess=filter_obspack)

# # Check for the sampling strategy
# ## Get the time in hours of each sample
# obspack['obs_length'] = (obspack['time'] - obspack['start_time'])
# obspack['obs_length'] = obspack['obs_length'].dt.seconds*2/(60*60)

# ## Convert that to the sampling strategy flag
# ## ss = place holder for sampling strategy
# obspack['ss'] = xr.DataArray(999*np.ones(len(obspack.obs)), dims=('obs'))

# ## Closest to 4 hours
# obspack['ss'] = obspack['ss'].where(obspack['obs_length'] > 5.25, 1)

# ## Closest to 90 minutes
# obspack['ss'] = obspack['ss'].where(obspack['obs_length'] > 2.75, 3)

# ## Closest to 1 hour
# obspack['ss'] = obspack['ss'].where(obspack['obs_length'] > 1.25, 2)

# ## Closest to instantaneous
# obspack['ss'] = obspack['ss'].where(obspack['obs_length'] > 0.5, 4)

# ## Cast to int
# obspack['ss'] = obspack['ss'].astype(int)

# # Rename and add attributes
# obspack = obspack.rename({'ss' : 'CT_sampling_strategy'})
# obspack['CT_sampling_strategy'].attrs = {'_FillValue' : -9,
#                                          'long_name' : 'model sampling strategy',
#                                          'values' : 'How to sample model. 1=4-hour avg; 2=1-hour avg; 3=90-min avg; 4=instantaneous'}

# # Other clean up
# obspack.attrs = {}
# obspack = obspack.drop(['obs_length', 'start_time', 'midpoint_time'])

# # And iterate through the unique days
# name_str = 'obspack_ch4.2019'
# for mm in range(1, 13):
#     for dd in range(1, 32):
#         # Subset for that day
#         daily = obspack.where((obspack['time'].dt.month == mm) &
#                               (obspack['time'].dt.day == dd), drop=True)

#         # If there is no data, continue
#         if len(daily.obs) == 0:
#             continue

#         # Data type fix
#         daily['obspack_id'] = daily['obspack_id'].astype('S200')
#         daily['platform'] = daily['platform'].astype('S50')

#         # Time fix
#         daily['time'].encoding['units'] = 'seconds since 1970-01-01 00:00:00 UTC'
#         daily['time'].encoding['calendar'] = 'proleptic_gregorian'

#         # daily['time_ltc'].encoding['units'] = 'seconds since 1970-01-01 00:00:00 UTC'
#         # daily['time_ltc'].encoding['calendar'] = 'proleptic_gregorian'

#         # One last rename
#         # daily = daily.rename({'value' : 'obs'})

#         # Otherwise, save out
#         print(f'Saving 2019-{mm:02d}-{dd:02d}')
#         daily.to_netcdf(f'{obspack_dir}/{name_str}{mm:02d}{dd:02d}.nc',
#                          unlimited_dims=['obs'])

# ------------------------------------------------------------------------ ##
# Process output
# ------------------------------------------------------------------------ ##
# # Load prior run output
# prior_data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_w37_edf/obspack'
# prior_files = glob.glob(f'{prior_data_dir}/*.nc4')
# prior_files.sort()

# def filter_obspack(data):
#     return data[['pressure', 'CH4']]

# obspack_prior = xr.open_mfdataset(prior_files, concat_dim='obs', 
#                                   combine='nested', chunks=1e4, 
#                                   mask_and_scale=False, 
#                                   preprocess=filter_obspack)

# posterior_data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_w37_edf_posterior/obspack'
# posterior_files = glob.glob(f'{posterior_data_dir}/*.nc4')
# posterior_files.sort()

# # Load posterior run output
# obspack_posterior = xr.open_mfdataset(posterior_files, concat_dim='obs', 
#                                       combine='nested', chunks=1e4, 
#                                       mask_and_scale=False, 
#                                       preprocess=filter_obspack)

# data_dir =  '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/obspack'
# files = glob.glob(f'{data_dir}/obspack_ch4.2019*.nc')
# files.sort()

# def filter_obspack(data):
#     return data[['obspack_id', 'value', 'altitude', 'latitude', 'longitude', 
#                  'time', 'utc_conv', 'platform']]

# obspack = xr.open_mfdataset(files, concat_dim='obs', 
#                             combine='nested', chunks=1e4, 
#                             mask_and_scale=False, 
#                             preprocess=filter_obspack)

# # Combine
# obspack_full = pd.DataFrame({'time' : obspack['time'].values,
#                              'utc_conv' : obspack['utc_conv'].values,
#                              'lat' : obspack['latitude'].values,
#                              'lon' : obspack['longitude'].values,
#                              'altitude' : obspack['altitude'].values,
#                              'pressure' : obspack_prior['pressure'].values,
#                              'id' : obspack['obspack_id'].values,
#                              'platform' : obspack['platform'].values,
#                              'obspack' : obspack['value'].values, 
#                              'prior' : obspack_prior['CH4'].values, 
#                              'post' : obspack_posterior['CH4'].values})

# # Adjust units to ppb
# obspack_full[['obspack', 'prior', 'post']] *= 1e9

# # # Remove scenes with 0 pressure
# # obspack_full = obspack_full[obspack_full['pressure'] != 0]

# obspack_full.to_csv(f'{data_dir}/obspack_ch4.2019.csv', header=True)

## ------------------------------------------------------------------------ ##
## Plot output
## ------------------------------------------------------------------------ ##
from copy import deepcopy as dc
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec
import shapefile
from shapely.geometry import Polygon, MultiPolygon
pd.set_option('display.max_columns', 20)

# Custom packages
import sys
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

# Directories
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()

# Get obs
y = xr.open_dataarray(f'{data_dir}ensemble/y.nc')
ya = xr.open_dataarray(f'{data_dir}ensemble/ya_w37_edf_nlc.nc') + 9.11
yhat = xr.open_dataarray(f'{data_dir}ensemble/ya_post.nc')

# Load data
data = pd.read_csv(f'{data_dir}obspack/obspack_ch4.2019.csv', index_col=0)

# Decode strings
for col in ['id', 'platform']:
    data[col] = data[col].str.strip('b\'\'')
data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
data['utc_conv'] = pd.to_timedelta(data['utc_conv'], unit='h')

# Get site
data['site'] = dc(data['id'].str.split('~ch4_').str[-1].str.split('_').str[0])

# Remove scenes with 0 pressure
data = data[data['pressure'] != 0]
data = data[data['pressure'] > 300]

# "For surface and tower measurements,we use only daytime (10:00–16:00 LT,
# local time) observations and average them to the corresponding daytime mean 
# values. We exclude outliers at individual sites that depart by more than 3 
# standard deviations from the mean." Lu et al. (2021)
## Subset to only look at surface and tower for now
data_st = dc(data[data['platform'].isin(['surface', 'tower'])])

## Convert to local time zone and subset to include only daytime measurements
data_st['time_ltc'] = data_st['time'] + data_st['utc_conv']
data_st = data_st.loc[(data_st['time'].dt.hour >= 10) & 
                      (data_st['time'].dt.hour <= 16)]

## Now exclude outliers
### First get the mean and standard dev for each day and site
data_st_mean = data_st.groupby([data_st['site'], 
                                data_st['time'].dt.normalize()])
data_st_mean = data_st_mean.agg(['mean', 'std'])['obspack']
data_st_mean = data_st_mean.rename(columns={'mean' : 'obspack_mean',
                                            'std' : 'obspack_std'})

### Now merge these in
data_st = pd.merge(data_st, data_st_mean, 
                  left_on=[data_st['site'], data_st['time'].dt.normalize()],
                  right_index=True)
data_st = data_st.drop(columns=['key_0', 'key_1'])

### Now exclude from outliers
data_st = data_st[(data_st['obspack'] - data_st['obspack_mean']).abs() <= 
                  3*data_st['obspack_std']]
# data_st = data_st[~data_st['site'].isin(['wgc'])]
# print(data_st[data_st['obspack'] < 1700])
# print(data_st[data_st['obspack'] > 2500])

### Now recalculate the mean
data_st_mean = data_st.groupby([data_st['site'], 
                                data_st['time'].dt.normalize()])

# And subset
data_st_mean = data_st_mean.mean()[['lat', 'lon', 'obspack', 'prior', 'post']]
data_st_mean = data_st_mean.reset_index(drop=True)

# Get the remaining data
data_rest = dc(data[~data['platform'].isin(['surface', 'tower'])])

# Remove a single anomalously large observation
# data_rest = data_rest[data_rest['obspack'] < 3.5e3]

# Remove aircore data
data_rest = data_rest[data_rest['platform'] != 'aircore']

# Subset
data_rest = data_rest[['lat', 'lon', 'obspack', 'prior', 'post']].reset_index(drop=True)

# And concatenate them!
data = pd.concat([data_rest, data_st_mean]).reset_index(drop=True)

# Plot locations
fig, ax = fp.get_figax(cols=2, maps=True, lats=clusters.lat, lons=clusters.lon)
c = ax[0].scatter(data['lon'], data['lat'], c=data['prior'] - data['obspack'], 
                  vmin=-30, vmax=30, cmap='RdBu_r', s=1)
c = ax[1].scatter(data['lon'], data['lat'], c=data['post'] - data['obspack'], 
                  vmin=-30, vmax=30, cmap='RdBu_r', s=1)
for i in range(2):
    ax[i] = fp.format_map(ax[i], clusters.lat, clusters.lon)
cax = fp.add_cax(fig, ax[1])
cb = fig.colorbar(c, cax=cax)
cb = fp.format_cbar(cb, cbar_title='GEOS-Chem - ObsPack XCH4 (ppb)')
fp.save_fig(fig, plot_dir, 'obspack_map')

# Plot the scatter plots
fig, ax = fp.get_figax(cols=2, rows=2, aspect=1, sharey='row')
plt.subplots_adjust(wspace=0.2, hspace=0.4)

op_min = 1.8e3
op_max = 2.15e3
op_vmin = 0
op_vmax = 2.5e3
xs = np.array([op_min, op_max])

# Get basic statistics
rmse_prior = gc.rmse(data['prior'] - data['obspack'])
rmse_post = gc.rmse(data['post'] - data['obspack'])
stats_prior = gc.comparison_stats(data['obspack'].values, data['prior'].values)
m_prior, b_prior, r_prior, bias_prior, std_prior = stats_prior
stats_post = gc.comparison_stats(data['obspack'].values, data['post'].values)
m_post, b_post, r_post, bias_post, std_post = stats_post

rma_prior = gc.rma(data['obspack'].values, data['prior'].values)
rma_post = gc.rma(data['obspack'].values, data['post'].values)

print(f'OBSPACK            Prior     Posterior')
print(f'RMSE               {rmse_prior:<10.2f}{rmse_post:<10.2f}')
print(f'R^2                {r_prior**2:<10.2f}{r_post**2:<10.2f}')
print(f'Bias               {bias_prior:<10.2f}{bias_post:<10.2f}')
print(f'Spatial bias       {std_prior:<10.2f}{std_post:<10.2f}')
print(f'RMA slope          {rma_prior[0]:<10.2f}{rma_post[0]:<10.2f}')
print(f'RMA intercept      {rma_prior[1]:<10.2f}{rma_post[1]:<10.2f}')
print('')

ax[1, 0].hexbin(data['obspack'].values, data['prior'].values, gridsize=40,
                cmap=fp.cmap_trans('inferno'), vmin=op_vmin, vmax=op_vmax, 
                extent=(op_min, op_max, op_min, op_max),
                linewidths=0, zorder=-1)
# ax[1, 0].plot(xs, rma_prior[0]*xs + rma_prior[1], color='grey', ls='-', 
#               zorder=20)
c = ax[1, 1].hexbin(data['obspack'].values, data['post'].values, gridsize=40,
                    cmap=fp.cmap_trans('inferno'), vmin=op_vmin, vmax=op_vmax, 
                    extent=(op_min, op_max, op_min, op_max),
                    linewidths=0, zorder=-1)
# ax[1, 1].plot(xs, rma_post[0]*xs + rma_post[1], color='grey', ls='-', 
#               zorder=20)

stats = [(r_prior, rmse_prior, bias_prior, std_prior), 
         (r_post, rmse_post, bias_post, std_post)]
for i in range(2, 4):
    axis = ax.flatten()[i]
    axis.text(0.05, 0.95, r'R$^2$ = %.2f' % stats[i % 2][0]**2, va='top', 
              ha='left', fontsize=config.LABEL_FONTSIZE*config.SCALE,
              transform=axis.transAxes)
    axis.text(0.05, 0.85, r'RMSE = %.1f ppb' % stats[i % 2][1], va='top', 
              ha='left', fontsize=config.LABEL_FONTSIZE*config.SCALE,
              transform=axis.transAxes)
    axis.text(0.05, 0.75, r'Bias = %.1f ppb' % stats[i % 2][2], va='top', 
              ha='left', fontsize=config.LABEL_FONTSIZE*config.SCALE,
              transform=axis.transAxes)

    axis.set_xlim(op_min, op_max)
    axis.set_ylim(op_min, op_max)
    axis = fp.plot_one_to_one(axis)

cax = fp.add_cax(fig, ax[1, 1])
cb = fig.colorbar(c, cax=cax)
cb = fp.format_cbar(cb, cbar_title='Count')

trop_min = 1.75e3
trop_max = 1.95e3
trop_vmin = 0
trop_vmax = 5e4
xs = np.array([trop_min, trop_max])

# Get basic statistics
rmse_prior = gc.rmse(ya.values - y.values)
rmse_post = gc.rmse(yhat.values - y.values)
stats_prior = gc.comparison_stats(y.values, ya.values)
m_prior, b_prior, r_prior, bias_prior, std_prior = stats_prior
stats_post = gc.comparison_stats(y.values, yhat.values)
m_post, b_post, r_post, bias_post, std_post = stats_post

rma_prior = gc.rma(y.values, ya.values)
rma_post = gc.rma(y.values, yhat.values)

print(f'TROPOMI            Prior     Posterior')
print(f'RMSE               {rmse_prior:<10.2f}{rmse_post:<10.2f}')
print(f'R^2                {r_prior**2:<10.2f}{r_post**2:<10.2f}')
print(f'Bias               {bias_prior:<10.2f}{bias_post:<10.2f}')
print(f'Spatial bias       {std_prior:<10.2f}{std_post:<10.2f}')
print(f'RMA slope          {rma_prior[0]:<10.2f}{rma_post[0]:<10.2f}')
print(f'RMA intercept      {rma_prior[1]:<10.2f}{rma_post[1]:<10.2f}')
print('')

ax[0, 0].hexbin(y.values, ya.values, gridsize=40,
                cmap=fp.cmap_trans('inferno'), vmin=trop_vmin, vmax=trop_vmax, 
                extent=(trop_min, trop_max, trop_min, trop_max),
                linewidths=0, zorder=-1)
# ax[0, 0].plot(xs, rma_prior[0]*xs + rma_prior[1], color='grey', ls='-', 
#               zorder=20)
c = ax[0, 1].hexbin(y.values, yhat.values, gridsize=40,
                    cmap=fp.cmap_trans('inferno'), 
                    vmin=trop_vmin, vmax=trop_vmax, 
                    extent=(trop_min, trop_max, trop_min, trop_max),
                    linewidths=0, zorder=-1)
# ax[0, 1].plot(xs, rma_post[0]*xs + rma_post[1], color='grey', ls='-', 
#               zorder=20)

stats = [(r_prior, rmse_prior, bias_prior, std_prior, rma_prior), 
         (r_post, rmse_post, bias_post, std_post, rma_post)]
for i in range(2):
    axis = ax.flatten()[i]
    axis.text(0.05, 0.95, r'R$^2$ = %.2f' % stats[i][0]**2, va='top', 
              ha='left', fontsize=config.LABEL_FONTSIZE*config.SCALE,
              transform=axis.transAxes)
    axis.text(0.05, 0.85, r'RMSE = %.1f ppb' % stats[i][1], va='top', 
              ha='left', fontsize=config.LABEL_FONTSIZE*config.SCALE,
              transform=axis.transAxes)
    axis.text(0.05, 0.75, r'Bias = %.1f ppb' % stats[i][2], va='top', 
              ha='left', fontsize=config.LABEL_FONTSIZE*config.SCALE,
              transform=axis.transAxes)
    axis.set_xlim(trop_min, trop_max)
    axis.set_ylim(trop_min, trop_max)
    axis = fp.plot_one_to_one(axis)

cax = fp.add_cax(fig, ax[0, 1])
cb = fig.colorbar(c, cax=cax)
cb = fp.format_cbar(cb, cbar_title='Count')

ax[1, 0] = fp.add_labels(ax[1, 0], 'ObsPack CH4 (ppb)', 'GEOS-Chem CH4 (ppb)',
                         labelpad=config.LABEL_PAD/2)
ax[1, 1] = fp.add_labels(ax[1, 1], 'ObsPack CH4 (ppb)', '',
                         labelpad=config.LABEL_PAD/2)
ax[0, 0] = fp.add_labels(ax[0, 0], 'TROPOMI CH4 (ppb)', 'GEOS-Chem CH4 (ppb)',
                         labelpad=config.LABEL_PAD/2)
ax[0, 1] = fp.add_labels(ax[0, 1], 'TROPOMI CH4 (ppb)', '',
                         labelpad=config.LABEL_PAD/2)


ax[0, 0] = fp.add_title(ax[0, 0], 'Prior simulation')
ax[0, 1] = fp.add_title(ax[0, 1], 'Posterior simulation')

fp.save_fig(fig, plot_dir, 'obspack_evaluation')

# Okay, so we ran the wrong things. But we can still get things started on the plotting side of things.


