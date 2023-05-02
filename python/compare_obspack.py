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
lats = clusters.lat.values
lons = clusters.lon.values

# Load CONUS mask
conus_mask = np.load(f'{data_dir}countries/CONUS_mask.npy').reshape((-1,))
conus_mask = ip.match_data_to_clusters(conus_mask, clusters, default_value=0)
conus_mask = conus_mask.to_dataframe().reset_index()
conus_mask = conus_mask[conus_mask['Clusters'] > 0].reset_index(drop=True)

# Get obs
ll = gc.load_obj(f'{data_dir}2019_corrected.pkl')[['DATE', 'MONTH', 
                                                   'LAT', 'LON',
                                                   'LAT_CENTER', 'LON_CENTER']]
ll = ll.rename(columns={'LAT_CENTER' : 'lat_center', 
                        'LON_CENTER' : 'lon_center'})
y = xr.open_dataarray(f'{data_dir}ensemble/y.nc')
ya = xr.open_dataarray(f'{data_dir}ensemble/ya_w37_edf_nlc.nc') + 9.11
yhat = xr.open_dataarray(f'{data_dir}ensemble/ya_post.nc')

ya = ya - (1.85 + 0.195*ll['LAT'].values)
yhat = yhat - (1.85 + 0.195*ll['LAT'].values)

# ll['lat_center'] = lats[gc.nearest_loc(ll['LAT'].values, lats)]
# ll['lon_center'] = lons[gc.nearest_loc(ll['LON'].values, lons)]
ll['y'] = y
ll['ya'] = ya
ll['yhat'] = yhat
ll['ya_diff'] = ya - y
ll['yhat_diff'] = yhat - y
ll['DATE'] = pd.to_datetime(ll['DATE'].astype(int), format='%Y%m%d')
ll = ll[ll['lat_center'].isin(conus_mask['lat']) & 
        ll['lon_center'].isin(conus_mask['lon'])]

# Load data
data = pd.read_csv(f'{data_dir}obspack/obspack_ch4.2019.csv', index_col=0)

# Decode strings
for col in ['id', 'platform']:
    data[col] = data[col].str.strip('b\'\'')
data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
data['utc_conv'] = pd.to_timedelta(data['utc_conv'], unit='h')

# Get site
data['site'] = dc(data['id'].str.split('~ch4_').str[-1].str.split('_').str[0])

# Get GC grid
data['lat_center'] = lats[gc.nearest_loc(data['lat'].values, lats)]
data['lon_center'] = lons[gc.nearest_loc(data['lon'].values, lons)]

# Remove scenes with 0 pressure
data = data[data['pressure'] != 0]
data = data[data['pressure'] > 300]

# Remove scenes not in CONUS
data = data[data['lat_center'].isin(conus_mask['lat']) & 
            data['lon_center'].isin(conus_mask['lon'])]

# Correct GEOS-chem
data['prior'] = data['prior'] - (1.85 + 0.195*data['lat'])
data['post'] = data['post'] - (1.85 + 0.195*data['lat'])

# "For surface and tower measurements,we use only daytime (10:00–16:00 LT,
# local time) observations and average them to the corresponding daytime mean 
# values. We exclude outliers at individual sites that depart by more than 3 
# standard deviations from the mean." Lu et al. (2021)
## Subset to only look at surface and tower for now
# data = dc(data[data['platform'].isin(['surface', 'tower'])])
data = data[data['platform'].isin(['surface', 'tower'])]

## Convert to local time zone and subset to include only daytime measurements
data['time_ltc'] = data['time'] + data['utc_conv']
data = data.loc[(data['time'].dt.hour >= 10) & (data['time'].dt.hour <= 16)]

## Now exclude outliers
### First get the mean and standard dev for each day and site
data_mean = data.groupby([data['site'], data['time'].dt.normalize()])
                         # data['time'].dt.month])
data_mean = data_mean.agg(['mean', 'std'])['obspack']
data_mean = data_mean.rename(columns={'mean' : 'obspack_mean',
                                      'std' : 'obspack_std'})

### Now merge these in
data = pd.merge(data, data_mean, 
                left_on=[data['site'], data['time'].dt.normalize()],
                  # left_on=[data['site'], data['time'].dt.month],
                  right_index=True)
data = data.drop(columns=['key_0', 'key_1'])

### Now exclude from outliers
data = data[(data['obspack'] - data['obspack_mean']).abs() <= 
             3*data['obspack_std']]
data = data[~data['site'].isin(['abt'])] #'wgc', 'sgp'

### Now recalculate the mean
data = data.groupby([data['site'], 
                     # data['time'].dt.normalize()])
                     data['time'].dt.month])

# And subset
data = data.mean()[['lat', 'lon', 'obspack', 'prior', 'post']]
data = data.reset_index(drop=True)

## ------------------------------------------------------------------------ ##
## Plot monthly bias
## ------------------------------------------------------------------------ ##
# Get obs


# llc = ll.groupby(['DATE']).count()['y']
# llc = llc.rolling(30).mean()
# ll = ll.groupby(['DATE']).agg(['mean', 'std'])
# llr = ll.rolling(10).mean()
# # print(llr)
# print(llc)

# fig, ax = fp.get_figax(aspect=3)
# ax.axhline(0, c='grey', lw=0.5)
# ax.plot(ll.index, ll['ya_diff']['mean'].values, fp.color(5), lw=0.5, ls=':')#, yerr=ll['ya_diff']['std'])
# ax.plot(llr.index, llr['ya_diff']['mean'].values, fp.color(5))
# ax.plot(ll.index, ll['yhat_diff']['mean'].values, fp.color(7), lw=0.5, ls='--')#, yerr=ll['yhat_diff']['std'])
# ax.plot(llr.index, llr['yhat_diff']['mean'].values, fp.color(7))

# ax2 = ax.twinx()
# ax2.plot(llc.index, llc.values, fp.color(0))

# # ax.plot(ll['y']['mean'].values, c=fp.color(3), lw=0.5, ls='-',
# #         label='TROPOMI')
# # ax.plot(ll['ya']['mean'].values, c=fp.color(5), lw=0.5, ls=':',
# #         label='Prior GEOS-Chem')
# # ax.plot(ll['yhat']['mean'].values, c=fp.color(7), lw=0.5, ls='--',
# #         label='Posterior GEOS-Chem')
# # ax.set_xlim(0, 365)
# # ax.set_ylim(1820, 1880)
# ax.set_xlim(np.datetime64('2019-01-01'), np.datetime64('2020-01-01'))

# ax = fp.add_legend(ax, loc='upper left', ncol=3)

# fp.save_fig(fig, plot_dir, 'prior_posterior_monthly_bias')

## ------------------------------------------------------------------------ ##
## Plot scatter plots
## ------------------------------------------------------------------------ ##
fig, ax = fp.get_figax(cols=2, rows=2, aspect=1, sharey='row')
plt.subplots_adjust(wspace=0.2, hspace=0.4)

op_min = 1.8e3
op_max = 2.25e3
op_vmin = 0
op_vmax = 3
xs = np.array([op_min, op_max])

# Get basic statistics
print(data['prior'] - data['obspack'])
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
print(rma_prior)
print(rma_post)
print('')

ax[1, 0].scatter(data['obspack'].values, data['prior'].values, 
                 color=fp.color(1, cmap='inferno'), s=5)

# ax[1, 0].hexbin(data['obspack'].values, data['prior'].values, gridsize=40,
#                 cmap=fp.cmap_trans('inferno'), vmin=op_vmin, vmax=op_vmax, 
#                 extent=(op_min, op_max, op_min, op_max),
#                 linewidths=0, zorder=-1)
ax[1, 0].plot(xs, rma_prior[0]*xs + rma_prior[1], color='red', ls='-', 
              zorder=20)

ax[1, 1].scatter(data['obspack'].values, data['post'].values, 
                 color=fp.color(1, cmap='inferno'), s=5)
# c = ax[1, 1].hexbin(data['obspack'].values, data['post'].values, gridsize=40,
#                     cmap=fp.cmap_trans('inferno'), vmin=op_vmin, vmax=op_vmax, 
#                     extent=(op_min, op_max, op_min, op_max),
#                     linewidths=0, zorder=-1)
ax[1, 1].plot(xs, rma_post[0]*xs + rma_post[1], color='red', ls='-', 
              zorder=20)

stats = [(r_prior, rmse_prior, rma_prior[0], m_prior), 
         (r_post, rmse_post, rma_post[0], m_post)]
for i in range(2, 4):
    axis = ax.flatten()[i]
    axis.text(0.05, 0.95, r'R$^2$ = %.2f' % stats[i % 2][0]**2, va='top', 
              ha='left', fontsize=config.LABEL_FONTSIZE*config.SCALE,
              transform=axis.transAxes)
    axis.text(0.05, 0.85, r'RMSE = %.1f ppb' % stats[i % 2][1], va='top', 
              ha='left', fontsize=config.LABEL_FONTSIZE*config.SCALE,
              transform=axis.transAxes)
    axis.text(0.05, 0.75, r'RMA slope = %.1f ppb' % stats[i % 2][2], va='top', 
              ha='left', fontsize=config.LABEL_FONTSIZE*config.SCALE,
              transform=axis.transAxes)
    axis.text(0.05, 0.65, r'OLS slope = %.1f ppb' % stats[i % 2][3], va='top', 
              ha='left', fontsize=config.LABEL_FONTSIZE*config.SCALE,
              transform=axis.transAxes)

    axis.set_xlim(op_min, op_max)
    axis.set_ylim(op_min, op_max)
    axis = fp.plot_one_to_one(axis)

# cax = fp.add_cax(fig, ax[1, 1])
# cb = fig.colorbar(c, cax=cax)
# cb = fp.format_cbar(cb, cbar_title='Count')

trop_min = 1.75e3
trop_max = 1.95e3
trop_vmin = 0
trop_vmax = 5e4
xs = np.array([trop_min, trop_max])

# Get basic statistics
rmse_prior = gc.rmse(ll['ya'].values - ll['y'].values)
rmse_post = gc.rmse(ll['yhat'].values - ll['y'].values)
stats_prior = gc.comparison_stats(ll['y'].values, ll['ya'].values)
m_prior, b_prior, r_prior, bias_prior, std_prior = stats_prior
stats_post = gc.comparison_stats(ll['y'].values, ll['yhat'].values)
m_post, b_post, r_post, bias_post, std_post = stats_post
rma_prior = gc.rma(ll['y'].values, ll['ya'].values)
rma_post = gc.rma(ll['y'].values, ll['yhat'].values)

print(f'TROPOMI            Prior     Posterior')
print(f'RMSE               {rmse_prior:<10.2f}{rmse_post:<10.2f}')
print(f'R^2                {r_prior**2:<10.2f}{r_post**2:<10.2f}')
print(f'Bias               {bias_prior:<10.2f}{bias_post:<10.2f}')
print(f'Spatial bias       {std_prior:<10.2f}{std_post:<10.2f}')
print(f'RMA slope          {rma_prior[0]:<10.2f}{rma_post[0]:<10.2f}')
print(f'RMA intercept      {rma_prior[1]:<10.2f}{rma_post[1]:<10.2f}')
print(rma_prior)
print(rma_post)

ax[0, 0].hexbin(ll['y'].values, ll['ya'].values, gridsize=40,
                cmap=fp.cmap_trans('inferno'), vmin=trop_vmin, vmax=trop_vmax, 
                extent=(trop_min, trop_max, trop_min, trop_max),
                linewidths=0, zorder=-1)
ax[0, 0].plot(xs, rma_prior[0]*xs + rma_prior[1], color='red', ls='-', 
              zorder=20)
c = ax[0, 1].hexbin(ll['y'].values, ll['yhat'].values, gridsize=40,
                    cmap=fp.cmap_trans('inferno'), 
                    vmin=trop_vmin, vmax=trop_vmax, 
                    extent=(trop_min, trop_max, trop_min, trop_max),
                    linewidths=0, zorder=-1)
ax[0, 1].plot(xs, rma_post[0]*xs + rma_post[1], color='red', ls='-', 
              zorder=20)

stats = [(r_prior, rmse_prior, rma_prior[0], m_prior), 
         (r_post, rmse_post, rma_post[0], m_post)]
for i in range(2):
    axis = ax.flatten()[i]
    axis.text(0.05, 0.95, r'R$^2$ = %.2f' % stats[i][0]**2, va='top', 
              ha='left', fontsize=config.LABEL_FONTSIZE*config.SCALE,
              transform=axis.transAxes)
    axis.text(0.05, 0.85, r'RMSE = %.1f ppb' % stats[i][1], va='top', 
              ha='left', fontsize=config.LABEL_FONTSIZE*config.SCALE,
              transform=axis.transAxes)
    axis.text(0.05, 0.75, r'RMA slope = %.1f ppb' % stats[i][2], va='top', 
              ha='left', fontsize=config.LABEL_FONTSIZE*config.SCALE,
              transform=axis.transAxes)
    axis.text(0.05, 0.65, r'OLS slope = %.1f ppb' % stats[i % 2][3], va='top', 
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

# ------------------------------------------------------------------------ ##
## Plot map
## ------------------------------------------------------------------------ ##

# fig, ax = fp.get_figax(rows=3, maps=True, lats=clusters.lat, lons=clusters.lon)

# # y = xr.open_dataarray(f'{data_dir}ensemble/y.nc')
# # ya = xr.open_dataarray(f'{data_dir}ensemble/ya_w37_edf_nlc.nc') + 9.11
# # yhat = xr.open_dataarray(f'{data_dir}ensemble/ya_post.nc')


# ll = gc.load_obj(f'{data_dir}2019_corrected.pkl')
# ll = ll[['LAT', 'LON']]
# ll['lat_center'] = lats[gc.nearest_loc(ll['LAT'].values, lats)]
# ll['lon_center'] = lons[gc.nearest_loc(ll['LON'].values, lons)]
# ll = ll.drop(columns=['LAT', 'LON'])

# ll['y'] = y
# ll['ya'] = ya
# ll['yhat'] = yhat

# ll = ll.groupby(['lat_center', 'lon_center']).mean()
# ll = ll.to_xarray()

# ax[0].set_title('Prior GEOS-Chem - TROPOMI')
# c = (ll['ya'] - ll['y']).plot(ax=ax[0], cmap='RdBu_r', vmin=-15, vmax=15, 
#                               add_colorbar=False)
# cax = fp.add_cax(fig, ax[0])
# cb = fig.colorbar(c, cax=cax)
# cb = fp.format_cbar(cb, cbar_title='')

# ax[1].set_title('Posterior GEOS-Chem - TROPOMI')
# c = (ll['yhat'] - ll['y']).plot(ax=ax[1], cmap='RdBu_r', vmin=-15, vmax=15, 
#                                 add_colorbar=False)
# cax = fp.add_cax(fig, ax[1])
# cb = fig.colorbar(c, cax=cax)
# cb = fp.format_cbar(cb, cbar_title='')

# ax[2].set_title('Posterior GEOS-Chem - Prior GEOS-Chem')
# c = (ll['yhat'] - ll['ya']).plot(ax=ax[2], cmap='RdBu_r', vmin=-5, vmax=5, 
#                                  add_colorbar=False)
# cax = fp.add_cax(fig, ax[2])
# cb = fig.colorbar(c, cax=cax)
# cb = fp.format_cbar(cb, cbar_title='')

# fp.save_fig(fig, plot_dir, 'prior_posterior_maps')


# # ya = ya - (1.85 + 0.195*ll)
# # yhat = yhat - (1.85 + 0.195*ll)