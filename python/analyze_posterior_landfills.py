from os.path import join
from os import listdir
import sys
import glob
from copy import deepcopy as dc
import math
import xarray as xr
import numpy as np
import pandas as pd
from sklearn import linear_model
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy
from collections import OrderedDict

pd.set_option('display.max_columns', 20)

# Custom packages
sys.path.append('.')
import config
# config.SCALE = config.PRES_SCALE
# config.BASE_WIDTH = config.PRES_WIDTH
# config.BASE_HEIGHT = config.PRES_HEIGHT
import gcpy as gc
import troppy as tp
import invpy as ip
import format_plots as fp
import inversion_settings as s

valid_color = '#FAC645'

## ------------------------------------------------------------------------ ##
## Directories
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'

## ------------------------------------------------------------------------ ##
## Load posterior files
## ------------------------------------------------------------------------ ##
# Define DOFS filter
DOFS_filter = 0.2

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()

## Dig into Shreveport
shreveport = int(clusters.sel(lat=32.25, lon=-93.75).values)
puente = int(clusters.sel(lat=34, lon=-118.125).values)

# Load area (km2)
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load ensemble members (relative posterior and DOFS), all of which are
# previously filtered on DOFS and have the BC elements removed as needed
dofs = pd.read_csv(f'{data_dir}ensemble/dofs.csv', index_col=0)
dofs = dofs.mean(axis=1)
xhat = pd.read_csv(f'{data_dir}ensemble/xhat.csv', index_col=0)
ensemble = list(xhat.columns)
 
# Load weighting matrices in units Gg/yr
w = pd.read_csv(f'{data_dir}w_w37_edf.csv')[['landfills']].T*1e-3
conus_mask = np.load(f'{data_dir}countries/CONUS_mask.npy').reshape((-1,))
w *= conus_mask

w_full = pd.read_csv(f'{data_dir}w_w37_edf.csv').T*1e-3
print('-'*75)
print('Shreveport problems')
print(w_full.iloc[:, shreveport - 1])
print('Puente Hills problems')
print(w_full.iloc[:, puente - 1])
print('-'*75)
w_frac = w_full.loc['landfills']/w_full.loc[['livestock', 'coal', 'ong', 'landfills', 'wastewater', 'other_anth']].sum(axis=0)

# Get the posterior xhat_abs (this is n x 15)
xhat_abs = w @ xhat
xhat_abs = (w.values.T*xhat).mean(axis=1)
xhat_abs_min = xhat_abs - (w.values.T*xhat).min(axis=1)
xhat_abs_max = (w.values.T*xhat).max(axis=1) - xhat_abs
xhat_abs_ensemble = (w.values.T*xhat)

# Put it on a grid 
xhat_abs = ip.match_data_to_clusters(xhat_abs, clusters, default_value=0)
xhat_abs = xhat_abs.to_dataset().rename({'Clusters' : 'post_mean'})
xhat_abs['post_min'] = ip.match_data_to_clusters(xhat_abs_min, clusters, 
                                            default_value=0)
xhat_abs['post_max'] = ip.match_data_to_clusters(xhat_abs_max, clusters, 
                                            default_value=0)
for member in ensemble:
    xhat_abs[member] = ip.match_data_to_clusters(xhat_abs_ensemble[member], 
                                                 clusters, default_value=0)

xhat_abs['dofs'] = ip.match_data_to_clusters(dofs, clusters, default_value=0)
xhat_abs['xa'] = ip.match_data_to_clusters(w.values.reshape(-1,), clusters,
                                           default_value=0)
xhat_abs['frac'] = ip.match_data_to_clusters(w_frac, clusters, default_value=0)

# And collapse it into a dataframe
xhat_abs = xhat_abs.to_dataframe().reset_index()
xhat_abs = xhat_abs[xhat_abs['post_mean'] > 0].reset_index(drop=True)
xhat_abs = xhat_abs.rename(columns={'lat' : 'lat_center', 
                                    'lon' : 'lon_center'})
xhat_abs = xhat_abs.set_index(['lat_center', 'lon_center'])
xhat_abs = xhat_abs[['xa', 'post_mean', 'post_min', 'post_max',
                     'dofs', 'frac'] + list(xhat.columns)]

## ------------------------------------------------------------------------ ##
## Load landfill file
## ------------------------------------------------------------------------ ##
# Load and clean
ghgrp = pd.read_csv(f'{data_dir}landfills/ghgrp.csv')
ghgrp = ghgrp.rename(columns={'REPORTING YEAR' : 'year',
                              'FACILITY NAME' : 'name', 
                              'GHGRP ID' : 'id',
                              'LATITUDE' : 'lat',
                              'LONGITUDE' : 'lon',
                              'CITY NAME' : 'city',
                              'COUNTY NAME' : 'county',
                              'STATE' : 'state',
                              'PARENT COMPANIES' : 'company',
                              'GHG QUANTITY (METRIC TONS CO2e)' : 'ghgrp',
                              'SUBPARTS' : 'subparts'})
ghgrp = ghgrp.drop(columns=['ZIP CODE', 'REPORTED ADDRESS'])

# Clean up methane
ghgrp = ghgrp.sort_values(by='ghgrp', ascending=False)
ghgrp = ghgrp.reset_index(drop=True)
ghgrp['ghgrp'] /= (25*1e3) # Convert to Gg methane

# Limit ourselves to landfills emitting more than 1 Gg/yr
print(ghgrp.shape[0], 'landfills are in the original GHGRP file.')
ghgrp = ghgrp[ghgrp['ghgrp'] >= 2.5]
print(ghgrp.shape[0], 'landfills remain that emit more than 2.5 Gg/yr')

# Put them onto the GC grid
lats, lons = gc.create_gc_grid(*s.lats, s.lat_delta,*s.lons, s.lon_delta,
                               centers=False, return_xarray=False)
ghgrp['lat_center'] = lats[gc.nearest_loc(ghgrp['lat'].values, lats)]
ghgrp['lon_center'] = lons[gc.nearest_loc(ghgrp['lon'].values, lons)]

# We will need to aggregate emissions by grid cell
ghgrp_short = dc(ghgrp).groupby(['lat_center', 'lon_center'])
ghgrp_short = ghgrp_short.agg({'name' : 'count',
                               'ghgrp' : 'sum',
                               'id' : 'sum'})
ghgrp_short = ghgrp_short.rename(columns={'name' : 'count'})

# Join in xhat_abs
ghgrp_short = ghgrp_short.join(xhat_abs, how='left')

# Drop NAs, filter on DOFS, and limit to grid cells with 50% of emissions
# explained by landfills
temp = ghgrp_short[ghgrp_short['dofs'] < DOFS_filter]
fig, ax = fp.get_figax(maps=True, lats=clusters.lat, lons=clusters.lon)
ax.scatter(temp.reset_index()['lon_center'], 
           temp.reset_index()['lat_center'])
ax = fp.format_map(ax, clusters.lat, clusters.lon)
fp.save_fig(fig, plot_dir, 'multiple_landfills')

ghgrp_short = ghgrp_short.dropna()
ghgrp_short = ghgrp_short[ghgrp_short['dofs'] >= DOFS_filter]
print(ghgrp_short.shape[0], 'landfills remain that have A >= ', DOFS_filter)

# ghgrp_short = ghgrp_short[ghgrp_short['post_mean'] >= 5]
ghgrp_short = ghgrp_short[ghgrp_short['frac'] >= 0.5]
print(ghgrp_short.shape[0], 'grid cells remain that have >= 50%% landfill emissions')

# Look at grid cells with multiple landfills
ghgrp_multi = ghgrp_short[ghgrp_short['count'] > 1]
cnt = ghgrp_multi['count'].sum()
print(f'{ghgrp_multi.shape[0]} grid cells with multiple landfills that TROPOMI constrains, removing {cnt} landfills')

# Drop grid cells with multiple landfills, which we couldn't distinguish
ghgrp_short = ghgrp_short[ghgrp_short['count'] < 2]
ghgrp_short = ghgrp_short.drop(columns=['count'])

print(f'{ghgrp_short.shape[0]} is the final number of landfills')

# Join in the rest of ghgrp
ghgrp = ghgrp.set_index(['lat_center', 'lon_center'])
ghgrp = ghgrp_short.join(ghgrp.drop(columns=['ghgrp', 'id']), how='left')

# Rank by largest discrepancy with the GHGRP
ghgrp['diff_abs'] = ghgrp['post_mean'] - ghgrp['ghgrp']
ghgrp['diff_rel'] = ghgrp['diff_abs']/ghgrp['ghgrp']
ghgrp = ghgrp.sort_values(by='post_mean', ascending=False)

# And finally, fix indexing
ghgrp = ghgrp.reset_index()

# Print some things
ghgrp_mean = ghgrp['ghgrp'].mean()
post_mean = ghgrp['post_mean'].mean()
xhat_mean = (ghgrp['post_mean']/ghgrp['ghgrp']).mean()
xhat_median = (ghgrp['post_mean']/ghgrp['ghgrp']).median()
print(ghgrp['post_mean']/ghgrp['ghgrp'])
fig, ax = fp.get_figax()
ax.hist(ghgrp['post_mean']/ghgrp['ghgrp'], bins=30, color=fp.color(3))
ax.axvline(1, color='grey')
fp.save_fig(fig, plot_dir, 'ghgrp_his t')
dofs_mean = ghgrp['dofs'].mean()
frac_mean = ghgrp['frac'].mean()
frac_std = ghgrp['frac'].std()
nl = ghgrp.shape[0]

print(f'The average sensitivity across these {nl} landfills in GHGRP is {dofs_mean:.2f}.')
print(f'Landfills explain an average of {frac_mean*100:.2f} +/- {frac_std*100:.2f}% emissions in these {nl} grid cells.')
print(f'The average emissions across these {nl} landfills in GHGRP is {ghgrp_mean:.2f} Gg/yr.')
print(f'The average emissions across these {nl} landfills in the posterior is {post_mean:.2f} Gg/yr.')
print(f'The average increase across these {nl} landfills in the posterior is {xhat_mean:.2f} (median {xhat_median:.2f}).')
print('-'*75)

ghgrp.to_csv(f'{data_dir}landfills/ghgrp_processed.csv')

## ------------------------------------------------------------------------ ##
## Load LMOP file and calculate a recovery rate
## ------------------------------------------------------------------------ ##
lmop = pd.read_csv(f'{data_dir}landfills/lmop.csv')
lmop = lmop.rename(columns={'GHGRP ID' : 'id',
                            'Year Landfill Opened' : 'landfill_year_opened',
                            'Landfill Closure Year' : 'landfill_year_closed',
                            'Current Landfill Status' : 'landfill_status',
                            'Waste in Place (tons)' : 'waste_mass',
                            'LFG Collection System In Place?' : 'LFG',
                            'LFG Collected (mmscfd)' : 'LFG_collection_rate',
                            'LFG Flared (mmscfd)' : 'LFG_flare_rate',
                            'Current Project Status' : 'LFG_status',
                            'Project Start Date' : 'LFG_start_date',
                            'Project Shutdown Date' : 'LFG_shutdown_date',
                            'Project Type Category' : 'project_type',
                            'LFG Energy Project Type' : 'LFG_energy_type',
                            'RNG Delivery Method' : 'RNG_delivery_method',
                            'Actual MW Generation' : 'generation',
                            'Rated MW Capacity' : 'capacity',
                            'LFG Flow to Project (mmscfd)' : 'LFG_flow',
                            'Current Year Emission Reductions (MMTCO2e/yr) - Direct' : 'avoided_emissions'})

# # Subset for plants operational in 2019
lmop['LFG_start_date'] = pd.to_datetime(lmop['LFG_start_date'])
lmop['LFG_shutdown_date'] = pd.to_datetime(lmop['LFG_shutdown_date'])

lmop = lmop[(lmop['LFG_start_date'].dt.year <= 2019) & 
            ((lmop['LFG_status'].isin(['Operational'])) | 
             (lmop['LFG_shutdown_date'].dt.year >= 2019))]
lmop = lmop.reset_index(drop=True)

# Convert avoided emissions to Gg/yr
lmop['avoided_emissions'] *= 1000/25

# Fix NaNs
cats = ['LFG_collection_rate', 'LFG_flare_rate' , 'avoided_emissions', 
        'generation', 'capacity']
lmop[cats] = lmop[cats].fillna(0)

# Group by facility
# lmop_date = lmop[['id', 'LFG_start_date']].groupby(['id']).min()
# lmop = lmop.groupby(['id']).sum()[['LFG_collection_rate', 'LFG_flare_rate' ,
#                                    'avoided_emissions', 'generation', 
#                                    'capacity']]
lmop = lmop.groupby(['id']).agg({'LFG_start_date' : 'min',
                                 'LFG_collection_rate' : 'mean', 
                                 'LFG_flare_rate' : 'mean',
                                 'avoided_emissions' : 'sum', 
                                 'generation' : 'sum',
                                 'capacity' : 'sum'})
# lmop = pd.concat([lmop, lmop_date], axis=1)
lmop = lmop.reset_index()

# And join in ghgrp
lmop = pd.merge(lmop, ghgrp, on='id', how='inner')

# Calculate the recovery rates
lmop['ghgrp_recovery'] = 100*lmop['avoided_emissions']/(lmop['ghgrp'] + lmop['avoided_emissions'])
# lmop['mean_recovery'] = 100*lmop['avoided_emissions']/(lmop['post_mean'] + lmop['avoided_emissions'])
# lmop['max_recovery'] = 100*lmop['avoided_emissions']/(lmop['post_mean'] - lmop['post_min'] + lmop['avoided_emissions'])
# lmop['min_recovery'] = 100*lmop['avoided_emissions']/(lmop['post_mean'] + lmop['post_max'] + lmop['avoided_emissions'])

# Calculate recovery rates for each ensemble member
for member in ensemble:
    lmop[f'{member}_recovery'] = 100*lmop['avoided_emissions']/(lmop[member] +lmop['avoided_emissions'])

# Calculate the statistics of the recovery rates
lmop['mean_recovery'] = lmop[[f'{m}_recovery' for m in ensemble]].mean(axis=1)
lmop['min_recovery'] = lmop[[f'{m}_recovery' for m in ensemble]].min(axis=1)
lmop['max_recovery'] = lmop[[f'{m}_recovery' for m in ensemble]].max(axis=1)

# Save out
lmop.to_csv(f'{data_dir}landfills/lmop_processed.csv')

# Print some statistics
recovery_rates = lmop[[f'{m}_recovery' for m in ensemble]].mean(axis=0)

print(f'For {lmop.shape[0]} recovery facilities:')
print('Average LFG collected: ', lmop['LFG_collection_rate'].mean().round(1), ' +/-', lmop['LFG_collection_rate'].std().round(1), ' mmscfd')
print('Average LFG flared: ', lmop['LFG_flare_rate'].mean().round(1), 'mmscfd')

print('')
print('Mean GHGRP recovery rate: ', lmop['ghgrp_recovery'].mean().round(1))
print('Mean posterior recovery rate: ', recovery_rates.mean().round(1), '(', recovery_rates.min().round(1), ' - ', recovery_rates.max().round(1), ')')

# Adjust min and max for plotting
lmop['min_recovery'] = lmop['mean_recovery'] - lmop['min_recovery']
lmop['max_recovery'] = lmop['max_recovery'] - lmop['mean_recovery']

# Plot them against each other
fig, ax = fp.get_figax(aspect=1)
ax.errorbar(lmop['ghgrp_recovery'], lmop['mean_recovery'],
            yerr=lmop[['min_recovery', 'max_recovery']].T.values,
            fmt='none', markersize=1, ecolor='black', elinewidth=0.5,
            capsize=0, capthick=0)
c = ax.scatter(lmop['ghgrp_recovery'], lmop['mean_recovery'], 
               c=lmop['capacity'], vmin=1, vmax=25,
               marker='o', s=30, cmap='plasma',
               edgecolor='black', zorder=10)

# Sub data
lmop_sub = dc(lmop[(lmop['capacity'] == 0) | lmop['capacity'].isnull()])
ax.scatter(lmop_sub['ghgrp_recovery'], lmop_sub['mean_recovery'], 
           marker='o', s=30, facecolor='white', edgecolor='black', zorder=10)

cax = fp.add_cax(fig, ax)
cb = fig.colorbar(c, cax=cax)
cb = fp.format_cbar(cb, cbar_title='Facility capacity (MW)')

# Visuals
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax = fp.plot_one_to_one(ax)
ax = fp.add_labels(ax, 'GHGRP recovery rate', 'Posterior recovery rate')

# R2
_, _, r, _, _ = gc.comparison_stats(lmop['ghgrp_recovery'].values,
                                    lmop['mean_recovery'].values)
ax.text(0.025, 0.95, r'R$^2$ = %.2f' % r**2,
        fontsize=config.LABEL_FONTSIZE*config.SCALE,
        transform=ax.transAxes)

lmop_sub = dc(lmop[(lmop['capacity'] >= 5)])
recovery_rates = lmop_sub[[f'{m}_recovery' for m in ensemble]].mean(axis=0)

print('\n'f'For {lmop_sub.shape[0]} facilities with greater than or equal to 5 MW capacity:')
print('Mean GHGRP recovery rate (>5 MW): ', lmop_sub['ghgrp_recovery'].mean().round(1))
print('Mean posterior recovery rate (>5 MW): ', recovery_rates.mean().round(1), '(', recovery_rates.min().round(1), ' - ', recovery_rates.max().round(1), ')')

m, b, r, _, _ = gc.comparison_stats(lmop_sub['ghgrp_recovery'].values,
                                    lmop_sub['mean_recovery'].values)
ax.text(0.025, 0.9, r'R$^2$ (capacity $>$ 5 MW) = %.2f' % r**2,
        fontsize=config.LABEL_FONTSIZE*config.SCALE,
        transform=ax.transAxes)
# ax.plot([0, 100], [b, 100*m+b], color='black')
print(f'{m:.2f}*GHGRP_recovery + {b:.2f} = posterior_recovery')

lmop_sub = dc(lmop[(lmop['LFG_start_date'].dt.year >= 2005)])
recovery_rates = lmop_sub[[f'{m}_recovery' for m in ensemble]].mean(axis=0)

print('\n'f'For {lmop_sub.shape[0]} facilities with greater than or equal to 5 MW capacity:')
print('Mean GHGRP recovery rate (>5 MW): ', lmop_sub['ghgrp_recovery'].mean().round(1))
print('Mean posterior recovery rate (>5 MW): ', recovery_rates.mean().round(1), '(', recovery_rates.min().round(1), ' - ', recovery_rates.max().round(1), ')')

m, b, r, _, _ = gc.comparison_stats(lmop_sub['ghgrp_recovery'].values,
                                    lmop_sub['mean_recovery'].values)
ax.text(0.025, 0.85, r'R$^2$ (start date $>$ 2009) = %.2f' % r**2,
        fontsize=config.LABEL_FONTSIZE*config.SCALE,
        transform=ax.transAxes)
# ax.plot([0, 100], [b, 100*m+b], color='black')
print(f'{m:.2f}*GHGRP_recovery + {b:.2f} = posterior_recovery')


# lmop_sub = dc(lmop[(lmop['capacity'] < 5)])
# print('\n'f'For {lmop_sub.shape[0]} facilities with less than 5 MW capacity:')
# print('Mean GHGRP recovery rate (<5 MW): ', lmop_sub['ghgrp_recovery'].mean().round(1))
# print('Mean posterior recovery rate (<5 MW): ', lmop_sub['mean_recovery'].mean().round(1))

fp.save_fig(fig, plot_dir, 'landfills_recovery_lmop')

# Now plot correlations vs lots of things
fig, ax = fp.get_figax(rows=2, cols=2, aspect=1.5, sharey=True)
plt.subplots_adjust(hspace=0.75, wspace=0.1)
# lmop_sub = dc(lmop[lmop['capacity'] > 0])
lmop_sub = dc(lmop)
lmop_sub['diff_recovery'] = lmop_sub['mean_recovery'] - lmop_sub['ghgrp_recovery']
lmop_sub['capacity_factor'] = lmop_sub['generation']/lmop_sub['capacity']
for i, q in enumerate(['LFG_start_date', 'capacity', 
                       'generation', 'capacity_factor']):
    axis = ax.flatten()[i]
    qstr = q.capitalize().replace('_', ' ')
    m, b, r, _, _ = gc.comparison_stats(lmop_sub[q].values.astype(float),
                                        lmop_sub['diff_recovery'].values)
    if q in ['capacity', 'generation', 'capacity_factor']:
        data_sub = lmop_sub[lmop_sub['capacity'] != 0]
    else: 
        data_sub = lmop_sub
    axis.scatter(data_sub[q], data_sub['diff_recovery'], 
                 marker='o', s=30, facecolor='white', edgecolor='black', 
                 zorder=10)
    axis.plot([data_sub[q].min(), data_sub[q].max()], 
              [m*data_sub[q].values.min().astype(float) + b, 
               m*data_sub[q].values.max().astype(float) + b],
              color=fp.color(5), ls=':')

    axis.text(0.975, 0.95, r'R$^2$ = %.2f' % r**2, va='top', ha='right', 
              fontsize=config.LABEL_FONTSIZE*config.SCALE,
              transform=axis.transAxes)

    axis.axhline(0, c='0.1', lw=1, ls='--', alpha=0.5)
    axis.axhline(data_sub['diff_recovery'].mean(), c=fp.color(3), 
                 lw=1, ls='--')
    if i % 2 == 0:
        ylabel = 'Posterior - GHGRP\nrecovery rate'
    else:
        ylabel = ''
    axis = fp.add_labels(axis, qstr, ylabel)
    if i < 1:
        axis.tick_params(axis='x', labelrotation=45)
fp.save_fig(fig, plot_dir, 'landfills_recovery_mlr')

print('-'*75)

## ------------------------------------------------------------------------ ##
## Load validation sites
## ------------------------------------------------------------------------ ##
# Get validation sites
lf_studies = pd.read_csv(f'{data_dir}landfills/validation.csv')
lf_studies[['mean', 'std']] *= 24*365/1e6
lf_studies['min'] = lf_studies['mean'] - lf_studies['std']
lf_studies['max'] = lf_studies['mean'] + lf_studies['std']
lf_studies = lf_studies.groupby(['study', 'name', 'id']).agg({'mean' : 'mean',
                                                              'min' : 'min',
                                                              'max' : 'max'})
lf_studies['min'] = lf_studies['mean'] - lf_studies['min']
lf_studies['max'] = lf_studies['max'] - lf_studies['mean']
lf_studies = lf_studies.reset_index()
lf_studies = pd.merge(lf_studies, ghgrp[['id', 'post_mean']], 
                      on='id', how='inner')
lf_studies = lf_studies.sort_values(by='post_mean', ascending=False)
lf_studies = lf_studies.reset_index(drop=True)
lf_names = lf_studies['name'].unique()
print('Validation sites:')
print(lf_names)

## ------------------------------------------------------------------------ ##
## Plot results: map
## ------------------------------------------------------------------------ ##
CONUS_lats = [clusters.lat.min(), 50]
CONUS_lons = [-125, -65]
aspect = fp.get_aspect(rows=1, cols=1, maps=True, lats=CONUS_lats, 
                       lons=CONUS_lons)
figsize = fp.get_figsize(aspect*1.25, max_width=config.BASE_WIDTH - 2)
fig = plt.figure(figsize=figsize)

gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.25], hspace=0.1)
ax0 = plt.subplot(gs[0], projection=ccrs.PlateCarree())
ax1 = plt.subplot(gs[1])
ax = [ax0, ax1]

ax[0] = fp.format_map(ax[0], lats=CONUS_lats, lons=CONUS_lons)

# Add California inset
ca_lats = [32.2, 39.2]
ca_lons = [-122.9, -116.5]
ca_aspect = fp.get_aspect(rows=1, cols=1, maps=True, 
                          lats=ca_lats, lons=ca_lons)

ca_ax = inset_axes(ax[0], width=f'{40*ca_aspect:.0f}%', height='40%',
                   bbox_to_anchor=(-0.025, 0, 1, 1), loc=3,
                   bbox_transform=ax[0].transAxes,
                   axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                   axes_kwargs=dict(map_projection=cartopy.crs.PlateCarree()))
ca_ax = fp.format_map(ca_ax, ca_lats, ca_lons)

# Plot the inset outline
ax[0].plot([ca_lons[0], ca_lons[0]], ca_lats, color='0.6', lw=0.75)
ax[0].plot([ca_lons[1], ca_lons[1]], ca_lats, color='0.6', lw=0.75)
ax[0].plot(ca_lons, [ca_lats[0], ca_lats[0]], color='0.6', lw=0.75)
ax[0].plot(ca_lons, [ca_lats[1], ca_lats[1]], color='0.6', lw=0.75)
ax[0].text(ca_lons[0], ca_lats[0] + 0.5, 'Inset', rotation=90, ha='right', 
           va='bottom', color='0.6')

# Add Illinois/Indiana inset
ilin_lats = [38.6, 42.7]
ilin_lons = [-91.5, -83.9]
ilin_aspect = fp.get_aspect(rows=1, cols=1, maps=True, 
                            lats=ilin_lats, lons=ilin_lons)
ilin_to_ca = (ca_lons[1] - ca_lons[0])/(ilin_lons[1] - ilin_lons[0])
ilin_ax = inset_axes(ax[0], width=f'{40*ilin_to_ca*ilin_aspect:.0f}%', 
                     height=f'{40*ilin_to_ca:.0f}%',
                     bbox_to_anchor=(0, 0, 1.03, 1), loc=4,
                     bbox_transform=ax[0].transAxes,
                     axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                     axes_kwargs=dict(map_projection=cartopy.crs.PlateCarree()))
ilin_ax = fp.format_map(ilin_ax, ilin_lats, ilin_lons)

# Plot the inset outline
ax[0].plot([ilin_lons[0], ilin_lons[0]], ilin_lats, color='0.6', lw=0.75)
ax[0].plot([ilin_lons[1], ilin_lons[1]], ilin_lats, color='0.6', lw=0.75)
ax[0].plot(ilin_lons, [ilin_lats[0], ilin_lats[0]], color='0.6', lw=0.75)
ax[0].plot(ilin_lons, [ilin_lats[1], ilin_lats[1]], color='0.6', lw=0.75)
# ax[0].text(ilin_lons[0] + 0.5, ilin_lats[1], 'Inset', ha='left', 
#            va='bottom', color='0.6')
ax[0].text(ilin_lons[0] + 0.5, ilin_lats[0], 'Inset', ha='left', 
           va='top', color='0.6')
# ax[0].text(ilin_lons[0], ilin_lats[0] + 0.5, 'Inset', rotation=90, ha='right', 
#            va='bottom', color='0.75')

# xhat_abs_p = (w.values.T*xhat).mean(axis=1)
# xhat_abs_p = xhat_abs_p/area.reshape(-1,)*1e3 # Tg/yr -> Mg/km2/yr

# fig, ax, c = ip.plot_state(xhat_abs_p, clusters, title='Landfill emissions',
#                            default_value=0, vmin=0, vmax=10, 
#                            cmap=fp.cmap_trans('viridis'), cbar=False)

# pos_adjustments = ghgrp[ghgrp['diff_rel'] > 0]

# Plot scatter points
cs = []
for combustion in [False, True]:
    if combustion:
        data = ghgrp[ghgrp['id'].isin(lmop['id'])]
        marker = 'o'
        label = 'Gas collected'
    else:
        data = ghgrp[~ghgrp['id'].isin(lmop['id'])]
        marker = 'D'
        label = 'Gas not collected'

    # Validation sites
    data_sub = data[data['id'].isin(lf_studies['id'].values)]
    ax[0].scatter(data_sub['lon'], data_sub['lat'], c='white', 
                  edgecolor=valid_color, linewidth=3, marker=marker,
                  s=0.7*data_sub['post_mean'], label='Validation site')

    c = ax[0].scatter(data['lon'], data['lat'], c=data['diff_abs'], 
                      edgecolor='black', linewidth=0.5, marker=marker, 
                      vmin=-30, vmax=30, cmap='RdBu_r',
                      s=0.7*data['post_mean'], label=label)
    cs.append(c)

    ca_ax.scatter(data_sub['lon'], data_sub['lat'], c='white', 
                  edgecolor=valid_color, linewidth=3, marker=marker,
                  s=0.7*data_sub['post_mean'], label='Validation site')
    ca_ax.scatter(data['lon'], data['lat'], c=data['diff_abs'], 
                  edgecolor='black', linewidth=0.5, marker=marker, 
                  vmin=-30, vmax=30, cmap='RdBu_r',
                  s=0.7*data['post_mean'], label=label)

    ilin_ax.scatter(data_sub['lon'], data_sub['lat'], c='white', 
                    edgecolor=valid_color, linewidth=3, marker=marker,
                    s=0.7*data_sub['post_mean'], label='Validation site')
    ilin_ax.scatter(data['lon'], data['lat'], c=data['diff_abs'], 
                    edgecolor='black', linewidth=0.5, marker=marker, 
                    vmin=-30, vmax=30, cmap='RdBu_r',
                    s=0.7*data['post_mean'], label=label)

# Plot validation sites
ys = np.arange(1, len(lf_names) + 1)
# Edges
ax[1].barh(ys - 0.185, ghgrp[ghgrp['name'].isin(lf_names)]['ghgrp'], 
           height=0.3, color='white', edgecolor=s.sector_colors['landfills'], 
           zorder=10)
ax[1].barh(ys + 0.185, ghgrp[ghgrp['name'].isin(lf_names)]['post_mean'], 
           xerr=np.array(ghgrp[ghgrp['name'].isin(lf_names)][['post_min', 'post_max']]).T,
           error_kw={'ecolor' : s.sector_colors['landfills'], 
                     'lw' : 0.75, 'capsize' : 2,
                     'capthick' : 0.75},
           height=0.3, color='white', edgecolor=s.sector_colors['landfills'], 
           zorder=10)

ax[1].barh(ys - 0.185, ghgrp[ghgrp['name'].isin(lf_names)]['ghgrp'], 
           height=0.3, color=s.sector_colors['landfills'], alpha=0.3, 
           zorder=20)
ax[1].barh(ys + 0.185, ghgrp[ghgrp['name'].isin(lf_names)]['post_mean'], 
           xerr=np.array(ghgrp[ghgrp['name'].isin(lf_names)][['post_min', 'post_max']]).T,
           error_kw={'ecolor' : s.sector_colors['landfills'], 
                     'lw' : 0.75, 'capsize' : 2,
                     'capthick' : 0.75},
           height=0.3, color=s.sector_colors['landfills'], alpha=0.3, 
           zorder=20)

formats = ['o', 's', '^', 'D']
for study in lf_studies['study'].unique():
    studies = lf_studies[lf_studies['study'] == study]
    for lf in studies['name'].unique():
        result = studies[studies['name'] == lf]
        y = np.argwhere(lf == lf_names)[0][0]*np.ones(result.shape[0])
        print(y, lf)
        ax[1].errorbar(
            result['mean'].values, y + 1,
            xerr=np.array(result[['min', 'max']]).T, fmt=formats[i % 4],
            markersize=3.5, markeredgecolor='black',
            markerfacecolor=valid_color, ecolor='black', elinewidth=0.5,
            capsize=1, capthick=0.5, zorder=20, label=study)
    i += 1

ax[1].set_xlim(0, 55)
ax[1].set_xticks([0, 25, 50])
# for j in range(2):
#     ax[1].axvline(25*(j + 1), color='0.75', lw=0.5, zorder=-10)

ax[1].set_ylim(0.5, ys.max() + 0.5)
ax[1].set_yticks(ys)
ax[1].set_yticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
                      ha='right', fontsize=config.TICK_FONTSIZE*config.SCALE)

ax[1] = fp.add_labels(ax[1], r'Emissions (Gg a$^{-1}$)', '',
                      labelpad=config.LABEL_PAD/3.9)
ax[1].text(1, ys[0] - 0.155, 'EPA GHGRP', ha='left', va='center', 
           fontsize=config.TICK_FONTSIZE - 2, zorder=40)
ax[1].text(1, ys[0] + 0.205, 'Posterior',  #left['post_livestock'] + 0.25
           ha='left', va='center', 
           fontsize=config.TICK_FONTSIZE - 2, zorder=40)

ax[1].invert_yaxis()
# # plot numbers
# data = ghgrp[(ghgrp['post_mean'] - ghgrp['post_min'] > 1.5*ghgrp['ghgrp']) |
#              (ghgrp['post_mean'] + ghgrp['post_max'] < 0.5*ghgrp['ghgrp'])]
# data = data.sort_values(by='post_mean', ascending=False)
# data = data[~data['name'].isin(['MEADOW BRANCH LANDFILL', 
#                                 'LAUREL RIDGE LANDFILL', 'CACTUS LANDFILL'])]
# data = data.reset_index(drop=True)
# print(data)
# for i in range(data.shape[0]):
#     row = data.iloc[i, :]
#     ax.text(row['lon'], row['lat'], i + 1, color='black', fontsize=6, 
#             weight='black', va='center', ha='center', zorder=30)
#     ax.text(row['lon'], row['lat'], i + 1, color='white', fontsize=6, 
#             va='center', ha='center', zorder=30  )

# neg_adjustments = ghgrp[ghgrp['diff_rel'] < 0]
# ax.scatter(neg_adjustments['lon_center'], neg_adjustments['lat_center'], 
           # color='white', edgecolors='blue', s=15)

ax[0] = fp.add_title(ax[0], 'Landfill methane emissions')
ax[1] = fp.add_title(ax[1], 'Validation')
# ax[0] = fp.format_map(ax[0], lats=CONUS_lats, lons=CONUS_lons)
cax = fp.add_cax(fig, ax[0], cbar_pad_inches=0.25, horizontal=True)
cb = fig.colorbar(c, cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb,
                    cbar_title=r'$\Delta$ (Posterior - GHGRP) methane emissions (Gg a$^{-1}$)', horizontal='horizontal', y=-2.75)


# cax.get_position().x1
w0 = ax[1].get_position().x0
w1 = ax[1].get_position().x1
h0 = cax.get_position().y0
h1 = ax[0].get_position().y1
ax[1].set_position([w0, h0, w1 - w0, h1 - h0])
# ax[1].set_aspect(17.5)


# Add legend
h, l = ax[1].get_legend_handles_labels()
l = OrderedDict(zip(l, h))
h = l.values()
l = l.keys()

custom_h = [Line2D([0], [0], markersize=0, lw=0)] + \
           [(Line2D([0], [0], marker='D', markerfacecolor='0.7',
             markeredgecolor='black', markeredgewidth=0.5,
             color='w', lw=0, markersize=np.sqrt(0.7*ss)), 
             Line2D([0], [0], marker='o', markerfacecolor='0.7',
             markeredgecolor='black', markeredgewidth=0.5,
             color='w', lw=0, markersize=np.sqrt(0.7*ss))) 
             for ss in [10, 20, 40]] + \
            [Line2D([0], [0], markersize=0, lw=0),
             Line2D([0], [0], marker='o', markerfacecolor='white', 
              markeredgecolor=valid_color, markeredgewidth=3, color='w', lw=0, 
              markersize=np.sqrt(40)),
             Line2D([0], [0], marker='o', markerfacecolor='0.7', 
              markeredgecolor='black', markeredgewidth=0.5, color='w', lw=0, 
              markersize=np.sqrt(40)),
             Line2D([0], [0], marker='D', markerfacecolor='0.7', 
              markeredgecolor='black', markeredgewidth=0.5, color='w', lw=0, 
              markersize=np.sqrt(40))]

custom_l = ['Posterior emissions'] + \
           [f'{ss} Gg a'r'$^{-1}$' for ss in [10, 20, 40]] + \
           ['', 'Validation site', 'Gas collected', 'Gas not collected']

custom_h.extend(h)
custom_l.extend(l)

ax[0].legend(handles=custom_h, labels=custom_l, 
             handler_map={tuple: HandlerTuple(ndivide=None)},
             loc='upper left', bbox_to_anchor=(0.05, -0.35), frameon=False,
             fontsize=config.LABEL_FONTSIZE*config.SCALE, ncol=3)
             # columnspacing=0.5)

# ax[0].text(0.12, 0.317, r'Posterior emissions', va='center',
#            fontsize=config.LABEL_FONTSIZE*config.SCALE,
#            transform=ax[0].transAxes)
# ax[0].text(0.03, 0.31, r'Gas not collected', ha='left',
#            fontsize=config.LABEL_FONTSIZE*config.SCALE,
#            transform=ax[0].transAxes, rotation=90)
# ax[0].text(0.07, 0.31, r'Gas collected', ha='left',
#            fontsize=config.LABEL_FONTSIZE*config.SCALE,
#            transform=ax[0].transAxes, rotation=90)

fp.save_fig(fig, plot_dir, 'landfills_map', dpi=700)
print('-'*75)

## ------------------------------------------------------------------------ ##
## Plot results: scatter
## ------------------------------------------------------------------------ ##
fig, ax = fp.get_figax(aspect=1)
cm = plt.cm.get_cmap('viridis', 12)

m, b, r, bias, std = gc.comparison_stats(ghgrp['ghgrp'].values, 
                                         ghgrp['post_mean'].values)

ax.errorbar(ghgrp['ghgrp'], ghgrp['post_mean'], 
            yerr=ghgrp[['post_min', 'post_max']].T.values,
            fmt='none', elinewidth=0.5, capsize=0, capthick=0, 
            ecolor=fp.color(3))
for combustion in [True, False]:
    if combustion:
        data = ghgrp[ghgrp['id'].isin(lmop['id'])]
        marker = 'o'
        label = 'Gas collected'
        _, _, r_combust, _, _ = gc.comparison_stats(data['ghgrp'].values,
                                                    data['post_mean'].values)
    else:
        data = ghgrp[~ghgrp['id'].isin(lmop['id'])]
        marker = '^'
        label = 'Gas not collected'
        _, _, r_noncombust, _, _ = gc.comparison_stats(data['ghgrp'].values,
                                                       data['post_mean'].values)
    ax.scatter(data['ghgrp'], data['post_mean'], s=20, marker=marker,
               facecolor='white', edgecolor=fp.color(3),
               # c=ghgrp['dofs'], cmap=cm, vmin=0.2, vmax=0.8, 
               zorder=10, label=label)

# Limit it to landfills with 50% errors or greater
nold = ghgrp.shape[0]
print(f'{nold} landfills meet our analysis threshold.')

# ghgrp = ghgrp[np.abs(ghgrp['diff_rel']) > 0.5]
ghgrp = ghgrp[(ghgrp['post_mean'] - ghgrp['post_min'] > 1.5*ghgrp['ghgrp']) |
              (ghgrp['post_mean'] + ghgrp['post_max'] < 0.5*ghgrp['ghgrp'])]

# Print information on the subsetted data
print(f'{ghgrp.shape[0]} landfills have errors greater than 50% compared to GHGRP.')
frac_mean = ghgrp['frac'].mean()
frac_std = ghgrp['frac'].std()
print(f'Landfills explain an average of {frac_mean*100:.2f} +/- {frac_std*100:.2f}% emissions in these {ghgrp.shape[0]} grid cells.')
print(ghgrp[ghgrp['post_mean'] - ghgrp['post_min'] > 1.5*ghgrp['ghgrp']].shape[0], 'have positive errors (posterior > GHGRP)')
print('-'*75)

ax.errorbar(ghgrp['ghgrp'], ghgrp['post_mean'], 
            yerr=ghgrp[['post_min', 'post_max']].T.values,
            fmt='none', elinewidth=1, capsize=0, capthick=0, 
            ecolor=fp.color(5))
for combustion in [True, False]:
    if combustion:
        data = ghgrp[ghgrp['id'].isin(lmop['id'])]
        # print(data)
        print(f'{data.shape[0]} landfills with large errors collect gas.')
        print(data[data['post_mean'] - data['post_min'] > 1.5*data['ghgrp']].shape[0], 'have positive errors (posterior > GHGRP)')
        marker = 'o'
    else:
        data = ghgrp[~ghgrp['id'].isin(lmop['id'])]
        print('NO COMBUSTION')
        # print(data)
        print(f'{data.shape[0]} landfills with large errors do not collect gas.')
        print(data[data['post_mean'] - data['post_min'] > 1.5*data['ghgrp']].shape[0], 'have positive errors (posterior > GHGRP)')
        marker = '^'
    ax.scatter(data['ghgrp'], data['post_mean'], s=20, marker=marker,
               # c=ghgrp['dofs'], cmap=cm, vmin=0.2, vmax=0.8, 
               facecolor='white', edgecolor=fp.color(5), 
               zorder=10, label=r'$>$50\% errors')

# Plot one-to-one and also a ~50% error threshold
ax = fp.plot_one_to_one(ax)

# Add R2
ax.text(0.025, 0.95, r'R$^2$ (all) = %.2f' % r**2,
        fontsize=config.LABEL_FONTSIZE*config.SCALE,
        transform=ax.transAxes)
ax.text(0.025, 0.9, r'R$^2$ (gas collected) = %.2f' % r_combust**2,
        fontsize=config.LABEL_FONTSIZE*config.SCALE,
        transform=ax.transAxes)
ax.text(0.025, 0.85, 
        r'R$^2$ (gas not collected) = %.2f' % r_noncombust**2,
        fontsize=config.LABEL_FONTSIZE*config.SCALE,
        transform=ax.transAxes)

# data = ghgrp[ghgrp['ghgrp'] > 12]
# m_small, b_small, r_small, _, _ = gc.comparison_stats(data['ghgrp'].values,
#                                           data['post_mean'].values)
# ax.text(0.025, 0.8, 
#         r'R$^2$ ($>$12 Gg a$^{-1}$) = %.2f (%.2fx + %.2f)' % (r_small**2, m_small, b_small),
#         fontsize=config.LABEL_FONTSIZE*config.SCALE,
#         transform=ax.transAxes)


ax.fill_between([0, 50], [0, 35], [0, 55], color='0.1', alpha=0.15, zorder=-1,
                label='10\% errors')
ax.fill_between([0, 50], [0, 25], [0, 75], color='0.1', alpha=0.1, zorder=-1,
                label='50\% errors')
# ax.fill_between([0, 40], [0, 20], [0, 60], color='0.1', alpha=0.1, zorder=-1)

# Aesthetics
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax = fp.add_labels(ax, r'GHGRP reported emissions (Gg a$^{-1}$)', 
                   r'Posterior emissions (Gg a$^{-1}$)')


ax = fp.add_legend(ax, loc='lower right') 
                   # loc='center left', bbox_to_anchor=(1.01, 0.5))
# Colorbar
# cax = fp.add_cax(fig, ax)
# cb = fig.colorbar(c, cax=cax)
# cb = fp.format_cbar(cb, cbar_title='Averaging kernel senstivity')

# Save
fp.save_fig(fig, plot_dir, 'landfills_ghgrp')
