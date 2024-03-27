import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

import sys
sys.path.append('.')
from copy import deepcopy as dc
import gcpy as gc
import format_plots as fp
import invpy as ip
import config
import inversion_settings as s

data_dir = '../inversion_data/'
plot_dir = '../plots/'
seasons = ['DJF', 'MAM', 'JJA', 'SON']

# Open averaging kernel sensitivities
clusters = xr.open_dataarray(f'{data_dir}clusters.nc')
dofs = pd.read_csv(f'{data_dir}ensemble/dofs.csv', index_col=0)
dofs = dofs.mean(axis=1)
dofs = ip.match_data_to_clusters(dofs, clusters)

# Open kw sum
kw = xr.open_dataarray(f'{data_dir}/reduced_rank/kw_sum.nc')

# Normalize kw (also try log normalizing)
kw = kw/kw.max()

# kw.where(kw > 0.05).plot(norm=colors.LogNorm(vmin=0.05, vmax=2e4))
# plt.show()

# Open landfill lat lons
ghgrp = pd.read_csv(f'{data_dir}landfills/ghgrp_processed.csv')

# Open v19
v19 = {}
for seas in seasons:
    data = xr.open_dataset(f'{data_dir}observations/v19_{seas}.nc')
    data = data.sel(lat=slice(s.lat_min ,s.lat_max),
                    lon=slice(s.lon_min, s.lon_max))

    # Set to 0 where kw sum is 0
    data['obs'] = data['obs'].where(kw > 0.05)
    data['count'] = data['count'].where(kw > 0.05)

    v19[seas] = data

# Calculate a total for the sake of removing the mean bias between the two
v19_tot = v19[seasons[0]]['count']*v19[seasons[0]]['obs']
v19_tot = v19_tot.to_dataset(name='mean')
v19_tot['count'] = dc(v19[seasons[0]]['count'])
for seas in seasons[1:]:
    v19_tot['mean'] += v19[seas]['count']*v19[seas]['obs']
    v19_tot['count'] += v19[seas]['count']
v19_tot['mean'] = v19_tot['mean']/v19_tot['count']

# Open v14
v14 = gc.load_obj(f'{data_dir}observations/2019_corrected.pkl')

# Recalculate the lat/lon center for the ~50 observations for which it matters
# (where the lat/lons are smack dab in the middle and Nick and my methods
# produce different results--this shouldn't matter for any other analyses,
# but seems good to be consistent when comparing.) (Rename it while we're at
# so that the comparison works.)
v14['lat'] = np.round(v14['LAT']/s.lat_delta)*s.lat_delta
v14['lon'] = np.round(v14['LON']/s.lon_delta)*s.lon_delta

# Group
gb = ['SEASON', 'lat', 'lon']
v14_mean = v14.groupby(gb).mean()['OBS']
v14_counts = v14.groupby(gb).count()['OBS']
v14_tot = v14.groupby(['lat', 'lon']).mean()['OBS']

# Expand to match clusters
cdf = clusters.squeeze(drop=True)
cdf = clusters.to_dataframe()
cdf = cdf[cdf['Clusters'] > 0]

v14_mean = cdf.join(v14_mean)
v14_mean = v14_mean.fillna(0)
v14_mean = v14_mean.sort_values('Clusters')
v14_mean = v14_mean['OBS'].to_xarray()

v14_counts = cdf.join(v14_counts)
v14_counts = v14_counts.fillna(0)
v14_counts = v14_counts.sort_values('Clusters')
v14_counts = v14_counts['OBS'].to_xarray()

v14_tot = cdf.join(v14_tot)
v14_tot = v14_tot.fillna(0)
v14_tot = v14_tot.sort_values('Clusters')
v14_tot = v14_tot['OBS'].to_xarray()

# Set to 0 where kw = 0
v14_mean = v14_mean.where(kw > 0.05)
v14_tot = v14_tot.where(kw > 0.05)

# Calculate the annual mean difference
diff_mean = (v14_tot - v19_tot['mean']).mean().values


(v14_tot - v19_tot['mean']).squeeze(drop=True).plot(vmin=-10, vmax=10, 
                                                    cmap='RdBu_r')
plt.show()

# Subset to areas where everyone has data
v14_tot = v14_tot.sel(lon=slice(-128, -60))
v19_tot = v19_tot.sel(lon=slice(-128, -60))
dofs = dofs.sel(lon=slice(-128, -60))

# Compare
fig, ax = fp.get_figax(rows=2, cols=2, maps=True, 
                       lats=v19['DJF'].lat, lons=v19['DJF'].lon)
plt.subplots_adjust(hspace=0.2)
for i, seas in enumerate(seasons):
    v14_d = v14_mean.sel(SEASON=seas).squeeze(drop=True)
    v19_d = v19[seas]['obs']
    diff = v14_d - v19_d - diff_mean
    c = diff.plot(ax=ax.flatten()[i], cmap='RdBu_r', vmin=-20, vmax=20, 
                  add_colorbar=False)
    fp.add_title(ax.flatten()[i], seas)
    fp.format_map(ax.flatten()[i], lats=diff.lat, lons=diff.lon)

cax = fp.add_cax(fig, ax)
cb = fig.colorbar(c, ax=ax, cax=cax)
cb = fp.format_cbar(cb, r'v14 - v19')

fp.save_fig(fig, plot_dir, 'v14_v19_comparison_seasonal')

# Compare total
fig, ax = fp.get_figax(maps=True, lats=v19['DJF'].lat, lons=v19['DJF'].lon,
                       max_width=config.BASE_WIDTH/2)

# Calculate mean adjusted difference
diff = v14_tot - v19_tot['mean'] - diff_mean
print('-'*70)
diff = diff.squeeze(drop=True)
diff.to_netcdf(f'{data_dir}observations/v14_v19_diff.nc')
print('-'*70)

# Scale by kw
# diff = diff*kw

# Mask for DOFS
# diff = diff.where(dofs > 0, 0)

# Mask for differences of 10 ppb
# diff = diff.where(np.abs(diff) > 10, 0)
print((np.abs(diff) > 5).sum().values)
print((np.abs(diff) > 10).sum().values)
print((kw > 0.05).sum().values)

# Make diff just for landfills
diff_lf = diff.to_dataframe(name='diff_trop').reset_index()
diff_lf = diff_lf.rename(columns={'lat' : 'lat_center', 'lon' : 'lon_center'})
diff_lf = pd.merge(ghgrp, diff_lf, on=['lat_center', 'lon_center'], how='inner')

# Plot difference
c = diff.plot(ax=ax, cmap='RdBu_r', vmin=-10, vmax=10, add_colorbar=False)

# We could check how large the footprint of each observation is

# # Plot landfills
# c = ax.scatter(diff_lf['lon_center'], diff_lf['lat_center'], 
#                c=diff_lf['diff_trop'], marker='x', s=3,
#                vmin=-20, vmax=20, cmap='RdBu_r')

fp.format_map(ax, lats=diff.lat, lons=diff.lon)

cax = fp.add_cax(fig, ax, horizontal=True)
cb = fig.colorbar(c, ax=ax, cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, r'v14 - v19 (ppb)', horizontal=True)
fp.add_title(ax, 'v14 - v19 TROPOMI data annual average')

fp.save_fig(fig, plot_dir, 'v14_v19_comparison_annual')

# Scatter plot
xhat = pd.read_csv(f'{data_dir}ensemble/xhat.csv', index_col=0)
xhat_mean = xhat.mean(axis=1) # Same as xhat_abs_mean/xa_abs
xhat_mean = ip.match_data_to_clusters(xhat_mean, clusters, default_value=1)
xhat_mean = xhat_mean.to_dataset(name='xhat')
xhat_mean['diff'] = diff
mask = (xhat_mean['xhat'] != 1) & (~xhat_mean['diff'].isnull())
mask = mask.squeeze(drop=True)

fig, ax = fp.get_figax(aspect=1, max_width=config.BASE_WIDTH/3, 
                       max_height=config.BASE_HEIGHT/3)
xx = xhat_mean['diff'].where(mask, drop=True)
xx = xx.values[~np.isnan(xx.values)]

yy = xhat_mean['xhat'].where(mask, drop=True)
yy = yy.values[~np.isnan(yy.values)]
ax.hexbin(xx, yy, linewidths=0, cmap=fp.cmap_trans('inferno'),
          vmin=0, vmax=100, gridsize=50)
_, _, r, _, _ = gc.comparison_stats(xx, yy)
ax.text(0.025, 0.925, r'R$^2$ = %.2f' % r**2, transform=ax.transAxes,
        fontsize=config.LABEL_FONTSIZE*config.SCALE)

fp.add_labels(ax, 'v14 - v19 annual average', 'Posterior scaling factor')
fp.save_fig(fig, plot_dir, 'v14_v19_comparison_scatter')


