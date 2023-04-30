import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Open v19
v19 = {}
for seas in seasons:
    data = xr.open_dataset(f'{data_dir}v19_{seas}.nc')
    data = data.sel(lat=slice(s.lat_min ,s.lat_max),
                    lon=slice(s.lon_min, s.lon_max))
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
v14 = gc.load_obj(f'{data_dir}2019_corrected.pkl')

# Recalculate the lat/lon center for the ~50 observations for which it matters
# (where the lat/lons are smack dab in the middle and Nick and my methods
# produce different results--this shouldn't matter for any other analyses,
# but seems good to be consistent when comparing.) (Rename it while we're at
# so that the comparison works.)
v14['lat'] = np.round(v14['LAT']/s.lat_delta)*s.lat_delta
v14['lon'] = np.round(v14['LON']/s.lon_delta)*s.lon_delta

# Group
gb = ['SEASON', 'lat', 'lon']
v14_mean = v14.groupby(gb).mean()['OBS'].to_xarray()
v14_counts = v14.groupby(gb).count()['OBS'].to_xarray()
v14_tot = v14.groupby(['lat', 'lon']).mean()['OBS'].to_xarray()

# Calculate the annual mean difference
diff_mean = (v14_tot - v19_tot['mean']).mean().values

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
diff = v14_tot - v19_tot['mean'] - diff_mean
c = diff.plot(ax=ax, cmap='RdBu_r', vmin=-20, vmax=20, add_colorbar=False)
fp.format_map(ax, lats=diff.lat, lons=diff.lon)

cax = fp.add_cax(fig, ax, horizontal=True)
cb = fig.colorbar(c, ax=ax, cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, r'v14 - v19', horizontal=True)
fp.add_title(ax, 'v19 - v14 TROPOMI data annual average')

fp.save_fig(fig, plot_dir, 'v14_v19_comparison_annual')

# Scatter plot
clusters = xr.open_dataarray(f'{data_dir}clusters.nc')
xhat = pd.read_csv(f'{data_dir}ensemble/xhat.csv', index_col=0)
xhat_mean = xhat.mean(axis=1) # Same as xhat_abs_mean/xa_abs
xhat_mean = ip.match_data_to_clusters(xhat_mean, clusters, default_value=1)
xhat_mean = xhat_mean.to_dataset(name='xhat')
xhat_mean['diff'] = diff
mask = (xhat_mean['xhat'] != 1) & (~xhat_mean['diff'].isnull())

fig, ax = fp.get_figax(aspect=1, max_width=config.BASE_WIDTH/3, 
                       max_height=config.BASE_HEIGHT/3)
xx = xhat_mean['diff'].where(mask, drop=True)
xx = xx.values[~np.isnan(xx.values)]
yy = xhat_mean['xhat'].where(mask, drop=True)
yy = yy.values[~np.isnan(yy.values)]
ax.hexbin(xx, yy, linewidths=0, cmap=fp.cmap_trans('inferno'),
          vmin=0, vmax=150, gridsize=50)
_, _, r, _, _ = gc.comparison_stats(xx, yy)
ax.text(0.025, 0.925, r'R$^2$ = %.2f' % r**2, transform=ax.transAxes,
        fontsize=config.LABEL_FONTSIZE*config.SCALE)

fp.add_labels(ax, 'v14 - v19 annual average', 'Posterior scaling factor')
fp.save_fig(fig, plot_dir, 'v14_v19_comparison_scatter')


