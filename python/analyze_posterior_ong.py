import sys
import glob
from copy import deepcopy as dc
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec
import shapefile
from shapely.geometry import Polygon, MultiPolygon
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

# Colormaps
plasma_trans = fp.cmap_trans('plasma')
yor_trans = fp.cmap_trans('YlOrRd', nalpha=100)

sf_cmap_1 = plt.cm.PuOr_r(np.linspace(0.2, 0.5, 256))
sf_cmap_2 = plt.cm.PuOr_r(np.linspace(0.5, 1, 256))
sf_cmap = np.vstack((sf_cmap_1, sf_cmap_2))
sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
div_norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=1, vmax=3)

## ------------------------------------------------------------------------ ##
## Directories
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'
paper_dir = base_dir + 'paper/figures/'

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
label_map = {'permian' :'Permian', 
             'delaware' : 'Delaware', 
             'haynesville' : 'Haynesville',
             'barnett' : 'Barnett',
             'anadarko' : 'Anadarko',
             'northeast' : 'Marcellus',
             'eagle_ford' : 'Eagle Ford', 
             'san_juan' : 'San Juan', 
             'california' : 'California',
             'bakken' : 'Bakken',
             'wyoming' : 'Wyoming',
             'dj' : 'DJ',
             'alberta_west' : 'Alberta West',
             'fayetteville' : 'Fayetteville',
             'sw_pa' : 'SW Pennsylvania',
             'west_arkoma' : 'West Arkoma',
             'ne_pa': 'NE Pennsylvania',
             'alberta_east' : 'Alberta East',
             'uinta': 'Uinta'}

# Units all in GG/yr, errors are rrelative
shen2022 = {'permian' : (3700, 0.09), 'delaware' : (970, 0.18),
            'haynesville' : (656, 0.16), 'barnett' : (478, 0.13), 
            'anadarko' : (610, 0.22), 'northeast' : (613, 0.28),
            'eagle_ford' : (508, 0.18), 'san_juan' : (236, 0.22), 
            'california' : (244, 0.3), 'bakken' : (123, 0.26), 
            'wyoming' : (124, 0.29), 'dj' : (52, 0.45), 
            'alberta_west' : (68, 0.24), 'fayetteville' : (36, 0.48), 
            'sw_pa' : (44, 0.29), 'west_arkoma' : (51, 0.31), 
            'ne_pa' : (28, 0.34),
            'alberta_east' : (32, 0.51), 'uinta' : (96, 0.21)}


# Load Lu 2023
files = glob.glob(f'{data_dir}ong/Lu2023/Region_emission*')
files.sort()
cols = ['Year', 'Emissions [Tg]', 'Emission(Min)', 'Emission(Max)']
lu2023 = pd.DataFrame(columns=cols)
for f in files:
    data = pd.read_csv(f, usecols=cols)
    data = data[data['Year'] == 2019]
    data['Basin'] = f.split('_')[-1].split('.')[0].lower().replace(' ', '_')
    data = data.set_index('Basin')
    lu2023 = lu2023.append(data)
lu2023 = lu2023.drop(columns='Year')
lu2023 = lu2023.rename(index={'marcellus' : 'northeast'},
                       columns={'Emissions [Tg]' : 'mean',
                                'Emission(Min)' : 'min',
                                'Emission(Max)' : 'max'})
lu2023['min'] = lu2023['mean'] - lu2023['min']
lu2023['max'] = lu2023['max'] - lu2023['mean']
print(lu2023)
lu2023 = lu2023.drop('us')

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load the ONG mask
w_ong = pd.read_csv(f'{data_dir}ong/ong_mask.csv', header=0).T

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()

# fig, ax, c = ip.plot_state(w_ong.loc['permian'], clusters)
# ax = fp.format_map(ax, lats=clusters.lat.values, lons=clusters.lon.values)
# plt.show()

# Load area (km2)
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load ensemble members (relative posterior and DOFS), all of which are
# previously filtered on DOFS and have the BC elements removed as needed
dofs = pd.read_csv(f'{data_dir}ensemble/dofs.csv', index_col=0)
xhat = pd.read_csv(f'{data_dir}ensemble/xhat.csv', index_col=0)
ensemble = xhat.columns

# Load weighting matrices in units Tg/yr (we don't consider wetlands
# here, so it doesn't matter which W we use)
w = pd.read_csv(f'{data_dir}sectors/w.csv')
w = dc(w['ong']).T*1e-6

# Get the posterior xhat_abs (this is n x nensemble) (only ONG)
xhat_abs = w.values[:, None]*xhat

## ------------------------------------------------------------------------ ##
## Calculate state sectoral prior and posterior
## ------------------------------------------------------------------------ ##
# Calculate the area, prior emissions, posterior emissions, and xhat
# for each column, then concatenate it into one dataframe
## Area
area = pd.DataFrame((w_ong @ area.reshape(-1,)).rename('area'))

## Prior (the same for all ensemble members) 
## (w_ong = 19 x 23691, w.T = 23691 x sectors)
prior = pd.DataFrame((w_ong @ w.T).rename('prior'))

## Posterior (19 x n, n x ensemble)
post = (w_ong @ xhat_abs)

## Posterior ratios
xhat_ong = post/prior.values

## Get statistics
xhat_stats = ip.get_ensemble_stats(xhat_ong).add_prefix('xhat_')
post_stats = ip.get_ensemble_stats(post).add_prefix('post_')

## Aggregate
summ = pd.concat([area, prior, post_stats, xhat_stats], axis=1)
print(summ.loc['permian'])

# Sort by posterior anthropogenic emissions
summ = summ.sort_values(by='post_mean', ascending=False)

# Save out csv
summ.to_csv(f'{data_dir}/ong/summ_ong.csv', header=True, index=True)

# Get number of basins 
nb = w_ong.shape[0]

## ------------------------------------------------------------------------ ##
## Plot results : absolute emissions
## ------------------------------------------------------------------------ ##
# Define x range
xs = np.arange(1, nb + 1)

# Begin plotting!
fig, ax = fp.get_figax(aspect=3, sharex=True)

# Get labels
labels = [label_map[l] for l in summ.index.values]

# Reorder Shen and extract mean and error
shen2022_data = np.array([shen2022[l][0] for l in summ.index.values])
shen2022_err = np.array([shen2022[l][0]*shen2022[l][1] 
                         for l in summ.index.values])

# Adjust min/max definitions for error bars
summ['post_max'] = summ['post_max'] - summ['post_mean']
summ['post_min'] = summ['post_mean'] - summ['post_min']

print('-'*70)
print('Differences between mean emissions between Shen et al. (2022) and our work')
print(summ['post_mean'] - shen2022_data*1e-3)

# Plot bar
# ax.bar(xs - 0.175, summ['prior'], width=0.3, color=s.sector_colors['ong'], 
#        label='Prior')
ax.bar(xs, summ['post_mean'], 
       yerr=np.array(summ[['post_min', 'post_max']]).T,
       error_kw={'ecolor' : '0.65', 'lw' : 0.75, 'capsize' : 2, 
                 'capthick' : 0.75},
       width=0.3, color=s.sector_colors['ong'], alpha=0.5, 
       label='Posterior emissions')

# Add Shen data
ax.errorbar(xs - 0.075, shen2022_data*1e-3, yerr=shen2022_err*1e-3, fmt='o', 
            markersize=4, markerfacecolor='white', markeredgecolor='black', 
            ecolor='black', lw=0.5, capsize=1, capthick=0.5, 
            label='Shen et al. (2022)')

# Add Shen threshold
ax.fill_between([0, nb + 1], [0, 0], [0.5, 0.5], 
                color='0.3', alpha=0.1, label='Quantification threshold', 
                zorder=-5)

# Add Lu data
x = np.array([np.argwhere(b == summ.index.values)[0][0] + 1 for b in lu2023.index.values])
ax.errorbar(x + 0.075, lu2023['mean'], yerr=np.array(lu2023[['min', 'max']]).T,
            fmt='s', markersize=4, markerfacecolor='white', 
            markeredgecolor='black', ecolor='black', lw=0.5, 
            capsize=1, capthick=0.5, 
            label='Lu et al. (2023)')
print('Comparing total emissions in Lu et al. (2023)')
print('Our estimate :', summ.loc[lu2023.index.values]['post_mean'].sum(), 'Tg')
print('Their estimate :', lu2023['mean'].sum(), 'Tg') 
print('Our estimate :', summ.loc[lu2023.index.values]['post_mean'][summ.loc[lu2023.index.values]['post_mean'] > 0.5].sum(), 'Tg')
print('Their estimate :', lu2023['mean'][lu2023['mean'] > 0.5 ].sum(), 'Tg') 

# Add labels
ax.set_xticks(xs)
ax.set_xlim(0, nb + 1)
# ax.invert_yaxis()
ax.set_xticklabels(labels, ha='right', fontsize=config.TICK_FONTSIZE,
                   rotation=90)

ax.set_ylim(0, 4.1)

ax.tick_params(axis='both', labelsize=config.TICK_FONTSIZE)
# plt.setp(ax.get_xticklabels(), visible=False)

# Final aesthetics
ax = fp.add_labels(ax, '', r'Emissions (Tg a$^{-1}$)',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)

for i in range(5):
    ax.axhline(i + 1, color='0.75', lw=0.5, zorder=-10)

# Final aesthetics
for k in range(nb - 1):
    if i % 5 == 0:
        ls = '-'
    else:
        ls = ':'
    ax.axvline((k + 1) + 0.5, color='0.75', alpha=1, lw=0.5, ls=ls,
                zorder=-10)

# Legend for summary plot
ax = fp.add_legend(ax, ncol=1, fontsize=config.TICK_FONTSIZE, 
                   loc='upper right')

fp.save_fig(fig, plot_dir, f'ong_ensemble')
fp.save_fig(fig, paper_dir, 'figS03', for_acp=True)

# ------------------------------------------------------------------------ ##
# Permian comparison
# ------------------------------------------------------------------------ ##
# Combine the Permian clusters with the NA clusters
permian = xr.open_dataset(f'{data_dir}ong/clusters_permian.nc')['Clusters']
c = clusters.squeeze(drop=True).to_dataset()
c['Permian'] = permian

# Get the Permian basin indices (discard the buffer cells)
cell_idx, cell_cnt  = np.unique(c['Permian'], return_counts=True)
cell_idx = cell_idx[cell_cnt == 1]
cell_idx = cell_idx[~np.isnan(cell_idx)]
permian = permian.where(permian.isin(cell_idx), 0)

# # Subset over the Permian
# c = c.where(c['Permian'].isin(cell_idx))['Clusters']
# c = c.sel(lat=permian.lat, lon=permian.lon)

# # Flatten and create boolean
# permian_idx = (ip.clusters_2d_to_1d(permian, c) - 1).astype(int)
permian_idx = np.load(f'{data_dir}ong/permian_idx.npy')
# print(c)
# c[c > 0] = 1

nstate_permian = len(permian_idx)

# Subset the posterior and convert to Tg/yr
xhat_permian = ip.get_ensemble_stats(xhat.values[permian_idx, :])
xa_abs_permian = w.values[permian_idx]
xhat_abs_permian = ip.get_ensemble_stats(xhat_abs.values[permian_idx, :])
dofs_permian = ip.get_ensemble_stats(dofs.values[permian_idx, :])
area_permian = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))
area_permian = area_permian[permian_idx, :]

# Calculate total emissions
tot_prior_permian = xa_abs_permian.sum()
tot_post_permian = xhat_abs_permian['mean'].sum()
print(f'Total prior emissions            : {tot_prior_permian}')
print(f'Total posterior emissions        : {tot_post_permian}')
print(f'Difference                       : {(tot_post_permian - tot_prior_permian)}')

# Adjust back to kg/km2/hr
xa_abs_permian = xa_abs_permian/area_permian/1e-9/(365*24)
xhat_abs_permian = xhat_abs_permian/area_permian/1e-9/(365*24)
# print(xa_abs_permian)
# print(xhat_permian)

fig, axis = fp.get_figax(rows=2, cols=2, maps=True,
                         lats=permian.lat, lons=permian.lon,
                         max_width=config.BASE_WIDTH*2.5,
                         max_height=config.BASE_HEIGHT*2.5)
plt.subplots_adjust(hspace=-0.05, wspace=0.75)

# Plot prior
fig_kwargs = {'figax' : [fig, axis[0, 0]]}
xhat_kwargs = {'cmap' : yor_trans, 'vmin' : 0, 'vmax' : 13,
               'default_value' : 0,
               'fig_kwargs' : fig_kwargs}
title = f'Prior emissions' # ({f}\%)'
fig, axis[0, 0], c = ip.plot_state(xa_abs_permian, permian, title=title, 
                                   cbar=False, **xhat_kwargs)
axis[0, 0].text(0.05, 0.05, f'{tot_prior_permian:.1f} Tg/yr',
                fontsize=config.LABEL_FONTSIZE*config.SCALE*1,
                transform=axis[0, 0].transAxes)


# Plot posterior emissions
fig_kwargs = {'figax' : [fig, axis[0, 1]]}
xhat_cbar_kwargs = {'title' : r'Emissions\\(kg km$^{-2}$ h$^{-1}$)', 'x' : 4,
                    'cbar_pad_inches' : 0.1}
xhat_kwargs['fig_kwargs'] = fig_kwargs
xhat_kwargs['cbar_kwargs'] = xhat_cbar_kwargs
title = f'Posterior emissions' # ({f}\%)'
fig, axis[0, 0], c = ip.plot_state(xhat_abs_permian['mean'], 
                                   permian, title=title, 
                                   **xhat_kwargs)
axis[0, 1].text(0.05, 0.05, f'{tot_post_permian:.1f} Tg/yr',
                fontsize=config.LABEL_FONTSIZE*config.SCALE*1,
                transform=axis[0, 1].transAxes)

# # Plot posterior scaling factors
# sf_cmap_1 = plt.cm.Oranges(np.linspace(0, 1, 256))
# sf_cmap_2 = plt.cm.Purples(np.linspace(0.5, 0, 256))
# sf_cmap = np.vstack((sf_cmap_2, sf_cmap_1))
# sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
# div_norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=3)

xhat_cbar_kwargs = {'title' : r'Scale factor', 'x' : 4, 
                    'ticks' : np.arange(0, 3.1, 0.5), 
                    'cbar_pad_inches' : 0.1}
fig_kwargs = {'figax' : [fig, axis[1, 0]]}
xhat_kwargs = {'cmap' : sf_cmap, 'norm' : div_norm,
               'default_value' : 1,
               'cbar_kwargs' : xhat_cbar_kwargs,
               'fig_kwargs' : fig_kwargs}
title = f'Posterior\nscale factors' # ({f}\%)'
fig, axis[1, 0], c = ip.plot_state(xhat_permian['mean'], permian, title=title,
                                   **xhat_kwargs)
# axis[1, 0].text(0.05, 0.05,
#                 f'({(xhat_permian.min()):.1f}, {(xhat_permian.max()):.1f})',
#                 fontsize=config.LABEL_FONTSIZE*config.SCALE*1,
#                 transform=axis[1, 0].transAxes)

# Plot DOFS
fig_kwargs = {'figax' : [fig, axis[1, 1]]}
avker_cbar_kwargs = {'title' : r'$\partial\hat{x}_i/\partial x_i$', 'x' : 4,
                     'cbar_pad_inches' : 0.1}
avker_kwargs = {'cmap' : plasma_trans, 'vmin' : 0, 'vmax' : 1,
                'cbar_kwargs' : avker_cbar_kwargs,
                'fig_kwargs' : fig_kwargs}
title = f'Averaging kernel\nsensitivities' # ({f}\%)'
fig, axis[1, 1], c = ip.plot_state(dofs_permian['mean'], permian, title=title,
                                 **avker_kwargs)
dofs = dofs_permian['mean'].sum()
axis[1, 1].text(0.05, 0.05,
                f'DOFS = {dofs:.1f}',
                fontsize=config.LABEL_FONTSIZE*config.SCALE*1,
                transform=axis[1, 1].transAxes)


fp.save_fig(fig, plot_dir, f'permian_ensemble')
plt.close()

