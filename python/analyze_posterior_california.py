import sys
from copy import deepcopy as dc
import glob
import math
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
from matplotlib import gridspec
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import shapefile
from shapely.geometry import Polygon, MultiPolygon
from collections import OrderedDict
pd.set_option('display.max_columns', 10)

# Custom packages
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

# Define colormaps
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

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# DOFS_filter
DOFS_filter = 0.2

# Define emission categories
sectors = ['livestock', 'ong', 'coal', 'landfills', 'wastewater', 'other_anth']

# Load other studies
basin_studies = pd.read_csv(f'{data_dir}cities/other_studies.csv')
basin_studies = basin_studies.sort_values(by=['Publication year', 'Label'], 
                                        ascending=False).reset_index(drop=True)
basin_studies = basin_studies[basin_studies['City'].isin(['San Francisco', 
                                                          'Los Angeles'])]
basin_studies.loc[basin_studies['City'] == 'San Francisco', 'City'] = 'San Francisco Bay'
basin_studies.loc[basin_studies['City'] == 'Los Angeles', 'City'] = 'South Coast'

# Jeong et al
jeong = pd.read_csv(f'{data_dir}states/jeong.csv', index_col=0)

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load the cities mask
w_airbasin = pd.read_csv(f'{data_dir}states/airbasin_mask.csv', header=0).T

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()

# Load area (km2)
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load ensemble members (relative posterior and DOFS), all of which are
# previously filtered on DOFS and have the BC elements removed as needed
dofs = pd.read_csv(f'{data_dir}ensemble/dofs.csv', index_col=0)
xhat = pd.read_csv(f'{data_dir}ensemble/xhat.csv', index_col=0)
ensemble = xhat.columns

# # Load reduced DOFS
# a_files = glob.glob(f'{data_dir}ensemble/a2_*_urban.csv')
# a_files.sort()
# dofs_c = pd.DataFrame(index=w_airbasin.index, columns=xhat.columns)
# for f in a_files:
#     short_f = f.split('/')[-1][3:-19]
#     a_r = pd.read_csv(f, header=0, index_col=0)
#     dofs_c[short_f] = np.diag(a_r)
# dofs_stats_c = ip.get_ensemble_stats(dofs_c).add_prefix('dofs_')

# Load weighting matrices in units Gg/yr (we don't consider wetlands
# here, so it doesn't matter which W we use)
w = pd.read_csv(f'{data_dir}w_w37_edf.csv')
w = dc(w[sectors])
w['total'] = w.sum(axis=1)
w = w.T*1e-3

# Get the posterior xhat_abs (this is n x 15)
xhat_abs = (w.loc['total'].values[:, None]*xhat)
xhat_abs_sect = pd.DataFrame(columns=[f'post_{s}' for s in w.index[:-1]])
for sect in w.index:
    if sect != 'total':
        xx = (w.loc[sect].values[:, None]*xhat)
        xx = ip.get_ensemble_stats(xx)['mean']
        xhat_abs_sect[f'post_{sect}'] = xx

## ------------------------------------------------------------------------ ##
## Calculate metropolitan statistical areas sectoral prior and posterior
## ------------------------------------------------------------------------ ##
# Calculate the area, prior emissions, posterior emissions, and xhat
# for each column, then concatenate it into one dataframe
## Area
area_c = (w_airbasin @ area.reshape(-1,)).rename('area')

## Prior (the same for all ensemble members)
prior_c = (w_airbasin @ w.T).add_prefix('prior_')

## Posterior (We want a matrix that is ncities x 15)
post_c = (w_airbasin @ xhat_abs)
post_sect_c = (w_airbasin @ xhat_abs_sect)

## Difference
diff_c = post_c - prior_c['prior_total'].values[:, None]

## Posterior ratios
xhat_c = post_c/prior_c['prior_total'].values[:, None]

## Get statistics
xhat_stats_c = ip.get_ensemble_stats(xhat_c).add_prefix('xhat_')
post_stats_c = ip.get_ensemble_stats(post_c).add_prefix('post_')
diff_stats_c = ip.get_ensemble_stats(diff_c).add_prefix('diff_')

## Aggregate
summ_c = pd.concat([pd.DataFrame(area_c), prior_c, post_sect_c,
                    post_stats_c, xhat_stats_c, diff_stats_c], axis=1)
summ_c.to_csv(f'{data_dir}/states/summ_california.csv', header=True, 
              index=True)

## ------------------------------------------------------------------------ ##
## Plot results
## ------------------------------------------------------------------------ ##
# Subset
summ_c = summ_c[summ_c['post_mean'] > 50]
summ_c.to_csv(f'{data_dir}/states/summ_airbasins.csv', header=True, index=True)

# Adjust min/max definitions for error bars
summ_c['post_max'] = summ_c['post_max'] - summ_c['post_mean']
summ_c['post_min'] = summ_c['post_mean'] - summ_c['post_min']

# Define ys
ys = np.arange(1, summ_c.shape[0] + 1)

# And begin the plot!
fig, ax = fp.get_figax(aspect=1.5*7/summ_c.shape[0], 
                       max_width=config.BASE_WIDTH, 
                       max_height=config.BASE_HEIGHT) 
#                        # max_height=config.BASE_HEIGHT*config.SCALE)
# plt.subplots_adjust(wspace=0.1)

# Get labels
labels = summ_c.index.values

# Plot stacked bar
ax = fp.add_title(ax, 'California air basin emissions', 
                  fontsize=config.TITLE_FONTSIZE)

# Formatters
formats = ['o', 's', '^', 'D']
sizes = [4, 4, 4, 4]

# Prior
left_prior = np.zeros(len(ys))
left_post = np.zeros(len(ys))
for i, e in enumerate(sectors):
    l = list(s.sectors.keys())[list(s.sectors.values()).index(e)]
    ax.barh(ys - 0.175, summ_c[f'prior_{e}'], left=left_prior, 
               height=0.3, color=s.sector_colors[e], label=f'{l}')
    left_prior += summ_c[f'prior_{e}']

    # Posterior
    ax.barh(ys + 0.175, summ_c[f'post_{e}'], left=left_post,
            height=0.3, color=s.sector_colors[e])
    left_post += summ_c[f'post_{e}']

ax.errorbar(summ_c[f'post_mean'], ys + 0.175,
            xerr=np.array(summ_c[['post_min', 'post_max']]).T,
            ecolor='0.65', lw=0.75, capsize=2, capthick=0.75, fmt='none', 
            zorder=10)

# Other studies
i = 0
for study in basin_studies['Label'].unique():
    studies = basin_studies[basin_studies['Label'] == study]

    # Subset for only cities in the top nc and iterate through those
    for basin in studies['City'].unique():
        result = studies[studies['City'] == basin]
        y = np.argwhere(basin == labels)[0][0]
        ax.errorbar(
            result['Mean'].values, (y + 1)*np.ones(result.shape[0]), 
            xerr=result[['Min', 'Max']].values.T, fmt=formats[i % 4], 
            markersize=sizes[i % 4], markeredgecolor='black', 
            markerfacecolor=fp.color(math.ceil((i + 1)/4), 
                                     cmap='viridis', lut=4),
            ecolor='black', elinewidth=0.25, capsize=1, capthick=0.5, 
            zorder=10, label=study)
    i += 1

jeong = jeong.loc[summ_c.index]
jeong['med'] = (jeong['min'] + jeong['max'])/2
jeong['max'] = jeong['max'] - jeong['med']
jeong['min'] = jeong['med'] - jeong['min']
ax.errorbar(jeong['med'], ys, xerr=jeong[['min', 'max']].values.T, 
            fmt=formats[i % 4], markersize=sizes[i % 4], 
            markeredgecolor='black', 
            markerfacecolor=fp.color(math.ceil((i + 1)/4), 
                                     cmap='viridis', lut=4),
            ecolor='black', elinewidth=0.25, capsize=1, capthick=0.5, 
            zorder=10, label='Jeong et al. (2016)')

# Add labels
ax.set_yticks(ys)
ax.set_ylim(0.5, len(ys) + 0.5)
ax.invert_yaxis()
ax.tick_params(axis='both', labelsize=config.TICK_FONTSIZE)

# Deal with scales
ax.set_xlim(0, 1.2e3)

# Final aesthetics
ax.set_yticklabels(labels, ha='right', fontsize=config.TICK_FONTSIZE)
ax = fp.add_labels(ax, r'Emissions (Gg a$^{-1}$)', '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)

for i in range(20):
    ax.axvline((i + 1)*200, color='0.75', lw=0.5, zorder=-10)

# Horizontal grid lines
for i in range(len(ys) + 1):
    if i % 5 == 0:
        ls = '-'
    else:
        ls = ':'
    ax.axhline(i + 0.5, color='0.75', lw=0.5, ls=ls, zorder=-10)


# Legend for summary plot
# m_handles, m_labels = ax0.get_legend_handles_labels()
handles, labels = ax.get_legend_handles_labels()
reorder = list(np.arange(0, len(handles)))
handles = [handles[i] for i in reorder]
labels = [labels[i] for i in reorder]
# print(len(labels))
# Remove duplicates
# labels = OrderedDict(zip(labels, handles))
# handles = list(labels.values())
# labels = list(labels.keys())

# # Sort
# reorder = [0, 5, 10, 15, 1, 6, 11, 16, 2, 7, 12, 3, 8, 13, 4, 9, 14]
# handles = [handles[idx] for idx in reorder]
# labels = [labels[idx] for idx in reorder]

# Add legend
ax = fp.add_legend(ax, handles=handles, labels=labels, ncol=3,
                   fontsize=config.TICK_FONTSIZE, loc='upper center', 
                   bbox_to_anchor=(0.5, -0.2))

fp.save_fig(fig, plot_dir, f'california_ensemble')