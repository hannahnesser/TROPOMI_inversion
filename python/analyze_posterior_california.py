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
emis = {'Oil and natural gas' : 'ong', 
        'Coal' : 'coal', 
        'Landfills' : 'landfills', 
        'Wastewater' : 'wastewater', 
        'Livestock' : 'livestock',
        'Other anthropogenic' : 'other_anth'}

basin_studies = {
    'Fairley and Fischer (2015)' : {'San Francisco Bay' : [240, 60, 60]}, # 2009 - 2012 https://www.sciencedirect.com/science/article/pii/S1352231015000886
    'Jeong et al. (2016)' : {'South Coast' : [380, 79, 110],
                             'San Francisco Bay' : [245, 86, 95]}, # values are medians June 2013 - May 2014 https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/2016JD025404
    'Jeong et al. (2017)' : {'San Francisco Bay' : [226, 60, 63]}, # Median for Sept - Dec 2015, much of underestimation from landfills https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/2016GL071794
    'Guha et al. (2020)' : {'San Francisco Bay' : [222.3, 40, 40]}, # 88% increase over inventory, 2015 - 2019 measurements https://pubs-acs-org.ezp-prod1.hul.harvard.edu/doi/pdf/10.1021/acs.est.0c01212
    'Cui et al. (2015)' : {'South Coast' : [406, 81, 81]}, # mean of 6 flights flown over 2010 https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/2014JD023002
    'Wunch et al. (2016)' : {'South Coast' : [413, 86, 86]}, # 2007 - 2016 average, relatively steady methane emissions https://acp.copernicus.org/articles/16/14091/2016/acp-16-14091-2016.pdf
    'Yadav et al. (2019)' : {'South Coast' : [333, 89, 89]}, # 2015-2016 mean, excluding Aliso Canyon leak https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JD030062
    'Kuwayama et al. (2019)' : {'South Coast' : [177, 19, 19]}, #158 196
    'Cusworth et al. (2020)' : {'South Coast' : [274, 72, 72]}, # Multi-tiered inversion with CLARS-FTS and TROPOMI aand AVIRIS-NG for Jan 2017- Sept 2018 https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2020GL087869
    'Yadav et al. (2023)' : {'South Coast' : [251, 10, 10]}
    # 'Karion et al. (2015)' : {'Dallas' : [660, 110, 110]}, # DFW mass baalance from 8 different flight days in March and October 2013 https://pubs.acs.org/doi/full/10.1021/acs.est.5b00217
               }

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load the cities mask
w_airbasin = pd.read_csv(f'{data_dir}states/airbasin_mask.csv', header=0).T
print(w_airbasin)

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()

# Load area (km2)
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load ensemble members (relative posterior and DOFS), all of which are
# previously filtered on DOFS and have the BC elements removed as needed
dofs = pd.read_csv(f'{data_dir}ensemble/dofs.csv', index_col=0)
xhat = pd.read_csv(f'{data_dir}ensemble/xhat.csv', index_col=0)
ensemble = xhat.columns

# ID two priors
w404_cols = [s for s in ensemble if 'w404' in s]

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
w = pd.read_csv(f'{data_dir}w_w404_edf.csv')
w = dc(w[list(emis.values())])
w['total'] = w.sum(axis=1)
w = w.T*1e-3

# Get the posterior xhat_abs (this is n x 15)
xhat_abs = (w.loc['total'].values[:, None]*xhat)

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

## Difference
diff_c = post_c - prior_c['prior_total'].values[:, None]

## Posterior ratios
xhat_c = post_c/prior_c['prior_total'].values[:, None]

## Get statistics
xhat_stats_c = ip.get_ensemble_stats(xhat_c).add_prefix('xhat_')
post_stats_c = ip.get_ensemble_stats(post_c).add_prefix('post_')
diff_stats_c = ip.get_ensemble_stats(diff_c).add_prefix('diff_')

## Aggregate
summ_c = pd.concat([pd.DataFrame(area_c), prior_c,
                    post_stats_c, xhat_stats_c, diff_stats_c], axis=1)

## ------------------------------------------------------------------------ ##
## Plot results
## ------------------------------------------------------------------------ ##
# Subset
summ_c = summ_c[summ_c['post_mean'] > 50]

# Adjust min/max definitions for error bars
summ_c['post_max'] = summ_c['post_max'] - summ_c['post_mean']
summ_c['post_min'] = summ_c['post_mean'] - summ_c['post_min']

# Define ys
ys = np.arange(1, summ_c.shape[0] + 1)

# And begin the plot!
fig, ax = fp.get_figax(aspect=3*6/7, max_width=config.BASE_WIDTH) 
#                        # max_height=config.BASE_HEIGHT*config.SCALE)
# plt.subplots_adjust(wspace=0.1)

# Get labels
labels = summ_c.index.values

# Plot stacked bar
ax = fp.add_title(ax, 'California air basin emissions', 
                  fontsize=config.TITLE_FONTSIZE)

# Formatters
cc = [fp.color(i, lut=2*7) for i in [6, 10, 2, 8, 0, 12, 0]]
formats = ['o', 's', '^']
sizes = [4, 4, 4]

# Prior
left_prior = np.zeros(len(ys))
for i, (l, e) in enumerate(emis.items()):
    ax.barh(ys - 0.175, summ_c[f'prior_{e}'], left=left_prior, 
               height=0.3, color=cc[i], label=f'{l}')
    left_prior += summ_c[f'prior_{e}']

# # Prior #2
# left_prior = np.zeros(len(ys))
# for i, (l, e) in enumerate(emis.items()):
#     ax.barh(ys - 0.0875, summ_c[f'prior2019_{e}'], left=left_prior, 
#                height=0.125, color=cc[i])
#     left_prior += summ_c[f'prior2019_{e}']
#     if e == 'ong':
#         ax.barh(ys - 0.0875, (summ_c['pop_2010']/pop_conus)*(11.4/25*1e3),
#                    left=left_prior, height=0.125, color=cc[i], alpha=0.6,
#                    label='Post meter natural gas')
#         left_prior += summ_c[f'pop_2010']/pop_conus*(11.4/25*1e3)
# # ax.barh(ys, )

# Posterior
ax.barh(ys + 0.175, summ_c['post_mean'],
           xerr=np.array(summ_c[['post_min', 'post_max']]).T,
           error_kw={'ecolor' : '0.6', 'lw' : 0.5, 'capsize' : 1,
                     'capthick' : 0.5},
           height=0.3, color=fp.color(3), alpha=0.3, 
           label='Posterior total')

# Other studies
i = 0
for cs_name, cs in basin_studies.items():
    study_used = False
    for c_name, cd in cs.items():
        if c_name in labels:
            if ~study_used:
                label = cs_name
            else:
                label = None
            study_used = True
            # print(cs_name, i, formats[i])
            y = np.argwhere(c_name == labels)[0][0]
            ax.errorbar(cd[0], y + 1, xerr=np.array(cd[1:])[:, None], 
                           fmt=formats[i % 3], markersize=sizes[i % 3], 
                           markerfacecolor=fp.color(math.ceil((i + 1)/3), 
                                                    cmap='viridis', lut=5),
                           markeredgecolor='black',
                           ecolor='black', elinewidth=0.25, 
                           capsize=1, capthick=0.5, zorder=10,
                           label=label)
            print(cs_name, c_name, i, math.ceil((i + 1)/3), fp.color(math.ceil((i + 1)/3), cmap='viridis', lut=5))

    if study_used:
        i += 1

# Add labels
ax.set_yticks(ys)
ax.set_ylim(0.5, len(ys) + 0.5)
ax.invert_yaxis()
ax.tick_params(axis='both', labelsize=config.TICK_FONTSIZE)

# Deal with scales
# ax.set_xlim(0, 2e3)
# ax.set_xscale('log')
# ax.set_xlim(10, 2.1e3)
ax.set_xlim(0, 1.2e3)
# ax.set_xscale('log')

# Final aesthetics
ax.set_yticklabels(labels, ha='right', fontsize=config.TICK_FONTSIZE)
ax = fp.add_labels(ax, r'Emissions (Gg a$^{-1}$)', '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)

for i in range(20):
    ax.axvline((i + 1)*250, color='0.75', lw=0.5, zorder=-10)

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
ax = fp.add_legend(ax, handles=handles, labels=labels, ncol=2,
                   fontsize=config.TICK_FONTSIZE, loc='upper center', 
                   bbox_to_anchor=(0.5, -0.5))

fp.save_fig(fig, plot_dir, f'california_ensemble')