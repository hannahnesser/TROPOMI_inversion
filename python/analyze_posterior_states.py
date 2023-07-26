import sys
import glob
from copy import deepcopy as dc
import math
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
paper_dir = base_dir + 'paper/figures/'

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
sectors = ['ong', 'livestock', 'landfills', 'coal', 'wastewater', 'other_anth']
sectors = sectors[::-1]

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load the states mask
w_state = pd.read_csv(f'{data_dir}states/states_mask.csv', header=0).T
conus_mask = np.load(f'{data_dir}countries/CONUS_mask.npy').reshape((-1,))

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()

# Plot difference between conus mask and states mask
fig, ax, c = ip.plot_state(w_state.sum() - conus_mask, clusters)
fp.save_fig(fig, plot_dir, 'state_vs_conus_mask')

# Load area (km2)
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load ensemble members (relative posterior and DOFS), all of which are
# previously filtered on DOFS and have the BC elements removed as needed
dofs = pd.read_csv(f'{data_dir}ensemble/dofs.csv', index_col=0)
xhat = pd.read_csv(f'{data_dir}ensemble/xhat.csv', index_col=0)
ensemble = xhat.columns

# Load reduced DOFS
a_files = glob.glob(f'{data_dir}ensemble/a2_*_states.csv')
a_files.sort()
dofs_s = pd.DataFrame(index=w_state.index, columns=xhat.columns)
for f in a_files:
    short_f = f.split('/')[-1][3:-19]
    a_r = pd.read_csv(f, header=0, index_col=0)
    dofs_s[short_f] = np.diag(a_r)
dofs_s = ip.get_ensemble_stats(dofs_s).add_prefix('dofs_')

# Load weighting matrices in units Gg/yr (we don't consider wetlands
# here, so it doesn't matter which W we use)
w = pd.read_csv(f'{data_dir}sectors/w.csv')
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

# Load EPA state estimates
epa_s = pd.read_csv(f'{data_dir}states/states_epa.csv', header=0, index_col=0)

# Individual state inventories in Tg/yr
state_inventories = {
    'California' : 1.556, # 2022 inventory (value for 2019) https://ww2.arb.ca.gov/sites/default/files/classic/cc/inventory/ghg_inventory_bygas.pdf
    'Colorado' : 1.338, # 2021 GHG inventory (value for 2019) https://cdphe.colorado.gov/air-pollution/climate-change#inventory
    'Iowa' : 0.7724, # 2019 GHG inventory (value for 2019) https://www.iowadnr.gov/Portals/idnr/uploads/air/ghgemissions/Final%202019%20GHG%20REPORT.pdf
    'Louisiana' : 0.375, # 2021 GHG inventory (value for 2018) https://www.lsu.edu/ces/publications/2021/louisiana-2021-greehouse-gas-inventory-df-rev_reduced.pdf
    'Maryland' : 0.181, # 2020 GHG inventory (value for 2020) https://mde.maryland.gov/programs/Air/ClimateChange/Pages/GreenhouseGasInventory.aspx
    'New York' : 0.913, # 2021 GHG inventory (value for 2019) with out of state emissions subtracted https://www.dec.ny.gov/docs/administration_pdf/ghgsumrpt21.pdf
    'Pennsylvania' : 1.278, # 2022 GHG inventory (value for 2019) https://files.dep.state.pa.us/Energy/Office%20of%20Energy%20and%20Technology/OETDPortalFiles/ClimateChange/PennsylvaniaGreenhouseGasInventory2022.pdf
    'Wisconsin' : 0.588 # 2021 GHG inventory (value for 2018) https://dnr.wisconsin.gov/climatechange/science
                     }
# state_inventories = pd.DataFrame(state_inventories)*1e-3 # Convert to Tg

# Delaware does not have a specific CH4 estimate
# Illinois does not have a specific CH4 estimate
# Maine does not have a specific CH4 estimate
# Massachusetts does not have a specific CH4 estimate
# Nevada does not have a specific CH4 estimate
# New Hampshire had a broken link
# New Jersey does not have a specific CH4 estimate
# New Mexico does not have a consolidated CH4 estimate https://cnee.colostate.edu/wp-content/uploads/2021/01/New-Mexico-GHG-Inventory-and-Forecast-Report_2020-10-27_final.pdf
# North Carolina does not have a specific CH4 estimate
# Oregon does not have a specific CH4 estimate
# Rhode Island does not have a specific CH4 estimate
# Vermont does not have a specific CH4 estimate
# Virginia does not have a consolidated CH4 estimate
# Washington does not have a specific CH4 estimate

## ------------------------------------------------------------------------ ##
## Get list of population
## ------------------------------------------------------------------------ ##
pop = pd.read_csv(f'{data_dir}states/states_pop.csv', header=0,
                  dtype={'State' : str, '2019' : int})
pop = pop.rename(columns={'State' : 'name'})
pop = pop.iloc[5:, :].reset_index(drop=True)
pop = pop.sort_values(by='2019', ascending=False, ignore_index=True)
pop = pop.set_index('name')

## ------------------------------------------------------------------------ ##
## Calculate state sectoral prior and posterior
## ------------------------------------------------------------------------ ##
# Calculate the area, prior emissions, posterior emissions, and xhat
# for each column, then concatenate it into one dataframe
## Area
area_s = (w_state @ area.reshape(-1,)).rename('area')

## Prior (the same for all ensemble members) 
## (w_state = 48 x 23691, w.T = 23691 x sectors)
prior_s = (w_state @ w.T).add_prefix('prior_')

## Posterior
post_s = (w_state @ xhat_abs)
post_sect_s = (w_state @ xhat_abs_sect)

## Difference
diff_s = post_s - prior_s['prior_total'].values[:, None]

## Posterior ratios
xhat_s = post_s/prior_s['prior_total'].values[:, None]

# ## Fraction of optimized emissions
# dofs_mask = np.ones(dofs.shape)
# denom = w_state @ (w.loc['total'].values[:, None]*dofs_mask)
# dofs_mask[dofs < DOFS_filter] = 0
# numer = w_state @ (w.loc['total'].values[:, None]*dofs_mask)
# frac_s = numer/denom
# frac_stats_s = ip.get_ensemble_stats(frac_s).add_prefix('frac_')

## Get statistics
xhat_stats_s = ip.get_ensemble_stats(xhat_s).add_prefix('xhat_')
post_stats_s = ip.get_ensemble_stats(post_s).add_prefix('post_')
diff_stats_s = ip.get_ensemble_stats(diff_s).add_prefix('diff_')

## Aggregate
summ_s = pd.concat([pd.DataFrame(area_s), prior_s, post_sect_s,
                    post_stats_s, xhat_stats_s, diff_stats_s], axis=1)

# Merge DOFS into summ_c
summ_s = pd.concat([summ_s, dofs_s.loc[summ_s.index]], axis=1)

# Merge in population
summ_s['pop'] = pop.loc[summ_s.index]['2019']

# Add in EPA state estimates
epa_s = epa_s.add_prefix('epa_')/25*1e3
summ_s = summ_s.join(epa_s)
# summ_s['epa'] = epa_s.loc[summ_s.index][['total']]/25*1e3 # to Gg/yr
# summ_s['epa'] *= summ_s['post_mean'].sum()/summ_s['epa'].sum()

# Per capita methane emissions (Gg/person)
summ_s['post_mean_pc'] = summ_s['post_mean']/summ_s['pop']

# Sort by posterior anthropogenic emissions
summ_s = summ_s.sort_values(by='post_mean', ascending=False)

# Compare the EPA inventory to our posterior
print((summ_s['post_mean'] - summ_s['epa_total']).mean())
for n in [10, 25, 50]:
    epa = summ_s['epa_total'][:n]
    gepa = summ_s['prior_total'][:n]
    post = summ_s['post_mean'][:n]
    pct_change = (post - epa)/epa
    pct_change_g = (post - gepa)/gepa
    print(f'{n} states: {100*pct_change.mean():.2f} {100*pct_change_g.mean():.2f}')

print((post - epa).sort_values(ascending=False))

# Save out csv
summ_s.to_csv(f'{data_dir}/states/summ_states.csv', header=True, index=True)

# Subset for largest emissions
# summ_s = summ_s[summ_s['post_mean'] > 100]
rel_cumsum = np.cumsum(summ_s['post_mean'])/summ_s['post_mean'].sum()
summ_s = summ_s.loc[rel_cumsum <= 0.9]
ns = summ_s.shape[0]
print(f'Plotting results for {ns} states.') 

## ------------------------------------------------------------------------ ##
## Plot maps
## ------------------------------------------------------------------------ ##
# state = shapefile.Reader(f'{data_dir}states/2019_tl_us_state/tl_2019_us_state.shp')

# fig, ax = fp.get_figax(maps=True, lats=clusters.lat, lons=clusters.lon)
# for shape in state.shapeRecords():
#     if shape.record[6] in summ_s.index.values:
#         # Get edges of city
#         x = [i[0] for i in shape.shape.points[:]]
#         y = [i[1] for i in shape.shape.points[:]]
#         c_poly = Polygon(np.column_stack((x, y)))
#         color = sf_cmap(div_norm(summ_s.loc[shape.record[6]]['xhat_mean']))
#         ax.fill(x, y, facecolor=color, edgecolor='black')
# ax = fp.format_map(ax, lats=clusters.lat, lons=clusters.lon)
# cmap = plt.cm.ScalarMappable(cmap=sf_cmap, norm=div_norm)
# cax = fp.add_cax(fig, ax)
# cb = fig.colorbar(cmap, ax=ax, cax=cax)
# cb = fp.format_cbar(cb, 'Scale factor')
# fp.save_fig(fig, plot_dir, f'states_map')

# ## ------------------------------------------------------------------------ ##
# ## Plot results : histogram
# ## ------------------------------------------------------------------------ ##
# fig, ax = fp.get_figax(aspect=1.5)
# ax.hist(summ_s['post_mean'], bins=40, color=fp.color(3))
# fp.save_fig(fig, plot_dir, 'states_histogram')

## ------------------------------------------------------------------------ ##
## Plot results : absolute emissions
## ------------------------------------------------------------------------ ##
# Define x range
xs = np.arange(1, ns + 1)

# Begin plotting!
# fig, ax = fp.get_figax(rows=4, aspect=4, sharex=True,
#                        max_height=config.BASE_HEIGHT*config.SCALE)
figsize = fp.get_figsize(aspect=1.8, 
                         max_height=config.BASE_HEIGHT*config.SCALE,
                         max_width=config.BASE_WIDTH*config.SCALE*0.8)
fig = plt.figure(figsize=figsize)

ylim2 = [0, 2.25e3]
ylim1 = [3.6e3, 6.55e3]
# 6 total
total_height = (ylim2[1] - ylim2[0]) + (ylim1[1] - ylim1[0])
height_ratios = [(ylim1[1] - ylim1[0])/total_height/3, 
                 (ylim2[1] - ylim2[0])/total_height]

gs = gridspec.GridSpec(2, 1, height_ratios=[0.25, 1], hspace=0.1)
gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=height_ratios,
                                       subplot_spec=gs[1], hspace=0.1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs2[0], sharex=ax0)
ax2 = plt.subplot(gs2[1], sharex=ax0)
ax = [ax0, ax1, ax2]
# Top row: fraction of posterior emissions and prior emissions explained by
# each state (sectorally?) (or fraction and cumulative)
# Middle row: posterior and prior sectoral adjustments
# Bottom row: averaging kernel values

# Get labels
labels = summ_s.index.values

# Adjust min/max definitions for error bars
summ_s['post_max'] = summ_s['post_max'] - summ_s['post_mean']
summ_s['post_min'] = summ_s['post_mean'] - summ_s['post_min']
summ_s['dofs_max'] = summ_s['dofs_max'] - summ_s['dofs_mean']
summ_s['dofs_min'] = summ_s['dofs_mean'] - summ_s['dofs_min']
# summ_s['frac_max'] = summ_s['frac_max'] - summ_s['frac_mean']
# summ_s['frac_min'] = summ_s['frac_mean'] - summ_s['frac_min']

# Add title
ax[0] = fp.add_title(ax[0], 
                     'State emissions', 
                     fontsize=config.TITLE_FONTSIZE)

# Plot DOFS
ax[0].errorbar(xs, summ_s['dofs_mean'], #fmt='none',
                yerr=np.array(summ_s[['dofs_min', 'dofs_max']]).T,
                fmt='D', markersize=4, markerfacecolor='white', 
                markeredgecolor='0.65', 
                ecolor='0.65', elinewidth=0.75, capsize=2, capthick=0.75)
ax[0].set_ylim(0, 1)
ax[0].set_yticks([0, 0.5, 1])
ax[0].set_ylabel('Sensitivity', fontsize=config.TICK_FONTSIZE, labelpad=20)

# Plot stacked bar
bottom_prior = np.zeros(ns)
bottom_post = np.zeros(ns)
for i, e in enumerate(sectors):
    l = list(s.sectors.keys())[list(s.sectors.values()).index(e)]
    for j in [1, 2]:
        ax[j].bar(xs - 0.175, summ_s[f'epa_{e}'], bottom=bottom_prior, 
                  width=0.3, color=s.sector_colors[e], label=l)
        ax[j].bar(xs + 0.175, summ_s[f'post_{e}'], bottom=bottom_post, 
                  width=0.3, color=s.sector_colors[e])
        if i == 0:
            ax[j].errorbar(xs + 0.175, summ_s['post_mean'],
                           yerr=np.array(summ_s[['post_min', 'post_max']]).T,
                           ecolor='0.65', lw=0.75, capsize=2, 
                           capthick=0.75, fmt='none', zorder=10)


    bottom_prior += summ_s[f'epa_{e}']
    bottom_post += summ_s[f'post_{e}']

# Add the state estimates
for s, si in state_inventories.items():
    if s == 'California':
        label = 'State inventory'
    else:
        label = None
    try:
        x = np.argwhere(s == summ_s.index)[0][0]
    except:
        continue
    ax[2].scatter(x + 1, si*1e3, s=16, marker='o', color='white', 
                  edgecolor='black', zorder=10, label=label)

# for i in [1, 2]:
#     if i == 2:
#         label = 'EPA state inventory'
#     else:
#         label = None
#     # Get scaling so that total agrees
#     ax[i].scatter(xs, summ_s['epa'], s=10, marker='o', 
#                   color='white', edgecolor='black', zorder=10, label=label)

# Split the axis
ax[2].set_ylim(*ylim2)
ax[1].set_ylim(*ylim1)
# ax[2].set_ylim(0, 2.75e3)
# ax[1].set_ylim(4.25e3, 7e3)

ax[1].spines['bottom'].set_visible(False)
ax[2].spines['top'].set_visible(False)
# ax[1].xaxis.tick_top()
ax[1].tick_params(labeltop=False)
ax[2].xaxis.tick_bottom()
d = 0.005  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax[1].transAxes, color='k', clip_on=False,
              lw=0.5)
ax[1].plot((-d, +d), (-3*d, +3*d), **kwargs)        # top-left diagonal
ax[1].plot((1 - d, 1 + d), (-3*d, +3*d), **kwargs)  # top-right diagonal
kwargs.update(transform=ax[2].transAxes)  # switch to the bottom axes
ax[2].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax[2].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# # Final aesthetics
ax[1] = fp.add_labels(ax[1], '', r'Methane emissions (Gg a$^{-1}$)', 
                      fontsize=config.TICK_FONTSIZE, 
                      labelpad=20)

# Final aesthetics
for j in range(3):
    for k in range(ns + 1):
        if k % 5 == 0:
            ls = '-'
        else:
            ls = ':'
        ax[j].axvline((k) + 0.5, color='0.75', alpha=1, lw=0.5, ls=ls,
                      zorder=-10)
        # if j == 2:
        #     ax[j].plot(((k + 1) + 0.5, (k + 1) + 0.5), (2.25e3, 2.475e3),
        #                color='0.75', alpha=1, lw=0.5, clip_on=False)
    ax[j].set_xticks(xs)
    ax[j].set_xlim(0, ns + 1)
    ax[j].tick_params(axis='both', labelsize=config.TICK_FONTSIZE)
    if j < 2:
        plt.setp(ax[j].get_xticklabels(), visible=False)

# Add x_ticklabels
ax[-1].set_xticklabels(labels, ha='center', fontsize=config.TICK_FONTSIZE,
                      rotation=90)

# Add horizontal lines
for k in range(4):
    ax[0].axhline((k + 1)*0.25, color='0.75', lw=0.5, zorder=-10)
ax[0].set_yticks(np.arange(0, 5)/4)
for j in [1, 2]:
    for k in range(6):
        ax[j].axhline((k + 1)*1000, color='0.75', lw=0.5, zorder=-10)
ax[0].set_yticks([0, 0.5, 1])
ax[1].set_yticks(np.arange(math.ceil(ylim1[0]/1000)*1000, ylim1[1], 1000))
ax[2].set_yticks(np.arange(math.ceil(ylim2[0]/1000)*1000, ylim2[1], 1000))
ax[1].tick_params(axis='x', which='both', bottom=False)

# Legend for summary plot
handles, labels = ax[2].get_legend_handles_labels()
# handles.extend(m_handles)
# labels.extend(m_labels)
ax[1] = fp.add_legend(ax[1], handles=handles[::-1], labels=labels[::-1], 
                      ncol=1, fontsize=config.TICK_FONTSIZE, 
                      loc='upper right')
ax[1].set_zorder(10)

# Set label coords
y0 = (ax[0].get_position().y1 + ax[0].get_position().y0)/2
ax[0].yaxis.set_label_coords(0.07, y0, transform=fig.transFigure)
y1 = (ax[1].get_position().y1 + ax[-1].get_position().y0)/2
ax[1].yaxis.set_label_coords(0.07, y1, transform=fig.transFigure)

# Add labels
ax[1].text(0.625, summ_s['epa_total'][0], '2022 EPA GHGI for 2019', ha='right',
           #summ_s['prior_total'][1] + 100, 'Prior', ha='right', 
           va='top', rotation=90, fontsize=config.TICK_FONTSIZE - 2)
ax[1].text(1.5, summ_s['post_mean'][0], 'Posterior',
           #summ_s['post_total'][1] + 100, 'Posterior', 
           ha='left', va='top', rotation=90, 
           fontsize=config.TICK_FONTSIZE - 2)

fp.save_fig(fig, plot_dir, f'states_ensemble')
fp.save_fig(fig, paper_dir, 'fig06', for_acp=True)

## ------------------------------------------------------------------------ ##
## Get Permian-Texas overlap
## ------------------------------------------------------------------------ ##
permian_idx = np.load(f'{data_dir}ong/permian_idx.npy')
w_tx_nm = w_state.loc[['Texas','New Mexico']]
w_tx_nm = w_tx_nm.iloc[:, permian_idx]
xhat_abs_permian = xhat_abs.iloc[permian_idx, :]
xhat_abs_permian = w_tx_nm @ xhat_abs_permian
xhat_abs_permian = ip.get_ensemble_stats(xhat_abs_permian)
print(xhat_abs_permian/summ_s[['post_mean', 'post_min', 'post_max']].loc[['Texas', 'New Mexico']].values)

## ------------------------------------------------------------------------ ##
## Plot results : change in rankings
## ------------------------------------------------------------------------ ##
# summ_s = summ_s.sort_values(by='prior_total', ascending=False)
# prior_r = (summ_s['prior_total'].values).argsort()[::-1]
# post_r = (summ_s['post_mean'].values).argsort()[::-1]

# # print('-'*75)
# # print('Original ranking')
# # print([f'{i+1}. {j}' for i, j in enumerate(summ_s.index[prior_r])])
# # print('-'*75)
# # print('Updated ranking')
# # print([f'{i+1}. {j}' for i, j in enumerate(summ_s.index[post_r])])

# fig, ax = fp.get_figax(aspect=0.25)
# cmap_1 = plt.cm.RdBu_r(np.linspace(0, 0.5, 256))
# cmap_2 = plt.cm.RdBu_r(np.linspace(0.5, 1, 256))
# cmap = np.vstack((cmap_1, cmap_2))
# cmap = colors.LinearSegmentedColormap.from_list('cmap', cmap)
# div_norm = colors.TwoSlopeNorm(vmin=-28, vcenter=0, vmax=28)

# for i in range(ns):
#     color = cmap(div_norm(post_r[i] - prior_r[i]))
#     ax.plot([0, 1], [post_r[i], prior_r[i]], c=color, 
#             marker='o', markersize=5)

# ax1 = ax.twinx()
# ax.set_xticks([])
# for axis in [ax, ax1]:
#     axis.set_yticks(prior_r)
#     axis.set_ylim(ns + 0.5, -0.5)
#     axis.spines['bottom'].set_visible(False)
#     axis.spines['top'].set_visible(False)
#     axis.spines['right'].set_visible(False)
#     axis.spines['left'].set_visible(False)
# ax.set_yticklabels([f'{i+1}. {j}' for i, j in enumerate(summ_s.index[prior_r])],
#                     fontsize=config.TICK_FONTSIZE)
# ax1.set_yticklabels([f'{i+1}. {j}' for i, j in enumerate(summ_s.index[post_r])],
#                    fontsize=config.TICK_FONTSIZE)

# ax.set_xlim(-0.05, 1.05)
# fp.save_fig(fig, plot_dir, f'states_ranking')
