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
DOFS_filter = 0.05

# Number of states
ns = 48

# Define emission categories
emis = {'Livestock' : 'livestock', 
        'Oil and natural gas' : 'ong', 
        'Coal' : 'coal', 
        'Landfills' : 'landfills', 
        'Wastewater' : 'wastewater', 
        'Other anthropogenic' : 'other_anth'}

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load the states mask
w_state = pd.read_csv(f'{data_dir}states/states_mask.csv', header=0).T

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()

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
w = pd.read_csv(f'{data_dir}w_w404_edf.csv')
w = dc(w[list(emis.values())])
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
epa_s = pd.read_csv(f'{data_dir}states/states_epa.csv', 
                        header=0, index_col=0)

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

## Fraction of optimized emissions
dofs_mask = np.ones(dofs.shape)
denom = w_state @ (w.loc['total'].values[:, None]*dofs_mask)
dofs_mask[dofs < DOFS_filter] = 0
numer = w_state @ (w.loc['total'].values[:, None]*dofs_mask)
frac_s = numer/denom
frac_stats_s = ip.get_ensemble_stats(frac_s).add_prefix('frac_')

## Get statistics
xhat_stats_s = ip.get_ensemble_stats(xhat_s).add_prefix('xhat_')
post_stats_s = ip.get_ensemble_stats(post_s).add_prefix('post_')
diff_stats_s = ip.get_ensemble_stats(diff_s).add_prefix('diff_')

## Aggregate
summ_s = pd.concat([pd.DataFrame(area_s), prior_s, post_sect_s,
                    post_stats_s, xhat_stats_s, diff_stats_s,
                    frac_stats_s], axis=1)

# Merge DOFS into summ_c
summ_s = pd.concat([summ_s, dofs_s.loc[summ_s.index]], axis=1)

# Merge in population
summ_s['pop'] = pop.loc[summ_s.index]['2019']

# Per capita methane emissions (Gg/person)
summ_s['post_mean_pc'] = summ_s['post_mean']/summ_s['pop']

# Sort by posterior anthropogenic emissions
summ_s = summ_s.sort_values(by='post_mean', ascending=False)
epa_s = epa_s.loc[summ_s.index]

# Save out csv
summ_s.to_csv(f'{data_dir}/states/summ_states.csv', header=True, index=True)

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

## ------------------------------------------------------------------------ ##
## Plot results : absolute emissions
## ------------------------------------------------------------------------ ##
# Define x range
xs = np.arange(1, ns + 1)

# Begin plotting!
# fig, ax = fp.get_figax(rows=4, aspect=4, sharex=True,
#                        max_height=config.BASE_HEIGHT*config.SCALE)
figsize = fp.get_figsize(aspect=1, 
                         max_height=config.BASE_HEIGHT*config.SCALE)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.1)
gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[0.61, 0.39],
                                       subplot_spec=gs[1], hspace=0.1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs2[0], sharex=ax0)
ax2 = plt.subplot(gs2[1], sharex=ax0)
ax3 = plt.subplot(gs[2], sharex=ax0)
ax = [ax0, ax1, ax2, ax3]
# Top row: fraction of posterior emissions and prior emissions explained by
# each state (sectorally?) (or fraction and cumulative)
# Middle row: posterior and prior sectoral adjustments
# Bottom row: averaging kernel values

# Get labels
labels = summ_s.index.values

# Plot cumulative contribution and relative contribution
## Prior cumulative contribution
prior_tot = summ_s['prior_total'].values.sum()
prior_cum_sum = np.cumsum(summ_s['prior_total'])/prior_tot

## Posterior cumulative contribution
post_tot = post_s.loc[summ_s.index].sum(axis=0)
post_cum_sum = np.cumsum(post_s.loc[summ_s.index], axis=0)/post_tot

## Plot
for i, ensemble_member in enumerate(post_cum_sum.columns):
    if i == 0:
        label = 'Ensemble members'
    else:
        label = None
    ax[0].plot(xs, post_cum_sum[ensemble_member], c='0.5', alpha=0.5, lw=0.5,
               label=label)
ax[0].plot(xs, post_cum_sum.mean(axis=1), c='black', lw=1,
           label='Ensemble mean')

ax[0] = fp.add_labels(ax[0], '', 'Cumulative fraction of\nCONUS posterior emissions', 
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=20)
ax[0].set_ylim(0.15, 1.05)
ax[0] = fp.add_legend(ax[0], ncol=1,
                      fontsize=config.TICK_FONTSIZE, loc='lower right')


# Adjust min/max definitions for error bars
summ_s['post_max'] = summ_s['post_max'] - summ_s['post_mean']
summ_s['post_min'] = summ_s['post_mean'] - summ_s['post_min']
summ_s['dofs_max'] = summ_s['dofs_max'] - summ_s['dofs_mean']
summ_s['dofs_min'] = summ_s['dofs_mean'] - summ_s['dofs_min']
summ_s['frac_max'] = summ_s['frac_max'] - summ_s['frac_mean']
summ_s['frac_min'] = summ_s['frac_mean'] - summ_s['frac_min']

# Plot stacked bar
cc = [fp.color(i, lut=2*7) for i in [0, 6, 10, 2, 8, 12, 0]]
bottom_prior = np.zeros(ns)
bottom_post = np.zeros(ns)
for i, (l, e) in enumerate(emis.items()):
    for j in [1, 2]:
        ax[j].bar(xs - 0.175, summ_s[f'prior_{e}'], 
                  bottom=bottom_prior, width=0.3, color=cc[i], label=l)
        ax[j].bar(xs + 0.175, summ_s[f'post_{e}'], 
                  bottom=bottom_post, width=0.3, color=cc[i])
        if i == 0:
            ax[j].errorbar(xs + 0.175, summ_s['post_mean'],
                           yerr=np.array(summ_s[['post_min', 'post_max']]).T,
                           ecolor='0.6', lw=0.5, capsize=1, capthick=0.5,
                           fmt='none', zorder=10)


    bottom_prior += summ_s[f'prior_{e}']
    bottom_post += summ_s[f'post_{e}']

# Add the state estimates
for s, si in state_inventories.items():
    if s == 'California':
        label = 'State inventory'
    else:
        label = None
    x = np.argwhere(s == summ_s.index)[0][0]
    ax[2].scatter(x + 1, si*1e3, s=15, marker='^', color='white', 
                  edgecolor='black', zorder=10, label=label)

for i in [1, 2]:
    if i == 2:
        label = 'Scaled EPA state\ninventory'
    else:
        label = None
    # Get scaling so that total agrees
    scaling = summ_s['post_mean'].sum()/(epa_s['total']/25*1e3).sum()

    ax[i].scatter(xs, scaling*epa_s['total']/25*1e3, s=20/3, marker='o', 
                  color='white', edgecolor='black', zorder=10, label=label)

# Compare the EPA inventory to our posterior
# n = 10
# epa = scaling*epa_s['total'][:n]/25*1e3
# post = summ_s['post_mean'][:n]
# pct_change = (post - epa)/epa
# print(pct_change*100)
# print(f'{100*pct_change.mean():.2f}')

# n = 25
# epa = scaling*epa_s['total'][:n]/25*1e3
# post = summ_s['post_mean'][:n]
# pct_change = (post - epa)/epa
# print(f'{100*pct_change.mean():.2f}')

# n = 50
# epa = scaling*epa_s['total'][:n]/25*1e3
# post = summ_s['post_mean'][:n]
# pct_change = (post - epa)/epa
# print(f'{100*pct_change.mean():.2f}')

# print((post - epa).sort_values(ascending=False))


# Split the axis
ax[2].set_ylim(0, 2.25e3)
ax[1].set_ylim(3.5e3, 7e3)
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
ax[1].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax[1].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
kwargs.update(transform=ax[2].transAxes)  # switch to the bottom axes
ax[2].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax[2].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# # Final aesthetics
ax[1] = fp.add_labels(ax[1], '', r'Emissions (Gg a$^{-1}$)', 
                      fontsize=config.TICK_FONTSIZE, 
                      labelpad=20)

# Plot DOFS
# ax[-1].bar(xs, summ_s['dofs_mean'], color='0.3', width=0.5,
#            label='Averaging kernel sensitivities')
ax[-1].errorbar(xs, summ_s['dofs_mean'], #fmt='none',
                yerr=np.array(summ_s[['dofs_min', 'dofs_max']]).T,
                fmt='D', markersize=2.5, markerfacecolor='white', 
                markeredgecolor='black', 
                ecolor='0.6', elinewidth=0.5, capsize=1, capthick=0.5)

# # Plot percent of optimization
# ax[-1].errorbar(xs, summ_s['frac_mean'], 
#                 yerr=np.array(summ_s[['frac_min', 'frac_max']]).T,
#                 fmt='D', markersize=2.5, markerfacecolor='white', 
#                 markeredgecolor='black', ecolor='black', elinewidth=0.5, 
#                 capsize=1, capthick=0.5, 
#                 label='Optimized prior emissions')

# Labels
ax[-1].set_ylim(0, 1.15)
ax[-1].set_ylabel('State averaging\nkernel sensitivity',
                  fontsize=config.TICK_FONTSIZE, 
                  labelpad=20)
# ax[-1] = fp.add_legend(ax[-1], ncol=2, fontsize=config.TICK_FONTSIZE, 
#                        loc='upper right')


# Add x_ticklabels
ax[-1].set_xticklabels(labels, ha='center', fontsize=config.TICK_FONTSIZE,
                      rotation=90)

# Final aesthetics
for j in range(4):
    for k in range(ns - 1):
        if i % 5 == 0:
            ls = '-'
        else:
            ls = ':'
        ax[j].axvline((k + 1) + 0.5, color='0.75', alpha=1, lw=0.5, ls=ls,
                      zorder=-10)
        # if j == 2:
        #     ax[j].plot(((k + 1) + 0.5, (k + 1) + 0.5), (2.25e3, 2.475e3),
        #                color='0.75', alpha=1, lw=0.5, clip_on=False)
    ax[j].set_xticks(xs)
    ax[j].set_xlim(-0.5, ns+1+0.5)
    ax[j].tick_params(axis='both', labelsize=config.TICK_FONTSIZE)
    if j < 3:
        plt.setp(ax[j].get_xticklabels(), visible=False)

for j in [0, 3]:
    for k in range(5):
        ax[j].axhline((k + 1)*0.2, color='0.75', lw=0.5, zorder=-10)
for j in [1, 2]:
    for k in range(6):
        ax[j].axhline((k + 1)*1000, color='0.75', lw=0.5, zorder=-10)
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
ax[1].yaxis.set_label_coords(0.07, 0.5, transform=fig.transFigure)
y2 = (ax[-1].get_position().y1 + ax[-1].get_position().y0)/2
ax[-1].yaxis.set_label_coords(0.07, y2, transform=fig.transFigure)

# Add labels
ax[1].text(0.625, summ_s['prior_total'][0], 'Prior', ha='right',
           #summ_s['prior_total'][1] + 100, 'Prior', ha='right', 
           va='top', rotation=90, fontsize=config.TICK_FONTSIZE)
ax[1].text(1.5, summ_s['post_mean'][0], 'Posterior',
           #summ_s['post_total'][1] + 100, 'Posterior', 
           ha='left', va='top', rotation=90, 
           fontsize=config.TICK_FONTSIZE)

fp.save_fig(fig, plot_dir, f'states_ensemble')

## ------------------------------------------------------------------------ ##
## Plot results : change in rankings
## ------------------------------------------------------------------------ ##
summ_s = summ_s.sort_values(by='prior_total', ascending=False)
prior_r = (summ_s['prior_total'].values).argsort()[::-1]
post_r = (summ_s['post_mean'].values).argsort()[::-1]

# print('-'*75)
# print('Original ranking')
# print([f'{i+1}. {j}' for i, j in enumerate(summ_s.index[prior_r])])
# print('-'*75)
# print('Updated ranking')
# print([f'{i+1}. {j}' for i, j in enumerate(summ_s.index[post_r])])

fig, ax = fp.get_figax(aspect=0.25)
cmap_1 = plt.cm.RdBu_r(np.linspace(0, 0.5, 256))
cmap_2 = plt.cm.RdBu_r(np.linspace(0.5, 1, 256))
cmap = np.vstack((cmap_1, cmap_2))
cmap = colors.LinearSegmentedColormap.from_list('cmap', cmap)
div_norm = colors.TwoSlopeNorm(vmin=-28, vcenter=0, vmax=28)

for i in range(ns):
    color = cmap(div_norm(post_r[i] - prior_r[i]))
    ax.plot([0, 1], [post_r[i], prior_r[i]], c=color, 
            marker='o', markersize=5)

ax1 = ax.twinx()
ax.set_xticks([])
for axis in [ax, ax1]:
    axis.set_yticks(prior_r)
    axis.set_ylim(ns + 0.5, -0.5)
    axis.spines['bottom'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
ax.set_yticklabels([f'{i+1}. {j}' for i, j in enumerate(summ_s.index[prior_r])],
                    fontsize=config.TICK_FONTSIZE)
ax1.set_yticklabels([f'{i+1}. {j}' for i, j in enumerate(summ_s.index[post_r])],
                   fontsize=config.TICK_FONTSIZE)

ax.set_xlim(-0.05, 1.05)
fp.save_fig(fig, plot_dir, f'states_ranking')
