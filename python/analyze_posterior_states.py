import sys
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

# Define file names
f = 'rg2rt_10t_w404_edf_bc0_rf0.25_sax0.75_poi80.0'
xa_abs_file = 'xa_abs_w404_edf_bc0.nc'
w_file = 'w_w404_edf.csv'
optimize_BC = False

# Define emission categories
emis = {'Livestock' : 'livestock', 
        'Oil and natural gas' : 'ong', 
        'Coal' : 'coal', 
        'Landfills' : 'landfills', 
        'Wastewater' : 'wastewater', 
        'Other' : 'other_anth'}

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()

# Load prior (Mg/km2/yr)
xa_abs = xr.open_dataarray(f'{data_dir}{xa_abs_file}').values.reshape((-1, 1))
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load posterior and DOFS
dofs = np.load(f'{data_dir}ensemble/dofs2_{f}.npy').reshape((-1, 1))
xhat = np.load(f'{data_dir}ensemble/xhat_fr2_{f}.npy').reshape((-1, 1))

# Filter on DOFS filter
xhat[dofs < DOFS_filter] = 1
dofs[dofs < DOFS_filter] = 0

# BC alteration
if optimize_BC:
    print('-'*30)
    print('Boundary condition optimization')
    print(' N E S W')
    print('xhat : ', xhat[-4:])
    print('dofs : ', dofs[-4:])
    print('-'*30)
    xhat = xhat[:-4]
    dofs = dofs[:-4]

# Calculate xhat abs
xhat_abs = (xhat*xa_abs)

# Load reduced DOFS
a_r = pd.read_csv(f'{data_dir}states/a2_{f}_states.csv', header=0,
                  index_col=0)
dofs_r = pd.DataFrame({'name' : a_r.index, 'dofs' : np.diag(a_r)})

# Load weighting matrix in units Gg/yr
w = pd.read_csv(f'{data_dir}{w_file}') # Mg/yr
w = w[list(emis.values())]
w['total'] = w.sum(axis=1)
w['net'] = xa_abs*area
w = w.T*1e-3 # Mg/yr --> Gg/yr

# Load the states mask
w_state = pd.read_csv(f'{data_dir}states/states_mask.csv', header=0).T

## ------------------------------------------------------------------------ ##
## Get list of population
## ------------------------------------------------------------------------ ##
pop = pd.read_csv(f'{data_dir}states/states_pop.csv', header=0,
                  dtype={'State' : str, '2019' : int})
pop = pop.rename(columns={'State' : 'name'})
pop = pop.iloc[5:, :].reset_index(drop=True)
pop = pop.sort_values(by='2019', ascending=False, ignore_index=True)

## ------------------------------------------------------------------------ ##
## Calculate state sectoral prior and posterior
## ------------------------------------------------------------------------ ##
# Calculate the area, prior emissions, posterior emissions, and xhat
# for each column, then concatenate it into one dataframe
state_area = (w_state @ area.reshape(-1,)).rename('area')
state_prior = (w_state @ w.T).add_prefix('prior_')
state_post = (w_state @ (w.T*xhat)).add_prefix('post_')
state_summ = pd.concat([pd.DataFrame(state_area), state_prior, state_post], 
                       axis=1)

# Calculate xhat and difference
state_summ['xhat'] = (state_summ['post_net']/state_summ['prior_net'])
state_summ['diff'] = state_summ['post_total'] - state_summ['prior_total']

# Merge DOFS into state_summ
state_summ['dofs'] = dofs_r.set_index('name').loc[state_summ.index]

# Merge in population
state_summ['pop'] = pop.set_index('name').loc[state_summ.index]['2019']

# Per capita methane emissions (Gg/person)
state_summ['post_total_pc'] = state_summ['post_total']/state_summ['pop']

# Calculate difference
state_summ['diff'] = state_summ['post_total'] - state_summ['prior_total']

# Sort by posterior anthropogenic emissions
state_summ = state_summ.sort_values(by='post_total', ascending=False)

## ------------------------------------------------------------------------ ##
## Plot maps
## ------------------------------------------------------------------------ ##
state = shapefile.Reader(f'{data_dir}states/2019_tl_us_state/tl_2019_us_state.shp')

fig, ax = fp.get_figax(maps=True, lats=clusters.lat, lons=clusters.lon)
for shape in state.shapeRecords():
    if shape.record[6] in state_summ.index.values:
        # Get edges of city
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        c_poly = Polygon(np.column_stack((x, y)))
        color = sf_cmap(div_norm(state_summ.loc[shape.record[6]]['xhat']))
        ax.fill(x, y, facecolor=color, edgecolor='black')
ax = fp.format_map(ax, lats=clusters.lat, lons=clusters.lon)
cmap = plt.cm.ScalarMappable(cmap=sf_cmap, norm=div_norm)
cax = fp.add_cax(fig, ax)
cb = fig.colorbar(cmap, ax=ax, cax=cax)
cb = fp.format_cbar(cb, 'Scale factor')
fp.save_fig(fig, plot_dir, f'states_map')

## ------------------------------------------------------------------------ ##
## Plot results : absolute emissions
## ------------------------------------------------------------------------ ##
xs = np.arange(1, ns + 1)

# fig, ax = fp.get_figax(rows=4, aspect=4, sharex=True,
#                        max_height=config.BASE_HEIGHT*config.SCALE)
figsize = fp.get_figsize(aspect=1, 
                         max_height=config.BASE_HEIGHT*config.SCALE)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.1)
gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.1)
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
labels = state_summ.index.values

# Plot cumulative contribtion and relative contribution
post_tot = state_summ['post_total'].values.sum()
prior_tot = state_summ['prior_total'].values.sum()
state_summ['post_frac'] = state_summ['post_total']/post_tot
state_summ['prior_cum_sum'] = np.cumsum(state_summ['prior_total'])/prior_tot
state_summ['post_cum_sum'] = np.cumsum(state_summ['post_total'])/post_tot
# ax[0].plot(xs, state_summ['post_frac'], c=fp.color(0), 
#            label='Fraction of total emissions')
ax[0].plot(xs, state_summ['post_cum_sum'], c='black', lw=0.75,
           label='Cumulative posterior emissions')
# ax[0].plot(xs, state_summ['prior_cum_sum'], c='grey', ls='--', lw=0.75,
#            label='Prior cumulative contribution')
ax[0] = fp.add_labels(ax[0], '', 'Cumulative fraction of\nCONUS posterior emissions', 
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=20)
ax[0].set_ylim(0.15, 1.05)
# ax[0] = fp.add_legend(ax[0], ncol=1,
#                       fontsize=config.TICK_FONTSIZE, loc='lower right')


# Plot stacked bar
cc = [0, 6, 10, 3, 8, 12, 0]
# cmaps = 
bottom_prior = np.zeros(ns)
bottom_post = np.zeros(ns)
for i, (l, e) in enumerate(emis.items()):
    for j in [1, 2]:
        ax[j].bar(xs - 0.175, state_summ[f'prior_{e}'], 
                  bottom=bottom_prior, 
                  width=0.3, color=fp.color(cc[i], lut=2*len(cc)), label=l)
        ax[j].bar(xs + 0.175, state_summ[f'post_{e}'], 
                  bottom=bottom_post, width=0.3,
                  color=fp.color(cc[i], lut=2*len(cc)))
    bottom_prior += state_summ[f'prior_{e}']
    bottom_post += state_summ[f'post_{e}']

# Split the axis
ax[2].set_ylim(0, 2.25e3)
ax[1].set_ylim(4.25e3, 6.5e3)

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
ax[1] = fp.add_labels(ax[1], '', 'Emissions (Gg a$^{-1}$)', 
                      fontsize=config.TICK_FONTSIZE, 
                      labelpad=20)

# Plot DOFS
ax[-1].bar(xs, state_summ['dofs'], color='0.5', width=0.6,
          label='Averaging kernel sensitivities')
ax[-1].set_ylim(0, 1)
ax[-1].set_ylabel('Averaging kernel\nsensitivities',
                  fontsize=config.TICK_FONTSIZE, 
                  labelpad=20)

# Add x_ticklabels
ax[-1].set_xticklabels(labels, ha='center', fontsize=config.TICK_FONTSIZE,
                      rotation=90)

# Final aesthetics
for j in range(4):
    for k in range(9):
        ax[j].axvline((k + 1)*5 + 0.5, color='0.75', alpha=1, lw=0.5, 
                      zorder=-10)
        if j == 2:
            ax[j].plot(((k + 1)*5 + 0.5, (k + 1)*5 + 0.5), (2.25e3, 2.475e3),
                       color='0.75', alpha=1, lw=0.5, clip_on=False)
    ax[j].set_xticks(xs)
    ax[j].set_xlim(-0.5, ns+1+0.5)
    ax[j].tick_params(axis='both', labelsize=config.TICK_FONTSIZE)
    if j < 3:
        plt.setp(ax[j].get_xticklabels(), visible=False)

for j in [0, 3]:
    for k in range(4):
        ax[j].axhline((k + 1)*0.2, color='0.75', lw=0.5, zorder=-10)
for j in [1, 2]:
    for k in range(6):
        ax[j].axhline((k + 1)*1000, color='0.75', lw=0.5, zorder=-10)
ax[1].tick_params(axis='x', which='both', bottom=False)

# Legend for summary plot
handles, labels = ax[1].get_legend_handles_labels()
# handles.extend(m_handles)
# labels.extend(m_labels)
ax[1] = fp.add_legend(ax[1], handles=handles[::-1], labels=labels[::-1], 
                      ncol=1, fontsize=config.TICK_FONTSIZE, 
                      loc='upper right')

# Set label coords
y0 = (ax[0].get_position().y1 + ax[0].get_position().y0)/2
ax[0].yaxis.set_label_coords(0.07, y0, transform=fig.transFigure)
ax[1].yaxis.set_label_coords(0.07, 0.5, transform=fig.transFigure)
y2 = (ax[-1].get_position().y1 + ax[-1].get_position().y0)/2
ax[-1].yaxis.set_label_coords(0.07, y2, transform=fig.transFigure)

# Add labels
ax[1].text(0.625, state_summ['prior_total'][0], 'Prior', ha='right',
           #state_summ['prior_total'][1] + 100, 'Prior', ha='right', 
           va='top', rotation=90, fontsize=config.TICK_FONTSIZE)
ax[1].text(1.5, state_summ['post_total'][0], 'Posterior',
           #state_summ['post_total'][1] + 100, 'Posterior', 
           ha='left', va='top', rotation=90, 
           fontsize=config.TICK_FONTSIZE)

fp.save_fig(fig, plot_dir, f'states_{f}')
