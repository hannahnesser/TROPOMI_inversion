from os.path import join
from os import listdir
import sys
import glob
import copy
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
from scipy.stats import probplot as qq
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.feature as cfeature
import imageio
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

## ------------------------------------------------------------------------ ##
## Directories
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'

## ------------------------------------------------------------------------ ##
## Set plotting preferences
## ------------------------------------------------------------------------ ##
# Colormaps
plasma_trans = fp.cmap_trans('plasma')
plasma_trans_r = fp.cmap_trans('plasma_r')
rdbu_trans = fp.cmap_trans_center('RdBu_r', nalpha=70)
r_trans = fp.cmap_trans('Reds', nalpha=100)
viridis_trans_r = fp.cmap_trans('viridis_r')
viridis_trans = fp.cmap_trans('viridis')
magma_trans = fp.cmap_trans('magma')
# print(viridis_trans)

# Small (i.e. non-default) figure settings
small_map_kwargs = {'draw_labels' : False}
small_fig_kwargs = {'max_width' : 4,
                    'max_height' : 3.5}

## ------------------------------------------------------------------------ ##
## Set user preferences and load data
## ------------------------------------------------------------------------ ##
DOFS_filter = 0.05

# Decide which posterior to run with
f = f'rf1.0_sa1.0_poi80'

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc')

# Load prior
xa_abs = xr.open_dataarray(f'{data_dir}xa_abs.nc')
area = xr.open_dataarray(f'{data_dir}area.nc')
xa_abs = xa_abs*area*1e-6 # Tg/yr

# Load posterior and DOFS
dofs = np.load(f'{data_dir}dofs2_{f}.npy')
xhat = np.load(f'{data_dir}xhat2_{f}.npy')

# Filter on DOFS filter
xhat[dofs < DOFS_filter] = 1
dofs[dofs < DOFS_filter] = 0

## ------------------------------------------------------------------------ ##
## Regularization factor
## ------------------------------------------------------------------------ ##
# rfs = np.array([1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10])
# rfs = np.array([1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 2])
# dofs_threshold = np.array([0.01, 0.05, 0.1])
# # sas = np.array([0.1, 0.25, 0.5, 0.75, 1, 2])
# ja = np.load(f'{data_dir}/ja2_dofs_threshhold.npy')
# # ja2 = np.load(f'{data_dir}/ja2_long_2.npy')
# # ja = np.concatenate([ja, ja2], axis=1)

# # jo = np.load(f'{data_dir}/jo2_long_2.npy')
# # jo2 = np.load(f'{data_dir}/jo2_long_2.npy')
# # jo = np.concatenate([jo, jo2], axis=1)

# # Get state vector and observational dimensions
# nstate = gc.read_file(f'{data_dir}/xa.nc').shape[0]
# nstate_func = np.load(f'{data_dir}/n_functional_dofs_threshold.npy')
# # nstate_func2 = np.load(f'{data_dir}/n_functional_2.npy')
# # nstate_func = np.concatenate([nstate_func, nstate_func2], axis=1)
# # nobs = gc.read_file(f'{data_dir}/y.nc').shape[0]

# # Normalize for vector length
# print(nstate_func)
# ja = ja/nstate_func
# # jo = (jo/rfs.reshape((-1, 1)))/nobs

# print(ja)

# # lims = [[0, 1], [0, 1]]
# # label = ['ja', 'jo']
# # letter = ['A', 'O']
# # for i, var in enumerate([ja, jo]):
# #     # # Plot
# #     fig, ax = fp.get_figax(aspect=len(ja)/len(jo))
# #     cax = fp.add_cax(fig, ax)

# #     # Plot
# #     # c = ax.contour(rfs, sas, ja.T)
# #     c = ax.imshow(var, vmin=lims[i][0], vmax=lims[i][1])

# #     # Labels
# #     ax.set_xticks(np.arange(0, len(sas)))
# #     ax.set_xticklabels(sas)
# #     ax.set_ylim(-0.5, len(rfs)-0.5)
# #     ax.set_yticks(np.arange(0, len(rfs)))
# #     ax.set_yticklabels(rfs)
# #     ax = fp.add_labels(ax, 'Prior errors', 'Regularization factor')

# #     # Colorbar
# #     cb = fig.colorbar(c, cax=cax)#, ticks=np.arange(0, 6, 1))
# #     cb = fp.format_cbar(cb, cbar_title=r'$J_{A}(\hat{x})$')

# #     fp.save_fig(fig, plot_dir, f'fig_rfs_sas_{label[i]}')

# # Labels
# # ax = fp.add_labels(ax, '')
# fig, ax = fp.get_figax(aspect=2)
# lss = ['-', '--', ':']
# for i, dofs in enumerate(dofs_threshold):
#     ax.plot(rfs, ja[:, dofs == dofs_threshold], c=fp.color(3),
#             ls=lss[i], marker='.',
#             label=r'$J_{A}(\hat{x})$')
# # ax.plot(rfs, jo[:, sas == 0.5], c=fp.color(6), ls='-', marker='.',
# #         label=r'$J_{O}(\hat{x})$')
# ax.axhline(1, ls='--', color='grey')
# ax.set_xscale('log')
# ax.set_ylim(0, 0.1)

# # Labels
# ax = fp.add_legend(ax)
# ax = fp.add_labels(ax, 'Regularization factor', 'Cost function')

# # Save
# fp.save_fig(fig, plot_dir, 'fig_rfs')

## ------------------------------------------------------------------------ ##
## Second estimate
## ------------------------------------------------------------------------ ##
# # Standard plotting preferences
# sf_cmap_1 = plt.cm.PuOr_r(np.linspace(0, 0.5, 256))
# sf_cmap_2 = plt.cm.PuOr_r(np.linspace(0.5, 1, 256))
# sf_cmap = np.vstack((sf_cmap_1, sf_cmap_2))
# sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
# div_norm = colors.TwoSlopeNorm(vmin=-2.5, vcenter=1, vmax=4.5)

# avker_cbar_kwargs = {'title' : r'$\partial\hat{x}_i/\partial x_i$'}
# avker_kwargs = {'cmap' : plasma_trans, 'vmin' : 0, 'vmax' : 1,
#                 'cbar_kwargs' : avker_cbar_kwargs,
#                 'map_kwargs' : small_map_kwargs}

# xhat_cbar_kwargs = {'title' : r'Scale factor'}
# xhat_kwargs = {'cmap' : sf_cmap, 'norm' : div_norm,
#                'default_value' : 1,
#                'cbar_kwargs' : xhat_cbar_kwargs,
#                'map_kwargs' : small_map_kwargs}

# # for sa_i in ['0.5', '0.75', '1.0']:
# # for poi in ['50', '70', '75', '80', '90', '99.9']:
# #     print('-'*20)

# # # Subset to ID only large corrections
# # xhat_sub = copy.deepcopy(xhat)
# # xhat_sub[dofs < 0.01] = 1
# # xhat_sub[xhat_sub > (xhat[dofs >= 0.01].mean() + xhat[dofs >= 0.01].std())]
# # xhat[dofs >= 0.01]

# # Plot averaging kernel sensitivities
# title = f'Averaging kernel sensitivities' # ({f}\%)'
# fig, ax, c = ip.plot_state(dofs, clusters, title=title,
#                            **avker_kwargs)
# ax.text(0.025, 0.05, 'DOFS = %d' % round(dofs.sum()),
#         fontsize=config.LABEL_FONTSIZE*config.SCALE,
#         transform=ax.transAxes)
# fp.save_fig(fig, plot_dir, f'fig_est2_dofs_{f}')
# plt.close()

# # Plot posterior scaling factors
# title = f'Posterior emission scale factors' # ({f}\%)'
# fig, ax, c = ip.plot_state(xhat, clusters, title=title,
#                            **xhat_kwargs)
# fp.save_fig(fig, plot_dir, f'fig_est2_xhat_{f}')
# plt.close()

# # Reset cbar kwargs
# avker_kwargs['cbar_kwargs'] = {'title' : r'$\partial\hat{x}_i/\partial x_i$'}
# xhat_kwargs['cbar_kwargs'] = {'title' : r'Scale factor'}

## ------------------------------------------------------------------------ ##
## Sectoral attribution
## ------------------------------------------------------------------------ ##
# # List emissions categories
# emis = ['wetlands', 'livestock', 'coal', 'oil', 'gas', 'landfills',
#         'wastewater', 'other']
# emis_labels = ['Wetlands', 'Livestock', 'Coal', 'Oil', 'Gas', 'Landfills',
#                'Wastewater', 'Other']

# # Open masks and create a total_mask array as well as a mask dictionary
# mex_mask = np.load(f'{data_dir}Mexico_mask.npy')
# can_mask = np.load(f'{data_dir}Canada_mask.npy')
# conus_mask = np.load(f'{data_dir}CONUS_mask.npy')
# total_mask = mex_mask + can_mask + conus_mask
# masks = {'Canada' : can_mask, 'CONUS' : conus_mask, 'Mexico' : mex_mask}

# xhat_abs = (xhat*xa_abs)
# print(f'Total prior emissions            : {xa_abs.sum().values}')
# print(f'Total posterior emissions        : {xhat_abs.sum().values}')
# print(f'Difference                       : {(xhat_abs.sum().values - xa_abs.sum().values)}')
# print(f'Maximum scale factor             : {xhat.max()}')
# print(f'Minimum scale factor             : {xhat.min()}')
# print(f'Number of negative scale factors : {(xhat < 0).sum()}')

# fig, ax = fp.get_figax(aspect=2)
# xhat_sub = xhat[dofs >= DOFS_filter]
# ax.hist(xhat_sub, bins=500, density=True, color=fp.color(7))
# ax.axvline(xhat_sub.mean(), color='grey', ls='-')
# ax.axvline(xhat_sub.mean() + xhat_sub.std(), color='grey', ls='--')
# ax.axvline(xhat_sub.mean() - xhat_sub.std(), color='grey', ls='--')
# ax.set_xlim(0, 2)
# fp.save_fig(fig, plot_dir, 'fig_xhat_dist')

# # Sectoral emissions: Load the sectoral attribution matrix
# w = pd.read_csv(f'{data_dir}w.csv')
# w = w.T*area.values*1e-6 # Convert to Tg/yr
# # w = w/w.sum(axis=0) # Normalize
# # w[np.isnan(w)] = 0 # Deal with nans

# country_emis = {}
# for country, mask in masks.items():
#     print('-'*30)
#     print(country)
#     sect_emis = pd.DataFrame(columns=['prior', 'post', 'diff',
#                                       'prior_sub', 'post_sub', 'diff_sub'])
#                                       #'diff_sub_pos', 'diff_sub_neg'])
#     w_c = copy.deepcopy(w)*mask # convert to Tg/yr in the country

#     # Get prior and posterior absolute emissions
#     sect_emis['prior'] = w_c.sum(axis=1)
#     sect_emis['post'] = (w_c*xhat).sum(axis=1)
#     sect_emis['diff'] = sect_emis['post'] - sect_emis['prior']

#     # Get prior and posterior absolute emissions only where DOFS > DOFS_filter
#     sect_emis['prior_sub'] = w_c.loc[:, dofs >= DOFS_filter].sum(axis=1)
#     sect_emis['post_sub'] = (w_c*xhat).loc[:, dofs >= DOFS_filter].sum(axis=1)
#     sect_emis['diff_sub'] = sect_emis['post_sub'] - sect_emis['prior_sub']

#     country_emis[country] = sect_emis

# # Plot histogram (at least take #1)
# xs = np.arange(0, len(emis))
# fig, axis = fp.get_figax(aspect=2, cols=3)
# j = 0
# lims = [[-2, 12], [-2, 12], [-2, 3]]
# country_emis_sub = {key : country_emis[key]
#                     for key in ['Canada', 'CONUS', 'Mexico']}
# for country, sect_emis in country_emis_sub.items():
#     ax = axis[j]
#     ax.text(0.95, 0.95, country, ha='right', va='top',
#             fontsize=config.LABEL_FONTSIZE*config.SCALE,
#             transform=ax.transAxes)
#     ax.bar(xs - 0.16, sect_emis['prior'], width=0.3,
#            color='white', edgecolor=fp.color(2*j+2), label='Prior (all)')
#     ax.bar(xs - 0.16, sect_emis['prior_sub'], width=0.3, color=fp.color(2*j+2),
#            label='Prior (optimized)')
#     ax.bar(xs + 0.16, sect_emis['post'], width=0.3,
#            color='white', edgecolor=fp.color(2*j+2), alpha=0.5,
#            label='Posterior (all)')
#     ax.bar(xs + 0.16, sect_emis['post_sub'], width=0.3,
#            color=fp.color(2*j+2), alpha=0.5, label='Posterior (optimized)')
#     # for i, e in enumerate(emis):
#     #     ax.arrow(x=i + 0.13, y=sect_emis.at[e, 'prior_sub'],
#     #              dx=0, dy=sect_emis.at[e, 'diff_sub_pos'],
#     #              color=fp.color(2*j+2), width=0.002, head_width=0.0025)
#     #     ax.arrow(x=i + 0.19,
#     #              y=sect_emis.at[e, 'prior_sub'] + sect_emis.at[e, 'diff_sub_pos'],
#     #              dx=0, dy=sect_emis.at[e, 'diff_sub_neg'],
#     #              color=fp.color(2*j+2), width=0.002, head_width=0.0025)
#     ax.set_ylim(lims[j])
#     # ax.bar(xs + 0.2, sect_emis['post_sub_pos'], width=0.1, color=fp.color(7))
#     # ax = fp.add_legend(ax)

#     j += 1

# axis[0] = fp.add_labels(axis[0], '', 'Emissions\n' + r'[Tg a$^{-1}$]')
# for i in range(3):
#     if i > 0:
#         axis[i] = fp.add_labels(axis[i], '', '')
#     axis[i].set_xticks(xs)
#     axis[i].set_xticklabels(emis_labels, rotation=90, ha='center',
#                             fontsize=config.TICK_FONTSIZE*config.SCALE/1.5)

# fp.save_fig(fig, plot_dir, 'fig_sectoral_bar')

## ------------------------------------------------------------------------ ##
## Permian comparison
## ------------------------------------------------------------------------ ##
# Combine the Permian clusters with the NA clusters
permian = xr.open_dataset(f'{data_dir}clusters_permian.nc')
c = clusters.squeeze(drop=True).to_dataset()
c['Permian'] = permian['Clusters']

# Get the Permian basin indices (discard the buffer cells)
cell_idx, cell_cnt  = np.unique(c['Permian'], return_counts=True)
cell_idx = cell_idx[cell_cnt == 1]
cell_idx = cell_idx[~np.isnan(cell_idx)]
permian
# There are good reasons to move this onto the Permian cluster file

# Subset over the Permian
c = c.where(c['Permian'].isin(cell_idx))['Clusters']

print(c)

# Flatten and create boolean
c = ip.clusters_2d_to_1d(permian, c)
c[c > 0] = 1
# c = c.astype(bool)

for dofs_t in [0.01, 0.05, 0.1, 0.25]:
    xhat_sub = xhat[dofs >= dofs_t]
    ja = ((xhat_sub - 1)**2/4).sum()/(len(xhat_sub))
    print(f'{dofs_t:<5}{xhat_sub.min():.2f}  {xhat_sub.max():.2f}  {ja:.2f}')

# Subset the posterior
xhat_permian = xhat*c

# Calculate emissions
xa_abs_permian = xa_abs*c
xhat_abs_permian = xhat_permian*xa_abs_permian
print(f'Total prior emissions            : {xa_abs_permian.sum().values}')
print(f'Total posterior emissions        : {xhat_abs_permian.sum().values}')
print(f'Difference                       : {(xhat_abs_permian.sum().values - xa_abs_permian.sum().values)}')


# Plot posterior scaling factors
# xhat_cbar_kwargs = {'title' : r'Scale factor'}
# xhat_kwargs = {'cmap' : 'PuOr_r', 'vmin' : 0, 'vmax' : 2,
#                'default_value' : 1,
#                'cbar_kwargs' : xhat_cbar_kwargs,
#                'map_kwargs' : small_map_kwargs}
# title = f'Posterior emission scale factors: Permian' # ({f}\%)'
# fig, ax, c = ip.plot_state(xhat_permian, clusters, title=title,
#                            **xhat_kwargs)
# fp.save_fig(fig, plot_dir, f'fig_est2_xhat_permian_{f}')
# plt.close()

