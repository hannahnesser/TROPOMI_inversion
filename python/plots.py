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
## Set user preferences
## ------------------------------------------------------------------------ ##
# Local preferences
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'

# Colormaps
# c = plt.cm.get_cmap('inferno', lut=10)
plasma_trans = fp.cmap_trans('plasma')
plasma_trans_r = fp.cmap_trans('plasma_r')
rdbu_trans = fp.cmap_trans_center('RdBu_r', nalpha=70)
r_trans = fp.cmap_trans('Reds', nalpha=100)
viridis_trans_r = fp.cmap_trans('viridis_r')
viridis_trans = fp.cmap_trans('viridis')
magma_trans = fp.cmap_trans('magma')
# print(viridis_trans)

# Small (i.e. non-default) figure settings
# small_map_kwargs = {'draw_labels' : False}
small_fig_kwargs = {'max_width' : 4,
                    'max_height' : 3.5}

DOFS_filter = 0.05

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc')

########################################################################
### FIGURE : EIGENVALUES
########################################################################
# evals_q = np.load(f'{data_dir}evals_q0.npy')
# evals_h = np.load(f'{data_dir}evals_h0.npy')
# # evals_h[evals_h < 0] = 0
# snr = evals_h**0.5
# DOFS_frac = np.cumsum(evals_q)/evals_q.sum()

# fig, ax = fp.get_figax(aspect=3)

# # DOFS frac
# ax.plot(100*DOFS_frac, label='Information content spectrum',
#         c=fp.color(3), lw=2)
# for p in [0.5, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99]:
#     diff = np.abs(DOFS_frac - p)
#     rank = np.argwhere(diff == np.min(diff))[0][0]
#     print(p, rank)
#     ax.scatter(rank, 100*DOFS_frac[rank], marker='*', s=80, c=fp.color(3))
#     ax.text(rank, 100*DOFS_frac[rank]-5, f'{int(100*p):d}%%',
#             ha='left', va='top', c=fp.color(3))
# ax.set_xlabel('Eigenvector index', fontsize=config.LABEL_FONTSIZE*config.SCALE,
#               labelpad=config.LABEL_PAD)
# ax.set_ylabel('Percentage of DOFS', fontsize=config.LABEL_FONTSIZE*config.SCALE,
#               labelpad=config.LABEL_PAD, color=fp.color(3))
# ax.tick_params(axis='both', which='both',
#                labelsize=config.LABEL_FONTSIZE*config.SCALE)
# ax.tick_params(axis='y', labelcolor=fp.color(3))

# # SNR
# ax2 = ax.twinx()
# ax2.plot(snr, label='Signal-to-noise ratio spectrum', c=fp.color(6), lw=2)
# for r in [1, 2]:
#     diff = np.abs(snr - r)
#     rank = np.argwhere(diff == np.min(diff))[0][0]
#     ax2.scatter(rank, snr[rank], marker='.', s=80, c=fp.color(6))
#     ax2.text(rank-200, snr[rank], f'{r:d}', ha='right', va='center',
#              c=fp.color(6))
# ax2.set_ylabel('Signal-to-noise ratio',
#                fontsize=config.LABEL_FONTSIZE*config.SCALE,
#               labelpad=config.LABEL_PAD, color=fp.color(6))
# ax2.tick_params(axis='y', which='both',
#                 labelsize=config.LABEL_FONTSIZE*config.SCALE,
#                 labelcolor=fp.color(6))

# ax = fp.add_title(ax, 'Initial Estimate Information Content Spectrum')

# fp.save_fig(fig, plot_dir, 'eigenvalues_update')

########################################################################
### FIGURE : RESOLUTION CONTEXT
########################################################################
# # MA lat/lon lims
# lat_min = 40
# lat_max = 44
# lon_min = -74
# lon_max = -69

# # Get longitude edges
# lons_e = np.arange(lon_min, lon_max+0.3125, 0.3125)
# # lons_e = lons_e[(lons_e > lon_min) & (lons_e < lon_max)]

# # Get latitude edges
# lats_e = np.arange(lat_min, lat_max+0.25, 0.25)
# # lats_e = lats_e[(lats_e > lat_min) & (lats_e < lat_max)]
# print(lats_e)

# fig, ax = fp.get_figax(maps=True,
#                        lats=[lat_min, lat_max], lons=[lon_min, lon_max])
# ax = fp.format_map(ax, lats_e, lons_e, **small_map_kwargs)

# for lat in lats_e:
#     ax.axhline(lat, c=fp.color(4), lw=0.1)

# for lon in lons_e:
#     ax.axvline(lon, c=fp.color(4), lw=0.1)

# # Save
# fp.save_fig(fig, plot_dir, f'resolution')

########################################################################
### FIGURE : JACOBIAN COLUMN
########################################################################
# # Cluster = 11046

# 40.013712, -80.612467
# 40, -80.625
# 14412
# Load observations to get lat/lon
# obs = gc.read_file(f'{data_dir}2019.pkl')
# obs = obs[obs['MONTH'] == 6]

# # Load Jacobian column
# ki = gc.read_file(f'{data_dir}k1_m06_sample_column.nc').values
# print(ki.max())
# # print(ki.max())

# # Create plot
# fig, ax = fp.get_figax(maps=True, lats=clusters.lat, lons=clusters.lon)
# ax = fp.format_map(ax, clusters.lat, clusters.lon, **small_map_kwargs)

# # Plot
# c = ax.scatter(obs['LON'], obs['LAT'], c=ki,
#                cmap=r_trans, vmin=0, vmax=0.5,
#                s=0.25)

# # Add colorbar
# cax = fp.add_cax(fig, ax)
# cb = fig.colorbar(c, ax=ax, cax=cax)
# cb = fp.format_cbar(cb, 'Jacobian Value')

# # Save
# fp.save_fig(fig, plot_dir, f'fig_k1')

########################################################################
### FIGURE : JACOBIAN COLUMN KW
########################################################################
# # Remove emissions from buffer grid cells
# fig, ax = fp.get_figax(rows=2, cols=4,
#                        maps=True, lats=clusters.lat, lons=clusters.lon)
# plt.subplots_adjust(hspace=0.5)
# cax = fp.add_cax(fig, ax, cbar_pad_inches=0.5)

# # open kw
# kw = np.load(f'{data_dir}kw1_m06.npy')

# # Load observations to get lat/lon
# obs = gc.read_file(f'{data_dir}2019.pkl')
# obs = obs[obs['MONTH'] == 6]

# for i, axis in enumerate(ax.flatten()):
#     # fig, axis = fp.get_figax(maps=True, lats=emis.lat, lons=emis.lon)
#     axis = fp.format_map(axis, lats=clusters.lat, lons=clusters.lon)

#     c = axis.scatter(obs['LON'], obs['LAT'], c=kw[:,i],
#                    cmap=rdbu_trans, vmin=-1, vmax=1,
#                    s=0.25)
#     axis = fp.add_title(axis, f'{(i+1)}')

# cb = fig.colorbar(c, cax=cax)#, ticks=[])
# cb = fp.format_cbar(cb, cbar_title=r'Model Response (ppb)')

# fp.save_fig(fig, plot_dir, 'fig_kw')

########################################################################
### FIGURE : REGULARIZATION FACTOR
########################################################################
# # rfs = np.array([1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10])
# rfs = np.array([1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1])
# sas = np.array([0.1, 0.25, 0.5, 0.75, 1, 2])
# ja = np.load(f'{data_dir}/ja2_long_2.npy')
# # ja2 = np.load(f'{data_dir}/ja2_long_2.npy')
# # ja = np.concatenate([ja, ja2], axis=1)

# jo = np.load(f'{data_dir}/jo2_long_2.npy')
# # jo2 = np.load(f'{data_dir}/jo2_long_2.npy')
# # jo = np.concatenate([jo, jo2], axis=1)

# # Get state vector and observational dimensions
# nstate = gc.read_file(f'{data_dir}/xa.nc').shape[0]
# nstate_func = np.load(f'{data_dir}/n_functional_2.npy')
# # nstate_func2 = np.load(f'{data_dir}/n_functional_2.npy')
# # nstate_func = np.concatenate([nstate_func, nstate_func2], axis=1)
# nobs = gc.read_file(f'{data_dir}/y.nc').shape[0]

# # Normalize for vector length
# print(nstate_func)
# ja = ja/nstate_func
# jo = (jo/rfs.reshape((-1, 1)))/nobs

# print(ja)

# lims = [[0, 1], [0, 1]]
# label = ['ja', 'jo']
# letter = ['A', 'O']
# for i, var in enumerate([ja, jo]):
#     # # Plot
#     fig, ax = fp.get_figax(aspect=len(ja)/len(jo))
#     cax = fp.add_cax(fig, ax)

#     # Plot
#     # c = ax.contour(rfs, sas, ja.T)
#     c = ax.imshow(var, vmin=lims[i][0], vmax=lims[i][1])

#     # Labels
#     ax.set_xticks(np.arange(0, len(sas)))
#     ax.set_xticklabels(sas)
#     ax.set_ylim(-0.5, len(rfs)-0.5)
#     ax.set_yticks(np.arange(0, len(rfs)))
#     ax.set_yticklabels(rfs)
#     ax = fp.add_labels(ax, 'Prior errors', 'Regularization factor')

#     # Colorbar
#     cb = fig.colorbar(c, cax=cax)#, ticks=np.arange(0, 6, 1))
#     cb = fp.format_cbar(cb, cbar_title=r'$J_{A}(\hat{x})$')

#     fp.save_fig(fig, plot_dir, f'fig_rfs_sas_{label[i]}')

# # Labels
# # ax = fp.add_labels(ax, '')
# fig, ax = fp.get_figax(aspect=2)
# ax.plot(rfs, ja[:, sas == 0.5], c=fp.color(3), ls='-', marker='.',
#         label=r'$J_{A}(\hat{x})$')
# ax.plot(rfs, jo[:, sas == 0.5], c=fp.color(6), ls='-', marker='.',
#         label=r'$J_{O}(\hat{x})$')
# ax.axhline(1, ls='--', color='grey')
# ax.set_xscale('log')
# ax.set_ylim(0, 2)

# # Labels
# ax = fp.add_legend(ax)
# ax = fp.add_labels(ax, 'Regularization factor', 'Cost function')

# # Save
# fp.save_fig(fig, plot_dir, 'fig_rfs')

# ########################################################################
# ### FIGURE : PRIOR ERROR ESTIMATION
# ########################################################################
# def alpha(a0, ka, an, L, L0=0.1):
#     return a0*np.exp(-ka*(L-L0)) + an

# def beta(b0, kb, L, L0=0.1):
#     return b0*np.exp(-kb*(L-L0))

# livestock = [0.89, 3.1, 0.12, 0, 0]
# nat_gas = [0.28, 4.2, 0.25, 0.09, 3.9]
# landfills = [0, 0, 0.51, 0.08, 2.0]
# wastewater = [0.78, 1.4, 0.21, 0.06, 6.9]
# petroleum = [0, 0, 0.87, 0.04, 197]
# sources = {'livestock' : livestock, 'nat_gas' : nat_gas,
#            'landfills' : landfills, 'wastewater' : wastewater,
#            'petroleum' : petroleum}

# for s, coefs in sources.items():
#     a = alpha(coefs[0], coefs[1], coefs[2], 0.25)
#     b = beta(coefs[3], coefs[4], 0.25)
#     print(f'{s:<20}{a:.2f}  {b:.2f}')


# ########################################################################
# ### FIGURE : FIRST ESTIMATE
# ########################################################################
# # Standard plotting preferences
# avker_cbar_kwargs = {'title' : r'$\partial\hat{x}_i/\partial x_i$'}
# avker_kwargs = {'cmap' : plasma_trans, 'vmin' : 0, 'vmax' : 1,
#                 'cbar_kwargs' : avker_cbar_kwargs,
#                 'map_kwargs' : small_map_kwargs}

# xhat_cbar_kwargs = {'title' : r'Scale factor'}
# xhat_kwargs = {'cmap' : 'PuOr_r', 'vmin' : 0.75, 'vmax' : 1.25,
#                'default_value' : 1,
#                'cbar_kwargs' : xhat_cbar_kwargs,
#                'map_kwargs' : small_map_kwargs}


# # Fraction of information contents
# # fracs = [90, 95, 98, 99, 99.9]
# # fracs = [99.9]
# suffixes = ['poi50', 'poi90', 'poi95', 'poi99', 'poi99.9']

# # Load initial averaging kernel
# for f in suffixes:
#     dofs = np.load(f'{data_dir}dofs1_{f}.npy')
#     xhat = np.load(f'{data_dir}xhat1_{f}.npy')
#     # xhat += np.ones(xhat.shape)

#     # Filter
#     xhat[dofs < 0.01] = 1

#     # Plot averaging kernel sensitivities
#     title = f'First update of information content' # ({f}\%)'
#     fig, ax, c = ip.plot_state(dofs, clusters_plot, title=title,
#                                **avker_kwargs)
#     ax.text(0.025, 0.05, 'DOFS = %d' % round(dofs.sum()),
#             fontsize=config.LABEL_FONTSIZE*config.SCALE,
#             transform=ax.transAxes)
#     fp.save_fig(fig, plot_dir, f'fig_est1_dofs_{f}_update')

#     # Plot posterior scaling factors
#     title = f'First update of emissions scale factors' # ({f}\%)'
#     fig, ax, c = ip.plot_state(xhat, clusters_plot, title=title,
#                                **xhat_kwargs)
#     fp.save_fig(fig, plot_dir, f'fig_est1_xhat_{f}_update')

#     # Reset cbar kwargs
#     avker_kwargs['cbar_kwargs'] = {'title' : r'$\partial\hat{x}_i/\partial x_i$'}
#     xhat_kwargs['cbar_kwargs'] = {'title' : r'Scale factor'}




########################################################################
### FIGURE : SECOND ESTIMATE
########################################################################
# # Standard plotting preferences
# avker_cbar_kwargs = {'title' : r'$\partial\hat{x}_i/\partial x_i$'}
# avker_kwargs = {'cmap' : plasma_trans, 'vmin' : 0, 'vmax' : 1,
#                 'cbar_kwargs' : avker_cbar_kwargs,
#                 'map_kwargs' : small_map_kwargs}

# xhat_cbar_kwargs = {'title' : r'Scale factor'}
# xhat_kwargs = {'cmap' : 'PuOr_r', 'vmin' : 0, 'vmax' : 2,
#                'default_value' : 1,
#                'cbar_kwargs' : xhat_cbar_kwargs,
#                'map_kwargs' : small_map_kwargs}

# # Fraction of information contents
# # fracs = [90, 95, 98, 99, 99.9]
# # fracs = [99.9]
# # suffixes = ['poi50', 'poi80', 'poi90', 'poi99.9']
# # suffixes = glob.glob(f'{data_dir}/dofs2*poi80*')
# # suffixes = [s.split('/')[-1][6:-4] for s in suffixes]

# # for sa_i in ['0.5', '0.75', '1.0']:
f = f'rf0.5_sa2.0_poi80'

# # Load
# dofs = np.load(f'{data_dir}dofs2_{f}.npy')
# xhat = np.load(f'{data_dir}xhat2_{f}.npy')
# # xhat += np.ones(xhat.shape)

# # Filter
# xhat[dofs < 0.01] = 1
# dofs[dofs < 0.01] = 0

# # Subset to ID only large corrections
# xhat_sub = copy.deepcopy(xhat)
# xhat_sub[dofs < 0.01] = 1
# xhat_sub[xhat_sub > (xhat[dofs >= 0.01].mean() + xhat[dofs >= 0.01].std())]
# xhat[dofs >= 0.01]


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

# ########################################################################
# ### FIGURE : SECOND ESTIMATE BAR CHART
# ########################################################################
# # List emissions categories
# emis = ['wetlands', 'livestock', 'coal', 'oil', 'gas', 'landfills',
#         'wastewater', 'other']
# emis_labels = ['Wetlands', 'Livestock', 'Coal', 'Oil', 'Gas', 'Landfills',
#                'Wastewater', 'Other']

# # Define function to open masks
# def open_mask(country, data_dir=data_dir):
#     data = xr.open_dataset(f'{data_dir}{country}_Mask.001x001.nc')
#     data = data.squeeze(drop=True)['MASK']
#     return data

# # Define function to regrid the masks to the inversion resolution
# def regrid_mask(mask, clusters):
#     # Subset mask to be as small as possible
#     mask = mask.where(mask > 0, drop=True)
#     mask = mask.fillna(0)

#     # Regrid
#     rg = mask.interp(lat=clusters.lat, lon=clusters.lon, method='linear')

#     # Flatten and return
#     flat = ip.clusters_2d_to_1d(clusters, rg)
#     return flat

# # Open masks and create a total_mask array as well as a mask dictionary
# mex_mask = regrid_mask(open_mask('Mexico'), clusters)
# can_mask = regrid_mask(open_mask('Canada'), clusters)
# conus_mask = regrid_mask(open_mask('CONUS'), clusters)
# total_mask = mex_mask + can_mask + conus_mask
# masks = {'Canada' : can_mask, 'CONUS' : conus_mask, 'Mexico' : mex_mask}

# # Normalize with the total mask to deal with places that have bloopers
# for c, m in masks.items():
#     # fig, ax, _ = ip.plot_state(m, clusters)
#     # plt.show()
#     tmp = m/total_mask
#     tmp = np.nan_to_num(tmp, 0)
#     masks[c] = tmp

# # Recalculate the total mask
# total_mask = masks['Canada'] + masks['CONUS'] + masks['Mexico']

# # Define a mask for Central American and Caribbean countries
# other_mask = 1 - copy.deepcopy(total_mask)
# other_mask = ip.match_data_to_clusters(other_mask, clusters)
# other_countries_cond = (((other_mask.lon > -92)   & (other_mask.lat < 18.3)) |
#                         ((other_mask.lon > -90)   & (other_mask.lat < 19.8)) |
#                         ((other_mask.lon > -86)   & (other_mask.lat < 24))   |
#                         ((other_mask.lon > -79.5) & (other_mask.lat < 27))   |
#                         ((other_mask.lon > -66)   & (other_mask.lat < 36)))
# other_mask = other_mask.where(other_countries_cond, 0)
# other_mask = ip.clusters_2d_to_1d(clusters, other_mask)
# masks['Other'] = other_mask

# ## Now deal with off shore emissions and spare grid cells
# # Set up a mask that has identifying numbers for the country that
# # occupies most of the grid cell
# total_mask_id = np.zeros(total_mask.shape) # Default
# total_mask_id[masks['Mexico'] > 0] = 1 # Mexico
# total_mask_id[masks['CONUS'] > masks['Mexico']] = 2 # CONUS
# total_mask_id[masks['Canada'] > masks['CONUS']] = 3 # Canada

# # Match that to clusters and set areas where the mask == 0 to nan so
# # that those values can be interpolated using neareswt neighbors
# total_mask_id = ip.match_data_to_clusters(total_mask_id, clusters)
# total_mask_id = total_mask_id.where(total_mask_id > 0)
# total_mask_id = total_mask_id.interpolate_na(dim='lat', method='nearest')
# total_mask_id = ip.clusters_2d_to_1d(clusters, total_mask_id)

# # Replace values from "other" that were falsely filled
# total_mask_id[masks['Other'] > 0] = 4 # Other

# # Distribute into each country's mask
# for i, country in enumerate(['Mexico', 'CONUS', 'Canada']):
#     temp_mask = copy.deepcopy(total_mask_id)
#     temp_mask[temp_mask != (i + 1)] = 0
#     temp_mask[temp_mask > 0] = 1
#     temp_bool = (masks[country] == 0) & (temp_mask > 0)
#     masks[country][temp_bool] = temp_mask[temp_bool]

# # Recalculate the total mask
# total_mask = (masks['Canada'] + masks['CONUS'] + masks['Mexico'] +
#               masks['Other'])

# # Still need to confirm that these add to proper values!

# # Load prior
# xa_abs = xr.open_dataarray(f'{data_dir}xa_abs.nc')
# area = xr.open_dataarray(f'{data_dir}area.nc')
# xa_abs = xa_abs*area*1e-6 # Tg/yr

# # Calculate total emissions for the best case
# dofs = np.load(f'{data_dir}dofs2_{f}.npy')
# xhat = np.load(f'{data_dir}xhat2_{f}.npy')

# # # Filter on DOFS
# # xhat[dofs < DOFS_filter] = 1
# # dofs[dofs < DOFS_filter] = 0

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
#                                       'prior_sub', 'post_sub', 'diff_sub',
#                                       'err_sub'])
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


#     print(sect_emis)



#     for i, e in enumerate(emis):
#         # Open sectoral emissions and convert units
#         xa_abs_e = xr.open_dataarray(f'{data_dir}xa_{e}.nc')
#         xa_abs_e = xa_abs_e*area*1e-6*mask # Convert to Tg/yr in the country

#         # Save out prior and post emissions
#         sect_emis.at[e, 'prior'] = xa_abs_e.sum().values
#         sect_emis.at[e, 'post'] = (xhat*xa_abs_e).sum().values
#         sect_emis.at[e, 'diff'] = sect_emis['post'] - sect_emis['prior']

#         xa_abs_sub_e = xa_abs_e[dofs >= 0.01]
#         xhat_abs_sub_e = (xhat*xa_abs_e)[dofs >= 0.01]
#         sect_emis.at[e, 'prior_sub'] = xa_abs_sub_e.sum().values
#         sect_emis.at[e, 'post_sub'] = xhat_abs_sub_e.sum().values
#         sect_emis.at[e, 'diff_sub'] = sect_emis['post_sub'] - sect_emis['prior_sub']

#         pos = (xhat_abs_sub_e - xa_abs_sub_e)[xhat_sub >= 1].sum().values
#         neg = (xhat_abs_sub_e - xa_abs_sub_e)[xhat_sub < 1].sum().values
#         sect_emis.at[e, 'diff_sub_pos'] = pos
#         sect_emis.at[e, 'diff_sub_neg'] = neg

#     country_emis[country] = sect_emis

# # Plot histogram (at least take #1)
# xs = np.arange(0, len(emis))
# fig, axis = fp.get_figax(aspect=2, cols=3)
# j = 0
# lims = [[0, 12], [0, 12], [0, 3]]
# for country, sect_emis in country_emis.items():
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

# ########################################################################
# ### FIGURE : PERMIAN COMPARISON
# ########################################################################
# permian = xr.open_dataset(f'{data_dir}clusters_permian.nc')
# c = clusters.squeeze(drop=True).to_dataset()
# print(c)
# print(permian)
# c['permian'] = permian
# print(c)

# ########################################################################
# ### FIGURE : PRIOR ERROR STANDARD DEVIATION
# ########################################################################
# # Load prior error
# sa_vec = gc.read_file(f'{data_dir}sa_abs.nc')
# sd_vec = sa_vec**0.5
# cbar_kwargs = {'title' : 'Mg km$^2$ a$^{-1}$'}
# fig, ax, c = ip.plot_state(sd_vec, clusters,
#                            title='Prior error\nstandard deviation',
#                            cmap=fp.cmap_trans('viridis'), vmin=0, vmax=10,
#                            fig_kwargs=small_fig_kwargs,
#                            cbar_kwargs=cbar_kwargs,
#                            map_kwargs=small_map_kwargs)
# fp.save_fig(fig, plot_dir, 'fig_prior_error')

# ########################################################################
# ### FIGURE : OBSERVATIONAL DENSITY
# ########################################################################
# # Load observations
# obs = gc.read_file(f'{data_dir}2019_corrected.pkl')

# # Get latitude and longitude edges
# lat_e = np.arange(s.lat_min, s.lat_max + s.lat_delta, s.lat_delta)
# lon_e = np.arange(s.lon_min, s.lon_max + s.lon_delta, s.lon_delta)

# # Put each observation into a grid box
# obs['LAT_E'] = pd.cut(obs['LAT'], lat_e, precision=4)
# obs['LON_E'] = pd.cut(obs['LON'], lon_e, precision=4)

# # Group by grid box (could probably do this by iGC/jGC but oh well)
# obs_density = obs.groupby(['LAT_E', 'LON_E']).count()
# obs_density = obs_density['OBS'].reset_index()

# # Get the edges of each grid box and set these as the indices
# obs_density['lat'] = obs_density['LAT_E'].apply(lambda x: x.mid)
# obs_density['lon'] = obs_density['LON_E'].apply(lambda x: x.mid)
# obs_density = obs_density.set_index(['lat', 'lon'])['OBS']

# # Convert to xarray
# obs_density = obs_density.to_xarray()

# # Plot
# title = 'Observation density'
# viridis_trans_long = fp.cmap_trans('viridis', nalpha=25, ncolors=300)
# cbar_kwargs = {'ticks' : np.arange(0, 1100, 200),
#                'title' : 'Count'}
# fig, ax, c = ip.plot_state_format(obs_density, title=title,
#                                   vmin=0, vmax=1000, default_value=0,
#                                   cmap=viridis_trans_long,
#                                   fig_kwargs=small_fig_kwargs,
#                                   cbar_kwargs=cbar_kwargs,
#                                   map_kwargs=small_map_kwargs)
# fp.save_fig(fig, plot_dir, 'fig_tropomi_obs_density')

########################################################################
### FIGURE : INITIAL EIGENVECTORS
########################################################################
# # Eigenvectors
# # evecs = [f'{data_dir}evec_pert_{i:02d}.nc' for i in range(1, 10)]
# # i = 1
# # for v in evecs:
# #     ev = xr.open_dataarray(v)
# #     fig, ax, c = ip.plot_state_format(ev, cmap='RdBu_r',
# #                                       vmin=-0.1, vmax=0.1)
# #     fp.save_fig(fig, plot_dir, f'fig_est0_evec_{i:02d}')
# #     i += 1

# evecs = np.load(f'{data_dir}prolongation0.npy')
# clusters = xr.open_dataarray(f'{data_dir}clusters.nc')
# fig, ax, c = ip.plot_state_grid(evecs[:,:8], 2, 4, clusters,
#                                 titles=[f'{i}' for i in range(1, 10)],
#                                 cmap='RdBu_r', vmin=-0.01, vmax=0.01,
#                                 cbar_kwargs={'title' : 'Eigenvector\nvalue',
#                                              'ticks' : [-0.01, 0, 0.01]})
# fp.save_fig(fig, plot_dir, 'fig_est0_evecs')

########################################################################
### FIGURE : SECOND GUESS EIGENVECTORS
########################################################################
# Eigenvectors
# evecs = [f'{data_dir}evec_pert_{i:02d}.nc' for i in range(1, 10)]
# i = 1
# for v in evecs:
#     ev = xr.open_dataarray(v)
#     fig, ax, c = ip.plot_state_format(ev, cmap='RdBu_r',
#                                       vmin=-0.1, vmax=0.1)
#     fp.save_fig(fig, plot_dir, f'fig_est0_evec_{i:02d}')
#     i += 1

# rows = 4
# cols = 4
# nevecs = rows*cols
# n0 = 430
# evecs = np.load(f'{data_dir}prolongation1.npy')
# clusters = xr.open_dataarray(f'{data_dir}clusters.nc')
# fig, ax, c = ip.plot_state_grid(evecs[:,n0:(n0+nevecs)], rows, cols, clusters,
#                                 titles=[f'{n0+i}' for i in range(1, nevecs+1)],
#                                 cmap='RdBu_r', vmin=-0.01, vmax=0.01,
#                                 cbar_kwargs={'title' : 'Eigenvector\nvalue',
#                                              'ticks' : [-0.01, 0, 0.01]})
# fp.save_fig(fig, plot_dir, f'fig_est1_evecs_{n0}_{n0+nevecs}')

########################################################################
### FIGURE : FIGURE OUT EIGENVECTOR WEIGHTING
########################################################################
# # Load raw emissions data
# emis_file = f'{base_dir}prior/total_emissions/HEMCO_diagnostics.{s.year}.nc'
# emis = gc.read_file(emis_file)
# emis = emis['EmisCH4_Total']#*emis['AREA']

# # fig, ax = fp.get_figax(maps=True, lats=emis.lat, lons=emis.lon)
# # ax = fp.format_map(ax, lats=emis.lat, lons=emis.lon)
# # cax = fp.add_cax(fig, ax, cbar_pad_inches=0.5)

# # c = emis.plot(ax=ax, cmap=viridis_trans, vmin=0, vmax=1e-9,
# #               add_colorbar=False)
# # cb = fig.colorbar(c, cax=cax)#, ticks=np.arange(0, 6, 1))
# # cb = fp.format_cbar(cb, cbar_title=r'Emissions (?)')
# # ax = fp.add_title(ax, 'Test')

# # fp.save_fig(fig, plot_dir, 'evec_weighting_test_1')


# # Remove emissions from buffer grid cells
# fig, ax = fp.get_figax(rows=2, cols=4,
#                        maps=True, lats=emis.lat, lons=emis.lon)
# plt.subplots_adjust(hspace=0.5)
# cax = fp.add_cax(fig, ax, cbar_pad_inches=0.5)

# for i, axis in enumerate(ax.flatten()):
#     # i = 1
#     evec_g = ip.match_data_to_clusters(evecs[:, i], clusters, 0)
#     emis_sub = xr.where(np.isclose(evec_g, 0, atol=1e-8), 0, 1e-8*evec_g/emis)
#     # evec_g/emis

#     # fig, axis = fp.get_figax(maps=True, lats=emis.lat, lons=emis.lon)
#     axis = fp.format_map(axis, lats=emis.lat, lons=emis.lon)

#     c = emis_sub.plot(ax=axis, cmap='RdBu_r', vmin=-0.5, vmax=0.5,
#                       add_colorbar=False)
#     cb = fig.colorbar(c, cax=cax)#, ticks=[])
#     cb = fp.format_cbar(cb, cbar_title=r'Fraction of Emissions')
#     axis = fp.add_title(axis, f'Eigenvector {(i+1)}')

# fp.save_fig(fig, plot_dir, 'evec_weighting_test_2')

# # evecs = xr.open_dataarray(f'{data_dir}evec_pert_01.nc')
# # fig, ax = fp.get_figax(maps=True, lats=emis.lat, lons=emis.lon)
# # ax = fp.format_map(ax, lats=emis.lat, lons=emis.lon)
# # cax = fp.add_cax(fig, ax, cbar_pad_inches=0.5)
# # c = evecs.plot(ax=ax, cmap='RdBu_r', vmin=-0.01, vmax=0.01, add_colorbar=False)
# # cb = fig.colorbar(c, cax=cax)
# # cb = fp.format_cbar(cb, cbar_title=r'Eigenvector perturbations kg/m2/s')
# # fp.save_fig(fig, plot_dir, 'evec_formatting_test')


# # fig, ax = fp.get_figax()
# # ax.hist(emis_sub.values)
# # fp.save_fig(fig, plot_dir, 'evec_weighting_test_2b')


# # Adjust units to Mg/km2/yr
# # emis *= 1e-3*(60*60*24*365)*(1000*1000)

# # evals = 'evals_h0.npy'
# # evals = np.load(f'{data_dir}{evals}')
# # evals[evals < 0] = 0
# # diff = np.abs(evals**0.5 - 1.25)
# # rank = np.argwhere(diff == np.min(diff))[0][0]
# # print(rank)

#             # fig, ax, c = ip.plot_state(prolongation[i, :], clusters,
#             #                            title='First Pertubation')
#             # # pert['evec_pert'].plot()
#             # plt.show()


