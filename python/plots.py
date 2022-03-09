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

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc')

## ------------------------------------------------------------------------ ##
## Figure: Resolution context
## ------------------------------------------------------------------------ ##
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

## ------------------------------------------------------------------------ ##
## Figure: Jacobian column
## ------------------------------------------------------------------------ ##
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

## ------------------------------------------------------------------------ ##
## Figure: Kw column
## ------------------------------------------------------------------------ ##
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

## ------------------------------------------------------------------------ ##
## Figure: First estimate
## ------------------------------------------------------------------------ ##
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

## ------------------------------------------------------------------------ ##
## Figure: Prior error standard deviation
## ------------------------------------------------------------------------ ##
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

## ------------------------------------------------------------------------ ##
## Figure: Observational density
## ------------------------------------------------------------------------ ##
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

## ------------------------------------------------------------------------ ##
## Figure: Eigenvalues
## ------------------------------------------------------------------------ ##
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

## ------------------------------------------------------------------------ ##
## Figure: Initial eigenvectors
## ------------------------------------------------------------------------ ##
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

## ------------------------------------------------------------------------ ##
## Figure: Updated eigenvectors
## ------------------------------------------------------------------------ ##
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


