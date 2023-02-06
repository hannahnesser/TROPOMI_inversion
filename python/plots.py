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
# MA lat/lon lims
lat_min = 40
lat_max = 44
lon_min = -74
lon_max = -69

# Get longitude edges
lons_e = np.arange(lon_min, lon_max+0.3125, 0.3125)
print(lons_e)
# lons_e = lons_e[(lons_e > lon_min) & (lons_e < lon_max)]

# Get latitude edges
lats_e = np.arange(lat_min, lat_max+0.25, 0.25)
# lats_e = lats_e[(lats_e > lat_min) & (lats_e < lat_max)]
print(lats_e)

fig, ax = fp.get_figax(maps=True,
                       lats=[lat_min, lat_max], lons=[lon_min, lon_max])
ax = fp.format_map(ax, lats_e, lons_e, **small_map_kwargs)

for lat in lats_e:
    ax.axhline(lat, c=fp.color(4), lw=0.1)

for lon in lons_e:
    ax.axvline(lon, c=fp.color(4), lw=0.1)

# Save
fp.save_fig(fig, plot_dir, f'resolution')

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
# xhat_kwargs = {'cmap' : 'PuOr_r', 'vmin' : 0, 'vmax' : 2,
#                'default_value' : 1,
#                'cbar_kwargs' : xhat_cbar_kwargs,
#                'map_kwargs' : small_map_kwargs}


# # Fraction of information contents
# # fracs = [90, 95, 98, 99, 99.9]
# # fracs = [99.9]
# # suffixes = ['poi50', 'poi90', 'poi95', 'poi99', 'poi99.9']
# suffixes = ['poi90']

# # Load initial averaging kernel
# for f in suffixes:
#     dofs = np.load(f'{data_dir}posterior/archive/dofs1_{f}.npy')
#     xhat = np.load(f'{data_dir}posterior/archive/xhat1_{f}.npy')
#     # xhat += np.ones(xhat.shape)

#     # Filter
#     xhat[dofs < 0.01] = 1

#     # Plot averaging kernel sensitivities
#     title = f'First update of information content' # ({f}\%)'
#     fig, ax, c = ip.plot_state(dofs, clusters, title=title, **avker_kwargs)
#     ax.text(0.025, 0.05, 'DOFS = %d' % round(dofs.sum()),
#             fontsize=config.LABEL_FONTSIZE*config.SCALE,
#             transform=ax.transAxes)
#     fp.save_fig(fig, plot_dir, f'fig_est1_dofs_{f}_update')

#     # Plot posterior scaling factors
#     title = f'First update of emissions scale factors' # ({f}\%)'
#     fig, ax, c = ip.plot_state(xhat, clusters, title=title, **xhat_kwargs)
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
# # evals_q = np.load(f'{data_dir}evals_q0.npy')
# evals_h = np.load(f'{data_dir}evals_h2_rgrt.npy')
# # print(evals_h)
# evals_q = evals_h/(1+evals_h)
# # evals_h[evals_h < 0] = 0
# snr = evals_h**0.5
# DOFS_frac = np.cumsum(evals_q)/evals_q.sum()

# fig, ax = fp.get_figax(aspect=3)
# ax.set_xlim(0, 1952)

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
# # ax2.plot(snr, label='Signal-to-noise ratio spectrum', c=fp.color(6), lw=2)
# ax2.plot(evals_h, label='PPH eigenvalues', c=fp.color(6), lw=2)
# # for r in [1, 2]:
# #     diff = np.abs(snr - r)
# #     rank = np.argwhere(diff == np.min(diff))[0][0]
# #     ax2.scatter(rank, snr[rank], marker='.', s=80, c=fp.color(6))
# #     ax2.text(rank-200, snr[rank], f'{r:d}', ha='right', va='center',
# #              c=fp.color(6))
# # ax2.set_ylabel('Signal-to-noise ratio',
# ax2.set_ylabel('PPH eigenvalues',
#                fontsize=config.LABEL_FONTSIZE*config.SCALE,
#               labelpad=config.LABEL_PAD, color=fp.color(6))
# ax2.tick_params(axis='y', which='both',
#                 labelsize=config.LABEL_FONTSIZE*config.SCALE,
#                 labelcolor=fp.color(6))
# ax2.set_yscale('log')
# ax2.set_ylim(1e0, 1e5)
# ax = fp.add_title(ax, 'Final Estimate Information Content Spectrum')

# fp.save_fig(fig, plot_dir, 'eigenvalues_final')

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

# ------------------------------------------------------------------------ ##
# Figure: Error distribution comparison
# ------------------------------------------------------------------------ ##
# labels = ['Native', '1x1', '2x2', '3x3', '4x4']
# fig, ax = fig, ax = fp.get_figax(rows=1, cols=2, aspect=1.75, sharey=True)
# for i, grid in enumerate(['native', 'rg1', 'rg2', 'rg3', 'rg4']):
#     so = pd.Series(xr.open_dataarray(f'{data_dir}/so_{grid}.nc')**0.5)
#     so.plot(ax=ax[0], kind='kde', color=fp.color(2*i))
#                 # histtype='step', bins=50, density=True,
#                 # label=labels[i])

#     so_rt = pd.Series(xr.open_dataarray(f'{data_dir}/so_{grid}rt.nc')**0.5)
#     so_rt.plot(ax=ax[1], kind='kde', color=fp.color(2*i), label=labels[i])
#                    # histtype='step', bins=50, density=True,
#                # label=f'{labels[i]} seasonal')

# # Add in threshold
# so_rt = pd.Series(xr.open_dataarray(f'{data_dir}/so_rg2rt_10t.nc')**0.5)
# so_rt.plot(ax=ax[1], kind='kde', color=fp.color(2*(i+1)), label='2x2, threshold')


# ax[0] = fp.add_title(ax[0], 'Monthly averaged bias')
# ax[1] = fp.add_title(ax[1], 'Seasonally averaged bias')
# ax[0].set_xlim(0, 25)
# ax[1].set_xlim(0, 25)
# ax[0]= fp.add_labels(ax[0], 'Observational Error (ppb)', 'Count')
# ax[1]= fp.add_labels(ax[1], 'Observational Error (ppb)', '')
# ax[1] = fp.add_legend(ax[1], bbox_to_anchor=(1, 0.5), loc='center left', ncol=1)

# fp.save_fig(fig, plot_dir, f'observational_error_summary')

# ------------------------------------------------------------------------ ##
# Figure: Permian prior and observation comparison
# ------------------------------------------------------------------------ ##
# Open prior and clusterr files
# xa_abs = xr.open_dataarray(f'{data_dir}xa_abs_bc0.nc') # Mg/km2/yr --> kg/km2/hr
xa_abs = xr.open_dataarray(f'{data_dir}xa_abs.nc')#*1000/365/24
xa_abs_edf = xr.open_dataarray(f'{data_dir}xa_abs_edf.nc')#*1000/365/24
permian_idx = np.load(f'{data_dir}permian_idx.npy')
xa_abs_permian = xa_abs#[permian_idx]
xa_abs_edf_permian = xa_abs_edf#[permian_idx]

# Get the Permian cluster and basin indices (discard the buffer cells)
permian = xr.open_dataset(f'{data_dir}clusters_permian.nc')['Clusters']
c = clusters.squeeze(drop=True).to_dataset()
c['Permian'] = permian
cell_idx, cell_cnt  = np.unique(c['Permian'], return_counts=True)
cell_idx = cell_idx[cell_cnt == 1]
cell_idx = cell_idx[~np.isnan(cell_idx)]
permian = permian.where(permian.isin(cell_idx), 0)
p_lats = permian.where(permian > 0, drop=True).lat.values
p_lats = [p_lats[0] - 2, p_lats[-1] + 2]
p_lons = permian.where(permian > 0, drop=True).lon.values
p_lons = [p_lons[0] - 2, p_lons[-1] + 2]
print('-'*30)
print(p_lats)
print(p_lons)
print('-'*30)

# Colormap
yor_trans = fp.cmap_trans('YlOrRd', nalpha=100)

# Make figure
fig, ax = fp.get_figax(rows=1, cols=3, maps=True, lats=p_lats, lons=p_lons)
# plt.subplots_adjust(wspace=1)

# Prior (EPA)
fig_kwargs = {'figax' : [fig, ax[0]]}
cbar_kwargs = {'title' : 'Emissions\n' r'(Mg km$^{-2}$ a$^{-1}$)',
               'horizontal' : False}
obs_kw = {'cmap' : viridis_trans, 'vmin' : 0, 'vmax' : 20, 'default_value' : 0,
          'cbar' : False, 'map_kwargs' : small_map_kwargs,
          'fig_kwargs' : fig_kwargs}
title = 'Initial emissions\nestimate' # ({f}\%)'
fig, ax[0], c = ip.plot_state(xa_abs_permian, clusters, title=title, **obs_kw)
ax[0] = fp.format_map(ax[0], p_lats, p_lons, **small_map_kwargs)
# ax[0, 0].text(0.05, 0.05, f'{(xa_abs_permian.sum().values):.1f} Tg/yr',
#                 fontsize=config.LABEL_FONTSIZE*config.SCALE*0.75,
#                 transform=ax[0, 0].transAxes)

# Prior (EDF)
fig_kwargs = {'figax' : [fig, ax[1]]}
cbar_kwargs = {'title' : 'Emissions\n' r'(kg km$^2$ hr$^{-1}$)',
               'horizontal' : True}
obs_kw = {'cmap' : viridis_trans, 'vmin' : 0, 'vmax' : 20, 'default_value' : 0,
          'cbar' : False, 'map_kwargs' : small_map_kwargs,
          'fig_kwargs' : fig_kwargs}
obs_kw['fig_kwargs'] = {'figax' : [fig, ax[1]]}

title = f'High-resolution\ninventory' # ({f}\%)'
fig, ax[1], c = ip.plot_state(xa_abs_edf_permian, clusters,
                              title=title, **obs_kw)
ax[1] = fp.format_map(ax[1], p_lats, p_lons, **small_map_kwargs)

# Add cbar for ax 0 and 1
cax = fp.add_cax(fig, ax[0:2], horizontal=True)
cb = fig.colorbar(c, cax=cax, ax=ax[0:2], orientation='horizontal')
cb = fp.format_cbar(cb, cbar_kwargs['title'], horizontal=True)

# Observations
lat, lon = gc.create_gc_grid(permian.lat.min(), permian.lat.max(), s.lat_delta,
                             permian.lon.min(), permian.lon.max(), s.lon_delta,
                             centers=False, return_xarray=False)

obs = gc.read_file(join(data_dir, f'{s.year}_corrected.pkl'))

# Select only one month
obs = obs[obs['MONTH'] == 3]

# Save nearest latitude and longitude centers
lat_idx = gc.nearest_loc(obs['LAT'].values, lat)
obs.loc[:, 'LAT_CENTER'] = lat[lat_idx]
lon_idx = gc.nearest_loc(obs['LON'].values, lon)
obs.loc[:, 'LON_CENTER'] = lon[lon_idx]

groupby = ['LAT_CENTER', 'LON_CENTER']
obs = obs.groupby(groupby).mean()['OBS'].to_xarray()
obs = obs.rename({'LAT_CENTER' : 'lats', 'LON_CENTER' : 'lons'})

ax[-1] = fp.format_map(ax[-1], p_lats, p_lons, **small_map_kwargs)
c = obs.plot(ax=ax[-1], cmap='plasma', vmin=1800, vmax=1875, add_colorbar=False)
cax = fp.add_cax(fig, ax[-1], horizontal=True)
cb = fig.colorbar(c, ax=ax[-1], cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, 'Column mixing ratio\n(ppb)', horizontal=True)
ax[-1] = fp.add_title(ax[-1], 'TROPOMI\nmethane')

fp.save_fig(fig, plot_dir, f'fig_permian_prior_v_obs')
