from os.path import join
from os import listdir
import sys
import copy
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import inversion_settings as settings

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# Local preferences
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'

# Colormaps
c = plt.cm.get_cmap('inferno', lut=10)
plasma_trans = fp.cmap_trans('plasma')
plasma_trans_r = fp.cmap_trans('plasma_r')
rdbu_trans = fp.cmap_trans_center('RdBu_r', nalpha=70)
viridis_trans_r = fp.cmap_trans('viridis_r')
magma_trans = fp.cmap_trans('magma')
# print(viridis_trans)

# Small (i.e. non-default) figure settings
small_fig_kwargs = {'max_width' : 3.25,
                    'max_height' : 3}
small_map_kwargs = {'draw_labels' : False}

########################################################################
### FIGURE 2: AVERAGING KERNEL SENSITIVITY TO PRIOR AND OBSERVATIONS ###
########################################################################

# Load initial averaging kernel
dofs = np.loadtxt(f'{data_dir}dofs0_evec_method.csv', delimiter=',')
# dofs = dofs**0.
# dofs += 0.4
clusters_plot = xr.open_dataarray(f'{data_dir}clusters.nc')
# dofs = np.diag(a)

# Initial estimate averaging kernel
# title = 'Initial estimate averaging\nkernel sensitivities'
title = 'Initial estimate KT SO-1 K'
# avker_cbar_kwargs = {'title' : r'$\partial\hat{x}_i/\partial x_i$'}
avker_cbar_kwargs = {'title' : ''}
avker_kwargs = {'cmap' : plasma_trans, 'vmin' : 0, 'vmax' : 1,
                'cbar_kwargs' : avker_cbar_kwargs,
                'fig_kwargs' : small_fig_kwargs,
                'map_kwargs' : small_map_kwargs}
fig, ax, c = ip.plot_state(dofs, clusters_plot, title=title,
                             **avker_kwargs)
# ax.text(0.025, 0.05, 'DOFS = %d' % round(dofs.sum()),
#         fontsize=config.LABEL_FONTSIZE*config.SCALE,
#         transform=ax.transAxes)
fp.save_fig(fig, plot_dir, 'fig_est0_dofs_evec_method')

# Load evecs
# evecs = np.loadtxt(f'{data_dir}evecs.csv', delimiter=',')
# evecs0 = evecs[:, 0]
# cbar_kwargs = {'title' : 'Eigenvector Value'}
# fig, ax, c = ip.plot_state(evecs0, clusters_plot, title='Eigenvector 1',
#                            cmap=rdbu_trans, cbar_kwargs=cbar_kwargs,
#                            vmin=-0.1, vmax=0.1,
#                            fig_kwargs=small_fig_kwargs,
#                            map_kwargs=small_fig_kwargs)
# fp.save_fig(fig, plot_dir, 'fig_est0_evec0')

# # Prior error
# true.sd_vec = true.sa_vec**0.5
# true.sd_vec_abs = true.sd_vec*true.xa_abs
# cbar_kwargs = {'title' : 'Tg/month'}
# fig2c, ax, c = true.plot_state('sd_vec_abs', clusters_plot,
#                                 title='Prior error standard deviation',
#                                 cmap=fp.cmap_trans('viridis'),
#                                 vmin=0, vmax=15,
#                                 fig_kwargs=small_fig_kwargs,
#                                 cbar_kwargs=cbar_kwargs,
#                                 map_kwargs=small_map_kwargs)
# fp.save_fig(fig2c, plots, 'fig2c_prior_error')

# # Observational density
# lat_res = np.diff(clusters_plot.lat)[0]
# lat_edges = np.append(clusters_plot.lat - lat_res/2,
#                       clusters_plot.lat[-1] + lat_res/2)
# # lat_edges = lat_edges[::2]
# obs['lat_edges'] = pd.cut(obs['lat'], lat_edges, precision=4)

# lon_res = np.diff(clusters_plot.lon)[0]
# lon_edges = np.append(clusters_plot.lon - lon_res/2,
#                       clusters_plot.lon[-1] + lon_res/2)
# # lon_edges = lon_edges[::2]
# obs['lon_edges'] = pd.cut(obs['lon'], lon_edges, precision=4)

# obs_density = obs.groupby(['lat_edges', 'lon_edges']).count()
# obs_density = obs_density['Nobs'].reset_index()
# obs_density['lat'] = obs_density['lat_edges'].apply(lambda x: x.mid)
# obs_density['lon'] = obs_density['lon_edges'].apply(lambda x: x.mid)
# obs_density = obs_density.set_index(['lat', 'lon'])['Nobs']
# obs_density = obs_density.to_xarray()

# title = 'GOSAT observation density\n(July 2009)'
# viridis_trans_long = fp.cmap_trans('viridis', nalpha=90, ncolors=300)
# cbar_kwargs = {'ticks' : np.arange(0, 25, 5),
#                'title' : 'Count'}
# fig2d, ax, c = true.plot_state_format(obs_density, title=title,
#                                        vmin=0, vmax=10, default_value=0,
#                                        cmap=viridis_trans_long,
#                                        fig_kwargs=small_fig_kwargs,
#                                        cbar_kwargs=cbar_kwargs,
#                                        map_kwargs=small_map_kwargs)
# fp.save_fig(fig2d, plots, 'fig2d_gosat_obs_density')
