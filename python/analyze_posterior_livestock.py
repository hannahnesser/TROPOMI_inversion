# from os.path import join
# from os import listdir
# import sys
# import glob
# import copy
# import math
# import xarray as xr
# import xesmf as xe
# import numpy as np
# import pandas as pd
# from scipy.stats import probplot as qq
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# from matplotlib.patches import Patch as patch
# import cartopy.feature as cfeature
# import cartopy.io.shapereader as shpreader
# import cartopy.crs as ccrs
# import imageio
# pd.set_option('display.max_columns', 10)

# # Custom packages
# sys.path.append('.')
# import config
# config.SCALE = config.PRES_SCALE
# config.BASE_WIDTH = config.PRES_WIDTH
# config.BASE_HEIGHT = config.PRES_HEIGHT
# import gcpy as gc
# import troppy as tp
# import invpy as ip
# import format_plots as fp
# import inversion_settings as s

# ## ------------------------------------------------------------------------ ##
# ## Directories
# ## ------------------------------------------------------------------------ ##
# base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
# code_dir = base_dir + 'python/'
# data_dir = base_dir + 'inversion_data/'
# plot_dir = base_dir + 'plots/'

# ## ------------------------------------------------------------------------ ##
# ## Set plotting preferences
# ## ------------------------------------------------------------------------ ##
# # Colormaps
# plasma_trans = fp.cmap_trans('plasma')
# plasma_trans_r = fp.cmap_trans('plasma_r')
# rdbu_trans = fp.cmap_trans_center('RdBu_r', nalpha=70)
# r_trans = fp.cmap_trans('Reds', nalpha=100)
# yor_trans = fp.cmap_trans('YlOrRd', nalpha=100)
# viridis_trans_r = fp.cmap_trans('viridis_r')
# viridis_trans = fp.cmap_trans('viridis')
# magma_trans = fp.cmap_trans('magma')
# # print(viridis_trans)

# # sf_cmap_1 = plt.cm.Reds(np.linspace(0, 0.5, 256))
# # sf_cmap_2 = plt.cm.Blues(np.linspace(0.5, 1, 256))
# # sf_cmap = np.vstack((sf_cmap_1, sf_cmap_2))
# # sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
# # div_norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=6)

# sf_cmap_1 = plt.cm.PuOr_r(np.linspace(0.2, 0.5, 256))
# sf_cmap_2 = plt.cm.PuOr_r(np.linspace(0.5, 1, 256))
# sf_cmap = np.vstack((sf_cmap_1, sf_cmap_2))
# sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
# div_norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=1, vmax=3)

# diff_cmap_1 = plt.cm.bwr(np.linspace(0, 0.5, 256))
# diff_cmap_2 = plt.cm.bwr(np.linspace(0.5, 1, 256))
# diff_cmap = np.vstack((diff_cmap_1, diff_cmap_2))
# diff_cmap = colors.LinearSegmentedColormap.from_list('diff_cmap', diff_cmap)
# diff_div_norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=1)

# # Small (i.e. non-default) figure settings
# small_map_kwargs = {'draw_labels' : False}
# small_fig_kwargs = {'max_width' : 4,
#                     'max_height' : 3.5}

# ## ------------------------------------------------------------------------ ##
# ## Set user preferences
# ## ------------------------------------------------------------------------ ##
# # DOFS_filter
# DOFS_filter = 0.05

# # List emissions categories
# interest = {'livestock' : [('North Carolina', [33.5, 37, -84.5, -75.5], [0, 20],
#                             None),
#                            ('Midwest', [38.5, 46, -100, -81], [0, 20],
#                             None),
#                            ('Central Valley', [33, 40, -125, -115], [0, 40],
#                             None)]}

# # Define file names
# f = 'rg2rt_10t_w404_rf0.25_sax0.75_poi80.0'
# xa_abs_file = 'xa_abs_w404.nc' #'xa_abs_wetlands404_edf.nc' 
# w_file = 'w_w404.csv' #'w_wetlands404_edf.csv'
# optimize_BC = False

# ## ------------------------------------------------------------------------ ##
# ## Load files
# ## ------------------------------------------------------------------------ ##
# # Load clusters
# clusters = xr.open_dataarray(f'{data_dir}clusters.nc')

# # Load prior (Mg/km2/yr)
# xa_abs = xr.open_dataarray(f'{data_dir}{xa_abs_file}').values.reshape((-1, 1))
# soil = xr.open_dataarray(f'{data_dir}soil_abs.nc')
# area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# # Load weighting matrix (Mg/)
# w = pd.read_csv(f'{data_dir}{w_file}')
# w_rel = w.div(w.sum(axis=1), axis=0)
# w_rel = w_rel.fillna(0)

# w_mask = copy.deepcopy(w)
# w_mask = w_mask.where(w*area/1e3 > 0.5, 0)
# w_mask = w_mask.where(w*area/1e3 <= 0.5, 1)

# w['total'] = w.sum(axis=1)

# # Load posterior and DOFS
# dofs = np.load(f'{data_dir}posterior/dofs2_{f}.npy').reshape((-1, 1))
# xhat = np.load(f'{data_dir}posterior/xhat_fr2_{f}.npy').reshape((-1, 1))
# # dofs = np.nan_to_num(dofs, 0)

# # BC alteration
# if optimize_BC:
#     print('-'*100)
#     print('Boundary condition optimization')
#     print(' N E S W')
#     print('xhat : ', xhat[-4:])
#     print('dofs : ', dofs[-4:])
#     print('-'*100)
#     xhat = xhat[:-4]
#     dofs = dofs[:-4]

# # Print information
# print('-'*100)
# print(f'We optimize {(dofs >= DOFS_filter).sum():d} grid cells, including {xa_abs[dofs >= DOFS_filter].sum():.2E}/{xa_abs.sum():.2E} = {(xa_abs[dofs >= DOFS_filter].sum()/xa_abs.sum()*100):.2f}% of prior emissions. This\nproduces {dofs[dofs >= DOFS_filter].sum():.2f} ({dofs.sum():.2f}) DOFS with an xhat range of {xhat.min():.2f} to {xhat.max():.2f}. There are {len(xhat[xhat < 0]):d} negative values.')
# print('-'*100)

# # Filter on DOFS filter
# xhat[dofs < DOFS_filter] = 1
# dofs[dofs < DOFS_filter] = 0

# # Calculate xhat abs
# xhat_abs = (xhat*xa_abs)

# # Get county outlines for high resolution results
# reader = shpreader.Reader(f'{data_dir}counties/countyl010g.shp')
# counties = list(reader.geometries())
# COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

# # Get hogs 
# hogs = xr.open_dataset(f'{data_dir}livestock/Intermediate_EPA_v2_Hog_Map.nc')
# hogs = hogs['emi_ch4'] # Units are actually hogs
# # hogs = hogs*(16.04*1e-6/6.02214e23)*(1e4)*(60*60*24*365) #molec/cm2/s->Mg/m2/yr
# hogs = hogs.sel(lat=slice(s.lat_min, s.lat_max), 
#                 lon=slice(s.lon_min, s.lon_max))
# hogs.attrs['units'] = 'count'

# # Get hogs regridded
# hogs_rg = xr.open_dataarray(f'{data_dir}livestock/hogs.nc')

# # Total emissions as check
# ## Original
# delta_lon_ls = 0.01
# delta_lat_ls = 0.01
# Re = 6375e3 # Radius of the earth in m
# lon_e_ls = np.round(np.append(hogs.lon.values - delta_lon_ls/2,
#                               hogs.lon[-1].values + delta_lon_ls/2), 3)
# lat_e_ls = np.round(np.append(hogs.lat.values - delta_lat_ls/2,
#                               hogs.lat[-1].values + delta_lat_ls/2), 3)
# area_ls = Re**2*(np.sin(lat_e_ls[1:]/180*np.pi) - 
#                  np.sin(lat_e_ls[:-1]/180*np.pi))*delta_lon_ls/180*np.pi
# total = (hogs*area_ls[:, None, None]).sum(['lat', 'lon']) # Mg/m2/yr -> Mg/y

# ## Regridded
# lon_e_gc = np.append(clusters.lon.values - s.lon_delta/2,
#                      clusters.lon[-1].values + s.lon_delta/2)
# lat_e_gc = np.append(clusters.lat.values - s.lat_delta/2,
#                      clusters.lat[-1].values + s.lat_delta/2)
# area_gc = Re**2*(np.sin(lat_e_gc[1:]/180*np.pi) - 
#                  np.sin(lat_e_gc[:-1]/180*np.pi))*s.lon_delta/180*np.pi
# total_rg = (hogs_rg*area_gc[None, :, None]).sum(['lat', 'lon'])

# print('Total hogs 0.01x0.01 2012-2018 : ', total.values)
# print('Total hogs 0.25x0.3125 2012-2018 : ', total_rg.values)

# # ## ------------------------------------------------------------------------ ##
# # ## Regrid native resolution hogs dataset
# # ## ------------------------------------------------------------------------ ##
# # # Regrid hogs onto the model grid
# # ## Get grid cell edges and area to calculate total mass
# # grid_ls = {'lat' : hogs.lat, 'lon' : hogs.lon,
# #            'lat_b' : lat_e_ls, 'lon_b' : lon_e_ls}

# # ## Get GEOS-Chem grid
# # grid_gc = {'lat' : clusters.lat, 'lon' : clusters.lon,
# #            'lat_b' : lat_e_gc, 'lon_b' : lon_e_gc}

# # ## Get the regridder
# # # regridder = xe.Regridder(grid_ls, grid_gc, 'conservative')
# # # regridder.to_netcdf(f'{data_dir}livestock/regridder_0.01x0.01_0.25x0.3125.nc')
# # regridder = xe.Regridder(grid_ls, grid_gc, 'conservative', 
# #                          weights=f'{data_dir}livestock/regridder_0.01x0.01_0.25x0.3125.nc')
# # print('Regridder generated')

# # ## Regrid the data
# # hogs_rg = regridder(hogs)
# # hogs_rg.to_netcdf(f'{data_dir}livestock/hogs.nc')

# ## ------------------------------------------------------------------------ ##
# ## Plot
# ## ------------------------------------------------------------------------ ##
# # Livestock (Mg/km2/yr)
# xhat_diff_ls = (w*(xhat - 1))['livestock'].values

# fig, ax = fp.get_figax(cols=2, aspect=1, sharey=True)

# # Plot 2012
# hog_lim = 10
# hogs_p = ip.clusters_2d_to_1d(clusters, hogs_rg.sel(year=2012))
# mask = (xhat_diff_ls != 0) & (hogs_p > hog_lim) & (xhat_abs.reshape(-1,) > 0)
# xs = np.arange(0, 1000, 100)
# m, b, r, bias, std = gc.comparison_stats(hogs_p[mask], xhat_diff_ls[mask])
# print('Mean correction vs. 2012 hogs: ', xhat_diff_ls[mask].mean())
# ax[0].scatter(hogs_p[mask], xhat_diff_ls[mask], color=fp.color(3), s=5)
# ax[0].axhline(0, color='grey', ls='--')
# # ax0 = ax[0].twinx()
# # ax0.scatter(hogs_p[mask], xhat[mask], color=fp.color(4), s=5, marker='x')
# ax[0].plot(xs, m*xs + b, color=fp.color(2))
# ax[0].text(0.05, 0.95, r'R$^2$'f' = {r**2:.2f}', ha='left', va='top',
#            fontsize=config.LABEL_FONTSIZE*config.SCALE,
#            transform=ax[0].transAxes)

# # Plot 2018
# hogs_p = ip.clusters_2d_to_1d(clusters, hogs_rg.sel(year=2018))
# mask = (xhat_diff_ls != 0) & (hogs_p > hog_lim) & (xhat_abs.reshape(-1,) > 0)
# m, b, r, bias, std = gc.comparison_stats(hogs_p[mask], xhat_diff_ls[mask])
# print('Mean correction vs. 2018 hogs: ', xhat_diff_ls[mask].mean())
# ax[1].scatter(hogs_p[mask], xhat_diff_ls[mask], color=fp.color(7), s=5)
# ax[1].axhline(0, color='grey', ls='--')
# # ax1 = ax[1].twinx()
# # ax1.scatter(hogs_p[mask], xhat[mask], color=fp.color(8), s=5, marker='x')
# ax[1].plot(xs, m*xs + b, color=fp.color(6))
# ax[1].text(0.05, 0.95, r'R$^2$'f' = {r**2:.2f}', ha='left', va='top',
#            fontsize=config.LABEL_FONTSIZE*config.SCALE,
#            transform=ax[1].transAxes)

# # Aesthetics and save
# ax[0] = fp.add_labels(ax[0], '2012 EPA GHGI\nhog counts', 
#                       'Posterior emissions change')
# ax[1] = fp.add_labels(ax[1], '2018 EPA GHGI\nhog counts', '')
# fp.save_fig(fig, plot_dir, 'livestock_scatter')

# # Plot hog maps
# fig, ax, c = ip.plot_state(ip.clusters_2d_to_1d(clusters, hogs_rg.sel(year=2012)),
#                         clusters, title='2012 Hogs', cmap=viridis_trans)
# fp.save_fig(fig, plot_dir, 'hogs_2012')

# fig, ax, c = ip.plot_state(ip.clusters_2d_to_1d(clusters, hogs_rg.sel(year=2018)),
#                         clusters, title='2018 Hogs', cmap=viridis_trans)
# fp.save_fig(fig, plot_dir, 'hogs_2018')

# # hogs_rg.sel(year=2018).plot()
# # plt.show()
# # hogs_rg_d = {}
# # for y in hogs.year.values:
# #     print(y)
# #     print(hogs.sel(year=y).squeeze())
# #     hogs_rg_d[y] = regridder(hogs.sel(year=y).squeeze())
# # print(hogs_rg_d)


# # print(xhat_diff_ls)
# # print(hogs['emi_ch4'])

import sys
from copy import deepcopy as dc
import glob
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import shapefile
from shapely.geometry import Polygon, MultiPolygon
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
DOFS_filter = 0.05

# Define basins of interest
interest = {'Upper midwest' : [35, 55, -105, -80]}

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
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
w37_cols = [s for s in ensemble if 'w37' in s]
w404_cols = [s for s in ensemble if 'w404' in s]

# Load weighting matrices in units Mg/yr (we don't consider wetlands
# here, so it doesn't matter which W we use)
w = pd.read_csv(f'{data_dir}w_w404_edf.csv')['livestock'].T

# Get the posterior xhat_abs (this is n x 15)
xhat_diff_abs = (w.values[:, None]*(xhat - 1))

# Iterate through the regions
for name, reg in interest.items():
    c = clusters.where((clusters.lat > reg[0]) &
                       (clusters.lat < reg[1]) &
                       (clusters.lon > reg[2]) &
                       (clusters.lon < reg[3]), drop=True)
    c_idx = (c.values[c.values > 0] - 1).astype(int)

    tt_prior = w.values[c_idx].sum()*1e-6
    tt_post = xhat_diff_abs.values[c_idx, :].sum(axis=0)*1e-6
    print(f'{name:<20s}:')
    print(f'Prior         : {tt_prior:.2f} Tg/yr')
    print(f'Posterior     : {(tt_prior + tt_post.mean()):4.2f} ({(tt_prior + tt_post.min()):4.2f}, {(tt_prior + tt_post.max()):4.2f}) Tg/yr')
    print(f'Delta         : {tt_post.mean():4.2f} ({tt_post.min():4.2f}, {tt_post.max():4.2f}) Tg/yr')
    print(f'Percent change: {(tt_post.mean()/tt_prior*100):4.2f}%')

