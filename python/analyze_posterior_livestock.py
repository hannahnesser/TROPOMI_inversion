from os.path import join
from os import listdir
import sys
import glob
import copy
import math
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
from scipy.stats import probplot as qq
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Patch as patch
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
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
# Get county outlines for high resolution results
reader = shpreader.Reader(f'{data_dir}counties/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

viridis_trans = fp.cmap_trans('viridis')

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# DOFS_filter
DOFS_filter = 0.05

# Define basins of interest
interest = {'Upper midwest' : [35, 55, -105, -80]}

animal_files = {
    'Intermediate_EPA_v2_AnimalMaps_shplmb.nc' : 'sheep',
    # 'Intermediate_EPA_v2_AnimalMaps_animal.nc' : 'animals',
    'Intermediate_EPA_v2_AnimalMaps_brltrk.nc' : 'broilers_turkeys',
    'Intermediate_EPA_v2_AnimalMaps_ctlfed.nc' : 'cattle_feed',
    'Intermediate_EPA_v2_AnimalMaps_ctlinv.nc' : 'beef_bison_cattle',
    'Intermediate_EPA_v2_AnimalMaps_goat.nc'   : 'goats',
    'Intermediate_EPA_v2_AnimalMaps_hogpig.nc' : 'hogs',
    'Intermediate_EPA_v2_AnimalMaps_hrspny.nc' : 'horses',
    'Intermediate_EPA_v2_AnimalMaps_lyrplt.nc' : 'poultry',
    'Intermediate_EPA_v2_AnimalMaps_mlkcow.nc' : 'dairy'
                }

labels = {
    'sheep' : 'sheep',
    'broilers_turkeys' : 'broilers and turkeys',
    'cattle_feed' : 'cattle feed',
    'beef_bison_cattle' : 'beef, bison, and cattle',
    'goats' : 'goats',
    'hogs' : 'hogs',
    'horses' : 'horses',
    'poultry' : 'poultry',
    'dairy' : 'dairy',
    'manure_management' : 'manure management'
}

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

# Load weighting matrices in units Mg/yr
w = pd.read_csv(f'{data_dir}w_w37_edf.csv')['livestock'].T

# Get the posterior xhat_abs (this is n x 15)
xhat_diff_abs = (w.values[:, None]*(xhat - 1))
xhat_diff_abs = ip.get_ensemble_stats(xhat_diff_abs)
xhat = ip.get_ensemble_stats(xhat)

# GEOS-Chem grid
Re = 6375e3 # Radius of the earth in m
lon_e_gc = np.append(clusters.lon.values - s.lon_delta/2,
                     clusters.lon[-1].values + s.lon_delta/2)
lat_e_gc = np.append(clusters.lat.values - s.lat_delta/2,
                     clusters.lat[-1].values + s.lat_delta/2)
area_gc = Re**2*(np.sin(lat_e_gc[1:]/180*np.pi) - 
                 np.sin(lat_e_gc[:-1]/180*np.pi))*s.lon_delta/180*np.pi

# Load GEPA and convert from molec/cm2/sec to Mg/km2/yr
gepa = xr.open_dataset(f'{data_dir}livestock/GEPA_Annual.nc')
gepa = gepa[['emissions_4A_Enteric_Fermentation',
             'emissions_4B_Manure_Management']]
gepa = gepa.sel(lat=slice(s.lat_min, s.lat_max),
                lon=slice(s.lon_min, s.lon_max))
gepa *= (1/6.022e23)*(16.04e-6)*(1e4)*(60*60*24*365)
gepa_rg = xr.open_dataset(f'{data_dir}livestock/GEPA_regridded.nc')
gepa_rg *= (1/6.022e23)*(16.04e-6)*(1e4)*(60*60*24*365)

# Load the GEPA grid
lon_e_gepa = np.append(gepa.lon.values - 0.1/2, gepa.lon[-1].values + 0.1/2)
lon_e_gepa = np.round(lon_e_gepa, 3)
lat_e_gepa = np.append(gepa.lat.values - 0.1/2, gepa.lat[-1].values + 0.1/2)
lat_e_gepa = np.round(lat_e_gepa, 3)
area_gepa = Re**2*(np.sin(lat_e_gepa[1:]/180*np.pi) - 
                   np.sin(lat_e_gepa[:-1]/180*np.pi))*0.1/180*np.pi

# Compare counts
total = (gepa*area_gepa[:, None]).sum(['lat', 'lon'])*1e-6
total_rg = (gepa_rg*area_gc[:, None]).sum(['lat', 'lon'])*1e-6
print(f'Total GEPA 0.1x0.1     : ', total.values)
print(f'Total GEPA 0.25x0.3125 : ', total_rg.values)

# Get  livestock
delta_lon_ls = 0.01
delta_lat_ls = 0.01
ls_data = {}
for i, (file, animal) in enumerate(animal_files.items()):
    ls = xr.open_dataset(f'{data_dir}livestock/{file}')
    ls = ls['emi_ch4'] # Units are actually hogs
    ls = ls.sel(lat=slice(s.lat_min, s.lat_max), 
                    lon=slice(s.lon_min, s.lon_max))
    ls.attrs['units'] = 'count'

    ls_rg = xr.open_dataarray(f'{data_dir}livestock/{animal}.nc')
    ls_data[animal] = ls_rg

    # Livestock grid
    if i == 0:
        lon_e_ls = np.round(np.append(ls.lon.values - delta_lon_ls/2,
                                      ls.lon[-1].values + delta_lon_ls/2), 3)
        lat_e_ls = np.round(np.append(ls.lat.values - delta_lat_ls/2,
                                      ls.lat[-1].values + delta_lat_ls/2), 3)
        area_ls = Re**2*(np.sin(lat_e_ls[1:]/180*np.pi) - 
                         np.sin(lat_e_ls[:-1]/180*np.pi))*delta_lon_ls/180*np.pi

    # Calculate totals
    total = (ls*area_ls[:, None, None]).sum(['lat', 'lon']) # Mg/m2/yr -> Mg/y
    total_rg = (ls_rg*area_gc[None, :, None]).sum(['lat', 'lon'])

    print(f'Total {animal} 0.01x0.01 2012-2018   : ', total.values)
    print(f'Total {animal} 0.25x0.3125 2012-2018 : ', total_rg.values)

ls_data['manure_management'] = ls_data['dairy'] + ls_data['hogs'] + ls_data['poultry'] + ls_data['beef_bison_cattle']

## ------------------------------------------------------------------------ ##
## Regrid native resolution hogs dataset
## ------------------------------------------------------------------------ ##
## Regrid hogs onto the model grid
## Get grid cell edges and area to calculate total mass
# grid_ls = {'lat' : ls.lat, 'lon' : ls.lon,
#            'lat_b' : lat_e_ls, 'lon_b' : lon_e_ls}

# ## Get the GEPA grid
# grid_gepa = {'lat' : gepa.lat, 'lon' : gepa.lon, 
#              'lat_b' : lat_e_gepa, 'lon_b' : lon_e_gepa}

# ## Get GEOS-Chem grid
# grid_gc = {'lat' : clusters.lat, 'lon' : clusters.lon,
#            'lat_b' : lat_e_gc, 'lon_b' : lon_e_gc}

# ## Get the regridder (0.01)
# # regridder = xe.Regridder(grid_ls, grid_gc, 'conservative')
# # regridder.to_netcdf(f'{data_dir}livestock/regridder_0.01x0.01_0.25x0.3125.nc')
# regridder = xe.Regridder(grid_ls, grid_gc, 'conservative', 
#                          weights=f'{data_dir}livestock/regridder_0.01x0.01_0.25x0.3125.nc')

# ## Get the regridder (0.1)
# # regridder = xe.Regridder(grid_gepa, grid_gc, 'conservative')
# # regridder.to_netcdf(f'{data_dir}livestock/regridder_0.1x0.1_0.25x0.3125.nc')
# regridder = xe.Regridder(grid_gepa, grid_gc, 'conservative', 
#                          weights=f'{data_dir}livestock/regridder_0.1x0.1_0.25x0.3125.nc')

# print('Regridder generated')

# # ## Regrid the livestock data
# # for file, name in animal_files.items():
# #     print(f'Processing: {name}')
# #     ls = xr.open_dataset(f'{data_dir}livestock/{file}')
# #     ls = ls['emi_ch4'] # Units are actually hogs
# #     ls = ls.sel(lat=slice(s.lat_min, s.lat_max), 
# #                 lon=slice(s.lon_min, s.lon_max))
# #     ls.attrs['units'] = 'count'

# #     ls_rg = regridder(ls)
# #     ls_rg.to_netcdf(f'{data_dir}livestock/{name}.nc')

# # gepa_rg = regridder(gepa)
# # gepa_rg.to_netcdf(f'{data_dir}livestock/GEPA_regridded.nc')

## ------------------------------------------------------------------------ ##
## Plot
## ------------------------------------------------------------------------ ##
tot = xhat_diff_abs['mean'].sum()*1e-6
print(f'Total livestock correction: {tot:.2f} Tg/a')

# Plot GEPA
fig, ax = fp.get_figax(cols=2, aspect=1, sharey=True)

gepa_p = ip.clusters_2d_to_1d(clusters, 
                              gepa_rg['emissions_4A_Enteric_Fermentation'])
mask = (xhat_diff_abs['mean'] != 0) & (gepa_p*area_gc[] > 0.1)
xs = np.arange(0, 1000, 100)
print(gepa_p.max())
m, b, r, bias, std = gc.comparison_stats(gepa_p[mask], 
                                         xhat_diff_abs['mean'][mask].values)
tot_mask = xhat_diff_abs['mean'][mask]*1e-6
# print(f'Mean correction vs. 2012: {tot_mask.mean()*1e3:.2f} Gg/a')
# print(f'Total correction in {labels[a]} grid cells: {tot_mask.sum():.2f} Tg/a ({tot_mask.sum()/tot*100:.2f}%)')
ax[0].scatter(gepa_p[mask], xhat_diff_abs['mean'][mask], 
              color=fp.color(3), s=5)
ax[0].axhline(0, color='grey', ls='--')
ax[0].plot(xs, m*xs + b, color=fp.color(2))
ax[0].text(0.05, 0.95, r'R$^2$'f' = {r**2:.2f}', ha='left', va='top',
           fontsize=config.LABEL_FONTSIZE*config.SCALE,
           transform=ax[0].transAxes)

# Plot 2018
gepa_p = ip.clusters_2d_to_1d(clusters, 
                              gepa_rg['emissions_4B_Manure_Management'])
mask = (xhat_diff_abs['mean'] != 0) & (gepa_p > 0.1)
# xs = np.arange(0, 1000, 100)
m, b, r, bias, std = gc.comparison_stats(gepa_p[mask], 
                                         xhat_diff_abs['mean'][mask].values)
# tot_mask = xhat_diff_abs['mean'][mask]*1e-6
# print(f'Mean correction vs. 2018: {tot_mask.mean()*1e3:.2f} Gg/a')
# print(f'Total correction in {labels[a]} grid cells: {tot_mask.sum():.2f} ({tot_mask.sum()/tot*100:.2f}%)')
ax[1].scatter(gepa_p[mask], xhat_diff_abs['mean'][mask], color=fp.color(7), 
              s=5)
ax[1].axhline(0, color='grey', ls='--')
# ax1 = ax[1].twinx()
# ax1.scatter(ls_p[mask], xhat[mask], color=fp.color(8), s=5, marker='x')
ax[1].plot(xs, m*xs + b, color=fp.color(6))
ax[1].text(0.05, 0.95, r'R$^2$'f' = {r**2:.2f}', ha='left', va='top',
           fontsize=config.LABEL_FONTSIZE*config.SCALE,
           transform=ax[1].transAxes)

# Aesthetics and save
ax[0] = fp.add_labels(ax[0], f'Enteric fermentation\nemissions', 
                      'Posterior emissions change')
ax[1] = fp.add_labels(ax[1], f'Manure management\nemissions', '')
fp.save_fig(fig, plot_dir, f'gepa_scatter')
plt.close()

# for a, d in ls_data.items():
#     print('-'*60)
#     print(labels[a])

#     fig, ax = fp.get_figax(cols=2, aspect=1, sharey=True)

#     # Plot 2012
#     ls_p = ip.clusters_2d_to_1d(clusters, d.sel(year=2012))
#     animal_lim = np.percentile(ls_p[ls_p > 0].mean(), 10)
#     print(animal_lim)
#     mask = (xhat_diff_abs['mean'] != 0) & (ls_p > animal_lim)
#     xs = np.arange(0, 1000, 100)
#     m, b, r, bias, std = gc.comparison_stats(ls_p[mask], 
#                                              xhat_diff_abs['mean'][mask].values)
#     tot_mask = xhat_diff_abs['mean'][mask]*1e-6
#     print(f'Mean correction vs. 2012: {tot_mask.mean()*1e3:.2f} Gg/a')
#     print(f'Total correction in {labels[a]} grid cells: {tot_mask.sum():.2f} Tg/a ({tot_mask.sum()/tot*100:.2f}%)')
#     ax[0].scatter(ls_p[mask], xhat_diff_abs['mean'][mask], 
#                   color=fp.color(3), s=5)
#     ax[0].axhline(0, color='grey', ls='--')
#     # ax0 = ax[0].twinx()
#     # ax0.scatter(ls_p[mask], xhat[mask], color=fp.color(4), s=5, marker='x')
#     ax[0].plot(xs, m*xs + b, color=fp.color(2))
#     ax[0].text(0.05, 0.95, r'R$^2$'f' = {r**2:.2f}', ha='left', va='top',
#                fontsize=config.LABEL_FONTSIZE*config.SCALE,
#                transform=ax[0].transAxes)

#     # Plot 2018
#     ls_p = ip.clusters_2d_to_1d(clusters, d.sel(year=2018))
#     animal_lim = np.percentile(ls_p[ls_p > 0].mean(), 10)
#     mask = (xhat_diff_abs['mean'] != 0) & (ls_p > animal_lim)
#     m, b, r, bias, std = gc.comparison_stats(ls_p[mask], 
#                                              xhat_diff_abs['mean'][mask].values)
#     tot_mask = xhat_diff_abs['mean'][mask]*1e-6
#     print(f'Mean correction vs. 2018: {tot_mask.mean()*1e3:.2f} Gg/a')
#     print(f'Total correction in {labels[a]} grid cells: {tot_mask.sum():.2f} ({tot_mask.sum()/tot*100:.2f}%)')
#     ax[1].scatter(ls_p[mask], xhat_diff_abs['mean'][mask], color=fp.color(7), 
#                   s=5)
#     ax[1].axhline(0, color='grey', ls='--')
#     # ax1 = ax[1].twinx()
#     # ax1.scatter(ls_p[mask], xhat[mask], color=fp.color(8), s=5, marker='x')
#     ax[1].plot(xs, m*xs + b, color=fp.color(6))
#     ax[1].text(0.05, 0.95, r'R$^2$'f' = {r**2:.2f}', ha='left', va='top',
#                fontsize=config.LABEL_FONTSIZE*config.SCALE,
#                transform=ax[1].transAxes)

#     # Aesthetics and save
#     ax[0] = fp.add_labels(ax[0], f'2012 EPA GHGI\n{labels[a]} counts', 
#                           'Posterior emissions change')
#     ax[1] = fp.add_labels(ax[1], f'2018 EPA GHGI\n{labels[a]} counts', '')
#     fp.save_fig(fig, plot_dir, f'{a}_scatter')
#     plt.close()

#     # Plot hog maps
#     fig, ax, c = ip.plot_state(ip.clusters_2d_to_1d(clusters, d.sel(year=2012)),
#                             clusters, title=f'2012 {labels[a]}', 
#                             cmap=viridis_trans)
#     fp.save_fig(fig, plot_dir, f'{a}_2012')
#     plt.close()

#     fig, ax, c = ip.plot_state(ip.clusters_2d_to_1d(clusters, d.sel(year=2018)),
#                             clusters, title=f'2018 {labels[a]}', 
#                             cmap=viridis_trans)
#     fp.save_fig(fig, plot_dir, f'{a}_2018')
#     plt.close()

#     # # d.sel(year=2018).plot()
#     # # plt.show()
#     # # ls_data[a]_d = {}
#     # # for y in hogs.year.values:
#     # #     print(y)
#     # #     print(hogs.sel(year=y).squeeze())
#     # #     ls_data[a]_d[y] = regridder(hogs.sel(year=y).squeeze())
#     # # print(ls_data[a]_d)


#     # # print(xhat_diff_abs)
#     # # print(hogs['emi_ch4'])


