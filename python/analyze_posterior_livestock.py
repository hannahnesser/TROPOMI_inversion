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
from sklearn import linear_model
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 10)

# Custom packages
sys.path.append('.')
import config
# config.SCALE = config.PRES_SCALE
# config.BASE_WIDTH = config.PRES_WIDTH
# config.BASE_HEIGHT = config.PRES_HEIGHT
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
## Set user preferences
## ------------------------------------------------------------------------ ##
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

Re = 6375e3 # Radius of the earth in m

## ------------------------------------------------------------------------ ##
## Load posterior files
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

# Load weighting matrices in units Tg/yr
w = pd.read_csv(f'{data_dir}w_edf_hr.csv')[['enteric_fermentation', 
                                            'manure_management']].T*1e-6

# Get the posterior xhat_abs (this is n x 15)
xa_abs = w.sum(axis=1)
print('-'*75)
print('Prior (Tg/yr)')
print(xa_abs.round(2))

xhat_abs = w @ xhat
xhat_abs = ip.get_ensemble_stats(xhat_abs)
print('-'*75)
print('Posterior (Tg/yr)')
print(xhat_abs.round(2))

print('-'*75)
print('Posterior percentages (percent of all livestock)')
print((xhat_abs/xhat_abs.sum(axis=0)).round(2)*100)


xhat_sf = xhat_abs/xa_abs.values[:, None]
print('-'*75)
print('Posterior scaling factors')
print(xhat_sf.round(2))

xhat_diff_abs = w @ (xhat - 1)
xhat_diff_abs = ip.get_ensemble_stats(xhat_diff_abs)
print('-'*75)
print('Difference (Tg/yr)')
print(xhat_diff_abs.round(2))
print('-'*75)

## ------------------------------------------------------------------------ ##
## Plot enteric fermentation and manure management corrections
## ------------------------------------------------------------------------ ##
fig, ax = fp.get_figax(cols=2, maps=True, lats=clusters.lat, lons=clusters.lon)
plt.subplots_adjust(wspace=0.1)
for i, sect in enumerate(['enteric_fermentation', 'manure_management']):
    xhat_diff_abs = w.loc[sect].values[:, None]*(xhat - 1)
    xhat_diff_abs = xhat_diff_abs.mean(axis=1).values # Ensemble mean
    xhat_diff_abs = xhat_diff_abs/area.reshape(-1,)*1e6 # Tg/yr -> Mg/km2/yr
    
    title = sect.replace('_', ' ').capitalize()
    fig, ax[i], c = ip.plot_state(
        xhat_diff_abs, clusters, title=title,
        fig_kwargs={'figax' : [fig, ax[i]]}, default_value=0, 
        vmin=-5, vmax=5, cmap='RdBu_r', cbar=False)

cax = fp.add_cax(fig, ax, cbar_pad_inches=0.2)
cb = fig.colorbar(c, cax=cax)
cb = fp.format_cbar(cb, 
                    cbar_title=r'$\Delta$ Methane emissions'f'\n'r'(Mg km$^{-2}$ a$^{-1}$)')
fp.save_fig(fig, plot_dir, 'livestock_sector_xhat_diff')
print('-'*75)

## ------------------------------------------------------------------------ ##
## Compare to past studies
## ------------------------------------------------------------------------ ##
w_std = pd.read_csv(f'{data_dir}w_w37_edf.csv')[['livestock']].T*1e-6

# Iterate through the regions
for name, reg in interest.items():
    w_reg = gc.grid_shape_overlap(clusters,
                                  x=[reg[2], reg[2], reg[3], reg[3]],
                                  y=[reg[0], reg[1], reg[1], reg[0]],
                                  name=name)
    reg_prior = (w_std.values*w_reg).sum()
    reg_post = (w_std*w_reg) @ xhat
    reg_post = ip.get_ensemble_stats(reg_post)
    print('Comparison to Yu et al. (2021)')
    print(f'Prior emissions: {reg_prior:.2f}')
    print(f'Posterior emissions:')
    print(reg_post.round(2))
print('-'*75)

## ------------------------------------------------------------------------ ##
## Load animal files
## ------------------------------------------------------------------------ ##
# # GEOS-Chem grid
# lon_e_gc = np.append(clusters.lon.values - s.lon_delta/2,
#                      clusters.lon[-1].values + s.lon_delta/2)
# lat_e_gc = np.append(clusters.lat.values - s.lat_delta/2,
#                      clusters.lat[-1].values + s.lat_delta/2)
# area_gc = Re**2*(np.sin(lat_e_gc[1:]/180*np.pi) - 
#                  np.sin(lat_e_gc[:-1]/180*np.pi))*s.lon_delta/180*np.pi

# # Get  livestock
# delta_lon_ls = 0.01
# delta_lat_ls = 0.01
# ls_data_2012 = pd.DataFrame()
# ls_data_2018 = pd.DataFrame()
# for i, (file, animal) in enumerate(animal_files.items()):
#     ls = xr.open_dataset(f'{data_dir}livestock/{file}')
#     ls = ls['emi_ch4'] # Units are actually hogs
#     ls = ls.sel(lat=slice(s.lat_min, s.lat_max), 
#                     lon=slice(s.lon_min, s.lon_max))
#     ls.attrs['units'] = 'count'

#     ls_rg = xr.open_dataarray(f'{data_dir}livestock/{animal}.nc')

#     # Livestock grid
#     if i == 0:
#         lon_e_ls = np.round(np.append(ls.lon.values - delta_lon_ls/2,
#                                       ls.lon[-1].values + delta_lon_ls/2), 3)
#         lat_e_ls = np.round(np.append(ls.lat.values - delta_lat_ls/2,
#                                       ls.lat[-1].values + delta_lat_ls/2), 3)
#         area_ls = Re**2*(np.sin(lat_e_ls[1:]/180*np.pi) - 
#                          np.sin(lat_e_ls[:-1]/180*np.pi))*delta_lon_ls/180*np.pi

#     # Calculate totals
#     total = (ls*area_ls[:, None, None]).sum(['lat', 'lon']) # Mg/m2/yr -> Mg/y
#     total_rg = (ls_rg*area_gc[None, :, None]).sum(['lat', 'lon'])

#     print(f'Total {animal} 0.01x0.01 2012-2018   : ', total.values)
#     print(f'Total {animal} 0.25x0.3125 2012-2018 : ', total_rg.values)

#     ls_data_2012[animal] = ip.clusters_2d_to_1d(clusters, ls_rg.sel(year=2012)*total[0]/total_rg[0])
#     ls_data_2018[animal] = ip.clusters_2d_to_1d(clusters, ls_rg.sel(year=2018)*total[-1]/total_rg[-1])

# ls_data_2012.to_csv(f'{data_dir}livestock/livestock_2012_summary.csv')
# ls_data_2018.to_csv(f'{data_dir}livestock/livestock_2018_summary.csv')

ls_data_2012 = pd.read_csv(f'{data_dir}livestock/livestock_2012_summary.csv', 
                           index_col=0)
ls_data_2018 = pd.read_csv(f'{data_dir}livestock/livestock_2018_summary.csv', 
                           index_col=0)

## ------------------------------------------------------------------------ ##
## Do a multiple linear regression
## ------------------------------------------------------------------------ ##
mlr = linear_model.LinearRegression()

# ls_data_2012 = ls_data_2012[['cattle_feed', 'hogs', 'dairy']]
# ls_data_2018 = ls_data_2018[['cattle_feed', 'hogs', 'dairy']]

print(ls_data_2012.columns.values)

print('-'*75)
print('Enteric fermentation (2012)')
xhat_diff_abs = w.loc['enteric_fermentation'].values[:, None]*(xhat - 1)
xhat_diff_abs = xhat_diff_abs.mean(axis=1).values # Ensemble mean
xhat_diff_abs = xhat_diff_abs/area.reshape(-1,)*1e6 # Tg/yr -> Mg/km2/yr
mlr.fit(ls_data_2012, xhat_diff_abs)
print(mlr.coef_.round(2)*1000)
print(mlr.score(ls_data_2012, xhat_diff_abs))

print('-'*75)
print('Enteric fermentation (2018)')
mlr.fit(ls_data_2018, xhat_diff_abs)
print(mlr.coef_.round(2)*1000)
print(mlr.score(ls_data_2018, xhat_diff_abs))

print('-'*75)
print('Manure management (2012)')
xhat_diff_abs = w.loc['manure_management'].values[:, None]*(xhat - 1)
xhat_diff_abs = xhat_diff_abs.mean(axis=1).values # Ensemble mean
xhat_diff_abs = xhat_diff_abs/area.reshape(-1,)*1e6 # Tg/yr -> Mg/km2/yr
mlr.fit(ls_data_2012, xhat_diff_abs)
print(mlr.coef_.round(2)*1000)
print(mlr.score(ls_data_2012, xhat_diff_abs))

print('-'*75)
print('Manure management (2018)')
mlr.fit(ls_data_2018, xhat_diff_abs)
print(mlr.coef_.round(2)*1000)
print(mlr.score(ls_data_2018, xhat_diff_abs))

# ls_data['manure_management'] = ls_data['dairy'] + ls_data['hogs'] + ls_data['poultry'] + ls_data['beef_bison_cattle']


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

# ## ------------------------------------------------------------------------ ##
# ## Plot
# ## ------------------------------------------------------------------------ ##
# tot = xhat_diff_abs['mean'].sum()*1e-6
# print(f'Total livestock correction: {tot:.2f} Tg/a')

# # Plot GEPA
# fig, ax = fp.get_figax(cols=2, aspect=1, sharey=True)

# gepa_p = ip.clusters_2d_to_1d(clusters, 
#                               gepa_rg['emissions_4A_Enteric_Fermentation'])
# mask = (xhat_diff_abs['mean'] != 0) & (gepa_p*area_gc[] > 0.1)
# xs = np.arange(0, 1000, 100)
# print(gepa_p.max())
# m, b, r, bias, std = gc.comparison_stats(gepa_p[mask], 
#                                          xhat_diff_abs['mean'][mask].values)
# tot_mask = xhat_diff_abs['mean'][mask]*1e-6
# # print(f'Mean correction vs. 2012: {tot_mask.mean()*1e3:.2f} Gg/a')
# # print(f'Total correction in {labels[a]} grid cells: {tot_mask.sum():.2f} Tg/a ({tot_mask.sum()/tot*100:.2f}%)')
# ax[0].scatter(gepa_p[mask], xhat_diff_abs['mean'][mask], 
#               color=fp.color(3), s=5)
# ax[0].axhline(0, color='grey', ls='--')
# ax[0].plot(xs, m*xs + b, color=fp.color(2))
# ax[0].text(0.05, 0.95, r'R$^2$'f' = {r**2:.2f}', ha='left', va='top',
#            fontsize=config.LABEL_FONTSIZE*config.SCALE,
#            transform=ax[0].transAxes)

# # Plot 2018
# gepa_p = ip.clusters_2d_to_1d(clusters, 
#                               gepa_rg['emissions_4B_Manure_Management'])
# mask = (xhat_diff_abs['mean'] != 0) & (gepa_p > 0.1)
# # xs = np.arange(0, 1000, 100)
# m, b, r, bias, std = gc.comparison_stats(gepa_p[mask], 
#                                          xhat_diff_abs['mean'][mask].values)
# # tot_mask = xhat_diff_abs['mean'][mask]*1e-6
# # print(f'Mean correction vs. 2018: {tot_mask.mean()*1e3:.2f} Gg/a')
# # print(f'Total correction in {labels[a]} grid cells: {tot_mask.sum():.2f} ({tot_mask.sum()/tot*100:.2f}%)')
# ax[1].scatter(gepa_p[mask], xhat_diff_abs['mean'][mask], color=fp.color(7), 
#               s=5)
# ax[1].axhline(0, color='grey', ls='--')
# # ax1 = ax[1].twinx()
# # ax1.scatter(ls_p[mask], xhat[mask], color=fp.color(8), s=5, marker='x')
# ax[1].plot(xs, m*xs + b, color=fp.color(6))
# ax[1].text(0.05, 0.95, r'R$^2$'f' = {r**2:.2f}', ha='left', va='top',
#            fontsize=config.LABEL_FONTSIZE*config.SCALE,
#            transform=ax[1].transAxes)

# # Aesthetics and save
# ax[0] = fp.add_labels(ax[0], f'Enteric fermentation\nemissions', 
#                       'Posterior emissions change')
# ax[1] = fp.add_labels(ax[1], f'Manure management\nemissions', '')
# fp.save_fig(fig, plot_dir, f'gepa_scatter')
# plt.close()

# # for a, d in ls_data.items():
# #     print('-'*60)
# #     print(labels[a])

# #     fig, ax = fp.get_figax(cols=2, aspect=1, sharey=True)

# #     # Plot 2012
# #     ls_p = ip.clusters_2d_to_1d(clusters, d.sel(year=2012))
# #     animal_lim = np.percentile(ls_p[ls_p > 0].mean(), 10)
# #     print(animal_lim)
# #     mask = (xhat_diff_abs['mean'] != 0) & (ls_p > animal_lim)
# #     xs = np.arange(0, 1000, 100)
# #     m, b, r, bias, std = gc.comparison_stats(ls_p[mask], 
# #                                              xhat_diff_abs['mean'][mask].values)
# #     tot_mask = xhat_diff_abs['mean'][mask]*1e-6
# #     print(f'Mean correction vs. 2012: {tot_mask.mean()*1e3:.2f} Gg/a')
# #     print(f'Total correction in {labels[a]} grid cells: {tot_mask.sum():.2f} Tg/a ({tot_mask.sum()/tot*100:.2f}%)')
# #     ax[0].scatter(ls_p[mask], xhat_diff_abs['mean'][mask], 
# #                   color=fp.color(3), s=5)
# #     ax[0].axhline(0, color='grey', ls='--')
# #     # ax0 = ax[0].twinx()
# #     # ax0.scatter(ls_p[mask], xhat[mask], color=fp.color(4), s=5, marker='x')
# #     ax[0].plot(xs, m*xs + b, color=fp.color(2))
# #     ax[0].text(0.05, 0.95, r'R$^2$'f' = {r**2:.2f}', ha='left', va='top',
# #                fontsize=config.LABEL_FONTSIZE*config.SCALE,
# #                transform=ax[0].transAxes)

# #     # Plot 2018
# #     ls_p = ip.clusters_2d_to_1d(clusters, d.sel(year=2018))
# #     animal_lim = np.percentile(ls_p[ls_p > 0].mean(), 10)
# #     mask = (xhat_diff_abs['mean'] != 0) & (ls_p > animal_lim)
# #     m, b, r, bias, std = gc.comparison_stats(ls_p[mask], 
# #                                              xhat_diff_abs['mean'][mask].values)
# #     tot_mask = xhat_diff_abs['mean'][mask]*1e-6
# #     print(f'Mean correction vs. 2018: {tot_mask.mean()*1e3:.2f} Gg/a')
# #     print(f'Total correction in {labels[a]} grid cells: {tot_mask.sum():.2f} ({tot_mask.sum()/tot*100:.2f}%)')
# #     ax[1].scatter(ls_p[mask], xhat_diff_abs['mean'][mask], color=fp.color(7), 
# #                   s=5)
# #     ax[1].axhline(0, color='grey', ls='--')
# #     # ax1 = ax[1].twinx()
# #     # ax1.scatter(ls_p[mask], xhat[mask], color=fp.color(8), s=5, marker='x')
# #     ax[1].plot(xs, m*xs + b, color=fp.color(6))
# #     ax[1].text(0.05, 0.95, r'R$^2$'f' = {r**2:.2f}', ha='left', va='top',
# #                fontsize=config.LABEL_FONTSIZE*config.SCALE,
# #                transform=ax[1].transAxes)

# #     # Aesthetics and save
# #     ax[0] = fp.add_labels(ax[0], f'2012 EPA GHGI\n{labels[a]} counts', 
# #                           'Posterior emissions change')
# #     ax[1] = fp.add_labels(ax[1], f'2018 EPA GHGI\n{labels[a]} counts', '')
# #     fp.save_fig(fig, plot_dir, f'{a}_scatter')
# #     plt.close()

# #     # Plot hog maps
# #     fig, ax, c = ip.plot_state(ip.clusters_2d_to_1d(clusters, d.sel(year=2012)),
# #                             clusters, title=f'2012 {labels[a]}', 
# #                             cmap=viridis_trans)
# #     fp.save_fig(fig, plot_dir, f'{a}_2012')
# #     plt.close()

# #     fig, ax, c = ip.plot_state(ip.clusters_2d_to_1d(clusters, d.sel(year=2018)),
# #                             clusters, title=f'2018 {labels[a]}', 
# #                             cmap=viridis_trans)
# #     fp.save_fig(fig, plot_dir, f'{a}_2018')
# #     plt.close()

# #     # # d.sel(year=2018).plot()
# #     # # plt.show()
# #     # # ls_data[a]_d = {}
# #     # # for y in hogs.year.values:
# #     # #     print(y)
# #     # #     print(hogs.sel(year=y).squeeze())
# #     # #     ls_data[a]_d[y] = regridder(hogs.sel(year=y).squeeze())
# #     # # print(ls_data[a]_d)


# #     # # print(xhat_diff_abs)
# #     # # print(hogs['emi_ch4'])


