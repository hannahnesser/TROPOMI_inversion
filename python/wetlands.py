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
## Universal variables
## ------------------------------------------------------------------------ ##
Re = 6375e3 # Radius of the earth in m
days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

## ------------------------------------------------------------------------ ##
## Directories
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'prior/wetlands/'
plot_dir = base_dir + 'plots/'
wetland_file = 'WetCHARTs_Highest_performance_Ensemble_v1.3.1_2010_2019.nc'

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

sf_cmap_1 = plt.cm.PuOr_r(np.linspace(0, 0.5, 256))
sf_cmap_2 = plt.cm.PuOr_r(np.linspace(0.5, 1, 256))
sf_cmap = np.vstack((sf_cmap_1, sf_cmap_2))
sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
div_norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)

# Small (i.e. non-default) figure settings
small_map_kwargs = {'draw_labels' : False}
small_fig_kwargs = {'max_width' : 4,
                    'max_height' : 3.5}

## ------------------------------------------------------------------------ ##
## Open and subset wetlands file
## ------------------------------------------------------------------------ ##
# Units are mg CH4/m2/day
wl_wc = xr.open_dataset(f'{data_dir}{wetland_file}', decode_times=False)
print(wl_wc.attrs)
print('-'*100)

# Fix lat/lon bugaboo
wl_wc = wl_wc.assign_coords({'lon' : wl_wc['longitude'], 'lat' : wl_wc['latitude']})

# Subset and convert to GC units (kg/m2/s)
wl_wc = wl_wc['wetland_CH4_emissions']*1e-6/(24*60*60)
wl_wc = wl_wc.sel(lat=slice(s.lat_min, s.lat_max),
                  lon=slice(s.lon_min, s.lon_max),
                  time=slice(109, 120))

# Remove models 1923 and 2913
wl_wc_s = copy.deepcopy(wl_wc)
wl_wc_s = wl_wc_s.drop_sel(model=[1923, 2913])

# Before averaging, plot the model ensemble
fig, axis = fp.get_figax(rows=4, cols=9, maps=True,
                         lats=wl_wc.lat, lons=wl_wc.lon)
for ax in axis.flatten():
    ax = fp.format_map(ax, wl_wc.lat, wl_wc.lon, **small_map_kwargs)

days_s= [[31, 31, 28], [31, 30, 31], [30, 31, 31], [30, 31, 30]]
seasons = ['DJF', 'MAM', 'JJA', 'SON']
for i, seas in enumerate([[109, 119, 120], [110, 111, 112],
                          [113, 114, 115], [116, 117, 118]]):
    wl_s = wl_wc.sel(time=seas)
    d_s = np.array(days_s[i])
    wl_s = (wl_s*d_s[None, :, None, None]).sum(axis=1)/d_s.sum()
    wl_s *= 1e-3*(60*60*24*365)*(1000*1000) # change units to Mg/km2/yr
    axis[i, 0].text(-0.05, 0.5, seasons[i], rotation='vertical',
                    ha='right', va='center',
                    fontsize=config.LABEL_FONTSIZE*config.SCALE,
                    transform=axis[i, 0].transAxes)
    for j, mod in enumerate(wl_s.model.values):
        if i == 0:
            axis[i, j].text(0.5, 1.05, f'{(wl_s.model.values[j])}', ha='center', va='bottom',
                            fontsize=config.LABEL_FONTSIZE*config.SCALE,
                            transform=axis[i, j].transAxes)
        wl_s_m = wl_s.sel(model=mod)
        c= wl_s_m.plot(ax=axis[i, j], add_colorbar=False, vmin=0, vmax=10)
        fp.add_title(axis[i, j], '')

cax = fp.add_cax(fig, axis, horizontal=True)
cb = fig.colorbar(c, cax=cax, orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title=r'Emissions (Mg km$^2$ a$^{-1}$)', 
                    horizontal=True, y=-2.5)

fp.save_fig(fig, plot_dir, f'wetland_models')

# For both wl_wc and wl_wc_s, take the mean over the models and the
# year (we're going to compare regridded total emissions to HEMCO
# regridded total emissions)
wl_wc = wl_wc.mean(dim='model')
wl_wc = (wl_wc*days[:, None, None]).sum(axis=0)/365

wl_wc_s = wl_wc_s.mean(dim='model')
wl_wc_s = (wl_wc_s*days[:, None, None]).sum(axis=0)/365

# Units are still kg/m2/s

# Calculate grid cell edges and area to calculate the total mass and
# for regridding
delta_lon_wc = 0.5
delta_lat_wc = 0.5
lon_e_wc = np.append(wl_wc.lon - delta_lon_wc/2,
                     wl_wc.lon[-1] + delta_lon_wc/2)/180*np.pi
lat_e_wc = np.append(wl_wc.lat - delta_lat_wc/2,
                     wl_wc.lat[-1] + delta_lat_wc/2)/180*np.pi
area_wc = Re**2*(np.sin(lat_e_wc[1:]) - np.sin(lat_e_wc[:-1]))*delta_lon_wc/180*np.pi

# Calculate total mass over the year
tot_emis_wc = wl_wc*area_wc[:, None]*(60*60*24*365)*1e-9
tot_emis_wc = np.nan_to_num(tot_emis_wc.values, nan=0)
tot_emis_wc = tot_emis_wc.sum()
print(f'WetCHARTs 0.5 x 0.5 emissions: {tot_emis_wc:0.2f} Tg')

# Calculate total mass over the year for the sensitivity test
tot_emis_wc_s = wl_wc_s*area_wc[:, None]*(60*60*24*365)*1e-9
tot_emis_wc_s = np.nan_to_num(tot_emis_wc_s.values, nan=0)
tot_emis_wc_s = tot_emis_wc_s.sum()
print(f'WetCHARTs 0.5 x 0.5 emissions (sensitivity): {tot_emis_wc_s:0.2f} Tg')

## ------------------------------------------------------------------------ ##
## Open and subset HEMCO wetlands file
## ------------------------------------------------------------------------ ##
emis = gc.read_file(f'{base_dir}/prior/total_emissions/HEMCO_diagnostics_correct.2019.nc')

# Get area in m2
area_gc = emis['AREA']
wl_gc = emis['EmisCH4_Wetlands']

# Sum over the year
tot_emis_gc = wl_gc*area_gc*(60*60*24*365)*1e-9
tot_emis_gc = tot_emis_gc.sum().values
print(f'GEOSChem 0.25 x 0.125 emissions: {tot_emis_gc:0.2f} Tg')

## ------------------------------------------------------------------------ ##
## Regrid onto the model grid
## ------------------------------------------------------------------------ ##
# Get the grid to regrid onto
clusters = xr.open_dataset(f'{base_dir}/inversion_data/clusters.nc')['Clusters']

# Calculate grid cell edges and area
delta_lon_gc = s.lon_delta
delta_lat_gc = s.lat_delta
lon_e_gc = np.append(clusters.lon - delta_lon_gc/2,
                     clusters.lon[-1] + delta_lon_gc/2)/180*np.pi
lat_e_gc = np.append(clusters.lat - delta_lat_gc/2,
                     clusters.lat[-1] + delta_lat_gc/2)/180*np.pi

grid_wc = {'lat' : wl_wc.lat, 'lon' : wl_wc.lon,
           'lat_b' : lat_e_wc, 'lon_b' : lon_e_wc}
grid_gc = {'lat' : clusters.lat, 'lon' : clusters.lon,
           'lat_b' : lat_e_gc, 'lon_b' : lon_e_gc}

# Get the regridder
regridder = xe.Regridder(grid_wc, grid_gc, 'conservative')

# Regrid the data (kg/m2/s)
wl_wc_rg = regridder(wl_wc)
wl_wc_s_rg = regridder(wl_wc_s)

# Calculate total mass of emitted methane
tot_emis_wc_rg = wl_wc_rg*area_gc*(60*60*24*365)*1e-9
tot_emis_wc_rg = np.nan_to_num(tot_emis_wc_rg.values, nan=0)
tot_emis_wc_rg = tot_emis_wc_rg.sum()
print(f'WetCHARTs 0.25 x 0.125 emissions: {tot_emis_wc_rg:0.2f} Tg')

# Calculate total mass of emitted methane for the sensitivity ttest
tot_emis_wc_s_rg = wl_wc_s_rg*area_gc*(60*60*24*365)*1e-9
tot_emis_wc_s_rg = np.nan_to_num(tot_emis_wc_s_rg.values, nan=0)
tot_emis_wc_s_rg = tot_emis_wc_s_rg.sum()
print(f'WetCHARTs 0.25 x 0.125 emissions (sensitivity): {tot_emis_wc_s_rg:0.2f} Tg')

# # And one last test: compare the state vector emissions from
# # wetcharts regridded vs GC
wl_gc_v = ip.clusters_2d_to_1d(clusters, wl_gc)
wl_wc_rg_v = ip.clusters_2d_to_1d(clusters, wl_wc_rg)
wl_wc_s_rg_v = ip.clusters_2d_to_1d(clusters, wl_wc_s_rg)
# wl_rg_diff_v = wl_wc_rg_v - wl_gc_v
# print(f'Minimum difference in cluster emission rate: {wl_rg_diff_v.min()}')
# print(f'Maximum difference in cluster emission rate: {wl_rg_diff_v.max()}')
# print(f'Cluster emission rate range: ({wl_gc_v.min()}, {wl_gc_v.max()})')

# And finally, save the sensiivty test out
wl_wc_s_rg.to_netcdf(f'{data_dir}wetlands37.nc')

## ------------------------------------------------------------------------ ##
## Plot
## ------------------------------------------------------------------------ ##
# Plot both sensitivity tests
fig, axis = fp.get_figax(rows=1, cols=3, maps=True,
                         lats=clusters.lat, lons=clusters.lon)
for ax in axis.flatten():
    ax = fp.format_map(ax, clusters.lat, clusters.lon, **small_map_kwargs)
cax = fp.add_cax(fig, axis, horizontal=True)

fig, axis[0], c = ip.plot_state(wl_gc_v*1e-3*(60*60*24*365)*(1000*1000),
                                clusters, title='Standard',
                                fig_kwargs={'figax' : [fig, axis[0]]},
                                cmap=viridis_trans, vmin=0, vmax=5,
                                cbar=False, map_kwargs=small_map_kwargs)
# fig, axis[1], c = ip.plot_state(wl_wc_rg_v*1e-3*(60*60*24*365)*(1000*1000),
#                                 clusters, title='Standard\n(xesmf)',
#                                 fig_kwargs={'figax' : [fig, axis[1]]},
#                                 cmap=viridis_trans, vmin=0, vmax=5,
#                                 cbar=False, map_kwargs=small_map_kwargs)
fig, axis[1], c = ip.plot_state(wl_gc_v*1e-3*(60*60*24*365)*(1000*1000)/4.04,
                                clusters, title='75\% decrease',
                                fig_kwargs={'figax' : [fig, axis[1]]},
                                cmap=viridis_trans, vmin=0, vmax=5,
                                cbar=False, map_kwargs=small_map_kwargs)
fig, axis[2], c = ip.plot_state(wl_wc_s_rg_v*1e-3*(60*60*24*365)*(1000*1000),
                                clusters, title='Ensemble subset',
                                fig_kwargs={'figax' : [fig, axis[2]]},
                                cmap=viridis_trans, vmin=0, vmax=5,
                                cbar=False, map_kwargs=small_map_kwargs)
cb = fig.colorbar(c, cax=cax, ticks=np.arange(0,6, 1), orientation='horizontal')
cb = fp.format_cbar(cb, cbar_title=r'Emissions (Mg km$^2$ a$^{-1}$)',
                    horizontal=True, y=-2.5)

fp.save_fig(fig, plot_dir, f'wetlands_rg')



