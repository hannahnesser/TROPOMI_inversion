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
wl = xr.open_dataset(f'{data_dir}{wetland_file}', decode_times=False)

# Fix lat/lon bugaboo
wl = wl.assign_coords({'lon' : wl['longitude'], 'lat' : wl['latitude']})
wl = wl['wetland_CH4_emissions']

# Subset
wl = wl.sel(lat=slice(s.lat_min, s.lat_max), lon=slice(s.lon_min, s.lon_max),
            time=slice(109, 120))

## ------------------------------------------------------------------------ ##
## Plot
## ------------------------------------------------------------------------ ##
fig, axis = fp.get_figax(rows=4, cols=9, maps=True, lats=wl.lat, lons=wl.lon)
for ax in axis.flatten():
    ax = fp.format_map(ax, wl.lat, wl.lon, **small_map_kwargs)


seasons = ['DJF', 'MAM', 'JJA', 'SON']
for i, seas in enumerate([[109, 119, 120], [110, 111, 112],
                          [113, 114, 115], [116, 117, 118]]):
    wl_s = wl.sel(time=seas)
    wl_s = wl_s.mean(dim='time')
    axis[i, 0].text(-0.05, 0.5, seasons[i], rotation='vertical',
                    ha='right', va='center',
                    fontsize=config.LABEL_FONTSIZE*config.SCALE,
                    transform=axis[i, 0].transAxes)
    for j, mod in enumerate(wl_s.model.values):
        if i == 0:
            axis[i, j].text(0.5, 1.05, f'{(j+1)}', ha='center', va='bottom',
                            fontsize=config.LABEL_FONTSIZE*config.SCALE,
                            transform=axis[i, j].transAxes)
        wl_s_m = wl_s.sel(model=mod)
        c= wl_s_m.plot(ax=axis[i, j], add_colorbar=False, vmin=0, vmax=50)
        fp.add_title(axis[i, j], '')

cax = fp.add_cax(fig, axis)
c = fig.colorbar(c, cax=cax)
c = fp.format_cbar(c, '')

fp.save_fig(fig, plot_dir, f'wetlands')
