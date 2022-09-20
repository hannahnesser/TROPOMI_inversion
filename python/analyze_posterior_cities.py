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
## Set user preferences
## ------------------------------------------------------------------------ ##
# DOFS_filter
DOFS_filter = 0.05

# Define file names
f = 'rg2rt_10t_w404_rf0.25_sax0.75_poi80.0'
xa_abs_file = 'xa_abs_w404.nc'
w_file = 'w_w404.csv'
optimize_BC = False

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc')

# Load prior (Mg/km2/yr)
xa_abs = xr.open_dataarray(f'{data_dir}{xa_abs_file}').values.reshape((-1, 1))
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load weighting matrix
w = pd.read_csv(f'{data_dir}{w_file}')
w['total'] = w.sum(axis=1)

# Load posterior and DOFS
dofs = np.load(f'{data_dir}posterior/dofs2_{f}.npy').reshape((-1, 1))
xhat = np.load(f'{data_dir}posterior/xhat_fr2_{f}.npy').reshape((-1, 1))

# BC alteration
if optimize_BC:
    print('-'*30)
    print('Boundary condition optimization')
    print(' N E S W')
    print('xhat : ', xhat[-4:])
    print('dofs : ', dofs[-4:])
    print('-'*30)
    xhat = xhat[:-4]
    dofs = dofs[:-4]

# Print information
print('-'*30)
print(f'We optimize {(dofs >= DOFS_filter).sum():d} grid cells, including {xa_abs[dofs >= DOFS_filter].sum():.2E}/{xa_abs.sum():.2E} = {(xa_abs[dofs >= DOFS_filter].sum()/xa_abs.sum()*100):.2f}% of prior emissions. This\nproduces {dofs[dofs >= DOFS_filter].sum():.2f} ({dofs.sum():.2f}) DOFS with an xhat range of {xhat.min():.2f} to {xhat.max():.2f}. There are {len(xhat[xhat < 0]):d} negative values.')
print('-'*30)

# Filter on DOFS filter
xhat[dofs < DOFS_filter] = 1
dofs[dofs < DOFS_filter] = 0

# Calculate xhat abs
xhat_abs = (xhat*xa_abs)

# Get county outlines for high resolution results
reader = shpreader.Reader(f'{data_dir}counties/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

## ------------------------------------------------------------------------ ##
## Cities analysis
## ------------------------------------------------------------------------ ##
cities = pd.read_csv(f'{data_dir}/uscities.csv')
ncities = 25
# print(cities.columns)

# Order by population and select the top 100
cities = cities.sort_values(by='population', ascending=False).iloc[:ncities, :]
cities = cities[['city', 'state_id', 'state_name', 'lat', 'lng',
                 'population', 'density']]
cities = cities.rename(columns={'lat' : 'lat_hr', 'lng' : 'lon_hr'})
cities = cities.reset_index(drop=True)

# Add in lat centers/lon centers
lats, lons = gc.create_gc_grid(*s.lats, s.lat_delta, *s.lons, s.lon_delta,
                               centers=False, return_xarray=False)
cities['lat'] = lats[gc.nearest_loc(cities['lat_hr'].values, lats)]
cities['lon'] = lons[gc.nearest_loc(cities['lon_hr'].values, lons)]
cities['area'] = cities['population']/cities['density']

# Append posterior
xa_f = ip.match_data_to_clusters(xa_abs, clusters).rename('xa_abs')
xa_f = xa_f.to_dataframe().reset_index()
xhat_f = ip.match_data_to_clusters(xhat, clusters).rename('xhat')
xhat_f = xhat_f.to_dataframe().reset_index()

# Join
cities = cities.merge(xa_f, on=['lat', 'lon'], how='left')
cities = cities.merge(xhat_f, on=['lat', 'lon'], how='left')

# Remove areaas with 1 correction (no information content)
cities = cities[cities['xhat'] != 1]
print(cities[cities['xhat'] == 0])
cities = cities[cities['xhat'] != 0] # WHAT CITY IS THIS
city_mean = (cities['xhat'] - 1).mean()

# Plot
fig, ax = fp.get_figax(rows=1, cols=3, aspect=1.5, sharey=True)
# ax[0] = fp.add_title(ax[0], 'Population')
# ax[1] = fp.add_title(ax[1], 'Density')
# ax[2] = fp.add_title(ax[2], 'Area')
quantities = ['population', 'density', 'area']
for i, q in enumerate(quantities):
    ax[i] = fp.add_title(ax[i], q.capitalize())
    ax[i].scatter(cities[q], (cities['xhat'] - 1), s=1, color=fp.color(3))
    ax[i].axhline(city_mean, color='grey', ls='--', lw=0.5)
    ax[i].set_ylim(-1, 2.5)
    ax[i].set_xscale('log')
    # for j, city in cities.iterrows():
    #     ax[i].scatter(city[q], (city['xhat'] - 1), s=1)
    #     ax[i].annotate(city['city'], (city[q], (city['xhat'] - 1)),
    #                    textcoords='offset points', xytext=(0, 2),
    #                    ha='center', fontsize=7)
    # ax.set_xscale('log')
ax[0].set_ylabel('Posterior\ncorrection')
ax[-1].text(cities['area'].max()*1.3, city_mean, f'{(city_mean):.2f}', 
            va='center')
fp.save_fig(fig, plot_dir, f'cities_test_{f}')


# print(cities.groupby(['lat_center', 'lon_center']).count().shape)
# print(cities)
# for c in range(100):