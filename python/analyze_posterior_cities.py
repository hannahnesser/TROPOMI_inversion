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
from matplotlib.ticker import MultipleLocator
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import shapefile
from shapely.geometry import Polygon, MultiPolygon
# from shapely.plotting import plot_polygon, plot_line
# from shapely.geometry.base import geom_factory
# from shapely.validation import make_valid
# from shapely.geos import lgeos
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

# Number of cities
nc = 20

# Define file names
f = 'rg2rt_10t_w404_rf0.25_sax0.75_poi80.0'
xa_abs_file = 'xa_abs_w404.nc'
w_file = 'w_w404.csv'
optimize_BC = False

# Define emission categories
emis = {'Wetlands' : 'wetlands', 'Livestock' : 'livestock', 'Coal' : 'coal', 
        'Oil and natural gas' : 'ong', 'Landfills' : 'landfills', 
        'Wastewater' : 'wastewater', 'Other' : 'other'}

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()

# Load prior (Mg/km2/yr)
xa_abs = xr.open_dataarray(f'{data_dir}{xa_abs_file}').values.reshape((-1, 1))
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load weighting matrix
w = pd.read_csv(f'{data_dir}{w_file}')
w['total'] = w.sum(axis=1)
w['net'] = xa_abs

# Convert to Mg/yr since that's important for summing
w *= area

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
print('-'*120)
print(f'We optimize {(dofs >= DOFS_filter).sum():d} grid cells, including {xa_abs[dofs >= DOFS_filter].sum():.2E}/{xa_abs.sum():.2E} = {(xa_abs[dofs >= DOFS_filter].sum()/xa_abs.sum()*100):.2f}% of prior emissions. This\nproduces {dofs[dofs >= DOFS_filter].sum():.2f} ({dofs.sum():.2f}) DOFS with an xhat range of {xhat.min():.2f} to {xhat.max():.2f}. There are {len(xhat[xhat < 0]):d} negative values.')
print('-'*120)

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
## Get overlap between metropolitan statistical areas and posterior
## ------------------------------------------------------------------------ ##
# Open the 2019 city boundaries
city = shapefile.Reader(f'{data_dir}cities/2019_tl_us_cbsa/tl_2019_us_cbsa.shp')

# Create a pandas dataframe for the summary city data
city_summ = pd.DataFrame(columns=['name', 'area', 'prior', 'xhat'] + list(w.columns))

# Create a numpy array for the information content analysis
w_city = pd.DataFrame(columns=np.arange(len(xhat)))

# Iterate through each city
for j, shape in enumerate(city.shapeRecords()):
    if ((shape.record[5] == 'M1') and 
        (shape.record[3].split(', ')[-1] not in ['AK', 'HI', 'PR'])):
        # Add a row to w_city
        w_city.loc[shape.record[3]] = np.zeros(len(xhat))

        # Get edges of the combined statistical area
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]

        # Make a polygon
        c_poly = Polygon(np.column_stack((x, y)))
        if not c_poly.is_valid:
            print(f'Buffering {shape.record[3]}')
            c_poly = c_poly.buffer(0)

        # Get max lat and lon lims
        lat_lims = (np.min(y), np.max(y))
        lon_lims = (np.min(x), np.max(x))

        # Convert that to the GC grid (admittedly using grid cell centers 
        # instead of edges, but that should be consesrvative)
        c_lat_lims = (clusters.lat.values[clusters.lat < lat_lims[0]][-1],
                      clusters.lat.values[clusters.lat > lat_lims[1]][0])
        c_lon_lims = (clusters.lon.values[clusters.lon < lon_lims[0]][-1],
                      clusters.lon.values[clusters.lon > lon_lims[1]][0])
        c_clusters = clusters.sel(lat=slice(*c_lat_lims), 
                                  lon=slice(*c_lon_lims))
        c_cluster_list = c_clusters.values.flatten()
        c_cluster_list = c_cluster_list[c_cluster_list > 0]

        # Initialize tracking arrays
        c_emis_post = pd.DataFrame(0, index=[0], columns=w.columns)
        c_emis_prior = pd.DataFrame(0, index=[0], columns=w.columns)
        c_area = 0

        # Iterate through overlapping grid cells
        for i, gc in enumerate(c_cluster_list):
            # Get center of grid box
            gc_center = c_clusters.where(c_clusters == gc, drop=True)
            gc_center = (gc_center.lon.values[0], gc_center.lat.values[0])
            
            # Get corners
            gc_corners_lon = [gc_center[0] - s.lon_delta/2,
                              gc_center[0] + s.lon_delta/2,
                              gc_center[0] + s.lon_delta/2,
                              gc_center[0] - s.lon_delta/2]
            gc_corners_lat = [gc_center[1] - s.lat_delta/2,
                              gc_center[1] - s.lat_delta/2,
                              gc_center[1] + s.lat_delta/2,
                              gc_center[1] + s.lat_delta/2]

            # Make polygon
            gc_poly = Polygon(np.column_stack((gc_corners_lon, 
                                                  gc_corners_lat)))

            # Calculate overlap
            if gc_poly.intersects(c_poly):
                # Get area of overlap area and GC cell and calculate
                # the fractional contribution of the overlap area
                overlap_area = c_poly.intersection(gc_poly).area
                gc_area = gc_poly.area
                c_fraction = overlap_area/gc_area

                # Calculate the emissions adjustment in the grid cell
                # (All in Mg/yr)
                c_emis_post_i = copy.deepcopy(w*(xhat - 1)).iloc[int(gc) - 1, :]
                c_emis_prior_i = copy.deepcopy(w).iloc[int(gc) - 1, :]

                # Calculate the area end emissions in the city within the 
                # grid cell
                c_area_i = copy.deepcopy(area)[int(gc) - 1] # km2
                c_area_i *= c_fraction
                c_emis_post_i *= c_fraction # Mg/yr
                c_emis_prior_i *= c_fraction # Mg/yr

                # Add to base term
                c_emis_post += c_emis_post_i # Mg/yr
                c_emis_prior += c_emis_prior_i # Mg/yr
                c_area += c_area_i # km2

                # Update w_city
                w_city.iloc[-1, int(gc) - 1] = copy.deepcopy(c_fraction)

        # Having iterated through all the GC grid cells, append the city
        # information to the dataframe
        basic_info = {'name' : shape.record[3], 'area' : c_area[0], 
                      'prior' : c_emis_prior['net'][0]/c_area[0],
                      'xhat' : (c_emis_post['net']/c_emis_prior['net'])[0]}
        c_emis_post = {c_emis_post.columns[i] : (c_emis_post.values[0]/c_area)[i] for i in range(len(c_emis_post.columns))}
        city_summ =  city_summ.append({**basic_info, **c_emis_post},
                                      ignore_index=True)

# Calculate means
xhat_mean_1 = city_summ['xhat'].mean()
xhat_mean_2 = (city_summ['net']*city_summ['area']).sum()/(city_summ['prior']*city_summ['area']).sum()
print(f'Analyzed {city_summ.shape[0]} cities:')
print('  xhat mean        ', 1 + xhat_mean_1)
print('  xhat for cities  ', 1 + xhat_mean_2)

# Try out using w_city
# tmp = (w_city @ (w['total']*(xhat - 1).reshape(-1,)).values)
# print(w_city.values[w_city.values > 0])
# print((w['total'].values.reshape(-1,)*(xhat - 1).reshape(-1,))[w_city.values.reshape(-1,) > 0])
# print(city_summ['total'])
# print(tmp)

# Save out w_city so we can pass it to the inversion function on the
# cluster
w_city.to_csv(f'{data_dir}cities/w_cities.csv', header=True, index=True)

## ------------------------------------------------------------------------ ##
## Get list of metropolitan statistical area population
## ------------------------------------------------------------------------ ##
pop = pd.read_csv(f'{data_dir}cities/cbsa_pop_est.csv', header=0,
                  dtype={'City' : str, '2012' : int, '2018' : int, '2019' : int})
pop = pop.sort_values(by='2019', ascending=False, ignore_index=True)
pop = pop.iloc[2:nc+2, :].reset_index(drop=True)

# Get list of largest cities
largest = pop['City'].values
largest = [l[:-11] for l in largest]

# Subset city_summ
city_summ_sub = city_summ[city_summ['name'].isin(largest)]
city_summ_sub = city_summ_sub.set_index('name')
city_summ_sub = city_summ_sub.loc[largest]

## ------------------------------------------------------------------------ ##
## Plot results
## ------------------------------------------------------------------------ ##
ys = np.arange(1, nc + 1)

fig, ax = fp.get_figax(rows=3, aspect=1, sharex=True)
plt.subplots_adjust(hspace=0.05)

## Largest relative adjustments
# Sort the array
city_summ = city_summ.sort_values(by=['xhat'], 
                                  ascending=False).reset_index(drop=True)

# Get labels
labels = city_summ['name'].values
labels = ['%s (%s)' % (l.split(',')[0].split('-')[0], l.split(', ')[-1]) 
          for l in labels[:nc]]

# Plot stacked bar
cc = [1, 8, 3, 10, 5, 12, 7]
left = np.zeros(nc)
for i, (l, e) in enumerate(emis.items()):
    ax[0].barh(ys, city_summ.loc[:nc - 1, e], left=left,
              color=fp.color(cc[i], lut=2*len(emis)), label=l)
    left += city_summ.loc[:nc - 1, e]
# ax[0].barh(ys, city_summ.loc[:nc-1, 'xhat'] - 1, color=fp.color(3))

# Plot xhat values
ax0 = ax[0].twiny()
ax0.scatter(city_summ.loc[:nc - 1, 'xhat'] + 1, ys, marker='x', color='black',
            s=10, label='Posterior scale factor')
ax0.set_xlim(0.33, 4.33)
# ax0.set_xticklabels(fontsize=config.TICK_FONTSIZE)
ax0 = fp.add_labels(ax0, 'Posterior scale factor', '', 
                    fontsize=config.TICK_FONTSIZE, 
                    labelsize=config.TICK_FONTSIZE, labelpad=10)

# Add labels
ax[0].text(0.95, 0.05, 'Largest\nrelative\ncorrections', 
           fontsize=config.TITLE_FONTSIZE, ha='right', va='bottom',
           transform=ax[0].transAxes)
ax[0].set_yticks(ys)
ax[0].set_yticklabels(labels, ha='right', fontsize=config.TICK_FONTSIZE)
ax[0].invert_yaxis()
# ax[0].set_yticks(np.arange(-1, 3, 1))
# ax[0].yaxis.set_minor_locator(MultipleLocator(0.5))
# ax[0].set_yticklabels(['-100\%', '0\%', '+100\%', '+200\%'])

# # Largest absolute adjustments
# Sort the array
city_summ = city_summ.sort_values(by=['total'], 
                                  ascending=False).reset_index(drop=True)

# Get labels
labels = city_summ['name'].values
labels = ['%s (%s)' % (l.split(',')[0].split('-')[0], l.split(', ')[-1]) 
          for l in labels[:nc]]

# Plot stacked bar
left = np.zeros(nc)
for i, (l, e) in enumerate(emis.items()):
    ax[1].barh(ys, city_summ.loc[:nc - 1, e], left=left,
              color=fp.color(cc[i], lut=2*len(emis)), label=l)
    left += city_summ.loc[:nc - 1, e]
# ax[1].barh(ys, city_summ.loc[:nc-1, 'xhat'] - 1, color=fp.color(3))

# Plot xhat values
ax1 = ax[1].twiny()
ax1.scatter(city_summ.loc[:nc - 1, 'xhat'] + 1, ys, marker='x', color='black',
            s=10)
ax1.set_xlim(0.33, 4.33)
ax1.set_xticklabels('')

# Add labels
ax[1].text(0.95, 0.05, 'Largest\nabsolute\ncorrections', 
           fontsize=config.TITLE_FONTSIZE, ha='right', va='bottom',
           transform=ax[1].transAxes)
# ax[1] = fp.add_title(ax[1], 'Largest absolute\nurban corrections', fontsize=config.TITLE_FONTSIZE)
ax[1].set_yticks(ys)
ax[1].set_yticklabels(labels, ha='right', fontsize=config.TICK_FONTSIZE)
ax[1].invert_yaxis()
# labels = city_summ['name'].values
# labels = ['%s (%s)' % (l.split(',')[0].split('-')[0], l.split(', ')[-1]) for l in labels]
# ax[0] = fp.add_title(ax[0], 'Largest relative\nurban corrections', fontsize=config.TITLE_FONTSIZE)
# ax[0].barh(ys, city_summ.loc[:nc-1, 'xhat'] - 1, color=fp.color(3))
# ax[0].set_yticks(ys)
# ax[0].set_yticklabels(labels, ha='right',
#                       fontsize=config.TICK_FONTSIZE)
# ax[0].set_yticks(np.arange(-1, 3, 1))
# ax[0].yaxis.set_minor_locator(MultipleLocator(0.5))
# ax[0].set_yticklabels(['-100\%', '0\%', '+100\%', '+200\%'])

# Largest cities
labels = city_summ_sub.index.values
labels = ['%s (%s)' % (l.split('-')[0], l.split(', ')[-1]) for l in labels]

# Plot stacked bar
left = np.zeros(nc)
for i, (l, e) in enumerate(emis.items()):
    ax[2].barh(ys, city_summ_sub[e], left=left,
              color=fp.color(cc[i], lut=2*len(emis)), label=l)
    left += city_summ_sub[e]
# ax[2].barh(ys, city_summ_sub.loc[:nc-1, 'xhat'] - 1, color=fp.color(3))

# Plot xhat values
ax2 = ax[2].twiny()
ax2.scatter(city_summ_sub['xhat'] + 1, ys, marker='x', 
            color='black', s=10)
ax2.set_xlim(0.33, 4.33)
ax2.set_xticklabels('')

# Add labels
ax[2].text(0.95, 0.05, 'Corrections\nin largest\nurban areas', 
           fontsize=config.TITLE_FONTSIZE, ha='right', va='bottom',
           transform=ax[2].transAxes)
# ax[2] = fp.add_title(ax[2], 'Corrections in largest\nurban areas', fontsize=config.TITLE_FONTSIZE)
ax[2].set_yticks(ys)
ax[2].set_yticklabels(labels, ha='right', fontsize=config.TICK_FONTSIZE)
ax[2].invert_yaxis()

# Final aesthetics
ax[-1] = fp.add_labels(ax[-1], 
                       r'$\Delta$ Emissions (Mg km$^{-2}$ a$^{-1}$)', '',
                       fontsize=config.TICK_FONTSIZE,
                       labelsize=config.TICK_FONTSIZE,
                       labelpad=10)

delta_mean = (city_summ['total']*city_summ['area']).sum()/city_summ['area'].sum()
for i in range(3):
    ax[i].tick_params(axis='both', labelsize=config.TICK_FONTSIZE)
    ax[i].axvline(0, ls=':', lw=0.5, color='grey')
    # ax[i].axvline(city_summ['total'].mean(), ls='--', lw=1, color='grey',
    #               label='Mean emissions correction')
    ax[i].set_xlim(-7.5, 37.5)
# ax[1].text(nc + 1.35, xhat_mean_1 - 1, 'Mean correction 1',
#            ha='left', va='center', 
#            fontsize=config.TICK_FONTSIZE, color='grey')
# ax[1].text(nc + 1.35, xhat_mean_2 - 1, 'Mean correction 2',
#            ha='left', va='center', 
#            fontsize=config.TICK_FONTSIZE, color='grey')


# Legend for summary plot
m_handles, m_labels = ax0.get_legend_handles_labels()
handles, labels = ax[-1].get_legend_handles_labels()
handles.extend(m_handles)
labels.extend(m_labels)
ax[-1] = fp.add_legend(ax[-1], handles=handles, labels=labels,
                       fontsize=config.TICK_FONTSIZE, 
                       bbox_to_anchor=(1, -0.3), loc='upper right', ncol=3)

fp.save_fig(fig, plot_dir, f'cities_largest_corrections_{f}')

# # # Read shape file
# # for j, shape in enumerate(city.shapeRecords()):
# #     if ((shape.record[5] == 'M1') and 
# #         (shape.record[3].split(', ')[-1] not in ['AK', 'HI', 'PR'])):

# #         # Get edges of the combined statistical area
# #         x = [i[0] for i in shape.shape.points[:]]
# #         y = [i[1] for i in shape.shape.points[:]]

# #         c_poly = Polygon(np.column_stack((x, y)))
# #         if not c_poly.is_valid:
# #             fig, ax = fp.get_figax(cols=2, maps=True, 
# #                                    lats=clusters.lat, lons=clusters.lon)
# #             print(f'Making valid {shape.record[3]}')
# #             ax[0].plot(x, y)
# #             c_poly = c_poly.buffer(0)
# #             # c_poly = make_valid(c_poly)
# #             try:
# #                 ax[1].plot(*c_poly.exterior.xy)
# #             except:
# #                 for c_geom in c_poly.geoms:
# #                     ax[1].plot(*c_geom.exterior.xy)
# #             #         try:
# #             #             ax[1].plot(*c_geom.exterior.xy)
# #             #         except:
# #             #             for line in c_geom.geoms:
# #             #                 ax[1].plot(*line.xy)

# #             fig.suptitle(shape.record[3])
# #             fp.save_fig(fig, plot_dir, f'cities_map_{j}')
# #             plt.close()