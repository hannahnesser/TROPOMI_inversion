'''
This python script generates the initial estimate of the Jacobian matrix

Inputs:
    prior_emis      This contains the emissions of the prior run simulation.
                    It is a list of monthly output HEMCO diagnostics files
                    to account for monthly variations in
'''

from os.path import join
from os import listdir
import sys
import calendar as cal

import xarray as xr
import numpy as np
from numpy.polynomial import polynomial as p
import pandas as pd

import matplotlib.pyplot as plt

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'

# The prior_run can either be a list of files or a single file
# with all of the data for simulation
year = 2019
months = np.arange(1, 12, 1) # excluding December for now
days = np.arange(1, 32, 1)

# Files
emis_file = f'{base_dir}prior/total_emissions/HEMCO_diagnostics.{year}.nc'
cluster_file = f'{data_dir}clusters_0.25x0.3125.nc'

# Which analyses do you wish to perform?

# Information on the grid
lat_bins = np.arange(10, 65, 5)
lat_min = 9.75
lat_max = 60
lat_delta = 0.25
lon_min = -130
lon_max = -60
lon_delta = 0.3125
buffers = [3, 3, 3, 3]

## ------------------------------------------------------------------------ ##
## Import custom packages
## ------------------------------------------------------------------------ ##
# Custom packages
sys.path.append(code_dir)
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
import gcpy as gc
import troppy as tp
import format_plots as fp

## ------------------------------------------------------------------------ ##
## Define the inversion grid
## ------------------------------------------------------------------------ ##
# Get information on the lats and lons (edges of the domain) (this means
# removing the buffer grid cells)
lat_e, lon_e = gc.adjust_grid_bounds(lat_min, lat_max, lat_delta,
                                     lon_min, lon_max, lon_delta, buffers)

## ------------------------------------------------------------------------ ##
## Load the clusters
## ------------------------------------------------------------------------ ##
clusters = xr.open_dataset(cluster_file)
nstate = int(clusters['Clusters'].max().values)

## ------------------------------------------------------------------------ ##
## Load and process the emissions
## ------------------------------------------------------------------------ ##
emis = gc.load_files(emis_file)

# Remove emissions from buffer grid cells
emis = gc.subset_data_latlon(emis, *lat_e, *lon_e)

# Separate out total methane emissions by removing soil absorption
# to ask Daniel: should we only consider emissions (i.e. not the sink?)
# print(emis)
emis['EmisCH4_Total'] -= emis['EmisCH4_SoilAbsorb']

# Convert units from kg/m2/s to mol/s
MCH4 = 0.016 # kg/mol
emis['EmisCH4_Total'] = emis['EmisCH4_Total']*emis['AREA']/MCH4

# Subset
emis = emis[['EmisCH4_Total', 'AREA']]

# Join in clusters]
emis['Clusters'] = clusters['Clusters'].squeeze()

## ------------------------------------------------------------------------ ##
## Calculate emissions in ppbv
## ------------------------------------------------------------------------ ##
# Make assumptions about atmospheric conditions
Mair = 0.02897 # Molar mass of air, kg/mol
P = 1e5 # Surface pressure, Pa
g = 9.8 # Gravitational acceleration, m/s^2
U = 5*(1000/3600) # Average wind speed, m/s
W = emis['AREA']**0.5 # Wind distance, m

# Calculate ppbv
emis['EmisCH4_ppb'] = 1e9*Mair*emis['EmisCH4_Total']*g/(U*W*P)

# Subset
emis = emis[['Clusters', 'EmisCH4_ppb']]

## ------------------------------------------------------------------------ ##
## Build the Jacobian
## ------------------------------------------------------------------------ ##
# Create a row for each cluster number
def calculate_zone_emissions(emissions, zone, fraction_to_zone):
    '''
    This function calculates the emissions allocated to each grid cell in
    successive rings (zones) from the source grid cell. Zone 0 corresponds
    to the source grid cell, zone 1 to the 8 grid cells surrounding the
    source grid cell, zone 2 to the 16 grid cells surrounding zone 1, and
    so on.
    '''
    if zone == 0:
        ncells = 1
    else:
        ncells = 8*zone

    return emissions*fraction_to_zone/ncells

def get_latlon_bounds(data, source_loc, zone):
    # Get lower boundary (for lat or lon)
    try:
        mini = data.where(data < source_loc, drop=True).values[-zone]
    except:
        mini = source_loc

    # Get upper boundary (for lat or lon)
    try:
        maxi = data.where(data >= source_loc, drop=True).values[zone]
    except:
        maxi = source_loc

    return mini, maxi

def get_latlon_edges(data, source, mini, maxi):
    if mini == source:
        cond = (data == maxi)
    elif maxi == source:
        cond = (data == mini)
    else:
        cond = (data == mini) | (data == maxi)
    return cond

def get_zone_nstate_indices(data, nstate_idx, zone):
    '''
    ...
    '''
    # Get the latitude and longitude of the source grid box
    source = data.where(data['Clusters'] == nstate_idx, drop=True)
    source_lat = source.lat.values[0]
    source_lon = source.lon.values[0]

    # Get the latitude and longitude boundaries
    lat_min, lat_max = get_latlon_bounds(data.lat, source_lat, zone)
    lon_min, lon_max = get_latlon_bounds(data.lon, source_lon, zone)

    # Get the state vector indices corresponding to the upper and
    # lower latitude values
    lat_cond = get_latlon_edges(data.lat, source_lat, lat_min, lat_max)
    tmp = data.where(lat_cond & (data.lon >= lon_min) & (data.lon <= lon_max),
                     drop=True)#['Clusters'].values

    # Get the state vector indices corresponding to the upper and lower
    # longitude values
    lon_cond = get_latlon_edges(data.lon, source_lon, lon_min, lon_max)
    tmp2 = data.where(lon_cond & (data.lat > lat_min) & (data.lat < lat_max),
                      drop=True)
    # if
    print(lat_min, source_lat, lat_max)
    print(lon_min, source_lon, lon_max)
    print(tmp, tmp2)


for i in range(10): # range(nstate):
    # we should construct the jacobian for each month. Probably easier than
    # dealing with the memory constraints
    # emis[]
    ...

get_zone_nstate_indices(emis, 1, 1)

# fig1, ax = plt.subplots()
# temp.plot(vmin=0, vmax=1e-8, ax=ax, cmap='viridis')

# fig2, ax = plt.subplots()
# # emis = emis.mean(dim='time')
# emis['EmisCH4_Total'].plot(vmin=0, vmax=1e-8, ax=ax, cmap='viridis')
# plt.show()

# print((temp - emis['EmisCH4_Total']).max())
# # print((temp - emis['EmisCH4_Total'] + emis['EmisCH4_SoilAbsorb']).max())
# print(temp == emis['EmisCH4_Total'])

# Adjust units to Mg/km2/yr
# emis *= 0.001*60*60*24*365*1000*1000

