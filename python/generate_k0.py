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
# # Local
# base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
# code_dir = f'{base_dir}python/'
# data_dir = f'{base_dir}inversion_data/'
# plot_dir = f'{base_dir}plots/'

# Cannon
base_dir = '/n/seasasfs02/hnesser/TROPOMI_inversion/'
code_dir = '/n/home04/hnesser/TROPOMI_inversion/python/'
data_dir = f'{base_dir}inversion_data/'

# The prior_run can either be a list of files or a single file
# with all of the data for simulation
year = 2019
months = np.arange(1, 12, 1) # excluding December for now
days = np.arange(1, 32, 1)

# Files
emis_file = f'{base_dir}prior/total_emissions/HEMCO_diagnostics.{year}.nc'
obs_file = f'{base_dir}observations/{year}_corrected.pkl'
cluster_file = f'{data_dir}clusters_0.25x0.3125.nc'
k_nstate = f'{data_dir}k0_nstate.pkl'#None

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
print(f'Number of state vector elements : {nstate}')

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

## Calculate emissions in ppbv
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
## Load and process the observations
## ------------------------------------------------------------------------ ##
obs = gc.load_obj(obs_file)[['LON', 'LAT', 'MONTH']]
nobs = int(obs.shape[0] + 1)
print(f'Number of observations : {nobs}')

# Find the indices that correspond to each observation (i.e. the grid
# box in which each observation is found) (Yes, this information should
# be contained in the iGC and jGC columns in obs_file, but stupidly
# I don't have that information for the cluster files)

# First, find the cluster number of the grid box of the obs
lat_idx = gc.nearest_loc(obs['LAT'].values, emis.lat.values)
lon_idx = gc.nearest_loc(obs['LON'].values, emis.lon.values)
obs['CLUSTER'] = emis['Clusters'].values[lat_idx, lon_idx]

# Format
obs[['MONTH', 'CLUSTER']] = obs[['MONTH', 'CLUSTER']].astype(int)

## ------------------------------------------------------------------------ ##
## Build the Jacobian
## ------------------------------------------------------------------------ ##
# Create a row for each cluster number
def get_zone_emissions(emissions, ncells, emis_fraction):
    '''
    This function calculates the emissions allocated to each grid cell in
    successive rings (zones) from the source grid cell. Zone 0 corresponds
    to the source grid cell, zone 1 to the 8 grid cells surrounding the
    source grid cell, zone 2 to the 16 grid cells surrounding zone 1, and
    so on.
    '''
    # if zone == 0:
    #     ncells = 1
    # else:
    return (emissions*emis_fraction/ncells).values.reshape(-1,)

def get_latlon_bounds(data, source_loc, zone):
    # Get lower boundary (for lat or lon)
    try:
        mini = np.sort(data.where(data <= source_loc, drop=True).values)[::-1]
        mini = mini[zone]
    except:
        mini = source_loc

    # Get upper boundary (for lat or lon)
    try:
        maxi = np.sort(data.where(data >= source_loc, drop=True).values)
        maxi = maxi[zone]
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

def get_zone_indices(data, grid_cell_index, zone):
    '''
    ...
    In theory this could be vectorized but it's not worth it for a
    one time loop.
    '''
    # Get the latitude and longitude of the source grid box
    source = data.where(data['Clusters'] == grid_cell_index, drop=True)
    source_lat = source.lat.values[0]
    source_lon = source.lon.values[0]

    # Get the latitude and longitude boundaries
    lat_min, lat_max = get_latlon_bounds(data.lat, source_lat, zone)
    lon_min, lon_max = get_latlon_bounds(data.lon, source_lon, zone)

    # Get the state vector indices corresponding to the upper and
    # lower latitude values
    lat_cond = get_latlon_edges(data.lat, source_lat, lat_min, lat_max)
    tmp1 = data.where(lat_cond & (data.lon >= lon_min) & (data.lon <= lon_max),
                      drop=True)['Clusters'].values

    # Get the state vector indices corresponding to the upper and lower
    # longitude values
    lon_cond = get_latlon_edges(data.lon, source_lon, lon_min, lon_max)
    tmp2 = data.where(lon_cond & (data.lat >= lat_min) & (data.lat <= lat_max),
                      drop=True)['Clusters'].values

    # Combine the two lists and remove duplicates and zeros (corresponding
    # to grid cells that are not included in the state vector)
    idx = np.concatenate((tmp1.reshape(-1,), tmp2.reshape(-1,)))
    idx = idx[idx > 0]
    idx = np.unique(idx).astype(int)

    return idx

if k_nstate is None:
    ## First, obtain a column for every grid box. We will then duplicate
    ## rows where there are multiple observations for a given grid box and time.

    # Initialize the nstate x nstate x months Jacobian
    k_nstate = np.zeros((nstate, nstate, len(months)), dtype=np.float32)

    # Set the fractional mass decrease
    f = np.array([10, 6, 4, 3, 2.5])
    f /= f.sum()

    # Iterate through the columns
    for i in range(1, nstate+1):
        # Get the emissions for that grid cell
        emis_i = emis.where(emis['Clusters'] == i, drop=True)['EmisCH4_ppb']

        # Iterate through the zones
        for z in range(len(f)):
            # Get state vector indices corresponding to each grid box in the
            # given zone
            idx_z = get_zone_indices(emis, grid_cell_index=i, zone=z)

            # Calculate the emissions to be allocated to each of the grid cells
            # in the zone if there are grid cells in the zone
            if len(idx_z) > 0:
                emis_z = get_zone_emissions(emis_i, ncells=len(idx_z),
                                            emis_fraction=f[z])

            # Place those emissions in the nstate x nstate x months Jacobian
            k_nstate[i-1, idx_z-1, :] = emis_z

        # Keep track of progress
        if i % 1000 == 0:
            print(f'{i:-6d}/{nstate}', '-'*int(20*i/nstate))

    # Save
    gc.save_obj(k_nstate, join(data_dir, f'k0_nstate.pkl'))

else:
    k_nstate = gc.load_obj(join(data_dir, f'k0_nstate.pkl'))

# Delete superfluous memory
del(emis)
del(clusters)
obs = obs[['CLUSTER', 'MONTH']]

# Now create a Jacobian for each chunk of 30,000 observations (the full
# Jacobian is something like 3e6 x 2e4 and requires 300 GB even with float32)
i = 0
count = 0
nchunk = int(3e4)
state_vector_elements_remain = True
while state_vector_elements_remain:
    print(count)
    o = obs[i:i+nchunk]
    k_m = k_nstate[o['CLUSTER'], :, o['MONTH']]
    gc.save_obj(k_m, join(data_dir, f'k0_{count:03d}.pkl'))
    i += nchunk
    count += 1
    if i > nobs:
        state_vector_elements_remain = False

# EASY CHECK: USE THIS SCRIPT TO BUILD THE JACOBIAN FOR MY TEST CASE

