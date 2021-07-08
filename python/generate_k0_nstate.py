'''
This python script generates the initial estimate of the Jacobian matrix
which has dimension n x n x 12. It is the basis for the generate_k0.py
script.

Inputs:
    prior_emis      This contains the emissions of the prior run simulation.
                    It is a list of monthly output HEMCO diagnostics files
                    to account for monthly variations in
'''

from os.path import join
import sys

import xarray as xr
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# Cannon
base_dir = '/n/seasasfs02/hnesser/TROPOMI_inversion/'
code_dir = '/n/home04/hnesser/TROPOMI_inversion/python/'
data_dir = f'{base_dir}inversion_data/'
output_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion'

# Import custom packages
sys.path.append(code_dir)
import gcpy as gc
import inversion_settings as settings

# Files
emis_file = f'{base_dir}prior/total_emissions/HEMCO_diagnostics.{settings.year}.nc'
cluster_file = f'{data_dir}clusters.nc'

# Fraction of emissions to allocate to each ring
fractions = np.array([10, 6, 4, 3, 2.5, 2.25, 2.125])
fractions /= fractions.sum()

## ------------------------------------------------------------------------ ##
## Load the clusters
## ------------------------------------------------------------------------ ##
clusters = xr.open_dataset(cluster_file)
nstate = int(clusters['Clusters'].max().values)
print(f'Number of state vector elements : {nstate}')

## ------------------------------------------------------------------------ ##
## Load and process the emissions
## ------------------------------------------------------------------------ ##
emis = gc.read_netcdf_file(emis_file)

# Remove emissions from buffer grid cells
emis = gc.subset_data_latlon(emis, *settings.lats, *settings.lons)

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
## Define the functions needed to build the n x n x 12 Jacobian
## ------------------------------------------------------------------------ ##
def get_zone_conc(emissions, ncells, emis_fraction):
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

## ------------------------------------------------------------------------ ##
## Build the n x n x 12 Jacobian
## ------------------------------------------------------------------------ ##
## First, obtain a column for every grid box. We will then duplicate
## rows where there are multiple observations for a given grid box and time.

# Initialize the nstate x nstate x months Jacobian
k_nstate = np.zeros((nstate, nstate, len(months)), dtype=np.float32)

# Iterate through the state vector elements
for i in range(1, nstate+1):
    # Get the emissions for that grid cell
    emis_i = emis.where(emis['Clusters'] == i, drop=True)['EmisCH4_ppb']

    # Iterate through the zones
    for z, frac in enumerate(fractions):
        # Get state vector indices corresponding to each grid box in the
        # given zone
        idx_z = get_zone_indices(emis, grid_cell_index=i, zone=z)

        # Calculate the emissions (as observation) to be allocated to each
        # of the grid cells in the zone if there are grid cells in the zone
        if len(idx_z) > 0:
            conc_z = get_zone_conc(emis_i, ncells=len(idx_z),
                                   emis_fraction=frac)

        # Place those concentrations in the nstate x nstate x months
        # Jacobian
        k_nstate[idx_z-1, i-1, :] = conc_z

    # Keep track of progress
    if i % 1000 == 0:
        print(f'{i:-6d}/{nstate}', '-'*int(20*i/nstate))

# Save
k_nstate.to_netcdf(join(data_dir, 'k0_nstate.nc'),
                   dims=('nobs', 'nstate', 'month'))
