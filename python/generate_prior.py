'''
This script generates netcdfs of the absolute and relative prior emissions and error variances for use in an analytical inversion.

   **Inputs**

   | ----------------- | -------------------------------------------------- |
   | Input             | Description                                        |
   | ----------------- | -------------------------------------------------- |
   | emis_file         | A file or files containing information on methane  |
   |                   | emissions from the prior run. This is typically    |
   |                   | given by HEMCO_diagnostics. The input here can be  |
   |                   | either a list of monthly files or a single file    |
   |                   | with an annual average.                            |
   | ----------------- | -------------------------------------------------- |
   | clusters          | The cluster file generated by generate_clusters.py |
   |                   | that maps a unique key for every grid cell         |
   |                   | contained in the state vector to the latitude-     |
   |                   |longitude grid used in the forward model.           |
   | ----------------- | -------------------------------------------------- |
   | rel_err           | The relative error (standard deviation) value to   |
   |                   | be used in the relative prior error covariance     |
   |                   | matrix. The default is 0.5.                        |
   | ----------------- | -------------------------------------------------- |

   **Outputs**

   | ----------------- | -------------------------------------------------- |
   | Output            | Description                                        |
   | ----------------- | -------------------------------------------------- |
   | xa.nc             | A netcdf containing the relative prior (all ones)  |
   |                   | xa for use in the inversion.                       |
   | ----------------- | -------------------------------------------------- |
   | sa.nc             | A netcdf containing the relative prior error (all  |
   |                   | given by rel_err) for use in the inversion.        |
   | ----------------- | -------------------------------------------------- |
   | xa_abs.nc         | A netcdf containing the absolute prior (all ones)  |
   |                   | xa for use in the inversion.                       |
   | ----------------- | -------------------------------------------------- |
   | sa_abs.nc         | A netcdf containing the absolute prior error (all  |
   |                   | given by rel_err) for use in the inversion.        |
   | ----------------- | -------------------------------------------------- |
'''

from os.path import join
import sys

import math
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

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
import inversion_settings as settings

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python'
data_dir = base_dir + 'inversion_data'
plot_dir = base_dir + 'plots'

# The emissions can either be a list of files or a single file
# with an annual average
emis_file = [f'{base_dir}/prior/total_emissions/\
HEMCO_diagnostics.{settings.year:04d}{mm:02d}010000.nc'
             for mm in settings.months]
# emis_file = f'{base_dir}/prior/total_emissions/HEMCO_diagnostics.{settings.year}.nc'
clusters = f'{data_dir}/clusters.nc'
clusters = xr.open_dataarray(clusters)
nstate = int(clusters.max().values)
print(nstate)

# Set relative prior error covariance value
rel_err = 1

## ------------------------------------------------------------------------ ##
## Figure out what the relative prior errors should be set at
## ------------------------------------------------------------------------ ##
def alpha(a0, ka, an, L, L0=0.1):
    return a0*np.exp(-ka*(L-L0)) + an

def beta(b0, kb, L, L0=0.1):
    return b0*np.exp(-kb*(L-L0))

livestock = [0.89, 3.1, 0.12, 0, 0]
nat_gas = [0.28, 4.2, 0.25, 0.09, 3.9]
landfills = [0, 0, 0.51, 0.08, 2.0]
wastewater = [0.78, 1.4, 0.21, 0.06, 6.9]
petroleum = [0, 0, 0.87, 0.04, 197]
sources = {'livestock' : livestock, 'nat_gas' : nat_gas,
           'landfills' : landfills, 'wastewater' : wastewater,
           'petroleum' : petroleum}

for s, coefs in sources.items():
    a = alpha(coefs[0], coefs[1], coefs[2], 0.25)
    b = beta(coefs[3], coefs[4], 0.25)
    a2 = alpha(coefs[0], coefs[1], coefs[2], 0.3125)
    b2 = beta(coefs[3], coefs[4], 0.3125)

    print(f'0.25   {s:<20}{a:.2f}  {b:.2f}')
    print(f'0.3125 {s:<20}{a2:.2f}  {b2:.2f}')

## -------------------------------------------------------------------------##
## Load raw emissions data
## -------------------------------------------------------------------------##
emis = gc.read_file(*emis_file)

# Remove emissions from buffer grid cells
emis = gc.subset_data_latlon(emis, *settings.lats, *settings.lons)

if 'time' in emis.dims:
    # Average over time
    emis = emis.mean(dim='time')

    # Save summary file
    name = 'HEMCO_diagnostics.2019.nc'
    emis.to_netcdf(f'{base_dir}/prior/total_emissions/{name}')

# Save out area as km2
area = emis['AREA']/(1000*1000)
area = ip.clusters_2d_to_1d(clusters, area)
area = xr.DataArray(area, dims=('nstate'))
area.to_netcdf(join(data_dir, 'area.nc'))

# Print a summary table
summ = emis[[var for var in emis.keys() if var[:4] == 'Emis']]*emis['AREA']
summ *= 1e-9*(60*60*24*365) # Adjust units to Tg/yr
# summ = xr.where(clusters > 0)
summ = summ.sum(dim=['lat', 'lon'])
tally = 0
for k in summ.keys():
    if k != 'EmisCH4_Total':
        tally += summ[k].values
    print(f'{k:>20} {summ[k].values:.2f}')
print('-'*50)
print(f'               Total {tally:.2f}')
print('-'*50)

# Adjust units to Mg/km2/yr
emis *= 1e-3*(60*60*24*365)*(1000*1000)

# Isolate soil absorption
soil_abs = emis['EmisCH4_SoilAbsorb']
soil_abs = ip.clusters_2d_to_1d(clusters, soil_abs)
soil_abs = xr.DataArray(soil_abs, dims=('nstate'))
soil_abs.to_netcdf(join(data_dir, 'soil_abs.nc'))

# Calculate total emissions
tot_emis = emis['EmisCH4_Total'] - emis['EmisCH4_SoilAbsorb']
tot_emis = ip.clusters_2d_to_1d(clusters, tot_emis)
tot_emis = xr.DataArray(tot_emis, dims=('nstate'))
tot_emis.to_netcdf(join(data_dir, 'xa_abs.nc'))

## -------------------------------------------------------------------------##
## Group by sector
## -------------------------------------------------------------------------##
emissions = {'wetlands' : 'Wetlands',
             'livestock' : 'Livestock',
             'coal' : 'Coal',
             'oil' : 'Oil',
             'gas' : 'Gas',
             'landfills' : 'Landfills',
             'wastewater' : 'Wastewater',
             'other' : ['Termites', 'Seeps', 'BiomassBurn', 'Lakes', 'Rice', 'OtherAnth']}

w = pd.DataFrame(columns=emissions.keys())
for label, categories in emissions.items():
    # Get emissions
    if type(categories) == str:
        e = emis['EmisCH4_%s' % categories].squeeze()
    elif type(categories) == list:
        e = sum(emis['EmisCH4_%s' % em].squeeze()
                for em in categories)

    # Flatten
    e = ip.clusters_2d_to_1d(clusters, e)

    # Saveouut
    # e = xr.DataArray(e, dims=('nstate'))
    # e.to_netcdf(join(data_dir, f'xa_{label}.nc'))
    w[label] = e
    # print(label)
    # print(e)

w.to_csv(join(data_dir, f'w.csv'), index=False)

## -------------------------------------------------------------------------##
## Plot
## -------------------------------------------------------------------------##
if plot_dir is not None:
    emissions = ['Wetlands', 'Livestock',
                ['Coal', 'Oil', 'Gas'],
                ['Wastewater', 'Landfills'],
                ['Termites', 'Seeps', 'BiomassBurn', 'Lakes'],
                ['Rice', 'OtherAnth']]
    titles = ['Wetlands', 'Livestock', 'Coal, Oil, and\nNatural Gas',
              'Wastewater\nand Landfills', 'Other Biogenic\nSources',
              'Other Anthropogenic\nSources']

    # Set colormap
    colormap = fp.cmap_trans('viridis')

    ncategory = len(emissions)
    fig, ax = fp.get_figax(rows=2, cols=math.ceil(ncategory/2),
                           maps=True, lats=emis.lat, lons=emis.lon)
    plt.subplots_adjust(hspace=0.5)
    cax = fp.add_cax(fig, ax, cbar_pad_inches=0.5)
    for i, axis in enumerate(ax.flatten()):
        axis = fp.format_map(axis, lats=emis.lat, lons=emis.lon)
        if type(emissions[i]) == str:
            e = emis['EmisCH4_%s' % emissions[i]].squeeze()
        elif type(emissions[i]) == list:
            e = sum(emis['EmisCH4_%s' % em].squeeze()
                    for em in emissions[i])

        c = e.plot(ax=axis, cmap=colormap, vmin=0, vmax=5,
                   add_colorbar=False)
        cb = fig.colorbar(c, cax=cax, ticks=np.arange(0, 6, 1))
        cb = fp.format_cbar(cb, cbar_title=r'Emissions (Mg km$^2$ a$^{-1}$)')
        axis = fp.add_title(axis, titles[i])

    fp.save_fig(fig, plot_dir, 'prior_emissions_2019')

# Select total emissions
emis = emis['EmisCH4_Total']

print('The minimum positive emission is',
      np.abs(emis.where(emis > 0).min()).values)

## -------------------------------------------------------------------------##
## Generate prior and prior error covariance
## -------------------------------------------------------------------------##
# Relative prior
xa = np.ones(nstate)
xa = xr.DataArray(xa, dims=('nstate'))
xa.to_netcdf(join(data_dir, 'xa.nc'))

# Relative errors
sa = rel_err**2*np.ones(nstate)
sa = xr.DataArray(sa, dims=('nstate'))
sa.to_netcdf(join(data_dir, 'sa.nc'))

# # Absolute prior
# xa_abs = ip.clusters_2d_to_1d(clusters, emis)
# xa_abs = xr.DataArray(xa_abs, dims=('nstate'))
# xa_abs.to_netcdf(join(data_dir, 'xa_abs.nc'))

# Absolute errors
sa_abs = sa*tot_emis**2
sa_abs.to_netcdf(join(data_dir, 'sa_abs.nc'))

# Varying errors
sa_var = 1*tot_emis.mean()/tot_emis
sa_var[sa_var < 1] = 1
sa_var[sa_var > 5] = 5
sa_var = sa_var**2
sa_var.to_netcdf(join(data_dir, 'sa_var.nc'))

# s_a_vec = 0.5*x_a.mean()/x_a
# s_a_vec[s_a_vec < 0.5] = 0.5
