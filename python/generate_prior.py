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
from copy import deepcopy as dc

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
import inversion_settings as s

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python'
data_dir = base_dir + 'inversion_data'
plot_dir = base_dir + 'plots'

# The emissions can either be a list of files or a single file
# with an annual average
emis_file = [f'{data_dir}/prior/total_emissions/\
HEMCO_diagnostics.{s.year:04d}{mm:02d}010000.nc'
             for mm in s.months]
# emis_file = f'{data_dir}/prior/total_emissions/HEMCO_diagnostics.{s.year}.nc'
emis_hr_file = [f'{data_dir}/prior/total_emissions/\
HEMCO_sectoral_diagnostics.{s.year:04d}{mm:02d}010000.nc'
             for mm in s.months]
clusters = f'{data_dir}/clusters.nc'
clusters = xr.open_dataarray(clusters)
nstate = int(clusters.max().values)
print(f'n = {nstate}')

# Set relative prior error covariance value
rel_err = 1

# Length of months
days = xr.DataArray([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                    dims='time')

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

print('-'*75)
print('RESOLUTION DEPENDENT ERRORS')
print('-'*75)
print('RES    SECTOR              ALPHA BETA')
for ss, coefs in sources.items():
    a = alpha(coefs[0], coefs[1], coefs[2], 0.25)
    b = beta(coefs[3], coefs[4], 0.25)
    a2 = alpha(coefs[0], coefs[1], coefs[2], 0.3125)
    b2 = beta(coefs[3], coefs[4], 0.3125)

    print(f'0.25   {ss:<20}{a:.2f}  {b:.2f}')
    print(f'0.3125 {ss:<20}{a2:.2f}  {b2:.2f}')
print('-'*75)
print('\n')

## ------------------------------------------------------------------------ ##
## Define EPA errors 
## ------------------------------------------------------------------------ ##
EPA_err_min = pd.Series({'coal' : 0.06, 'gas' : 0.1, 'oil' : 0.12, 
                         'landfills' : 0.19, 'wastewater' : 0.01, 
                         'livestock' : 0.05, 'other' : 0.23,
                         'wetlands' : 0.5})

EPA_err_max = pd.Series({'coal' : 0.07, 'gas' : 0.15, 'oil' : 0.76, 
                         'landfills' : 0.33, 'wastewater' : 0.2, 
                         'livestock' : 0.07, 'other' : 0.5,
                         'wetlands' : 0.5})

## -------------------------------------------------------------------------##
## Load and process raw emissions data
## -------------------------------------------------------------------------##
emis_files = {'std' : emis_file, 'hr' : emis_hr_file}
emis = {}
for e_str, e_file in emis_files.items():
    emis[e_str] = gc.read_file(*e_file)

    # Remove emissions from buffer grid cells
    emis[e_str] = gc.subset_data_latlon(emis[e_str], *s.lats, *s.lons)

    # Drop hyam, hybm, and P0 (uneeded variables)
    emis[e_str] = emis[e_str].drop(['hyam', 'hybm', 'P0'])
    emis[e_str] = emis[e_str].squeeze()

    # Average over the year
    if 'time' in emis[e_str].dims:
        emis[e_str] = (emis[e_str]*days).sum(dim='time')/(days.sum())

        # Save summary files
        if e_str == 'std':
            name = f'HEMCO_diagnostics.2019.nc'
        elif e_str == 'hr':
            name = f'HEMCO_diagnostics_hr.2019.nc'
        emis[e_str].to_netcdf(f'{data_dir}/prior/total_emissions/{name}')

    # Adjust units to Mg/km2/yr (from kg/m2/s)
    if e_str == 'std':
        conv_factor = 1e-3*(60*60*24*365)*(1000*1000)
        emis[e_str] *= conv_factor
        emis[e_str]['AREA'] /= conv_factor*(1000*1000) # Fix area/convert to km2 from m2
    elif e_str == 'hr': 
        conv_factor = 1e-3*(60*60*24*365)*(1000*1000)*1e4
        emis[e_str] *= conv_factor
        emis[e_str]['AREA'] /= conv_factor*(1000*1000)


# Re = 6375e3 # Radius of the earth in m
# lon_e_gc = np.append(clusters.lon.values - s.lon_delta/2,
#                      clusters.lon[-1].values + s.lon_delta/2)
# lat_e_gc = np.append(clusters.lat.values - s.lat_delta/2,
#                      clusters.lat[-1].values + s.lat_delta/2)
# area_gc = Re**2*(np.sin(lat_e_gc[1:]/180*np.pi) - 
#                  np.sin(lat_e_gc[:-1]/180*np.pi))*s.lon_delta/180*np.pi

# print(area_gc/1e6)
# print(emis['hr']['AREA'].values[:, 0])
# print(emis['std']['AREA'].values[:, 0])

# Save out area as km2
area = ip.clusters_2d_to_1d(clusters, emis['std']['AREA'])
area = xr.DataArray(area, dims=('nstate'))
area.to_netcdf(f'{data_dir}/area.nc')

# Isolate soil absorption
soil_abs = emis['std']['EmisCH4_SoilAbsorb']
soil_abs = ip.clusters_2d_to_1d(clusters, soil_abs)
soil_abs = xr.DataArray(soil_abs, dims=('nstate'))
soil_abs.to_netcdf(f'{data_dir}/prior/xa_soil_abs.nc')

# Calculate total emissions
xa_abs = emis['std']['EmisCH4_Total'] #- emis['EmisCH4_SoilAbsorb']
xa_abs = ip.clusters_2d_to_1d(clusters, xa_abs)
xa_abs = xr.DataArray(xa_abs, dims=('nstate'))
xa_abs.to_netcdf(f'{data_dir}/prior/xa_abs.nc')

print('The minimum positive emission is',
      np.abs(emis['std']['EmisCH4_Total'].where(emis['std']['EmisCH4_Total'] > 0).min()).values)

## -------------------------------------------------------------------------##
## Tables
## -------------------------------------------------------------------------##
# Print a summary table
print('-'*75)
print('SECTORAL EMISSIONS')
print('-'*75)
summ = emis['std'][[var for var in emis['std'].keys()
                    if var[:4] == 'Emis']]*emis['std']['AREA']
summ *= 1e-6 # Adjust units to Tg/yr
# summ = summ.sum(dim=['lat', 'lon'])
tally = 0    # Total tally
tally_sv = 0 # The tally within the state vector
for k in summ.keys():
    # Get emissions within the state vector
    k_all = summ[k].sum(dim=['lat', 'lon']).values
    k_sv = ip.clusters_2d_to_1d(clusters, summ[k]).sum()
    if k != 'EmisCH4_Total':
        tally += k_all
        tally_sv += k_sv
    k_str = k.split('_')[-1]
    if k_all != 0:
        print(f'{k_str:>20} {k_all:5.2f} {k_sv:5.2f} {(100*k_sv/k_all):5.1f}%')
    else:
        print(f'{k_str:>20} {k_all:5.2f} {k_sv:5.2f}   ----')
print('-'*75)
print(f'               Total {tally:5.2f} {tally_sv:5.2f} {(100*tally_sv/tally):5.1f}%')
print('-'*75)

# Print a second summary table
print('-'*75)
print('SECTORAL NATIONAL EMISSIONS')
print('-'*75)

# Open masks and create a total_mask array as well as a mask dictionary
mex_mask = np.load(f'{data_dir}/countries/Mexico_mask.npy')
can_mask = np.load(f'{data_dir}/countries/Canada_mask.npy')
conus_mask = np.load(f'{data_dir}/countries/CONUS_mask.npy')
other_mask = np.load(f'{data_dir}/countries/Other_mask.npy')
total_mask = mex_mask + can_mask + conus_mask + other_mask
masks = pd.DataFrame.from_dict({'Canada' : can_mask, 'CONUS' : conus_mask, 
                                'Mexico' : mex_mask, 'Other' : other_mask})

# Map sectors to emission categories
e_table = {'Natural' : {'Wetlands' : ['Wetlands'],
                        'Open fires' : ['BiomassBurn'],
                        'Termites' : ['Termites'],
                        'Geological seeps' : ['Seeps'],
                        'Other natural' : ['Lakes']},
           'Anthropogenic' : {'Livestock' : ['Livestock'],
                              'Oil and gas' : ['Oil', 'Gas'],
                              'Landfills' : ['Landfills'],
                              'Coal mining' : ['Coal'],
                              'Wastewater' : ['Wastewater'],
                              'Rice cultivation' : ['Rice'],
                              'Other anthropogenic' : ['OtherAnth']}}

# Subset to the emission categories
summ = emis['std'][[var for var in emis['std'].keys() if var[:4] == 'Emis']]
summ *= 1e-6 # Tg/km2/yr
summ *= emis['std']['AREA'] # Tg/yr

c = masks.columns.values
print(f'Source                {c[0]:>10}{c[1]:>10}{c[2]:>10}{c[3]:>10}     Total')
print('-'*75)
# print(f'{'Canada':25}')
ttotal = [0, 0, 0, 0]
for label, big_cat in e_table.items():
    total = [0, 0, 0, 0]
    print(f'{label}')
    for cat, sources in big_cat.items():
        # Get gridded emissions sum
        e = sum(summ[f'EmisCH4_{s}'].squeeze() for s in sources)

        # Flatten
        e = ip.clusters_2d_to_1d(clusters, e).reshape((-1, 1))

        # Apply mask and sum
        e = (e*masks).sum(axis=0).values
        total = [total[i] + e[i] for i in range(4)]

        print(f'  {cat:20}{e[0]:10.3f}{e[1]:10.3f}{e[2]:10.3f}{e[3]:10.3f}{sum(e):10.3f}')
        #{e['CONUS']:5.2f}{e['Mexico']:5.2f}{e['Other']:5.2f}')
        # print(cat, e)
        # print(cat, sources)
    print(f'  Total               {total[0]:10.3f}{total[1]:10.3f}{total[2]:10.3f}{total[3]:10.3f}{sum(total):10.3f}')
    ttotal = [ttotal[i] + total[i] for i in range(4)]

print(f'Total                 {ttotal[0]:10.3f}{ttotal[1]:10.3f}{ttotal[2]:10.3f}{ttotal[3]:10.3f}{sum(ttotal):10.3f}')

print('-'*75)

## -------------------------------------------------------------------------##
## Group by sector
## -------------------------------------------------------------------------##
def sectoral_matrix(raw_emis, suffix, prefix='EmisCH4_', area=area,
                    clusters=clusters, emis_cats=s.sector_groups):
    w = pd.DataFrame(columns=emis_cats.keys())
    for label, categories in emis_cats.items():
        # Get emissions
        if type(categories) == str:
            e = raw_emis['%s%s' % (prefix, categories)].squeeze()
        elif type(categories) == list:
            e = sum(raw_emis['%s%s' % (prefix, em)].squeeze()
                    for em in categories)

        # Flatten
        e = ip.clusters_2d_to_1d(clusters, e)

        # Convert to Mg/yr
        e *= area

        # Saveout
        w[label] = e

    w.to_csv(join(data_dir, f'w{suffix}.csv'), index=False)
    return w

w = sectoral_matrix(emis['std'], '')

# High resolution sectoral W
w_hr = sectoral_matrix(emis['hr'], suffix='_hr', prefix='InvGEPA_CH4_', 
                       emis_cats=s.sector_groups_hr)

## And now correct for updates to oil and natural gas
### Scale natural gas emissions by 2020 GHGI emissions for 2018
w_hr['gas_distribution'] *= (473*1e3)/w_hr['gas_distribution'].sum()

### Convert the upstream emissions to match W --> but we need to do 
### this with the EDF emissions included, so we'll do it below.
### This is a place holder to check the arithmetic.
w_hr['ong_upstream'] = conus_mask*(w['ong'] - w_hr['gas_distribution'])

print('-'*75)
print('Standard emissions')
print(w.sum(axis=0)*1e-6)
print('total         ', w.values.sum()*1e-6 +
                        (soil_abs*area).values.sum()*1e-6)
print('total         ', (xa_abs*area).sum().values*1e-6)

print('-'*75)
print('Sectoral high resolution emissions')
print(w_hr.sum(axis=0)*1e-6)
print('total         ', w_hr.values.sum()*1e-6 +
                        (soil_abs*area).values.sum()*1e-6)
print('total         ', (xa_abs*area).sum().values*1e-6)

## -------------------------------------------------------------------------##
## Get sectoral errors
## -------------------------------------------------------------------------##
# sect_err_max = np.sqrt(((w*EPA_err_max)**2).sum(axis=1))/w.sum(axis=1).values
# sect_err_max = np.array(sect_err_max)
# sect_err_max = np.nan_to_num(sect_err_max, 0.5)
# sect_err_max[sect_err_max == 0.0] = 0.5

# sect_err_min = np.sqrt(((w*EPA_err_min)**2).sum(axis=1))/w.sum(axis=1).values
# sect_err_min = np.array(sect_err_min)
# sect_err_min = np.nan_to_num(sect_err_min, 0.5)
# sect_err_min[sect_err_min == 0] = 0.5

# # cbar_kwargs = {'title' : r'Relative errors'}
# # small_map_kwargs = {'draw_labels' : False}
# # small_fig_kwargs = {'max_width' : 4,
# #                     'max_height' : 3.5}
# # emis_kwargs = {'cmap' : fp.cmap_trans('viridis'), 'vmin' : 0, 'vmax' : 0.5,
# #                'default_value' : 0, 'cbar_kwargs' : cbar_kwargs, 
# #                'fig_kwargs' : small_fig_kwargs, 
# #                'map_kwargs' : small_map_kwargs}
# # # fig, ax = fp.get_figax(rows=1, cols=2, maps=True,
# # #                        lats=emis['std'].lat, lons=emis['std'].lon)
# # plt.subplots_adjust(hspace=0.5)
# # # fig, ax[0], c = ip.plot_state(sect_err_min, clusters, cbar=False,
# # #                               title='Minimum sectoral errors',
# # #                               fig_kwargs={'figax' : [fig, ax[0]]},
# # #                               **emis_kwargs)
# # fig, ax, c = ip.plot_state(sect_err_max, clusters, #cbar=False,
# #                               title='Sectoral errors',
# #                               # fig_kwargs={'figax' : [fig, ax[1]]},
# #                               **emis_kwargs)
# # # cax = fp.add_cax(fig, ax, cbar_pad_inches=0.5)
# # # cb = fig.colorbar(c, cax=cax, ticks=np.arange(0, 0.6, 0.1))
# # # cb = fp.format_cbar(cb, cbar_title=r'Relative errors') 
# # fp.save_fig(fig, plot_dir, 'sectoral_errors')

## -------------------------------------------------------------------------##
## Sensitivity tests: Boundary condition
## -------------------------------------------------------------------------##
# Get indices corresponding to the functional boundary condition correction
bc_idx = clusters.where(clusters.lat > s.lat_max - 3*s.lat_delta, drop=True)
bc_idx = bc_idx.values.astype(int)
bc_idx = bc_idx[bc_idx > 0] - 1
bc_idx.sort()

# Set these to zero in the absolute prior
xa_abs_bc0 = dc(xa_abs)
xa_abs_bc0[bc_idx] = 0
xa_abs_bc0.to_netcdf(join(data_dir, 'xa_abs_bc0.nc'))

# We can use the same sectoral breakdown as W.

# -------------------------------------------------------------------------##
## Sensitivity tests: Permian
## -------------------------------------------------------------------------##
# Load Permian inventory and replace EPA emissions
permian_ong_edf = xr.open_dataset(f'{data_dir}/prior/permian/permian_EDF_2019.nc') # Originally kg/m2/s
permian_ong_edf['EmisCH4_Oil'] *= 1e-3*(60*60*24*365)*(1000*1000) # -> Mg/km2/yr
permian_ong_edf['EmisCH4_Gas'] *= 1e-3*(60*60*24*365)*(1000*1000)
permian_condition = ((clusters.lat <= permian_ong_edf.lat.max()) &
                     (clusters.lat >= permian_ong_edf.lat.min()) &
                     (clusters.lon <= permian_ong_edf.lon.max()) &
                     (clusters.lon >= permian_ong_edf.lon.min()))
permian_idx = np.where(permian_condition.values)

# a. Subset the full clusters over the Permian
permian_clusters = clusters.where(permian_condition, drop=True)

# b. Flatten and create list of indices
permian_idx = permian_clusters.values.reshape(-1,)
permian_idx = np.sort(permian_idx)
permian_idx = permian_idx[permian_idx > 0]
permian_idx = (permian_idx - 1).astype(int)

# c. Subset total prior emissions and oil and natural gas emissions over
# the Permian (Mg/km2/yr)
xa_abs_permian = xa_abs[permian_idx]
xa_abs_permian_ong_epa = ip.clusters_2d_to_1d(clusters, emis['std']['EmisCH4_Oil'] + emis['std']['EmisCH4_Gas'])[permian_idx]

# d. Convert to 1D
xa_abs_permian_o_edf = ip.clusters_2d_to_1d(permian_clusters, permian_ong_edf['EmisCH4_Oil'])
xa_abs_permian_ng_edf = ip.clusters_2d_to_1d(permian_clusters, permian_ong_edf['EmisCH4_Gas'])

# e. Calculate the new total emissions over the Permian and replace them in
# the whole state vector
xa_abs_permian += (xa_abs_permian_o_edf + xa_abs_permian_ng_edf
                   - xa_abs_permian_ong_epa)
xa_abs_edf = dc(xa_abs)
xa_abs_edf[permian_idx] = xa_abs_permian

# g. Calculate a new sectoral attribution matrix
w_edf = dc(w)
# w_edf['oil'][permian_idx] = xa_abs_permian_o_edf
# w_edf['gas'][permian_idx] = xa_abs_permian_ng_edf
w_edf['ong'][permian_idx] = (xa_abs_permian_o_edf + xa_abs_permian_ng_edf)*area[permian_idx]

# h. Corrrect the high resolution inventory for EDF
w_hr['ong_upstream'] = conus_mask*(w_edf['ong'] - w_hr['gas_distribution'])

## Including by dealing with the negatives created by subtracting gas 
## distribution...
w_hr_neg = dc(w_hr['ong_upstream'])
w_hr_neg[w_hr_neg > 0] = 0
w_hr['gas_distribution'] += w_hr_neg
w_hr['ong_upstream'][w_hr['ong_upstream'] < 0] = 0
# tmp = dc(w_hr['ong_upstream']/area)
# tmp[tmp > 0] = 0
# print(tmp.min())
# print(tmp.max())
# ip.plot_state(tmp, clusters, cmap='RdBu_r', vmin=-1, vmax=1)
# plt.show()

# i. Save out
xa_abs_edf = xr.DataArray(xa_abs_edf, dims=('nstate'))
xa_abs_edf.to_netcdf(join(data_dir, 'xa_abs_edf.nc'))
w_edf.to_csv(join(data_dir, f'w_edf.csv'), index=False)
w_hr.to_csv(join(data_dir, f'w_edf_hr.csv'), index=False)

print('-'*75)
print('EDF inventory emissions')
print(w_edf.sum(axis=0)*1e-6)
print('total         ', w_edf.values.sum()*1e-6 + 
                        (soil_abs*area).values.sum()*1e-6)
print('total         ', (xa_abs_edf*area).sum().values*1e-6)

print('-'*75)
print('High-resolution EDF inventory emissions')
print(w_hr.sum(axis=0)*1e-6)
print('total         ', w_hr.values.sum()*1e-6 + 
                        (soil_abs*area).values.sum()*1e-6)
print('total         ', (xa_abs_edf*area).sum().values*1e-6)

## -------------------------------------------------------------------------##
## Sensitivity tests: Wetlands
## -------------------------------------------------------------------------##
## Remove ensemble members 3 (1923) and 7 (2913). The prep work for
## this test was done in wetlands.py
# Load in HEMCO_diagnostics (this code is commented above)
emis_w37_file = [f'{data_dir}/prior/wetlands/wetlands37/\
HEMCO_diagnostics.{s.year:04d}{mm:02d}010000.nc'
             for mm in s.months]
emis_w37 = gc.read_file(*emis_w37_file)
emis_w37 = gc.subset_data_latlon(emis_w37, *s.lats, *s.lons)
emis_w37 = emis_w37.drop(['hyam', 'hybm', 'P0'])
emis_w37 = (emis_w37*days).sum(dim='time')/(days.sum())
emis_w37.to_netcdf(f'{data_dir}/prior/wetlands/HEMCO_diagnostics_w37.2019.nc')
emis_w37 *= 1e-3*(60*60*24*365)*(1000*1000) # --> Mg/km2/yr
emis_w37['AREA'] /= 1e-3*(60*60*24*365)*(1000*1000)**2

xa_abs_w37 = ip.clusters_2d_to_1d(clusters, emis_w37['EmisCH4_Total'])
xa_abs_w37 = xr.DataArray(xa_abs_w37, dims=('nstate'))
xa_abs_w37.to_netcdf(join(data_dir, 'xa_abs_w37.nc'))

# Sectoral attribution matrix
w_w37 = sectoral_matrix(emis_w37, '_w37')

print('-'*75)
print('Removed wetland ensemble members emissions')
print(w_w37.sum(axis=0)*1e-6)
print('total         ', w_w37.values.sum()*1e-6 + 
                        (soil_abs*area).values.sum()*1e-6)
print('total         ', (xa_abs_w37*area).sum().values*1e-6)

## -------------------------------------------------------------------------##
## Sensitivity tests: Combination
## -------------------------------------------------------------------------##
# # EDF and BC0
# xa_abs_edf_bc0 = dc(xa_abs_edf)
# xa_abs_edf_bc0[bc_idx] = 0
# xa_abs_edf_bc0.to_netcdf(join(data_dir, 'xa_abs_edf_bc0.nc'))

# EDF and Wetlands 37
xa_abs_w37_edf = dc(xa_abs_edf) + (w_w37['wetlands'] - w['wetlands'])/area
xa_abs_w37_edf.to_netcdf(join(data_dir, 'xa_abs_w37_edf.nc'))

w_w37_edf = dc(w)
w_w37_edf['wetlands'] = dc(w_w37['wetlands'])
w_w37_edf['ong'] = dc(w_edf['ong'])
w_w37_edf.to_csv(join(data_dir, 'w_w37_edf.csv'), index=False)

print('-'*75)
print('Removed wetland ensemble members emissions and EDF inventory')
print(w_w37_edf.sum(axis=0)*1e-6)
print('total         ', w_w37_edf.values.sum()*1e-6 + 
                        (soil_abs*area).values.sum()*1e-6)
print('total         ', (xa_abs_w37_edf*area).sum().values*1e-6)
print('-'*75)

# EDF, Wetlands 37, and BC0
xa_abs_w37_edf_bc0 = dc(xa_abs_w37_edf)
xa_abs_w37_edf_bc0[bc_idx] = 0
xa_abs_w37_edf_bc0.to_netcdf(join(data_dir, 'xa_abs_w37_edf_bc0.nc'))

## -------------------------------------------------------------------------##
## Plot
## -------------------------------------------------------------------------##
if plot_dir is not None:
    plot_e = {'Total' : ['Total'],
              'Oil and gas' : ['OilAndGas'], #'Gas' : ['Gas'], 
              'Coal' : ['Coal'],
              'Livestock' : ['Livestock'], 
              'Landfills' : ['Landfills'],
              'Wastewater' : ['Wastewater'], 
              'Wetlands' : ['Wetlands37'],
              'Other' : ['Other']}#,
              # 'Open Fires' : ['BiomassBurn']}
    # Rice, Termites, Seeps, Lakes, OtherAnth are missing
    plot_e_terms = [item for sublist in plot_e.values() for item in sublist]

    # Set colormap
    colormap = fp.cmap_trans('viridis')

    # Update emissions
    e_c = dc(emis['std'])

    # Save out the wetlands that we actually use
    # e_c['EmisCH4_Wetlands404'] = e_c['EmisCH4_Wetlands']/4.04
    e_c['EmisCH4_Wetlands37'] = emis_w37['EmisCH4_Wetlands']

    # Bring in the EDF emissions
    e_c['EmisCH4_OilAndGas'] = ip.match_data_to_clusters(w_edf['ong']/area,
                                                         clusters)
    # e_c['EmisCH4_Gas'] = dc(summ['EmisCH4_Gas'])*1e6
    # e_c['EmisCH4_Oil'] = dc(summ['EmisCH4_Oil'])*1e6

    # Sum up other
    e_c['EmisCH4_Other'] = sum(e_c[f'EmisCH4_{s}'] for s 
                               in ['BiomassBurn', 'Termites', 'Seeps', 'Lakes',
                                   'Rice', 'OtherAnth'])

    # Update total
    e_c['EmisCH4_Total'] = sum(e_c[f'EmisCH4_{s}'] for s in plot_e_terms
                               if s not in 
                               ['Total'])
    # e_c['EmisCH4_Total'] += (e_c['EmisCH4_Wetlands404'] + e_c['EmisCH4_Wetlands37'])/2

    # Plot
    fig, ax = fp.get_figax(rows=2, cols=4, maps=True,
                           lats=e_c.lat, lons=e_c.lon)
    for axis, (title, emis_list) in zip(ax.flatten(), plot_e.items()):
        axis = fp.format_map(axis, lats=e_c.lat, lons=e_c.lon)
        e = sum(e_c[f'EmisCH4_{em}'].squeeze() for em in emis_list)
        c = e.plot(ax=axis, cmap=colormap, vmin=0, vmax=5,
                   add_colorbar=False)
        axis.text(0.025, 0.025, title, ha='left', va='bottom', 
                  fontsize=config.TICK_FONTSIZE,
                  transform=axis.transAxes)
    plt.subplots_adjust(hspace=-0.25, wspace=0.1)
    cax = fp.add_cax(fig, ax, horizontal=True)#, cbar_pad_inches=0.2)
    cb = fig.colorbar(c, cax=cax, ticks=np.arange(0, 6, 1), 
                      orientation='horizontal')
    cb = fp.format_cbar(cb, cbar_title=r'Methane emissions (Mg km$^2$ a$^{-1}$)',
                        horizontal=True, y=-3)
    fp.save_fig(fig, plot_dir, 'prior_emissions_2019')

    # Create another plot comparing orig and correct
    fig, ax = fp.get_figax(rows=1, cols=3, maps=True, 
                           lats=emis['std'].lat, lons=emis['std'].lon)
    for i, axis in enumerate(ax.flatten()):
        axis = fp.format_map(axis, lats=emis['std'].lat, lons=emis['std'].lon)