## ---------------------------------------------------------------------##
## Standard imports
## ---------------------------------------------------------------------##
import sys
import xarray as xr
import numpy as np
import pandas as pd
import glob

## ---------------------------------------------------------------------##
## Directories
## ---------------------------------------------------------------------##
# code_dir = '.'
code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
prior_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_final/ProcessedDir'
pert_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_mb/ProcessedDir'
data_dir = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/inversion_data'
exclude_dates = ['20191218', '20191219', '20191220', '20191221', '20191222', 
                 '20191223', '20191224', '20191225', '20191226', '20191227', 
                 '20191228', '20191229', '20191230', '20191231']

## ---------------------------------------------------------------------##
## Custom imports
## ---------------------------------------------------------------------##
sys.path.append(code_dir)
import gcpy as gc
import inversion_settings as s

## ---------------------------------------------------------------------##
## Load the observation filter
## ---------------------------------------------------------------------##
obs_filter = pd.read_csv(f'{data_dir}/obs_filter.csv', header=0)
months = obs_filter['MONTH']
obs_filter = obs_filter['FILTER']

## ---------------------------------------------------------------------##
## Load the data for the prior simulation
## ---------------------------------------------------------------------##
prior_files = glob.glob(f'{prior_dir}/{s.year:04d}????_GCtoTROPOMI.pkl')
prior_files = [p for p in prior_files
               if (int(p.split('/')[-1].split('_')[0][4:6]) in s.months)]
prior_files.sort()
prior = np.array([])
mask = np.array([], dtype=bool) # Deal with excluded dates
for f in prior_files:
    data = gc.load_obj(f)[:, 1]
    if f.split('/')[-1].split('_')[0] in exclude_dates:
        mask = np.concatenate((mask, np.full(len(data), False)))
    else:
        prior = np.concatenate((prior, data))
        mask = np.concatenate((mask, np.full(len(data), True)))

prior = prior[obs_filter[mask]]

## ---------------------------------------------------------------------##
## Load the data for the perturbation simulation
## ---------------------------------------------------------------------##
pert_files = glob.glob(f'{pert_dir}/{s.year:04d}????_GCtoTROPOMI.pkl')
pert_files = [p for p in pert_files
              if (int(p.split('/')[-1].split('_')[0][4:6]) in s.months)
              & (p.split('/')[-1].split('_')[0] not in exclude_dates)]
pert_files.sort()
# print(pert_files[-1])
pert = np.array([])
for f in pert_files:
    if f.split('/')[-1].split('_')[0] not in exclude_dates:
        pert = np.concatenate((pert, gc.load_obj(f)[:, 1]))

pert = pert[obs_filter[mask]]

## ---------------------------------------------------------------------##
## Define the Jacobian and other inversion quantities
## ---------------------------------------------------------------------##
k = (pert - prior).reshape((-1, 1))/0.5
xa = np.array([1])
sa = np.array([0.5**2])
y = xr.open_dataarray(f'{data_dir}/y.nc').values.reshape((-1, 1))[mask[obs_filter]]
ya = xr.open_dataarray(f'{data_dir}/ya.nc').values.reshape((-1, 1))[mask[obs_filter]] # Test ya_nlc.nc
ydiff = y - ya
so = xr.open_dataarray(f'{data_dir}/so_rg2rt_10t.nc').values.reshape((-1, 1))[mask[obs_filter]]

## ---------------------------------------------------------------------##
## Solve inversion
## ---------------------------------------------------------------------##
def solve_inv(k, xa, sa, ydiff, so, rf, scale_emis=1):
    # k[k < 0] = 0
    ydiff = ydiff - (k @ np.array([scale_emis - 1])).reshape((-1, 1))
    k = scale_emis*k
    shat = 1/(rf*(k.T/so.reshape(-1,) @ k + 1/sa))
    g = shat*rf*k.T/so.reshape(-1,)
    xhat = xa + g @ ydiff
    print(f'Using rf = {rf:4} and sa = {sa:6}, we find a net correction of {xhat[0][0]:.6f}.')

for rf_i in [0.01, 0.05, 0.1, 0.5, 1, 2]:
    for sa_i in [0.5, 1, 2, 4]:
        solve_inv(k, xa, sa_i**2, ydiff, so, rf_i)

# ## ---------------------------------------------------------------------##
# ## Plot emissions as a first check
# ## ---------------------------------------------------------------------##
# base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion'
# emis_file = [f'{base_dir}/prior/mass_balance_test/\
# HEMCO_diagnostics.{s.year:04d}{mm:02d}010000.nc'
#              for mm in s.months]
# emis_orig_file = f'{base_dir}/prior/total_emissions/HEMCO_diagnostics_correct.{s.year}.nc'

# # Open files
# emis = gc.read_file(*emis_file)
# emis = gc.subset_data_latlon(emis, *s.lats, *s.lons)
# days = xr.DataArray([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
#                     dims='time')
# emis = (emis*days).sum(dim='time')/(days.sum())
# emis = emis['EmisCH4_Total']

# emis_orig = xr.open_dataset(emis_orig_file)['EmisCH4_Total']

# # Calculate the ratio
# ratio = (emis/emis_orig)
# ratio = ratio.rename('Ratio')
# print(np.nanmax(ratio.values))
# print(ratio.values)

# # Plot
# import format_plots as fp
# fig, ax = fp.get_figax(maps=True, lats=ratio.lat, lons=ratio.lon)
# c = ratio.plot(ax=ax)
# ax = fp.format_map(ax, lats=ratio.lat, lons=ratio.lon)
# ax = fp.add_title(ax, 'Ratio')
# fp.save_fig(fig, '../plots/', 'mass_balance_ratio')

# ## ---------------------------------------------------------------------##
# ## Plot concentrations as a second check
# ## ---------------------------------------------------------------------##
# conc_orig = xr.open_dataset('../gc_outputs/species_conc/GEOSChem.SpeciesConc.20190101_0000z_updated.nc4')
# conc = xr.open_dataset('../prior/mass_balance_test/GEOSChem.SpeciesConc.20190101_0000z.nc4')
# clusters = xr.open_dataarray('../inversion_data/clusters.nc')

# print(clusters.lat.min(), clusters.lat.max())
# print(conc.lat.min(), conc.lat.max())

# ratio = conc['SpeciesConc_CH4']/conc_orig['SpeciesConc_CH4']
# ratio = ratio.sel(lev=ratio.lev[0], time=ratio.time[-1])
# ratio = ratio.rename('Concentration ratio')

# import format_plots as fp
# fig, ax = fp.get_figax(maps=True, lats=ratio.lat, lons=ratio.lon)
# ax = fp.format_map(ax, lats=ratio.lat, lons=ratio.lon)
# c = ratio.plot(ax=ax)
# ax = fp.add_title(ax, 'Concentration')

# fp.save_fig(fig, '../plots/', 'mass_balance_conc_ratio')

# print(ratio)
