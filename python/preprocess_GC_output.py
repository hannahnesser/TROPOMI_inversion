'''
This script preprocesses the output of any GEOS-Chem simulation to remove buffer grid cells and to replace any grid cells where anomalously (unphysical) high methane concentrations result from the violation of the CFL condition near the boundary conditions.

**Inputs**

| ----------------- | -------------------------------------------------- |
| Input             | Description                                        |
| ----------------- | -------------------------------------------------- |
| files             | A list of SpeciesConc files from a GEOS-Chem       |
|                   | simulation.                                        |
| ----------------- | -------------------------------------------------- |
| replacement_files | A list of SpeciesConc files from a GEOS-Chem       |
|                   | simulation without anomalous (unphysical) methane  |
|                   | concentrations (January through November files are |
|                   | from the standard prior simulation and December    |
|                   | files are from a prior simulation conducted with a |
|                   | halved time step).                                 |
| ----------------- | -------------------------------------------------- |
| profiles          | Monthly mean methane vertical profiles to identify |
|                   | anomalous methane concentrations. We replace       |
|                   | vertical levels where there are values that are    |
|                   | scale_factor times bigger or smaller than the mean |
|                   | profile if the level is in the top half of the GC  |
|                   |atmosphere.                                         |
| ----------------- | -------------------------------------------------- |
| scale_factor      | The comparison factor that determines whether the  |
|                   | GEOS-Chem output is significantly different from   |
|                   | the mean profile.
| ----------------- | -------------------------------------------------- |

**Outputs**

| ----------------- | -------------------------------------------------- |
| Output            | Description                                        |
| ----------------- | -------------------------------------------------- |
| GEOSChem. \       | The corrected and subset files.                    |
| SpeciesConc. \    |                                                    |
| YYYYMMDD_0000z.nc4|                                                    |
| ----------------- | -------------------------------------------------- |
'''

from os.path import join
from os import listdir
import os
import sys
import copy
import math
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom packages
sys.path.append('.')
import config
import gcpy as gc
import troppy as tp
import format_plots as fp
import inversion_settings as settings

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# Local preferences
# input_dir = sys.argv[1]
# base_dir = sys.argv[2]
# data_dir = join(base_dir, 'OutputDir')
# code_dir = sys.argv[3]

input_dir = '/n/seasasfs02/hnesser/TROPOMI_inversion/gc_outputs/'
base_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/'
data_dir = join(base_dir, 'OutputDir')
code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'

# Information about the files
files = join(data_dir, 'GEOSChem.SpeciesConc.YYYYMMDD_0000z.nc4')
replacement_files = join(input_dir, 'halfstep_outputs',
                         'GEOSChem.SpeciesConc.YYYYMMDD_0000z.nc4')
profiles = join(input_dir, 'vertical_profiles', 'mean_profile_YYYYMM.nc')

# The scale factor for comparison to the mean vertical profiles--how many
# times bigger or smaller than the mean profile does a methane concetnration
# need to be for the level to be replaced?
scale_factor = 5

## ------------------------------------------------------------------------ ##
## Replace stratosphere
## ------------------------------------------------------------------------ ##
# Iterate through the correct files
for month in settings.months:
    # Open the default vertical profile
    profile = xr.open_dataset(profiles.replace('YYYYMM',
                                               f'{settings.year}{month:02d}'))

    for day in settings.days:
        # Get file name
        file = files.replace('YYYYMMDD',
                             f'{settings.year}{month:02d}{day:02d}')

        # Open the file
        print(f'-'*75)
        print(f'Opening {file}')
        if file.split('/')[-1] not in listdir(file.rpartition('/')[0]):
            print(f'{file} is not in the data directory.')
            continue
        data = xr.open_dataset(join(data_dir, file))

        # Remove the buffer grid cells
        print('Removing buffer cells.')
        data = gc.subset_data_latlon(data, *settings.lats, *settings.lons)

        # Check for time dimensiom
        if len(data.time) != 24:
            print(f'Filling in first hour.')
            data = gc.fill_GC_first_hour(data)

        # Check if any values are more than 5x greater than or less than
        # the profile
        print('Checking for anomalous values in upper levels.')
        diff = np.abs(xr.ufuncs.log10(data['SpeciesConc_CH4'])/np.log10(scale_factor) -
                      xr.ufuncs.log10(profile['SpeciesConc_CH4'])/np.log10(scale_factor))

        # Replace the bottom level with 0s
        diff = diff.where(diff.lev < 0.5, 0)

        # If so, replace those values
        if (diff >= 1).any():
            print(f'Replacing data in {file}')

            # Move the original file
            os.rename(join(data_dir, file), join(data_dir, f'{file}_orig'))

            # The original file, only where the problem values are
            old = data['SpeciesConc_CH4'].where(diff >=1, drop=True).squeeze()

            # Open the new file (that will replace the problem values)
            rfile = replacement_files.replace('YYYYMMDD',
                                              f'{settings.year}{month:02d}{day:02d}')
            new = xr.open_dataset(rfile)['SpeciesConc_CH4']
            new = gc.subset_data_latlon(new, *settings.lats, *settings.lons)

            # Check for time dimension
            if len(new.time) != 24:
                print(f'Filling in first hour.')
                new = gc.fill_GC_first_hour(new)

            # Fill in the data by replacing the entire level
            cond = data.lev.isin(old.lev.values)
            data['SpeciesConc_CH4'] = data['SpeciesConc_CH4'].where(~cond, new)

            # Print out information
            print(f'  Level       Old Min     New Min     Old Max     New Max')
            for l in old.lev.values:
                old_dat = old.sel(lev=l).values
                new_dat = data.sel(lev=l)['SpeciesConc_CH4'].values
                print(f'  {l:<12.2e}{np.nanmin(old_dat):<12.2e}{new_dat.min():<12.2e}{np.nanmax(old_dat):<12.2e}{new_dat.max():<12.2e}')

        # save out file
        data.to_netcdf(join(data_dir, file))
