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
year = 2019
months = np.arange(1, 13, 1)
days = np.arange(1, 32, 1)
files = join(data_dir, 'GEOSChem.SpeciesConc.YYYYMMDD_0000z.nc4')
replacement_files = join(input_dir, 'halfstep_outputs',
                         'GEOSChem.SpeciesConc.YYYYMMDD_0000z.nc4')
profiles = join(input_dir, 'vertical_profiles', 'mean_profile_YYYYMM.nc')

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
import gcpy as gc
import troppy as tp
import format_plots as fp

## ------------------------------------------------------------------------ ##
## Replace stratosphere
## ------------------------------------------------------------------------ ##
lat_e, lon_e = gc.adjust_grid_bounds(lat_min, lat_max, lat_delta,
                                     lon_min, lon_max, lon_delta,
                                     buffers)

# Iterate through the correct files
for month in months:
    # Open the default vertical profile
    profile = xr.open_dataset(profiles.replace('YYYYMM', f'{year}{month:02d}'))

    for day in days:
        # Get file name
        file = files.replace('YYYYMMDD', f'{year}{month:02d}{day:02d}')

        # Open the file
        print(f'-'*75)
        print(f'Opening {file}')
        if file.split('/')[-1] not in listdir(file.rpartition('/')[0]):
            print(f'{file} is not in the data directory.')
            continue
        data = xr.open_dataset(join(data_dir, file))

        # Remove the buffer grid cells
        print('Removing buffer cells.')
        data = gc.subset_data_latlon(data, *lat_e, *lon_e)

        # Check for time dimensiom
        if len(data.time) != 24:
            print(f'Filling in first hour.')
            data = gc.fill_GC_first_hour(data)

        # Check if any values are more than 5x greater than or less than
        # the profile
        print('Checking for anomalous values in upper levels.')
        diff = np.abs(xr.ufuncs.log10(data['SpeciesConc_CH4'])/np.log10(5) -
                      xr.ufuncs.log10(profile['SpeciesConc_CH4'])/np.log10(5))

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
                                              f'{year}{month:02d}{day:02d}')
            new = xr.open_dataset(rfile)['SpeciesConc_CH4']
            new = gc.subset_data_latlon(new, *lat_e, *lon_e)

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

            # # Create a plot to check
            # data_summ = data['SpeciesConc_CH4'].mean(dim=['lat', 'lon'])
            # fig, ax = fp.get_figax()
            # ax.plot(profile['SpeciesConc_CH4'], profile.lev,
            #         c=fp.color(0), lw=3)
            # ax.plot(data_summ.T, data_summ.lev, c=fp.color(4),
            #         alpha=0.5, lw=0.5)

            # title = file.split('_')[0].split('.')[-1]
            # ax = fp.add_title(ax, f'{title}')
            # ax = fp.add_labels(ax, r'XCH4 (mol mol$^{-1}$)',
            #                    'Hybrid Level at Midpoints')
            # ax.set_xlim(0, 3e-6)
            # ax.set_ylim(1, 0)

            # fp.save_fig(fig, data_dir, title)

            # save out file
        data.to_netcdf(join(data_dir, file))

            # # Optional plotting
            # for t in data.time:
            #     datas = data['SpeciesConc_CH4']
            #     ds = datas.sel(time=t)
            #     for i, l in enumerate(ds.lev):
            #         if l < 0.3:
            #             new_title = f'{title}-{i:02d}'
            #             if f'{new_title}.png' not in listdir(data_dir):
            #                 datas = ds.sel(lev=l)
            #                 fig, ax = fp.get_figax(maps=True, lats=datas.lat, lons=datas.lon)
            #                 ax = fp.format_map(ax, lats=datas.lat, lons=datas.lon)
            #                 c = datas.plot(ax=ax, cmap=fp.cmap_trans('viridis'),
            #                               add_colorbar=False, vmin=0, vmax=0.0000025)

            #                 # Add colorbar
            #                 cax = fp.add_cax(fig, ax)
            #                 cb = fig.colorbar(c, cax=cax)
            #                 cb = fp.format_cbar(cb, cbar_title='')

            #                 ax = fp.add_title(ax, f'{new_title}')
            #                 fp.save_fig(fig, data_dir, f'{new_title}')
            #                 plt.close()

            # # Create a gif
            # images = []
            # files = [f for f in listdir(data_dir) if (f[-3:] == 'png') and (len(f) == 15)]
            # files.sort()
            # # files = files[::-1]
            # print(files)
            # for f in files:
            #     images.append(imageio.imread(join(data_dir, f)))
            # imageio.mimsave(join(data_dir, f'{title}.gif'),
            #                 images, duration=0.5)