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
import imageio

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# Local preferences
input_dir = sys.argv[1]
base_dir = sys.argv[2]
# base_dir = sys.argv[1]
# code_dir = sys.argv[2]
data_dir = join(base_dir, 'OutputDir')
code_dir = sys.argv[3]

# Information about the files
year = 2019
months = np.arange(12, 13, 1)
days = np.arange(1, 32, 1)
files = [f'GEOSChem.SpeciesConc.{year}{mm:02d}{dd:02d}_0000z.nc4'
         for mm in months for dd in days]
files.sort()
profile = join(input_dir, 'default_profile.nc')

# Information on the grid # I think this is irrelevant?
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
# config.SCALE = config.PRES_SCALE
# config.BASE_WIDTH = config.PRES_WIDTH
# config.BASE_HEIGHT = config.PRES_HEIGHT
import gcpy as gc
import troppy as tp
import format_plots as fp

## ------------------------------------------------------------------------ ##
## Replace stratosphere
## ------------------------------------------------------------------------ ##
# Open default vertical profile
profile = xr.open_dataset(profile)

# Iterate through the correct files
for file in files:
    # Open the file
    if file not in listdir(data_dir):
        print(f'{file} is not in the data directory.')
        continue
    print(f'-'*75)
    print(f'Checking for anomalous values in {file}')
    data = xr.open_dataset(join(data_dir, file))

    # Check for time dimensiom
    if len(data.time) != 24:
        print(f'Filling in first hour.')
        data = gc.fill_GC_first_hour(data)

    # Check if any values are more than 5x greater than or less than
    # the profile
    diff = np.abs(xr.ufuncs.log10(data['SpeciesConc_CH4'])/np.log10(5) -
                  xr.ufuncs.log10(profile['SpeciesConc_CH4'])/np.log10(5))

    # If so, replace those values
    if (diff >= 1).any():
        print(f'Replacing data in {file}')

        # The original file, only where the problem values are
        old = data['SpeciesConc_CH4'].where(diff >=1, drop=True).squeeze()

        # Open the new file (that will replace the problem values)
        new = xr.open_dataset(join(input_dir, file))['SpeciesConc_CH4']
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

        # Create a plot to check
        data_summ = data['SpeciesConc_CH4'].mean(dim=['lat', 'lon'])
        fig, ax = fp.get_figax()
        ax.plot(profile['SpeciesConc_CH4'], profile.lev,
                c=fp.color(0), lw=3)
        ax.plot(data_summ.T, data_summ.lev, c=fp.color(4), alpha=0.5, lw=0.5)

        title = file.split('_')[0].split('.')[-1]
        ax = fp.add_title(ax, f'{title}')
        ax = fp.add_labels(ax, r'XCH4 (mol mol$^{-1}$)',
                           'Hybrid Level at Midpoints')
        ax.set_xlim(0, 3e-6)
        ax.set_ylim(1, 0)

        fp.save_fig(fig, data_dir, title)

        # save out file
        data.to_netcdf(join(data_dir, f'{file}_fixed.nc'))

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
