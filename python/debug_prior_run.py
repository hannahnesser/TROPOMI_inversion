import xarray as xr
import numpy as np
import pandas as pd
from os import listdir
from os.path import join
import sys
import matplotlib.pyplot as plt
import imageio

# Information on the grid
lat_min = 9.75
lat_max = 60
lat_delta = 0.25
lon_min = -130
lon_max = -60
lon_delta = 0.3125
buffers = [6, 6, 6, 6]

## ------------------------------------------------------------------------ ##
## Import custom packages
## ------------------------------------------------------------------------ ##
# Custom packages
sys.path.append('.')
import config
import gcpy as gc
import troppy as tp
import invpy as ip
import format_plots as fp

# Other plot details
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'AppleGothic'
rcParams['font.size'] = config.LABEL_FONTSIZE*config.SCALE
rcParams['text.usetex'] = True
# rcParams['mathtext.fontset'] = 'stixsans'
rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'
rcParams['axes.titlepad'] = 0

## -------------------------------------------------------------------------##
## Save out absolute and relative priors
## -------------------------------------------------------------------------##

## ------------------------------------------------------------------------ ##
## Define the inversion grid
## ------------------------------------------------------------------------ ##
# Get information on the lats and lons (edges of the domain) (this means
# removing the buffer grid cells)
lat_e, lon_e = gc.adjust_grid_bounds(lat_min, lat_max, lat_delta,
                                     lon_min, lon_max, lon_delta, buffers)


data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/gc_outputs'

# List files
files = [f for f in listdir(data_dir) if (f[0] != '.') and (f[-3:] == 'nc4')]
# files = [f for f in files if int(f.split('_')[0][-2:]) >= 17]
files.sort()

# for f in files:
#     # f = files[2]
#     d = xr.open_dataset(join(data_dir, f))['SpeciesConc_CH4']
#     # Remove emissions from buffer grid cells
#     d = gc.subset_data_latlon(d, *lat_e, *lon_e)

#     print(d.where(d==d.max(), drop=True))

# f = files[19]
# print(f)

# lev = 1.13378455e-03
# d = xr.open_dataset(join(data_dir, f))['SpeciesConc_CH4']
# # d = gc.subset_data_latlon(d, *lat_e, *lon_e)
# dtmp = d.sel(time='2019-12-20T02:00:00.000000000', lev=0.006684756500000001)
# print(dtmp.min().values, dtmp.max().values)

# f = files[1]
# mins = []
# maxs = []
# times = []
# for f in files[10:]:
#     print(f)
#     d = xr.open_dataset(join(data_dir, f))['SpeciesConc_CH4']
#     for t in d.time:
#         dtmp = d.sel(time=t)
#         mins.append(float(d.min().values))
#         maxs.append(float(d.max().values))
#         times.append(t.values)

# # print(mins)
# # df = pd.DataFrame({'time' : times, 'min' : mins, 'max' : maxs})
# # df.to_csv(join(data_dir, 'minmax.csv'))

# df = pd.read_csv(join(data_dir, 'minmax.csv'), index_col=0)
# df['time'] = pd.to_datetime(df['time'])
# print(df[df['time'].dt.day == 18])

# fig, ax = fp.get_figax()
# ax.plot(df['time'], df['min'], c=fp.color(3))
# # ax.set_ylim(0, 5e-7)
# ax.set_yscale('log')
# ax.set_ylabel('Minimum Hourly XCH4', color=fp.color(3))

# ax2 = ax.twinx()
# ax2.plot(df['time'], df['max'], c=fp.color(6))
# ax2.set_yscale('log')
# ax2.set_ylabel('Maximum Hourly XCH4', color=fp.color(6))

# from matplotlib.dates import DateFormatter
# ax.set_xlim(np.datetime64('2019-12-10T00:00:00.000000000'),
#             np.datetime64('2019-12-31T00:00:00.000000000'))
# date_form = DateFormatter('%m-%d')
# ax.xaxis.set_major_formatter(date_form)

# plt.show()

for f in files:
    d = xr.open_dataset(join(data_dir, f))['SpeciesConc_CH4']
    d = gc.subset_data_latlon(d, *lat_e, *lon_e)
    d = d.where(d.lev >= 0.19, drop=True)
    print(f.split('.')[-2].split('_')[0], ':', d.max().values)

# d = d.sel(lev=lev)
# for t in d.time:
# for t in d.lev:
#     # data = d.sel(time=t)
#     # if str(t.values)[:3] == '0.0':
#     data = d.sel(lev=t)
#     fig, ax = fp.get_figax(maps=True, lats=data.lat, lons=data.lon)
#     ax = fp.format_map(ax, lats=data.lat, lons=data.lon)
#     c = data.plot(ax=ax, cmap=fp.cmap_trans('viridis'),
#                   add_colorbar=False, vmin=0, vmax=0.0000025)

#     # Add colorbar
#     cax = fp.add_cax(fig, ax)
#     cb = fig.colorbar(c, cax=cax)
#     cb = fp.format_cbar(cb, cbar_title='')
#     # print(t)
#     # title = str(t.values).split('T')[1].split(':')[0]
#     title = t.values
#     ax = fp.add_title(ax, f'{title}')
#     fp.save_fig(fig, data_dir, f'{title}')
#     plt.close()
#     # print(t)

# # Create a gif
# images = []
# files = [f for f in listdir(data_dir) if (f[0] != '.') and
#                                          (f[-3:] == 'png') and
#                                          (f[0].isdigit())]
#                                          # (f[:3] == '0.0')]
# files.sort()
# files = files[::-1]
# print(files)
# for f in files:
#     images.append(imageio.imread(join(data_dir, f)))
# imageio.mimsave(join(data_dir, 'prior_bug_levels_12-26-12h.gif'),
#                 images, duration=0.5)

# print(d)
# for l in d.lev[:1]:
#     data = d.sel(lev=l)
#     data.plot.line(x='time', add_legend=False, color='blue', alpha=0.5)
#     print(f'{l.values}')
#     plt.savefig(join(data_dir, f'{l.values}.png'))
