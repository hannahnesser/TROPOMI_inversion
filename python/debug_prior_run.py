import xarray as xr
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
print(lat_e, lon_e)



data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/gc_outputs'

# List files
files = [f for f in listdir(data_dir) if (f[0] != '.') and (f[-3:] == 'nc4')]
files = [f for f in files if int(f.split('_')[0][-2:]) >= 18]
files.sort()

# for f in files:
#     # f = files[2]
#     d = xr.open_dataset(join(data_dir, f))['SpeciesConc_CH4']
#     # Remove emissions from buffer grid cells
#     d = gc.subset_data_latlon(d, *lat_e, *lon_e)

#     print(d.where(d==d.max(), drop=True))

f = files[2]
lev = 1.13378455e-03
d = xr.open_dataset(join(data_dir, f))['SpeciesConc_CH4']
d = d.sel(lev=lev)
for t in d.time:
    data = d.sel(time=t)
    fig, ax = fp.get_figax(maps=True, lats=data.lat, lons=data.lon)
    ax = fp.format_map(ax, lats=data.lat, lons=data.lon)
    c = data.plot(ax=ax, cmap=fp.cmap_trans('viridis'),
                  add_colorbar=False, vmin=0, vmax=0.004)

    # Add colorbar
    cax = fp.add_cax(fig, ax)
    cb = fig.colorbar(c, cax=cax)
    cb = fp.format_cbar(cb, cbar_title='')
    print(t)
    title = str(t.values).split('T')[1].split(':')[0]
    ax = fp.add_title(ax, f'{title}')
    fp.save_fig(fig, data_dir, f'{title}')
    # print(t)

# Create a gif
images = []
files = [f for f in listdir(data_dir) if (f[0] != '.') and (f[-3:] == 'png')]
files.sort()
for f in files:
    images.append(imageio.imread(join(data_dir, f)))
imageio.mimsave(join(data_dir, 'prior_bug_12-20.gif'), images, duration=0.5)

# print(d)
# for l in d.lev[:1]:
#     data = d.sel(lev=l)
#     data.plot.line(x='time', add_legend=False, color='blue', alpha=0.5)
#     print(f'{l.values}')
#     plt.savefig(join(data_dir, f'{l.values}.png'))
