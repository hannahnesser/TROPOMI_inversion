from os import listdir, mkdir, getcwd, environ
from os.path import join, dirname, realpath
from os import environ
import sys

import pandas as pd
import numpy as np
import xarray as xr
import math

import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cartopy import feature as cf
sys.path.append('.')
import config as config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT

# config.TITLE_PAD /= 2
import format_plots as fp

# from fastkml import kml
# from lxml import etree
# from simplekml import (Kml, OverlayXY, ScreenXY, Units, RotationXY,
#                        AltitudeMode, Camera)

import cartopy.crs as ccrs
import cartopy

import shapefile as shp

# Tell matplotlib not to look for an X-window
# environ['QT_QPA_PLATFORM']='offscreen'

colors = plt.cm.get_cmap('inferno', lut=8)

# rcParams['font.family'] = 'sans-serif'
# Other font details
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'AppleGothic'
rcParams['font.size'] = config.LABEL_FONTSIZE*config.SCALE
rcParams['text.usetex'] = True
rcParams['mathtext.fontset'] = 'stixsans'
rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'
rcParams['axes.titlepad'] = config.TITLE_PAD/2

# some global variables
season_dict = {'DJF' : 'Winter', 'MAM' :  'Spring', 'JJA' : 'Summer',
               'SON' : 'Fall'}

def plot_TROPOMI(data, latlim, lonlim, res, figax=None, title='', genkml=False, vals='xch4', **plot_options):
    # Set default plot options]
    if 'vmin' not in plot_options:
        plot_options['vmin'] = 1700
    if 'vmax' not in plot_options:
        plot_options['vmax'] = 1900
    if 'cmap' not in plot_options:
        plot_options['cmap'] = 'inferno'

    res_prec = len(str(res).split('.')[1])

    # Define edges
    latlim = [round(l, res_prec) for l in latlim]
    lat_steps = int((latlim[1]-latlim[0])/res) + 1

    lonlim = [round(l, res_prec) for l in lonlim]
    lon_steps = int((lonlim[1]-lonlim[0])/res) + 1

    lats = np.around(np.linspace(latlim[0], latlim[1], lat_steps) + res/2,
                     res_prec + 1)
    lats_s = pd.DataFrame({'idx' : np.ones(len(lats)-1), 'lat' : lats[:-1]})
    lons = np.around(np.linspace(lonlim[0], lonlim[1], lon_steps) + res/2,
                     res_prec + 1)
    lons_s = pd.DataFrame({'idx' : np.ones(len(lons)-1), 'lon' : lons[:-1]})

    df = pd.merge(lats_s, lons_s, on='idx').drop(columns='idx')
    data = pd.merge(df, data, on=['lat', 'lon'], how='left')

    d_p = data.pivot(index='lat', columns='lon', values=vals)
    lon, lat = np.meshgrid(lons-res/2, lats-res/2)

    if figax is None:
        fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection' : ccrs.PlateCarree()})
    else:
        fig, ax = figax

    c = ax.pcolormesh(lons, lats, d_p,
                      vmin=plot_options['vmin'], vmax=plot_options['vmax'],
                      cmap=plot_options['cmap'], edgecolors=None, snap=True)

    return fig, ax, c

if __name__ == '__main__':
    # DATA_DIR = sys.argv[1]
    # REGION   = str(sys.argv[2].split(',')[0])
    DATA_DIR = '../observations/oversampling'
    PLOT_DIR = '../plots/'
    REGION = 'northamerica'
    print('Processing %s' % REGION)

    # latlim = np.array(sys.argv[2].split(',')[1:3]).astype(float)
    # lonlim = np.array(sys.argv[2].split(',')[3:5]).astype(float)
    latlim = np.array([10.375, 59.375]).astype(float)
    lonlim = np.array([-129.21875, -60.78125]).astype(float)
    res = 0.01
    count_min = 1

    files = listdir(DATA_DIR)
    files = [f for f in files if f[-3:] == 'csv']
    files.sort()

    for f in files:
        data = pd.read_csv(join(DATA_DIR, f))
        data = data[(data['lon'] >= lonlim[0]) &
                    (data['lon'] <= lonlim[1]) &
                    (data['lat'] >= latlim[0]) &
                    (data['lat'] <= latlim[1])]

        print('Plotting %s' % f)
        full_date = f.split('_')[0]
        year = full_date[:4]
        if len(full_date) == 6:
            month = full_date[-2:]
            month_name = datetime.date(1900, int(month), 1).strftime('%B')
            title = f'{month_name} {year}'
        else:
            continue
            # title = f'{season_dict[full_date[-3:]]} {year}'

        fig, ax = fp.get_figax(maps=True, lats=latlim, lons=lonlim)
        ax = fp.format_map(ax, latlim, lonlim, draw_labels=False)

        # calculate vmin and vmax
        vmin = 1800
        vmax = 1900
        plot_options = {'vmin' : vmin,
                        'vmax' : vmax,
                        'cmap' : 'plasma'}

        fig, ax, c = plot_TROPOMI(data, latlim, lonlim, res,
                                  figax=[fig, ax],
                                  vals='xch4',
                                  **plot_options)

        cax = fp.add_cax(fig, ax)
        cbar = fig.colorbar(c, cax=cax)
        cbar = fp.format_cbar(cbar, 'XCH4 (ppb)')
        fp.add_title(ax, title=title)

        # Save plot
        fp.save_fig(fig, PLOT_DIR, f'{REGION}_{full_date}')
        plt.close()
        print('\n')
