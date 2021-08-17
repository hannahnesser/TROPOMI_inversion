import xarray as xr
import numpy as np
from numpy.linalg import inv, norm, eigh
from scipy.sparse import diags, identity
from scipy.stats import linregress
import pandas as pd
import copy
from os.path import join

# clustering
from sklearn.cluster import KMeans

import math
import itertools
from collections import OrderedDict

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams, colorbar, colors
from matplotlib.colors import LinearSegmentedColormap
import mpl_toolkits.mplot3d
from matplotlib.collections import PolyCollection, LineCollection
import cartopy.crs as ccrs
import cartopy
import cartopy.feature
from cartopy.mpl.patch import geos_to_path

# sys.path.append('.')
import config

# Other font details
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'AppleGothic'
rcParams['font.size'] = config.LABEL_FONTSIZE*config.SCALE
rcParams['text.usetex'] = True
# rcParams['mathtext.fontset'] = 'stixsans'
rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'
rcParams['axes.titlepad'] = config.TITLE_PAD

from matplotlib.font_manager import findfont, FontProperties
font = findfont(FontProperties(family=['sans-serif']))

def color(k, cmap='plasma', lut=10):
    c = plt.cm.get_cmap(cmap, lut=lut)
    return colors.to_hex(c(k))

def cmap_trans(cmap, ncolors=300, nalpha=20):
    color_array = plt.get_cmap(cmap)(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.append(np.linspace(0.0, 1.0, nalpha),
                                  np.ones(ncolors-nalpha))

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name=str(cmap) + '_trans',colors=color_array)

    return map_object

def cmap_trans_center(cmap, ncolors=300, nalpha=20):
    color_array = plt.get_cmap(cmap)(range(ncolors))

    # change alpha values
    half_l = math.floor((ncolors - nalpha)/2)
    half_r = math.ceil((ncolors - nalpha)/2)
    color_array[:,-1] = np.concatenate((np.ones(half_l),
                                        np.linspace(1, 0, int(nalpha/2)),
                                        np.linspace(0, 1, int(nalpha/2)),
                                        np.ones(half_r)))

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name=str(cmap) + '_ctrans' ,
                                                   colors=color_array)

    return map_object

def get_figsize(aspect, rows, cols, **fig_kwargs):
    # Set default kwarg values
    max_width = fig_kwargs.get('max_width', config.BASE_WIDTH*config.SCALE)
    max_height = fig_kwargs.get('max_height', config.BASE_HEIGHT*config.SCALE)

    # Get figsize
    if aspect > 1: # width > height
        figsize = (max_width,
                   max_width/aspect)
    else: # width < height
        figsize = (max_height*aspect,
                   max_height)
    return figsize

def get_aspect(rows, cols, aspect=None,
               maps=False, lats=None, lons=None):
    if maps:
        aspect = np.cos(np.mean([np.min(lats), np.max(lats)])*np.pi/180)
        xsize = np.ptp([np.max(lons), np.min(lons)])*aspect
        ysize = np.ptp([np.max(lats), np.min(lats)])
        aspect = xsize/ysize
    return aspect*cols/rows

def make_axes(rows=1, cols=1, aspect=None,
              maps=False, lats=None, lons=None,
              **fig_kwargs):
    aspect = get_aspect(rows, cols, aspect, maps, lats, lons)
    figsize = get_figsize(aspect, rows, cols, **fig_kwargs)
    kw = {}
    if maps:
        kw['subplot_kw'] = {'projection' : ccrs.PlateCarree()}
    # if (rows + cols) > 2:
    #     kw['constrained_layout'] = True
        # figsize = tuple(f*1.5 for f in figsize)
    fig, ax = plt.subplots(rows, cols, figsize=figsize, **kw)
    # plt.subplots_adjust(right=1)
    return fig, ax

def add_cax(fig, ax, cbar_pad_inches=0.25, horizontal=False):
    # should be updated to infer cbar width and cbar_pad_inches
    if not horizontal:
        try:
            axis = ax[-1, -1]
            height = ax[0, -1].get_position().y1 - ax[-1, -1].get_position().y0
            ax_width = ax[0, -1].get_position().x1 - ax[0, 0].get_position().x0
        except IndexError:
            axis = ax[-1]
            # height = ax[-1].get_position().height
            height = ax[0].get_position().y1 - ax[-1].get_position().y0
            ax_width = ax[-1].get_position().x1 - ax[0].get_position().x0
        except TypeError:
            axis = ax
            height = ax.get_position().height
            ax_width = ax.get_position().width

        # x0
        fig_width = fig.get_size_inches()[0]
        x0_init = axis.get_position().x1
        x0 = (fig_width*x0_init + cbar_pad_inches*config.SCALE)/fig_width

        # y0
        y0 = axis.get_position().y0

        # Width
        width = 0.1*config.SCALE/fig_width
    else:
        try:
            axis = ax[-1, 0]
            width = ax[-1, -1].get_position().x1 - ax[-1, 0].get_position().x0
        except IndexError:
            axis = ax[0]
            width = ax[-1].get_position().x1 - ax[0].get_position().x0
        except TypeError:
            axis = ax
            width = ax.get_position().width

        # x0
        x0 = axis.get_position().x0

        # y0
        fig_height = fig.get_size_inches()[1]
        y0_init = axis.get_position().y0
        y0 = (fig_height*y0_init - cbar_pad_inches*config.SCALE)/fig_height

        # Height
        height = 0.1*config.SCALE/fig_height

    # Make axis
    cax = fig.add_axes([x0, y0, width, height])

    # if return_coords:
    #     return cax, x0, y0, width, height
    # else:
    return cax

def get_figax(rows=1, cols=1, aspect=1,
              maps=False, lats=None, lons=None,
              figax=None, **fig_kwargs):
    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = make_axes(rows, cols, aspect, maps, lats, lons, **fig_kwargs)

    if (rows > 1) or (cols > 1):
        for axis in ax.flatten():
            axis.set_facecolor('0.98')
        # plt.subplots_adjust(hspace=0.1, wspace=0.4)
    else:
        ax.set_facecolor('0.98')

    return fig, ax

def add_labels(ax, xlabel, ylabel, **label_kwargs):
    # Set default values
    label_kwargs['fontsize'] = label_kwargs.get('fontsize',
                                                config.LABEL_FONTSIZE*config.SCALE)
    label_kwargs['labelpad'] = label_kwargs.get('labelpad',
                                                config.LABEL_PAD)
    labelsize = label_kwargs.pop('labelsize',
                                 config.TICK_FONTSIZE*config.SCALE)

    # Set labels
    ax.set_xlabel(xlabel, **label_kwargs)
    ax.set_ylabel(ylabel, **label_kwargs)
    ax.tick_params(axis='both', which='both', labelsize=labelsize)
    return ax

def add_legend(ax, **legend_kwargs):
    legend_kwargs['frameon'] = legend_kwargs.get('frameon', False)
    legend_kwargs['fontsize'] = legend_kwargs.get('fontsize',
                                                  (config.LABEL_FONTSIZE*
                                                   config.SCALE))

    # Remove duplicates from legend
    handles, labels = ax.get_legend_handles_labels()
    labels = OrderedDict(zip(labels, handles))

    ax.legend(labels.values(), labels.keys(), **legend_kwargs)
    return ax

def add_title(ax, title, **title_kwargs):
    title_kwargs['y'] = title_kwargs.get('y', config.TITLE_LOC)
    title_kwargs['pad'] = title_kwargs.get('pad', config.TITLE_PAD)
    title_kwargs['fontsize'] = title_kwargs.get('fontsize',
                                                config.TITLE_FONTSIZE*config.SCALE)
    title_kwargs['va'] = title_kwargs.get('va', 'bottom')
    ax.set_title(title, **title_kwargs)
    return ax

def add_subtitle(ax, subtitle, **kwargs):
    y = kwargs.pop('y', config.TITLE_LOC)
    kwargs['ha'] = kwargs.get('ha', 'center')
    kwargs['va'] = kwargs.get('va', 'bottom')
    kwargs['xytext'] = kwargs.get('xytext', (0, config.TITLE_PAD))
    kwargs['textcoords'] = kwargs.get('textcoords', 'offset points')
    kwargs['fontsize'] = kwargs.get('fontsize',
                                    config.SUBTITLE_FONTSIZE*config.SCALE)
    subtitle = ax.annotate(s=subtitle, xy=(0.5, y), xycoords='axes fraction',
                           **kwargs)
    return ax

def get_square_limits(xdata, ydata, **kw):
    # Get data limits
    dmin = min(np.min(xdata), np.min(ydata))
    dmax = max(np.max(xdata), np.max(ydata))
    pad = (dmax - dmin)*0.05
    dmin -= pad
    dmax += pad

    try:
        # get lims
        ylim = kw.pop('lims')
        xlim = ylim
        xy = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    except:
        # set lims
        xlim = ylim = xy = (dmin, dmax)

    return xlim, ylim, xy, dmin, dmax

def format_map(ax, lats, lons,
               fontsize=config.TICK_FONTSIZE*config.SCALE,
               **gridline_kwargs):
    # Get kwargs
    gridline_kwargs['draw_labels'] = gridline_kwargs.get('draw_labels', True)
    gridline_kwargs['color'] = gridline_kwargs.get('color', 'grey')
    gridline_kwargs['linestyle'] = gridline_kwargs.get('linestyle', ':')

    # Format
    ax.set_ylim(min(lats), max(lats))
    ax.set_xlim(min(lons), max(lons))
    ax.add_feature(cartopy.feature.OCEAN, facecolor='0.98', linewidth=0.5)
    ax.add_feature(cartopy.feature.LAND, facecolor='0.98', linewidth=0.5)
    ax.add_feature(cartopy.feature.STATES, edgecolor='0.3', linewidth=0.2)
    ax.add_feature(cartopy.feature.LAKES, facecolor='none', edgecolor='0.3 ',
                   linewidth=0.2)
    ax.coastlines(color='0.2', linewidth=0.5)

    # gl = ax.gridlines(**gridline_kwargs)
    # gl.xlabel_style = {'fontsize' : fontsize}
    # gl.ylabel_style = {'fontsize' : fontsize}
    return ax

def format_cbar(cbar, cbar_title='', horizontal=False):
    # cbar.set_label(cbar_title, fontsize=BASEFONT*config.SCALE,
    #                labelpad=CBAR_config.LABEL_PAD)
            # x0
    if horizontal:
        x = 0.5
        y = -4
        rotation = 'horizontal'
    else:
        x = 6
        y = 0.5
        rotation = 'vertical'

    cbar.ax.tick_params(axis='both', which='both',
                        labelsize=config.TICK_FONTSIZE*config.SCALE)
    cbar.ax.text(x, y, cbar_title, ha='center', va='center', rotation=rotation,
                 fontsize=config.LABEL_FONTSIZE*config.SCALE,
                 transform=cbar.ax.transAxes)

    return cbar

def plot_one_to_one(ax):
    xlim, ylim, _, _, _ = get_square_limits(ax.get_xlim(),
                                            ax.get_ylim())
    ax.plot(xlim, xlim, c='0.1', lw=2, ls=':',
            alpha=0.5, zorder=0)
    return ax

def save_fig(fig, loc, name, **kwargs):
    fig.savefig(join(loc, name + '.png'),
                bbox_inches='tight', dpi=500,
                transparent=True, **kwargs)
    print('Saved %s' % name + '.png')
