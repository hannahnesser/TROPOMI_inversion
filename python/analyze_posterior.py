from os.path import join
from os import listdir
import sys
import glob
import copy
import math
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
from scipy.stats import probplot as qq
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Patch as patch
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import imageio
pd.set_option('display.max_columns', 10)

# Custom packages
sys.path.append('.')
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
import gcpy as gc
import troppy as tp
import invpy as ip
import format_plots as fp
import inversion_settings as s

## ------------------------------------------------------------------------ ##
## Directories
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'

## ------------------------------------------------------------------------ ##
## Set plotting preferences
## ------------------------------------------------------------------------ ##
# Colormaps
plasma_trans = fp.cmap_trans('plasma')
plasma_trans_r = fp.cmap_trans('plasma_r')
rdbu_trans = fp.cmap_trans_center('RdBu_r', nalpha=70)
r_trans = fp.cmap_trans('Reds', nalpha=100)
yor_trans = fp.cmap_trans('YlOrRd', nalpha=100)
viridis_trans_r = fp.cmap_trans('viridis_r')
viridis_trans = fp.cmap_trans('viridis')
magma_trans = fp.cmap_trans('magma')
# print(viridis_trans)

# sf_cmap_1 = plt.cm.Reds(np.linspace(0, 0.5, 256))
# sf_cmap_2 = plt.cm.Blues(np.linspace(0.5, 1, 256))
# sf_cmap = np.vstack((sf_cmap_1, sf_cmap_2))
# sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
# div_norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=6)

sf_cmap_1 = plt.cm.PuOr_r(np.linspace(0.2, 0.5, 256))
sf_cmap_2 = plt.cm.PuOr_r(np.linspace(0.5, 1, 256))
sf_cmap = np.vstack((sf_cmap_1, sf_cmap_2))
sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
div_norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=1, vmax=3)

diff_cmap_1 = plt.cm.bwr(np.linspace(0, 0.5, 256))
diff_cmap_2 = plt.cm.bwr(np.linspace(0.5, 1, 256))
diff_cmap = np.vstack((diff_cmap_1, diff_cmap_2))
diff_cmap = colors.LinearSegmentedColormap.from_list('diff_cmap', diff_cmap)
diff_div_norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=1)

# Small (i.e. non-default) figure settings
small_map_kwargs = {'draw_labels' : False}
small_fig_kwargs = {'max_width' : 4,
                    'max_height' : 3.5}

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# DOFS_filter
DOFS_filter = 0.05

# List emissions categories
emis = {'Total' : 'total', 'Oil and gas' : 'ong', 'Coal' : 'coal',
        'Livestock' : 'livestock', 'Landfills' : 'landfills', 
        'Wastewater' : 'wastewater', 'Wetlands' : 'wetlands', 
        'Other' : 'other'}
interest = {'total' : [('New York', [40, 45.5, -80, -73], [0, 100], None)],
            'livestock' : [('North Carolina', [33.5, 37, -84.5, -75.5], [0, 20],
                            None),
                           ('Midwest', [38.5, 46, -100, -81], [0, 20],
                            None),
                           ('Central Valley', [33, 40, -125, -115], [0, 40],
                            None)],
            'coal' : [('Illinois Basin', [36.25, 41.25, -91.5, -85], [0, 40],
                        None),
                      ('Powder River Basin', [40, 46, -111.5, -103.5], [0, 40],
                        None)],
            'ong' : [('Permian Basin', [27.5, 36, -107.3, -98], [0, 40],
                      None),
                     ('Haynesville Shale', [29, 33.5, -95, -89], [0, 40],
                      None),
                     ('Anadarko Shale', [33, 37.5, -103.5, -94], [0, 20], None),
                     ('Marcellus Shale', [37, 43, -83, -75], [0, 20], None),
                     ('Saskatchewan', [48, 56, -110.5, -101], [0, 20],
                      {'North Battleford' : [52.7765, -108.2975]})]}
           # 'wetlands' : None,
               # 'landfills' : None,
               # 'wastewater' : None,
               # 'other' : None}

# Define file names
# # f = 'bc_rg2rt_10t_w404_edf_bc0_rf0.25_sax0.75_poi80.0'
# f = 'rg2rt_10t_w404_edf_rf0.25_sax0.75_poi80.0'
# # f2 = 'bc_rg2rt_10t_w404_edf_rf0.25_sax0.75_poi80.0'
# xa_abs_file = 'xa_abs_w404_edf.nc'
# w_file = 'w_w404_edf.csv'
# optimize_BC = False
# # optimize_BC2 = True
# title_str = 'Standard'
ensemble = glob.glob(f'{data_dir}ensemble/xhat_fr2*')
ensemble.sort()
ensemble = [f.split('/')[-1][9:] for f in ensemble]
print(ensemble)

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc')

# Load prior (Mg/km2/yr)
xa_w404 = xr.open_dataarray(f'{data_dir}xa_abs_w404_edf.nc').reshape((-1, 1))
xa_w37 = xr.open_dataarray(f'{data_dir}xa_abs_w37_edf.nc').reshape((-1, 1))
xa_abs = {'w404_edf' : xa_w404, 'w37_edf' : xa_w37}
# xa_abs = xr.open_dataarray(f'{data_dir}{xa_abs_file}').values.reshape((-1, 1))
# soil = xr.open_dataarray(f'{data_dir}soil_abs.nc')
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load weighting matrix (Mg/yr)
w_w404 = pd.read_csv(f'{data_dir}w_w404_edf.csv')
w_w37 = pd.read_csv(f'{data_dir}w_w37_edf.csv')
w = {'w404_edf' : w_w404, 'w37_edf' : w_w37}
for wkey, ww in w.items():
    w[wkey]['total'] = ww.sum(axis=1)
    w[wkey]['net'] = xa_abs[wkey]*area
    w[wkey]['other'] = ww['other_bio'] + w['other_anth']
    w[wkey] = w[wkey].drop(columns=['other_bio', 'other_anth'])

print(w)
# w = pd.read_csv(f'{data_dir}{w_file}')
# w['total'] = w.sum(axis=1)
# w['net'] = xa_abs*area
# w['other'] = w['other_bio'] + w['other_anth']
# w = w.drop(columns=['other_bio', 'other_anth'])

# # Sectoral masks
# w_mask = copy.deepcopy(w)
# w_mask = w_mask.where(w/1e3 > 0.5, 0)
# w_mask = w_mask.where(w/1e3 <= 0.5, 1)

# # Load posterior and DOFS
# dofs = np.load(f'{data_dir}ensemble/dofs2_{f}.npy').reshape((-1, 1))
# xhat = np.load(f'{data_dir}ensemble/xhat_fr2_{f}.npy').reshape((-1, 1))

# # BC alteration
# if optimize_BC:
#     print('-'*75)
#     print('Boundary condition optimization')
#     print(' N E S W')
#     print('xhat : ', xhat[-4:].reshape(-1,))
#     print('dofs : ', dofs[-4:].reshape(-1,))
#     print('-'*75)
#     xhat = xhat[:-4]
#     dofs = dofs[:-4]

# # Print information
# print('-'*75)
# print(f'We optimize {(dofs >= DOFS_filter).sum():d} grid cells, including {xa_abs[dofs >= DOFS_filter].sum():.2E}/{xa_abs.sum():.2E} = {(xa_abs[dofs >= DOFS_filter].sum()/xa_abs.sum()*100):.2f}% of prior\nemissions. This produces {dofs[dofs >= DOFS_filter].sum():.2f} ({dofs.sum():.2f}) DOFS with an xhat range of {xhat.min():.2f}\nto {xhat.max():.2f}. There are {len(xhat[xhat < 0]):d} negative values.')
# print('-'*75)

# # Filter on DOFS filter
# xhat[dofs < DOFS_filter] = 1
# dofs[dofs < DOFS_filter] = 0

# # Calculate xhat abs
# xhat_abs = (xhat*xa_abs)

# # Get observations
# y = xr.open_dataarray(f'{data_dir}y.nc')
# ya = xr.open_dataarray(f'{data_dir}ya.nc')

# # Get county outlines for high resolution results
# reader = shpreader.Reader(f'{data_dir}counties/countyl010g.shp')
# counties = list(reader.geometries())
# COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

# # # # Get standard results
# # dofs2 = np.load(f'{data_dir}posterior/dofs2_{f2}.npy').reshape((-1, 1))
# # xhat2 = np.load(f'{data_dir}posterior/xhat_fr2_{f2}.npy').reshape((-1, 1))
# # xhat2[dofs2 < DOFS_filter] = 1
# # dofs2[dofs2 < DOFS_filter] = 0
# # if optimize_BC2:
# #     print('-'*75)
# #     print('Boundary condition optimization')
# #     print(' N E S W')
# #     print('xhat : ', xhat2[-4:].reshape(-1,))
# #     print('dofs : ', dofs2[-4:].reshape(-1,))
# #     print('-'*75)
# #     xhat2 = xhat2[:-4]
# #     dofs2 = dofs2[:-4]

# ## ------------------------------------------------------------------------ ##
# ## Second estimate posterior
# ## ------------------------------------------------------------------------ ##
# fig, ax = ip.plot_posterior(xhat, dofs, clusters)
# fp.save_fig(fig, plot_dir, f'fig_est2_posterior_{f}')

# # Make sectoral correction maps
# d_avker_cbar_kwargs = {'title' : r'$\Delta \partial\hat{x}_i/\partial x_i$'}
# avker_cbar_kwargs = {'title' : r'$\partial\hat{x}_i/\partial x_i$',
#                      'horizontal' : False}
# avker_kwargs = {'cmap' : plasma_trans, 'vmin' : 0, 'vmax' : 1,
#                 'cbar_kwargs' : avker_cbar_kwargs,
#                 'map_kwargs' : small_map_kwargs}

# d_xhat_cbar_kwargs = {'title' : r'$\Delta$ Scale factor'}
# xhat_cbar_kwargs = {'title' : r'Scale factor', 'horizontal' : False}
# xhat_kwargs = {'cmap' : sf_cmap, 'norm' : div_norm,
#                # 'cmap' : 'RdBu_r', 'vmin' : 0, 'vmax' : 6,
#                'default_value' : 1,
#                'cbar_kwargs' : xhat_cbar_kwargs,
#                'map_kwargs' : small_map_kwargs}

# # Plot averaging kernel sensitivities
# title = f'Averaging kernel sensitivities' # ({f}\%)'
# fig, ax, c = ip.plot_state(dofs, clusters, title=title,
#                            **avker_kwargs)
# ax.text(0.025, 0.05, 'DOFS = %d' % round(dofs.sum()),
#         fontsize=config.LABEL_FONTSIZE*config.SCALE,
#         transform=ax.transAxes)
# fp.save_fig(fig, plot_dir, f'fig_est2_dofs_{f}')
# plt.close()

# # Plot posterior scaling factors
# title = f'Posterior emission scale factors' # ({f}\%)'
# fig, ax, c = ip.plot_state(xhat, clusters, title=title,
#                            **xhat_kwargs)
# fp.save_fig(fig, plot_dir, f'fig_est2_xhat_{f}')
# plt.close()


# # ## Difference plots
# # title = f'Wetlands/4.04 - reduced wetland ensemble\nposterior emission scale factors'
# # fig, ax, c = ip.plot_state(xhat - xhat2, clusters, title=title,
# #                            cmap='RdBu_r', vmin=-0.5, vmax=0.5, default_value=0,
# #                            cbar_kwargs=d_xhat_cbar_kwargs,
# #                            map_kwargs=small_map_kwargs)
# # fp.save_fig(fig, plot_dir, f'fig_est2_xhat_diff_{f}')
# # plt.close()

# # title = f'Wetlands/4.04 - reduced wetland ensemble\naveraging kernel sensitivities'
# # fig, ax, c = ip.plot_state(dofs - dofs2, clusters, title=title,
# #                            cmap='RdBu_r', vmin=-0.75, vmax=0.75, default_value=0,
# #                            cbar_kwargs=d_avker_cbar_kwargs,
# #                            map_kwargs=small_map_kwargs)
# # ax.text(0.025, 0.05, r'$\Delta$DOFS = %d' % round(dofs.sum() - dofs2.sum()),
# #         fontsize=config.LABEL_FONTSIZE*config.SCALE,
# #         transform=ax.transAxes)
# # fp.save_fig(fig, plot_dir, f'fig_est2_dofs_diff_{f}')
# # plt.close()

# # ## ------------------------------------------------------------------------ ##
# # ## Sectoral attribution maps
# # ## ------------------------------------------------------------------------ ##
# # # Get sectoral xhat difference from prior (in units Mg/km2/yr)
# # xhat_diff_sect =  w*(xhat - 1)/area

# # ul = 10
# # ncategory = len(emis)
# # fig, ax = fp.get_figax(rows=2, cols=3, maps=True,
# #                        lats=clusters.lat, lons=clusters.lon)
# # plt.subplots_adjust(hspace=0.3)

# # d_xhat_cbar_kwargs = {'title' : r'$\Delta$ Emissions (Mg km$^{-2}$ a$^{-1}$)'}
# # d_xhat_kwargs = {'cmap' : 'RdBu_r', 
# #                'cbar_kwargs' : d_xhat_cbar_kwargs,
# #                'map_kwargs' : small_map_kwargs}


# # for axis, (title, emis_label) in zip(ax.flatten(), emis.items()):
# #     # Get sectoral values (Mg/km2/yr)
# #     xhat_diff_sect_i = xhat_diff_sect[emis_label].values

# #     if title != 'Total':
# #         xhat_sect_i = (xhat - 1)*w_mask[emis_label].values.reshape(-1, 1) + 1
# #         fig2b, ax2b, c2b = ip.plot_state(xhat_sect_i, clusters, 
# #                                          title=f'{title} scaling factors',
# #                                          **xhat_kwargs)
# #         fp.save_fig(fig2b, plot_dir, f'fig_est2_xhat_sf_{emis_label}_{f}')


# #     fig, axis, c = ip.plot_state(xhat_diff_sect_i, clusters,
# #                                  fig_kwargs={'figax' : [fig, axis]},
# #                                  title=title, default_value=0,
# #                                  vmin=-ul, vmax=ul,
# #                                  cmap='RdBu_r', cbar=False)

# #     fig2, ax2, c2 = ip.plot_state(xhat_diff_sect_i, clusters, default_value=0,
# #                                  vmin=-ul, vmax=ul, title=title, 
# #                                  **d_xhat_kwargs)

# #     d_xhat_kwargs['cbar_kwargs'] = {'title' : r'$\Delta$ Emissions (Mg km$^{-2}$ a$^{-1}$)'}

# #     # Calculate annual difference
# #     diff_sec_tot = (xhat_diff_sect_i.reshape((-1, 1))*area)
# #     diff_sec_tot_pos = diff_sec_tot[diff_sec_tot > 0].sum()*1e-6
# #     diff_sec_tot_neg = diff_sec_tot[diff_sec_tot <= 0].sum()*1e-6
# #     diff_sec_tot = diff_sec_tot.sum()*1e-6
# #     axis.text(0.05, 0.05, f'{diff_sec_tot:.2f} Tg', ha='left', va='bottom',
# #               fontsize=config.LABEL_FONTSIZE*config.SCALE,
# #               transform=axis.transAxes)
# #     ax2.text(0.05, 0.05, f'{diff_sec_tot:.2f} Tg ({diff_sec_tot_neg:.2f} Tg, {diff_sec_tot_pos:.2f} Tg)', ha='left', va='bottom',
# #              fontsize=config.LABEL_FONTSIZE*config.SCALE,
# #              transform=ax2.transAxes)

# #     # Save out individual plot
# #     fp.save_fig(fig2, plot_dir, f'fig_est2_xhat_{emis_label}_{f}')
# #     plt.close(fig2)

# #     # Plot regions of interst
# #     if (emis_label in interest.keys()):
# #         for j, reg in enumerate(interest[emis_label]):
# #             fig3, ax3 = fp.get_figax(rows=2, cols=2, maps=True,
# #                                      lats=reg[1][:2], lons=reg[1][2:],
# #                                      max_height=config.BASE_WIDTH*config.SCALE)
# #             figw, figh = fig3.get_size_inches()
# #             plt.subplots_adjust(hspace=0.5/figh, wspace=6/figw)
# #             # fig3.set_figheight(figh + 0.5)
# #             # fig3.set_figwidth(figw + 7)

# #             c = clusters.where((clusters.lat > reg[1][0]) &
# #                                (clusters.lat < reg[1][1]) &
# #                                (clusters.lon > reg[1][2]) &
# #                                (clusters.lon < reg[1][3]), drop=True)
# #             c_idx = (c.values[c.values > 0] - 1).astype(int)

# #             fig3, ax3[0, 0], c32 = ip.plot_state(
# #                 w.loc[:, emis_label]/area.reshape(-1,), clusters, 
# #                 title='Sector prior', cbar=False, cmap=viridis_trans, 
# #                 vmin=reg[2][0], vmax=reg[2][1], default_value=0, 
# #                 fig_kwargs={'figax' : [fig3, ax3[0, 0]]}, 
# #                 map_kwargs=small_map_kwargs)
# #             fig3, ax3[0, 1], _ = ip.plot_state(
# #                 xhat_diff_sect_i + w.loc[:, emis_label]/area.reshape(-1,),
# #                 clusters, title='Sector posterior', cbar=False, 
# #                 cmap=viridis_trans, vmin=reg[2][0], vmax=reg[2][1],
# #                 default_value=0, fig_kwargs={'figax' : [fig3, ax3[0, 1]]},
# #                 map_kwargs=small_map_kwargs)
# #             fig3, ax3[1, 0], c30 = ip.plot_state(
# #                 xhat, clusters, title=f'Scale factors', 
# #                 cbar=False, cmap=sf_cmap, norm=div_norm, default_value=1,
# #                 fig_kwargs={'figax' : [fig3, ax3[1, 0]]},
# #                 map_kwargs=small_map_kwargs)
# #             fig3, ax3[1, 1], c31 = ip.plot_state(
# #                 xhat_diff_sect_i, clusters, title='Emissions change', 
# #                 cbar=False, vmin=-reg[2][1]/4, vmax=reg[2][1]/4,
# #                 fig_kwargs={'figax' : [fig3, ax3[1, 1]]}, **d_xhat_kwargs)

# #             tt = (xhat_diff_sect_i.reshape(-1,)*area.reshape(-1,))[c_idx].sum()*1e-6
# #             ax3[1, 1].text(0.05, 0.05, f'{tt:.1f} Tg/yr',
# #                             fontsize=config.LABEL_FONTSIZE*config.SCALE,
# #                             transform=ax3[1, 1].transAxes)

# #             if reg[3] is not None:
# #                 for label, point in reg[3].items():
# #                     for axis3 in ax3.flatten():
# #                         axis3.scatter(point[1], point[0], s=10, c='black')
# #                         axis3.text(point[1], point[0], r'$~~$'f'{label}',
# #                                    fontsize=config.LABEL_FONTSIZE*config.SCALE/2)

# #             for k, axis3 in enumerate(ax3.flatten()):
# #                 axis3.set_ylim(reg[1][:2])
# #                 axis3.set_xlim(reg[1][2:])
# #                 axis3.add_feature(COUNTIES, facecolor='none', 
# #                                    edgecolor='0.1', linewidth=0.1)
# #             cax30 = fp.add_cax(fig3, ax3[1, 0], cbar_pad_inches=0.1)
# #             cb30 = fig.colorbar(c30, cax=cax30,
# #                                 ticks=np.arange(0, 3, 1))
# #             cb30 = fp.format_cbar(cb30,
# #                                  cbar_title='Scale factor', x=4)

# #             cax31 = fp.add_cax(fig3, ax3[1, 1], cbar_pad_inches=0.1)
# #             cb31 = fig.colorbar(c31, cax=cax31,
# #                                 ticks=np.arange(-reg[2][1]/4, reg[2][1]/4+1, 
# #                                                 reg[2][1]/8))
# #             cbar_str = r'$\Delta$ Emissions\\(Mg km$^{-2}$ a$^{-1}$)'
# #             cb31 = fp.format_cbar(cb31,
# #                                  cbar_title=cbar_str, x=4)
# #             if emis_label == 'total':
# #                 step = 10
# #             else:
# #                 step = 5
# #             cax32 = fp.add_cax(fig3, ax3[0, 1], cbar_pad_inches=0.1)
# #             cb32 = fig.colorbar(
# #                 c32, cax=cax32, 
# #                 ticks=np.arange(reg[2][0], reg[2][1]+step, step))
# #             cb32 = fp.format_cbar(
# #                 cb32, cbar_title=r'Emissions\\(Mg km$^{-2}$ a$^{-1}$)', x=4)

# #             fp.save_fig(fig3, plot_dir, 
# #                         f'fig_est2_xhat_{emis_label}_reg{j}_{f}')
# #             plt.close(fig3)

# # cax = fp.add_cax(fig, ax, cbar_pad_inches=0.3, horizontal=True)
# # cb = fig.colorbar(c, cax=cax, ticks=np.arange(-ul, ul+1, ul/2),
# #                   orientation='horizontal')
# # cb = fp.format_cbar(cb, cbar_title=r'$\Delta$ Emissions (Mg km$^{-2}$ a$^{-1}$)',
# #                     horizontal=True)
# # # axis = fp.add_title(axis, titles[i])

# # fp.save_fig(fig, plot_dir, f'fig_est2_xhat_sectoral_{f}')
# # plt.close(fig)

# ## ------------------------------------------------------------------------ ##
# ## Sectoral attribution bar chart
# ## ------------------------------------------------------------------------ ##
# # w = w*area.values*1e-6 # Convert to Tg/yr

# # Open masks and create a total_mask array as well as a mask dictionary
# mex_mask = np.load(f'{data_dir}Mexico_mask.npy').reshape((-1, 1))
# can_mask = np.load(f'{data_dir}Canada_mask.npy').reshape((-1, 1))
# conus_mask = np.load(f'{data_dir}CONUS_mask.npy').reshape((-1, 1))
# total_mask = mex_mask + can_mask + conus_mask
# masks = {'Canada' : can_mask, 'CONUS' : conus_mask, 'Mexico' : mex_mask}

# print('-'*75)
# print(f'Total prior emissions (Tg/yr)    : {(xa_abs*area*1e-6).sum():.2f}')
# print(f'Total posterior emissions        : {(xhat_abs*area*1e-6).sum():.2f}')
# print(f'Difference                       : {((xhat_abs - xa_abs)*area*1e-6).sum():.2f}')
# print(f'Maximum scale factor             : {xhat.max():.2f}')
# print(f'Minimum scale factor             : {xhat.min():.2f}')
# print(f'Number of negative scale factors : {(xhat < 0).sum()}')
# print('-'*75)

# xiao = {'Canada' : {'livestock' : [1.4], 'ong' : [3.2], 'coal' : [0.0], 
#                     'landfills' : [0.69], 'wastewater' : [0.0], 
#                     'wetlands' : [9.9], 'other' : [1.01]},
#         'CONUS' : {'livestock' : [10.6], 'ong' : [14.5], 'coal' : [2.8], 
#                    'landfills' : [7.2], 'wastewater' : [0.63], 
#                    'wetlands' : [8.4], 'other' : [2.17]},
#         'Mexico' : {'livestock' : [2.5], 'ong' : [1.26], 'coal' : [0.26], 
#                     'landfills' : [1.0], 'wastewater' : [0.8], 
#                     'wetlands' : [0.57], 'other' : [0.41]}}

# country_emis = {}
# for country, mask in masks.items():
#     sect_emis = pd.DataFrame(columns=['prior', 'post', 'diff',
#                                       'prior_sub', 'post_sub', 'diff_sub', 
#                                       'lu'])
#                                       #'diff_sub_pos', 'diff_sub_neg'])
#     w_c = copy.deepcopy(w)*mask*1e-6 # convert to Tg/yr in the country

#     # Get prior and posterior absolute emissions
#     sect_emis['prior'] = w_c.sum(axis=0)
#     sect_emis['post'] = (w_c*xhat).sum(axis=0)
#     sect_emis['diff'] = sect_emis['post'] - sect_emis['prior']

#     # Get prior and posterior absolute emissions only where DOFS > DOFS_filter
#     sect_emis['prior_sub'] = w_c.loc[dofs >= DOFS_filter, :].sum(axis=0)
#     sect_emis['post_sub'] = (w_c*xhat).loc[dofs >= DOFS_filter, :].sum(axis=0)
#     sect_emis['diff_sub'] = sect_emis['post_sub'] - sect_emis['prior_sub']

#     # Get positive and negative adjustments
#     sect_emis['diff_pos'] = (w_c*(xhat - 1))[xhat >= 1].sum(axis=0)
#     sect_emis['diff_neg'] = np.abs(w_c*(xhat - 1))[xhat < 1].sum(axis=0)

#     # Reorder sect_emis
#     sect_emis = sect_emis.loc[list(emis.values())[1:]]

#     # Add in Xiao
#     sect_emis['lu'] = pd.DataFrame.from_dict(xiao[country], orient='index').loc[list(emis.values())[1:]]

#     country_emis[country] = sect_emis

#     # Print info
#     c_dofs = dofs*mask
#     c_emis = w_c['total']
#     print('-'*75)
#     print(f'In {country}, we optimize {(c_dofs >= DOFS_filter).sum():d} grid cells including {c_emis[dofs.reshape(-1,) >= DOFS_filter].sum():.2E}/{c_emis.sum():.2E} = {(c_emis[dofs.reshape(-1,) >= DOFS_filter].sum()/c_emis.sum()*100):.2f}%\nof emissions. This produces {c_dofs.sum():.2f} DOFS with an xhat range of {(xhat*mask).min():.2f} to {(xhat*mask).max():.2f}.\nThere are {len((xhat*mask)[(xhat*mask) < 0]):d} negative values.')
#     print('-'*75)

# # Plot histogram (at least take #1)
# xs = np.arange(0, len(emis.keys()) - 1)
# fig, axis = fp.get_figax(aspect=2, cols=3)
# j = 0
# lims = [[0, 4], [0, 15], [0, 4]]
# country_emis_sub = {key : country_emis[key]
#                     for key in ['Canada', 'CONUS', 'Mexico']}
# for country, sect_emis in country_emis_sub.items():
#     ax = axis[j]
#     diff_summ = sect_emis['diff'].sum()
#     ax.text(0.05, 0.95, country, ha='left', va='top',
#             fontsize=config.LABEL_FONTSIZE*config.SCALE,
#             transform=ax.transAxes)
#     ax.text(0.95, 0.95, f'({diff_summ:0.2f} Tg)', ha='right', va='top',
#             fontsize=config.TICK_FONTSIZE*config.SCALE/1.5,
#             transform=ax.transAxes)
#     ax.bar(xs - 0.12, sect_emis['prior'], width=0.2,
#            color='white', edgecolor=fp.color(2*j+2), label='Prior (all)')
#     ax.bar(xs - 0.12, sect_emis['prior_sub'], width=0.2, color=fp.color(2*j+2),
#            label='Prior (optimized)')
#     # ax.bar(xs - 0.12, sect_emis['prior_sub2'], width=0.2, color=fp.color(2*j+2),
#     #        label='Prior (more optimized)')
#     ax.bar(xs + 0.12, sect_emis['post'], width=0.2,
#            color='white', edgecolor=fp.color(2*j+2), alpha=0.5,
#            label='Posterior (all)')
#     ax.bar(xs + 0.12, sect_emis['post_sub'], width=0.2,
#            color=fp.color(2*j+2), alpha=0.5, 
#            label='Posterior (optimized)')

#     ax.bar(xs + 0.35, sect_emis['lu'], width=0.1, 
#           color=fp.color(2*j+2), alpha=0.25, label='Lu et al. (2022)')

#     # ax.bar(xs + 0.11, sect_emis['post_sub2'], width=0.2,
#     #        color=fp.color(2*j+2), alpha=0.5, 
#     #        label='Posterior (more optimized)')
#     # print(sect_emis)
#     # for i, e in enumerate(list(emis.values())[1:]):
#     #     ax.arrow(x=i + 0.06, y=sect_emis.at[e, 'prior'], dx=0, 
#     #              dy=sect_emis.at[e, 'diff_pos'],
#     #              color=fp.color(2*j+2), alpha=1, width=0.002, 
#     #              head_width=0.0025)
#     #     ax.arrow(x=i + 0.18, 
#     #              y=sect_emis.at[e, 'prior'] + sect_emis.at[e, 'diff_pos'],
#     #              dx=0, dy=-sect_emis.at[e, 'diff_neg'],
#     #              color=fp.color(2*j+2), alpha=1, width=0.002, 
#     #              head_width=0.0025)
#     ax.set_ylim(lims[j])
#     # ax.bar(xs + 0.2, sect_emis['post_sub_pos'], width=0.1, color=fp.color(7))
#     # ax = fp.add_legend(ax)

#     j += 1

# axis[0] = fp.add_labels(axis[0], '', 'Emissions\n' + r'[Tg a$^{-1}$]',
#                         fontsize=config.TICK_FONTSIZE*config.SCALE/1.5,
#                         labelsize=config.TICK_FONTSIZE*config.SCALE/1.5)
# for i in range(3):
#     if i > 0:
#         axis[i] = fp.add_labels(axis[i], '', '',
#                                 labelsize=config.TICK_FONTSIZE*config.SCALE/1.5)
#     axis[i].set_xticks(xs)
#     axis[i].set_xticklabels(list(emis.keys())[1:], rotation=90, ha='center',
#                             fontsize=config.TICK_FONTSIZE*config.SCALE/1.5)
#     axis[i].set_xlim(-0.5, 7)

# custom_patches = [patch(color=fp.color(0), alpha=1), 
#                   patch(color=fp.color(0), alpha=0.5),
#                   # patch(color=fp.color(0), alpha=0.25),
#                   # patch(color='grey'), 
#                   patch(facecolor='white', edgecolor=fp.color(0)),
#                   patch(color=fp.color(0), alpha=0.25)]
# custom_labels = ['Prior', 'Posterior', #'Lu et al. (2022)',
#                  r'Not optimized (A$_{ii}$ $<$'f' {DOFS_filter})',
#                  'Lu et al. (2022)']
#                  # r'Averaging kernel sensitivities $\ge$'f' {DOFS_filter}', 
#                  # r'Averaging kernel sensitivities $<$'f' {DOFS_filter}']
# # custom_labels = ['Numerical solution', 'Predicted solution']
# # handles, labels = ax_summ.get_legend_handles_labels()
# # custom_lines.extend(handles)
# # custom_labels.extend(labels)
# fp.add_legend(axis[1], handles=custom_patches, labels=custom_labels,
#               bbox_to_anchor=(0.5, -0.75), loc='upper center', ncol=2,
#               fontsize=config.TICK_FONTSIZE*config.SCALE/1.5)


# fp.save_fig(fig, plot_dir, f'fig_sectoral_bar_{f}')
# plt.close()