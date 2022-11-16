from os.path import join
from os import listdir
import sys
import glob
from copy import deepcopy as dc
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

# Plotting preferences
d_avker_cbar_kwargs = {'title' : r'$\Delta \partial\hat{x}_i/\partial x_i$'}
avker_cbar_kwargs = {'title' : r'$\partial\hat{x}_i/\partial x_i$',
                     'horizontal' : False}
avker_kwargs = {'cmap' : plasma_trans, 'vmin' : 0, 'vmax' : 1,
                'cbar_kwargs' : avker_cbar_kwargs,
                'map_kwargs' : small_map_kwargs}

d_xhat_cbar_kwargs = {'title' : r'$\Delta$ Scale factor'}
xhat_cbar_kwargs = {'title' : r'Scale factor', 'horizontal' : False}
xhat_kwargs = {'cmap' : sf_cmap, 'norm' : div_norm,
               # 'cmap' : 'RdBu_r', 'vmin' : 0, 'vmax' : 6,
               'default_value' : 1,
               'cbar_kwargs' : xhat_cbar_kwargs,
               'map_kwargs' : small_map_kwargs}

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

# Get ensemble values
ensemble = glob.glob(f'{data_dir}ensemble/xhat_fr2*')
ensemble.sort()
ensemble = [f.split('/')[-1][9:] for f in ensemble]

# ID two priors
w37_cols = [s[:-12] for s in ensemble if 'w37' in s]
w404_cols = [s[:-12] for s in ensemble if 'w404' in s]

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Default
optimize_BC = False

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc')

# Load prior (Mg/km2/yr)
xa_w404 = xr.open_dataarray(f'{data_dir}xa_abs_w404_edf.nc').values
xa_w37 = xr.open_dataarray(f'{data_dir}xa_abs_w37_edf.nc').values
xa_abs_dict = {'w404_edf' : xa_w404, 'w37_edf' : xa_w37}
area = xr.open_dataarray(f'{data_dir}area.nc').values
nstate = area.shape[0]

# Load weighting matrix (Mg/yr)
w_w404 = pd.read_csv(f'{data_dir}w_w404_edf.csv')
w_w37 = pd.read_csv(f'{data_dir}w_w37_edf.csv')
w = {'w404_edf' : w_w404, 'w37_edf' : w_w37}
w_mask = {}
for wkey, ww in w.items():
    w[wkey]['total'] = w[wkey].sum(axis=1)
    w[wkey]['net'] = xa_abs_dict[wkey]*area
    w[wkey]['other'] = w[wkey]['other_bio'] + w[wkey]['other_anth']
    w[wkey] = w[wkey].drop(columns=['other_bio', 'other_anth'])
    w[wkey] = w[wkey].T

    # Sectoral masks
    w_mask[wkey] = dc(w[wkey])
    w_mask[wkey] = w_mask[wkey].where(w[wkey]/1e3 > 0.5, 0)
    w_mask[wkey] = w_mask[wkey].where(w[wkey]/1e3 <= 0.5, 1)

# Create dataframes for the ensemble data
dofs = pd.DataFrame(columns=[s[:-12] for s in ensemble])
xa_abs = pd.DataFrame(columns=[s[:-12] for s in ensemble])
xhat = pd.DataFrame(columns=[s[:-12] for s in ensemble])
xhat_abs = pd.DataFrame(columns=[s[:-12] for s in ensemble])
shat_abs = pd.DataFrame(columns=[s[:-12] for s in ensemble])
dofs_bc = pd.DataFrame(columns=[s[:-12] for s in ensemble if s[:2] == 'bc'],
                       index=['N', 'S', 'E', 'W'])
xhat_bc = pd.DataFrame(columns=[s[:-12] for s in ensemble if s[:2] == 'bc'],
                       index=['N', 'S', 'E', 'W'])
shat_bc = pd.DataFrame(columns=[s[:-12] for s in ensemble if s[:2] == 'bc'],
                       index=['N', 'S', 'E', 'W'])

# Iterate throuugh the ensemble to load the data
for suff in ensemble:
    # Get string information about the ensemble member
    short_suff = suff.split('rg2rt_10t_')[-1].split('_bc0')[0]
    sa_scale = float(suff.split('_sax')[-1].split('_')[0])

    # Load the files
    dofs_s = np.load(f'{data_dir}ensemble/dofs2_{suff}')
    xhat_s = np.load(f'{data_dir}ensemble/xhat_fr2_{suff}')
    shat_s = np.load(f'{data_dir}ensemble/shat_kpi2_{suff}')

    # Filter on the DOFS filter
    xhat_s[dofs_s < DOFS_filter] = 1
    dofs_s[dofs_s < DOFS_filter] = 0
    shat_s[dofs_s < DOFS_filter] = sa_scale**2

    # If the ensemble member optimizes the boundary conditions, save
    # out the boundary condition and grid cell elements separately
    if suff[:2] == 'bc':
        optimize_bc = True

        # Add BC elements
        dofs_bc[suff[:-12]] = dofs_s[-4:]
        xhat_bc[suff[:-12]] = xhat_s[-4:]
        shat_bc[suff[:-12]] = shat_s[-4:]

        # Shorten results
        xhat_s = xhat_s[:-4]
        dofs_s = dofs_s[:-4]
        shat_s = shat_s[:-4]

    # Save out the resulting values to the dataframe
    dofs[suff[:-12]] = dofs_s
    xa_abs[suff[:-12]] = xa_abs_dict[short_suff]
    xhat[suff[:-12]] = xhat_s
    xhat_abs[suff[:-12]] = xhat_s*xa_abs_dict[short_suff]
    shat_abs[suff[:-12]] = shat_s*(xa_abs_dict[short_suff]**2)

# Calculate the statistics of the posterior solution
dofs_mean = dofs.mean(axis=1)
xa_abs_mean = xa_abs.mean(axis=1)
xhat_abs_mean = xhat_abs.mean(axis=1)
xhat_mean = xhat_abs_mean/xa_abs_mean

# # Calculate the posterior error covariance from the ensemble
# shat_e = (xhat_abs - xhat_abs_mean) @ (xhat_abs - xhat_abs_mean).T
# print(shat_e.shape)

# Get BC statistics
bc_stats = pd.DataFrame({'mean' : xhat_bc.mean(axis=1),
                         'min' : xhat_bc.min(axis=1),
                         'max' : xhat_bc.max(axis=1)})

# BC alteration
if optimize_bc:
    print('-'*75)
    print('Boundary condition optimization')
    print(bc_stats.round(2))
    print('-'*75)

# Calculate statistics and print results
xa_abs_opt_frac = xa_abs[dofs > DOFS_filter].sum(axis=0)/xa_abs.sum(axis=0)
xhat_abs_tot = (xhat_abs*area.reshape((-1, 1))*1e-6).sum(axis=0)
n_opt = (dofs > DOFS_filter).sum(axis=0)
negs = (xhat_abs[xa_abs >= 0] < 0).sum(axis=0)

print(f'We optimize {n_opt.mean():.0f} ({n_opt.min():d}, {n_opt.max():d}) grid cells if we analyze each ensemble member\nindividually. If we consider those grid cells that are included in the\nensemble mean, we optimize {(dofs_mean > DOFS_filter).sum():d} ({(dofs_mean > 0).sum():d}) grid cells.')
print('')
print(f'Across the ensemble, we optimize {(xa_abs_opt_frac.mean()*100):.1f} ({(xa_abs_opt_frac.min()*100):.1f} - {(xa_abs_opt_frac.max()*100):.1f})% of prior emissions.')
print('')
print(f'This produces a mean of of {dofs.sum(axis=0).mean():.2f} ({dofs.sum(axis=0).min():.2f}, {dofs.sum(axis=0).max():.2f}) DOFS.')
print('')
print(f'There are {negs.mean():.0f} ({negs.min():d}, {negs.max():d}) new negative values. If we consider those grid cells\nthat are included in the ensemble mean, there are {(xhat_abs_mean[xa_abs_mean >= 0] < 0).sum():d} new negative values.')
print('')
# print(f'Total prior emissions (Tg/yr)    : {(xa_abs_mean*area*1e-6).sum():.2f}')
print(f'Total posterior emissions : {xhat_abs_tot.mean():.2f} ({xhat_abs_tot.min():.2f}, {xhat_abs_tot.max():.2f}) Tg/yr')
print('-'*75)

# Get county outlines for high resolution results
reader = shpreader.Reader(f'{data_dir}counties/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

## ------------------------------------------------------------------------ ##
## Plot pposterior
## ------------------------------------------------------------------------ ##
fig, ax = ip.plot_posterior(xhat_mean, dofs_mean, clusters)
fp.save_fig(fig, plot_dir, f'fig_est2_posterior_ensemble')

# # Plot averaging kernel sensitivities
# title = f'Averaging kernel sensitivities' # ({f}\%)'
# fig, ax, c = ip.plot_state(dofs_mean, clusters, title=title,
#                            **avker_kwargs)
# ax.text(0.025, 0.05, 'DOFS = %d' % round(dofs_mean.sum()),
#         fontsize=config.LABEL_FONTSIZE*config.SCALE,
#         transform=ax.transAxes)
# fp.save_fig(fig, plot_dir, f'fig_est2_dofs_ensemble')
# plt.close()

# # Plot posterior scaling factors
# title = f'Posterior emission scale factors' # ({f}\%)'
# fig, ax, c = ip.plot_state(xhat_mean, clusters, title=title,
#                            **xhat_kwargs)
# fp.save_fig(fig, plot_dir, f'fig_est2_xhat_ensemble')
# plt.close()

# ------------------------------------------------------------------------ ##
# Sectoral attribution maps
# ------------------------------------------------------------------------ ##
# # Get sectoral xhat difference from prior (in units Mg/km2/yr)
# xhat_diff_sect = pd.DataFrame(columns=[s[:-12] for s in ensemble])
# for suff in ensemble:
#     short_suff = suff.split('rg2rt_10t_')[-1].split('_bc0')[0]
#     xhat_diff_sect[suff] = w[short_suff]*(xhat[suff] - 1)/area
# xhat_diff_sect = pd.DataFrame({'mean' : xhat_diff_sect.mean(axis=1),
#                                'min' : xhat_diff_sect.min(axis=1),
#                                'max' : xhat_diff_sect.max(axis=1)})

# ul = 10
# ncategory = len(emis)
# fig, ax = fp.get_figax(rows=2, cols=3, maps=True,
#                        lats=clusters.lat, lons=clusters.lon)
# plt.subplots_adjust(hspace=0.3)

# d_xhat_cbar_kwargs = {'title' : r'$\Delta$ Emissions (Mg km$^{-2}$ a$^{-1}$)'}
# d_xhat_kwargs = {'cmap' : 'RdBu_r', 
#                'cbar_kwargs' : d_xhat_cbar_kwargs,
#                'map_kwargs' : small_map_kwargs}


# for axis, (title, emis_label) in zip(ax.flatten(), emis.items()):
#     # Get sectoral values (Mg/km2/yr)
#     xhat_diff_sect_i = xhat_diff_sect[emis_label].values

#     if title != 'Total':
#         xhat_sect_i = (xhat - 1)*w_mask[emis_label].values.reshape(-1, 1) + 1
#         fig2b, ax2b, c2b = ip.plot_state(xhat_sect_i, clusters, 
#                                          title=f'{title} scaling factors',
#                                          **xhat_kwargs)
#         fp.save_fig(fig2b, plot_dir, f'fig_est2_xhat_sf_{emis_label}_{f}')


#     fig, axis, c = ip.plot_state(xhat_diff_sect_i, clusters,
#                                  fig_kwargs={'figax' : [fig, axis]},
#                                  title=title, default_value=0,
#                                  vmin=-ul, vmax=ul,
#                                  cmap='RdBu_r', cbar=False)

#     fig2, ax2, c2 = ip.plot_state(xhat_diff_sect_i, clusters, default_value=0,
#                                  vmin=-ul, vmax=ul, title=title, 
#                                  **d_xhat_kwargs)

#     d_xhat_kwargs['cbar_kwargs'] = {'title' : r'$\Delta$ Emissions (Mg km$^{-2}$ a$^{-1}$)'}

#     # Calculate annual difference
#     diff_sec_tot = (xhat_diff_sect_i.reshape((-1, 1))*area)
#     diff_sec_tot_pos = diff_sec_tot[diff_sec_tot > 0].sum()*1e-6
#     diff_sec_tot_neg = diff_sec_tot[diff_sec_tot <= 0].sum()*1e-6
#     diff_sec_tot = diff_sec_tot.sum()*1e-6
#     axis.text(0.05, 0.05, f'{diff_sec_tot:.2f} Tg', ha='left', va='bottom',
#               fontsize=config.LABEL_FONTSIZE*config.SCALE,
#               transform=axis.transAxes)
#     ax2.text(0.05, 0.05, f'{diff_sec_tot:.2f} Tg ({diff_sec_tot_neg:.2f} Tg, {diff_sec_tot_pos:.2f} Tg)', ha='left', va='bottom',
#              fontsize=config.LABEL_FONTSIZE*config.SCALE,
#              transform=ax2.transAxes)

#     # Save out individual plot
#     fp.save_fig(fig2, plot_dir, f'fig_est2_xhat_{emis_label}_{f}')
#     plt.close(fig2)

#     # Plot regions of interst
#     if (emis_label in interest.keys()):
#         for j, reg in enumerate(interest[emis_label]):
#             fig3, ax3 = fp.get_figax(rows=2, cols=2, maps=True,
#                                      lats=reg[1][:2], lons=reg[1][2:],
#                                      max_height=config.BASE_WIDTH*config.SCALE)
#             figw, figh = fig3.get_size_inches()
#             plt.subplots_adjust(hspace=0.5/figh, wspace=6/figw)
#             # fig3.set_figheight(figh + 0.5)
#             # fig3.set_figwidth(figw + 7)

#             c = clusters.where((clusters.lat > reg[1][0]) &
#                                (clusters.lat < reg[1][1]) &
#                                (clusters.lon > reg[1][2]) &
#                                (clusters.lon < reg[1][3]), drop=True)
#             c_idx = (c.values[c.values > 0] - 1).astype(int)

#             fig3, ax3[0, 0], c32 = ip.plot_state(
#                 w.loc[:, emis_label]/area.reshape(-1,), clusters, 
#                 title='Sector prior', cbar=False, cmap=viridis_trans, 
#                 vmin=reg[2][0], vmax=reg[2][1], default_value=0, 
#                 fig_kwargs={'figax' : [fig3, ax3[0, 0]]}, 
#                 map_kwargs=small_map_kwargs)
#             fig3, ax3[0, 1], _ = ip.plot_state(
#                 xhat_diff_sect_i + w.loc[:, emis_label]/area.reshape(-1,),
#                 clusters, title='Sector posterior', cbar=False, 
#                 cmap=viridis_trans, vmin=reg[2][0], vmax=reg[2][1],
#                 default_value=0, fig_kwargs={'figax' : [fig3, ax3[0, 1]]},
#                 map_kwargs=small_map_kwargs)
#             fig3, ax3[1, 0], c30 = ip.plot_state(
#                 xhat, clusters, title=f'Scale factors', 
#                 cbar=False, cmap=sf_cmap, norm=div_norm, default_value=1,
#                 fig_kwargs={'figax' : [fig3, ax3[1, 0]]},
#                 map_kwargs=small_map_kwargs)
#             fig3, ax3[1, 1], c31 = ip.plot_state(
#                 xhat_diff_sect_i, clusters, title='Emissions change', 
#                 cbar=False, vmin=-reg[2][1]/4, vmax=reg[2][1]/4,
#                 fig_kwargs={'figax' : [fig3, ax3[1, 1]]}, **d_xhat_kwargs)

#             tt = (xhat_diff_sect_i.reshape(-1,)*area.reshape(-1,))[c_idx].sum()*1e-6
#             ax3[1, 1].text(0.05, 0.05, f'{tt:.1f} Tg/yr',
#                             fontsize=config.LABEL_FONTSIZE*config.SCALE,
#                             transform=ax3[1, 1].transAxes)

#             if reg[3] is not None:
#                 for label, point in reg[3].items():
#                     for axis3 in ax3.flatten():
#                         axis3.scatter(point[1], point[0], s=10, c='black')
#                         axis3.text(point[1], point[0], r'$~~$'f'{label}',
#                                    fontsize=config.LABEL_FONTSIZE*config.SCALE/2)

#             for k, axis3 in enumerate(ax3.flatten()):
#                 axis3.set_ylim(reg[1][:2])
#                 axis3.set_xlim(reg[1][2:])
#                 axis3.add_feature(COUNTIES, facecolor='none', 
#                                    edgecolor='0.1', linewidth=0.1)
#             cax30 = fp.add_cax(fig3, ax3[1, 0], cbar_pad_inches=0.1)
#             cb30 = fig.colorbar(c30, cax=cax30,
#                                 ticks=np.arange(0, 3, 1))
#             cb30 = fp.format_cbar(cb30,
#                                  cbar_title='Scale factor', x=4)

#             cax31 = fp.add_cax(fig3, ax3[1, 1], cbar_pad_inches=0.1)
#             cb31 = fig.colorbar(c31, cax=cax31,
#                                 ticks=np.arange(-reg[2][1]/4, reg[2][1]/4+1, 
#                                                 reg[2][1]/8))
#             cbar_str = r'$\Delta$ Emissions\\(Mg km$^{-2}$ a$^{-1}$)'
#             cb31 = fp.format_cbar(cb31,
#                                  cbar_title=cbar_str, x=4)
#             if emis_label == 'total':
#                 step = 10
#             else:
#                 step = 5
#             cax32 = fp.add_cax(fig3, ax3[0, 1], cbar_pad_inches=0.1)
#             cb32 = fig.colorbar(
#                 c32, cax=cax32, 
#                 ticks=np.arange(reg[2][0], reg[2][1]+step, step))
#             cb32 = fp.format_cbar(
#                 cb32, cbar_title=r'Emissions\\(Mg km$^{-2}$ a$^{-1}$)', x=4)

#             fp.save_fig(fig3, plot_dir, 
#                         f'fig_est2_xhat_{emis_label}_reg{j}_{f}')
#             plt.close(fig3)

# cax = fp.add_cax(fig, ax, cbar_pad_inches=0.3, horizontal=True)
# cb = fig.colorbar(c, cax=cax, ticks=np.arange(-ul, ul+1, ul/2),
#                   orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title=r'$\Delta$ Emissions (Mg km$^{-2}$ a$^{-1}$)',
#                     horizontal=True)
# # axis = fp.add_title(axis, titles[i])

# fp.save_fig(fig, plot_dir, f'fig_est2_xhat_sectoral_{f}')
# plt.close(fig)

## ------------------------------------------------------------------------ ##
## Sectoral attribution bar chart
## ------------------------------------------------------------------------ ##
w = {wkey : ww*1e-6 for wkey, ww in w.items()}  # Convert to Tg/yr

# Open masks and create a total_mask array as well as a mask dictionary
mex_mask = np.load(f'{data_dir}Mexico_mask.npy').reshape((-1,))
can_mask = np.load(f'{data_dir}Canada_mask.npy').reshape((-1,))
conus_mask = np.load(f'{data_dir}CONUS_mask.npy').reshape((-1,))
total_mask = mex_mask + can_mask + conus_mask
masks = {'Canada' : can_mask, 'CONUS' : conus_mask, 'Mexico' : mex_mask}

# # xiao = {'Canada' : {'livestock' : [1.4], 'ong' : [3.2], 'coal' : [0.0], 
# #                     'landfills' : [0.69], 'wastewater' : [0.0], 
# #                     'wetlands' : [9.9], 'other' : [1.01]},
# #         'CONUS' : {'livestock' : [10.6], 'ong' : [14.5], 'coal' : [2.8], 
# #                    'landfills' : [7.2], 'wastewater' : [0.63], 
# #                    'wetlands' : [8.4], 'other' : [2.17]},
# #         'Mexico' : {'livestock' : [2.5], 'ong' : [1.26], 'coal' : [0.26], 
# #                     'landfills' : [1.0], 'wastewater' : [0.8], 
# #                     'wetlands' : [0.57], 'other' : [0.41]}}

## LEFT OFF HERE
country_emis = {}
for country, mask in masks.items():
    w_w37_c = (dc(w['w37_edf'])*mask) @ xhat[w37_cols]
    w_w404_c = (dc(w['w404_edf'])*mask) @ xhat[w404_cols]
    w_c = pd.concat([w_w37_c, w_w404_c], axis=1)
print(w_c)



    # sect_emis_mean = pd.DataFrame(columns=['prior', 'post', 'diff',
    #                                        'prior_sub', 'post_sub', 'diff_sub', 
    #                                        'lu'])
    # sect_emis_min = pd.DataFrame(columns=['prior', 'post'])
    # sect_emis_max = pd.DataFrame(columns=['prior', 'post'])

    # for suff in ensemble:
    #     short_suff = suff.split('rg2rt_10t_')[-1].split('_bc0')[0]
    #                                       #'diff_sub_pos', 'diff_sub_neg'])
    #     w_c = dc(w[short_suff])*mask*1e-6 # convert to Tg/yr in the country

    #     # Get prior and posterior absolute emissions
    #     sect_emis['prior'] = w_c.sum(axis=0)
    #     sect_emis['post'] = (w_c*xhat[suff]).sum(axis=0)
    #     sect_emis['diff'] = sect_emis['post'] - sect_emis['prior']

    #     # Get prior and posterior absolute emissions only where DOFS > DOFS_filter
    #     sect_emis['prior_sub'] = w_c.loc[dofs[suff] >= DOFS_filter, :].sum(axis=0)
    #     sect_emis['post_sub'] = (w_c*xhat[suff]).loc[dofs[suff] >= DOFS_filter, :].sum(axis=0)
    #     sect_emis['diff_sub'] = sect_emis['post_sub'] - sect_emis['prior_sub']

    #     # # Get positive and negative adjustments
    #     # sect_emis['diff_pos'] = (w_c*(xhat - 1))[xhat >= 1].sum(axis=0)
    #     # sect_emis['diff_neg'] = np.abs(w_c*(xhat - 1))[xhat < 1].sum(axis=0)

    #     # Reorder sect_emis
    #     sect_emis = sect_emis.loc[list(emis.values())[1:]]

    #     # Add in Xiao
    #     sect_emis['lu'] = pd.DataFrame.from_dict(xiao[country], orient='index').loc[list(emis.values())[1:]]

    #     # Save 
    #     country_range[suff] = sect_emis
    # print(country_range)

    # country_emis[country] = sect_emis

        # # Print info
        # c_dofs = dofs*mask
        # c_emis = w_c['total']
        # print('-'*75)
        # print(f'In {country}, we optimize {(c_dofs >= DOFS_filter).sum():d} grid cells including {c_emis[dofs.reshape(-1,) >= DOFS_filter].sum():.2E}/{c_emis.sum():.2E} = {(c_emis[dofs.reshape(-1,) >= DOFS_filter].sum()/c_emis.sum()*100):.2f}%\nof emissions. This produces {c_dofs.sum():.2f} DOFS with an xhat range of {(xhat*mask).min():.2f} to {(xhat*mask).max():.2f}.\nThere are {len((xhat*mask)[(xhat*mask) < 0]):d} negative values.')
        # print('-'*75)

# # # # Plot histogram (at least take #1)
# # # xs = np.arange(0, len(emis.keys()) - 1)
# # # fig, axis = fp.get_figax(aspect=2, cols=3)
# # # j = 0
# # # lims = [[0, 4], [0, 15], [0, 4]]
# # # country_emis_sub = {key : country_emis[key]
# # #                     for key in ['Canada', 'CONUS', 'Mexico']}
# # # for country, sect_emis in country_emis_sub.items():
# # #     ax = axis[j]
# # #     diff_summ = sect_emis['diff'].sum()
# # #     ax.text(0.05, 0.95, country, ha='left', va='top',
# # #             fontsize=config.LABEL_FONTSIZE*config.SCALE,
# # #             transform=ax.transAxes)
# # #     ax.text(0.95, 0.95, f'({diff_summ:0.2f} Tg)', ha='right', va='top',
# # #             fontsize=config.TICK_FONTSIZE*config.SCALE/1.5,
# # #             transform=ax.transAxes)
# # #     ax.bar(xs - 0.12, sect_emis['prior'], width=0.2,
# # #            color='white', edgecolor=fp.color(2*j+2), label='Prior (all)')
# # #     ax.bar(xs - 0.12, sect_emis['prior_sub'], width=0.2, color=fp.color(2*j+2),
# # #            label='Prior (optimized)')
# # #     # ax.bar(xs - 0.12, sect_emis['prior_sub2'], width=0.2, color=fp.color(2*j+2),
# # #     #        label='Prior (more optimized)')
# # #     ax.bar(xs + 0.12, sect_emis['post'], width=0.2,
# # #            color='white', edgecolor=fp.color(2*j+2), alpha=0.5,
# # #            label='Posterior (all)')
# # #     ax.bar(xs + 0.12, sect_emis['post_sub'], width=0.2,
# # #            color=fp.color(2*j+2), alpha=0.5, 
# # #            label='Posterior (optimized)')

# # #     ax.bar(xs + 0.35, sect_emis['lu'], width=0.1, 
# # #           color=fp.color(2*j+2), alpha=0.25, label='Lu et al. (2022)')

# # #     # ax.bar(xs + 0.11, sect_emis['post_sub2'], width=0.2,
# # #     #        color=fp.color(2*j+2), alpha=0.5, 
# # #     #        label='Posterior (more optimized)')
# # #     # print(sect_emis)
# # #     # for i, e in enumerate(list(emis.values())[1:]):
# # #     #     ax.arrow(x=i + 0.06, y=sect_emis.at[e, 'prior'], dx=0, 
# # #     #              dy=sect_emis.at[e, 'diff_pos'],
# # #     #              color=fp.color(2*j+2), alpha=1, width=0.002, 
# # #     #              head_width=0.0025)
# # #     #     ax.arrow(x=i + 0.18, 
# # #     #              y=sect_emis.at[e, 'prior'] + sect_emis.at[e, 'diff_pos'],
# # #     #              dx=0, dy=-sect_emis.at[e, 'diff_neg'],
# # #     #              color=fp.color(2*j+2), alpha=1, width=0.002, 
# # #     #              head_width=0.0025)
# # #     ax.set_ylim(lims[j])
# # #     # ax.bar(xs + 0.2, sect_emis['post_sub_pos'], width=0.1, color=fp.color(7))
# # #     # ax = fp.add_legend(ax)

# # #     j += 1

# # # axis[0] = fp.add_labels(axis[0], '', 'Emissions\n' + r'[Tg a$^{-1}$]',
# # #                         fontsize=config.TICK_FONTSIZE*config.SCALE/1.5,
# # #                         labelsize=config.TICK_FONTSIZE*config.SCALE/1.5)
# # # for i in range(3):
# # #     if i > 0:
# # #         axis[i] = fp.add_labels(axis[i], '', '',
# # #                                 labelsize=config.TICK_FONTSIZE*config.SCALE/1.5)
# # #     axis[i].set_xticks(xs)
# # #     axis[i].set_xticklabels(list(emis.keys())[1:], rotation=90, ha='center',
# # #                             fontsize=config.TICK_FONTSIZE*config.SCALE/1.5)
# # #     axis[i].set_xlim(-0.5, 7)

# # # custom_patches = [patch(color=fp.color(0), alpha=1), 
# # #                   patch(color=fp.color(0), alpha=0.5),
# # #                   # patch(color=fp.color(0), alpha=0.25),
# # #                   # patch(color='grey'), 
# # #                   patch(facecolor='white', edgecolor=fp.color(0)),
# # #                   patch(color=fp.color(0), alpha=0.25)]
# # # custom_labels = ['Prior', 'Posterior', #'Lu et al. (2022)',
# # #                  r'Not optimized (A$_{ii}$ $<$'f' {DOFS_filter})',
# # #                  'Lu et al. (2022)']
# # #                  # r'Averaging kernel sensitivities $\ge$'f' {DOFS_filter}', 
# # #                  # r'Averaging kernel sensitivities $<$'f' {DOFS_filter}']
# # # # custom_labels = ['Numerical solution', 'Predicted solution']
# # # # handles, labels = ax_summ.get_legend_handles_labels()
# # # # custom_lines.extend(handles)
# # # # custom_labels.extend(labels)
# # # fp.add_legend(axis[1], handles=custom_patches, labels=custom_labels,
# # #               bbox_to_anchor=(0.5, -0.75), loc='upper center', ncol=2,
# # #               fontsize=config.TICK_FONTSIZE*config.SCALE/1.5)


# # # fp.save_fig(fig, plot_dir, f'fig_sectoral_bar_{f}')
# # # plt.close()