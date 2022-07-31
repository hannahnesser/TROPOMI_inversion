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

sf_cmap_1 = plt.cm.Reds(np.linspace(0, 0.5, 256))
sf_cmap_2 = plt.cm.Blues(np.linspace(0.5, 1, 256))
sf_cmap = np.vstack((sf_cmap_1, sf_cmap_2))
sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
div_norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=6)

diff_cmap_1 = plt.cm.bwr(np.linspace(0, 0.5, 256))
diff_cmap_2 = plt.cm.bwr(np.linspace(0.5, 1, 256))
diff_cmap = np.vstack((diff_cmap_1, diff_cmap_2))
diff_cmap = colors.LinearSegmentedColormap.from_list('diff_cmap', diff_cmap)
diff_div_norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=1)

# Small (i.e. non-default) figure settings
small_map_kwargs = {'draw_labels' : False}
small_fig_kwargs = {'max_width' : 4,
                    'max_height' : 3.5}

def heat_map(x, y, data, fig, ax, cmap, n_cmap, vmin, vmax):
    scale = 0.6
    hm_cmap = plt.cm.get_cmap(cmap, n_cmap)
    c = ax.imshow(data, cmap=hm_cmap, vmin=vmin, vmax=vmax)
    for k in range(len(x)):
        for l in range(len(y)):
            text = ax.text(k, l, f'{data[l, k]}', 
                           ha='center', va='center', color='w', 
                           fontsize=config.TICK_FONTSIZE*scale)
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, fontsize=config.TICK_FONTSIZE*scale)
    ax.set_xlabel('Regularization factor', fontsize=config.TICK_FONTSIZE*scale)
    ax.set_yticks(np.arange(len(y)))
    ax.set_yticklabels(y, fontsize=config.TICK_FONTSIZE*scale)
    ax.set_ylabel('Prior error', fontsize=config.TICK_FONTSIZE*scale)
    cax = fp.add_cax(fig, ax, cbar_pad_inches=0.1)
    cb = fig.colorbar(c, cax=cax)
    cb.ax.tick_params(labelsize=config.TICK_FONTSIZE*scale)
    return fig, ax


## ------------------------------------------------------------------------ ##
## Set user preferences and load data
## ------------------------------------------------------------------------ ##
DOFS_filter = 0.1

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc')

# Load prior (Mg/km2/yr)
xa_abs = xr.open_dataarray(f'{data_dir}xa_abs_wetlands404_edf_bc0.nc').values.reshape((-1, 1)) # Mg/km2/yr
# xa_abs = xr.open_dataarray(f'{data_dir}xa_abs_correct.nc').values.reshape((-1, 1))
xa_abs_base = xr.open_dataarray(f'{data_dir}xa_abs.nc').values.reshape((-1, 1))
xa_ratio = xa_abs/xa_abs_base
xa_ratio[(xa_abs_base == 0) & (xa_abs == 0)] = 1 # Correct for the grid cell with 0 emisisons
soil = xr.open_dataarray(f'{data_dir}soil_abs.nc')

# Scale by area
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))
# xa_abs = xa_abs*area*1e-6 # Tg/yr
# soil = soil*area*1e-6 # Tg/yr

# # List emissions categories
# emis = ['wetlands', 'livestock', 'coal', 'oil', 'gas', 'landfills',
#         'wastewater', 'other']
# emis_labels = ['Wetlands', 'Livestock', 'Coal', 'Oil', 'Gas', 'Landfills',
#                'Wastewater', 'Other']
emis = {'Total' : 'total', 'Oil and gas' : 'ong', 'Coal' : 'coal',
        'Livestock' : 'livestock', 'Landfills' : 'landfills', 
        'Wastewater' : 'wastewater', 'Wetlands' : 'wetlands', 
        'Other' : 'other'}
# emis_labels = ['Wetlands', 'Livestock', 'Coal', 'Oil and gas', 'Landfills',
#                'Wastewater', 'Other']
               # regions of interest
interest = {'livestock' : [('North Carolina', [33.5, 37, -84.5, -75.5], [0, 20]),
                           ('Midwest', [38.5, 46, -100, -81], [0, 20])],
            'coal' : [('Illinois Basin', [36.25, 41.25, -91.5, -85], [0, 30]),
                      ('Powder River Basin', [40, 46, -111.5, -103.5], [0, 40])],
            'ong' : [('Permian Basin', [27.5, 36, -107.3, -98], [0, 40]),
                     ('Haynesville Shale', [29, 33.5, -95.5, -89], [0, 30]),
                     ('Anadarko Shale', [33, 37.5, -103.5, -94], [0, 20])]}
           # 'wetlands' : None,
               # 'landfills' : None,
               # 'wastewater' : None,
               # 'other' : None}

# Select posterior and DOFS
# Decide which posterior to run with
# xfiles = glob.glob(f'{data_dir}posterior/xhat2_rg?rt_rf1.0_sax1.0_poi80.npy')
# xfiles.sort()
# fs = [f.split('/')[-1][6:-4] for f in xfiles]
# print(fs)
# xhat_fr2_bc_rg2rt_10t_wetlands404_edf_nlc_bc0_rf1.0_sax1.0_poi80.0.npy
f = 'rg2rt_10t_wetlands404_edf_rf1.0_sax0.5_poi80.0'
title_str = 'Scaled wetlands + EDF + NLC'
# Load files
dofs = np.load(f'{data_dir}posterior/dofs2_{f}.npy').reshape((-1, 1))
xhat = 1 + np.load(f'{data_dir}posterior/xhat_fr2_{f}.npy').reshape((-1, 1))
w = pd.read_csv(f'{data_dir}w_wetlands404_edf.csv')
# dofs = np.nan_to_num(dofs, 0)

# # BC alteration
# xhat = xhat[:-4]
# dofs = dofs[:-4]

# # Filter on DOFS filter
xhat[dofs < DOFS_filter] = 1
dofs[dofs < DOFS_filter] = 0
print(f, xhat.max(), xhat.min(), dofs.sum(), len(xhat[xhat < 0]))

# Get xhat abs
xhat_abs = (xhat*xa_abs)

# Adjust W units
# w = w.T*area.values*1e-6 # Convert to Tg/yr
# w_rel = w/w.sum(axis=0) # Normalize
# w_rel[np.isnan(w_rel)] = 0 # Deal with nans

# # Get standard results
# f2 = 'rg2rt_10t_rf1.0_sax1.0_poi80'
# dofs2 = np.load(f'{data_dir}posterior/dofs2_{f2}.npy').reshape((-1, 1))
# xhat2 = np.load(f'{data_dir}posterior/xhat2_{f2}.npy').reshape((-1, 1))
# xhat2[dofs2 < DOFS_filter] = 1
# dofs2[dofs2 < DOFS_filter] = 0
# # xhat2 = xhat2[:-4]
# # dofs2 = dofs2[:-4]
# print(f2, xhat2.max(), xhat2.min(), dofs2.sum(), len(xhat2[xhat2 < 0]))

# # Get county outlines for high resolution results
reader = shpreader.Reader(f'{data_dir}counties/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

# ## ------------------------------------------------------------------------ ##
# ## Regularization factor
# ## ------------------------------------------------------------------------ ##
x_files = glob.glob(f'{data_dir}posterior/xhat_fr2_rg2rt_10t_wetlands404_edf_*poi80.0.npy')
a_files = glob.glob(f'{data_dir}posterior/dofs2_rg2rt_10t_wetlands404_edf_*poi80.0.npy')
x_files.sort()
a_files.sort()

# # # xa_abs_base_rf = gc.read_file(f'{data_dir}xa_abs.nc')
# # # xa_abs_rf = gc.read_file(f'{data_dir}xa_abs_correct.nc')
# # # xa = xa_abs_rf/xa_abs_base_rf # Ratio of standard prior to input prior
# # # xa[(xa_abs_base_rf == 0) & (xa_abs_rf == 0)] = 1 # Correct for the grid cell with 0 emisisons
# # # xa = xa.values.reshape((-1,))
# # # print(xa)

# ja = pd.DataFrame(columns=['sa', 'ja'])
sas = ['0.5']#['0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
rfs = ['0.01', '0.015','0.05', '0.1', '0.5', '1.0']
# print('DOFS  |   rf |   sa |  poi |      n |      Ja/n |   neg |   avg diff | DOFS ')
# print('-'*81)
for DD in [0.05, 0.1, 0.15, 0.2]:
    ja = np.zeros((len(sas), len(rfs)))
    negs = np.zeros((len(sas), len(rfs)))
    avgs = np.zeros((len(sas), len(rfs)))
    nns = np.zeros((len(sas), len(rfs)))
    for i, ss in enumerate(sas):
        for j, rr in enumerate(rfs):
            sfx = f'rg2rt_10t_wetlands404_edf_rf{rr}_sax{ss}_poi80.0'
            xf = f'{data_dir}posterior/xhat_fr2_{sfx}.npy'
            af = f'{data_dir}posterior/dofs2_{sfx}.npy'

            # Get error value
            ss = float(ss)
            # pp = float(x_files[i].split('poi')[1].split('.')[0])
            rr = float(rr)

            # Load files
            xx = 1 + np.load(xf)
            aa = np.load(af)

            # Calculate n func
            nn = (aa >= DD).sum()
            nns[i, j] = nn

            # Subset xx
            xx[aa < DD] = 1
            aa[aa < DD] = 0

            # Calculate Ja(xhat)/n
            diff = (xx - 1)**2
            diff = (diff/ss**2).sum()
            if DD == 0.1:
                print(sfx, diff/nn, (xx < 0).sum())
            ja[i, j] = diff/nn

            # Get number of negative grid cells
            negs[i, j] = (xx < 0).sum()

            # Get mean of corrected grid cells
            avgs[i, j] = xx[aa >= DD].mean()
    fig, ax = fp.get_figax(rows=2, cols=2, aspect=len(rfs)/len(sas))
    plt.subplots_adjust(wspace=0.65, hspace=2)

    # Plot ja
    ja = np.around(ja, 1)
    fig, ax[0, 0] = heat_map(rfs, sas, ja, fig, ax[0, 0], 
                             'viridis', 10, -0.5, 9.5)
    fp.add_title(ax[0, 0], 'Ja/n', fontsize=config.SUBTITLE_FONTSIZE*0.6)

    # Plot negs
    negs = negs.astype(int)
    fig, ax[0, 1] = heat_map(rfs, sas, negs, fig, ax[0, 1], 
                          'plasma', 100, -0.5, 1000)
    fp.add_title(ax[0, 1], 'Negative values', 
                 fontsize=config.SUBTITLE_FONTSIZE*0.6)

    # Plot mean adjustment
    avgs = np.around(avgs, 2)
    fig, ax[1, 0] = heat_map(rfs, sas, avgs, fig, ax[1, 0], 
                          'inferno', 5, 0.75, 1.25)
    fp.add_title(ax[1, 0], r'Mean $\hat{x}$', 
                 fontsize=config.SUBTITLE_FONTSIZE*0.6)

    # Plot functional ns
    nns = nns.astype(int)
    fig, ax[1, 1] = heat_map(rfs, sas, nns, fig, ax[1, 1], 
                          'cividis', 100, -0.5, 6e3)
    fp.add_title(ax[1, 1], r'Functional n', 
                 fontsize=config.SUBTITLE_FONTSIZE*0.6)

    # Save
    fp.save_fig(fig, plot_dir, f'fig_rfs_sas_2_{DD}')

            # Print
            # print(f'{DD:5} |{rr:5} |{ss:5} |{pp:5} |{nn:7} |{diff:10.4f} |{neg:6d} | {avg:10.4f} | {aa.sum():10.1f}')



# # # lims = [[0, 1], [0, 1]]
# # # label = ['ja', 'jo']
# # # letter = ['A', 'O']
# # # for i, var in enumerate([ja, jo]):
# # #     # # Plot
# # #     fig, ax = fp.get_figax(aspect=len(ja)/len(jo))
# # #     cax = fp.add_cax(fig, ax)

# # #     # Plot
# # #     # c = ax.contour(rfs, sas, ja.T)
# # #     c = ax.imshow(var, vmin=lims[i][0], vmax=lims[i][1])

# # #     # Labels
# # #     ax.set_xticks(np.arange(0, len(sas)))
# # #     ax.set_xticklabels(sas)
# # #     ax.set_ylim(-0.5, len(rfs)-0.5)
# # #     ax.set_yticks(np.arange(0, len(rfs)))
# # #     ax.set_yticklabels(rfs)
# # #     ax = fp.add_labels(ax, 'Prior errors', 'Regularization factor')

# # #     # Colorbar
# # #     cb = fig.colorbar(c, cax=cax)#, ticks=np.arange(0, 6, 1))
# # #     cb = fp.format_cbar(cb, cbar_title=r'$J_{A}(\hat{x})$')

# # #     fp.save_fig(fig, plot_dir, f'fig_rfs_sas_{label[i]}')



## ------------------------------------------------------------------------ ##
## Second estimate
## ------------------------------------------------------------------------ ##
# Identify anomalous corrections
# Standard deviation of absolute adjustments, ignoring avker = 0
xhat_abs_corr = (xhat - 1)*xa_abs
xhat_abs_std = xhat_abs_corr[dofs != 0].std()
xhat_abs_mu = xhat_abs_corr[dofs != 0].mean()

# Mg/km2/yr absolute adjustment relative to the prior
# Used to be xhat_g
z_abs = (xhat_abs_corr - xhat_abs_mu)/xhat_abs_std
z_abs_g = ip.match_data_to_clusters(z_abs, clusters, default_value=0)

# Corrections more than two std away
idx_gt = np.where(z_abs_g > 1.645)
idx_lt = np.where(z_abs_g < -1.645)

# Standard deviation of relative adjustments
xhat_corr = (xhat - 1)
xhat_std = xhat_corr[xhat != 1].std()
xhat_mu = xhat_corr[xhat != 1].mean()
z_rel = (xhat_corr - xhat_mu)/xhat_std
z_rel_g = ip.match_data_to_clusters(z_rel, clusters, default_value=0)
idx_rel_gt = np.where(z_rel_g > 1.645)
idx_rel_lt = np.where(z_rel_g < -1.645)

# Make sectoral correction maps
d_avker_cbar_kwargs = {'title' : r'$\Delta \partial\hat{x}_i/\partial x_i$'}
avker_cbar_kwargs = {'title' : r'$\partial\hat{x}_i/\partial x_i$',
                     'horizontal' : True}
avker_kwargs = {'cmap' : plasma_trans, 'vmin' : 0, 'vmax' : 1,
                'cbar_kwargs' : avker_cbar_kwargs,
                'map_kwargs' : small_map_kwargs}

d_xhat_cbar_kwargs = {'title' : r'$\Delta$ Scale factor'}
xhat_cbar_kwargs = {'title' : r'Scale factor', 'horizontal' : True}
xhat_kwargs = {'cmap' : 'PuOr_r', 'vmin' : 0, 'vmax' : 2,
            # 'cmap' : sf_cmap, 'norm' : div_norm,
               # 'cmap' : 'RdBu_r', 'vmin' : 0, 'vmax' : 6,
               'default_value' : 1,
               'cbar_kwargs' : xhat_cbar_kwargs,
               'map_kwargs' : small_map_kwargs}

# # Subset to ID only large corrections
# xhat_sub = copy.deepcopy(xhat)
# xhat_sub[dofs < 0.01] = 1
# xhat_sub[xhat_sub > (xhat[dofs >= 0.01].mean() + xhat[dofs >= 0.01].std())]
# xhat[dofs >= 0.01]

# Plot averaging kernel sensitivities
title = f'Averaging kernel sensitivities' # ({f}\%)'
fig, ax, c = ip.plot_state(dofs, clusters, title=title,
                           **avker_kwargs)
# for i in range(len(idx_gt[0])):
#     ax.scatter(clusters.lon[idx_gt[1][i]], clusters.lat[idx_gt[0][i]],
#                marker='x', s=10, color='red')
# for i in range(len(idx_rel_gt[0])):
#     ax.scatter(clusters.lon[idx_rel_gt[1][i]], clusters.lat[idx_rel_gt[0][i]],
#                marker='.', s=10, color='red')
# for i in range(len(idx_lt[0])):
#     ax.scatter(clusters.lon[idx_lt[1][i]], clusters.lat[idx_lt[0][i]],
#                marker='x', s=10, color='blue')
# for i in range(len(idx_rel_lt[0])):
#     ax.scatter(clusters.lon[idx_rel_lt[1][i]], clusters.lat[idx_rel_lt[0][i]],
#                marker='.', s=10, color='blue')

ax.text(0.025, 0.05, 'DOFS = %d' % round(dofs.sum()),
        fontsize=config.LABEL_FONTSIZE*config.SCALE,
        transform=ax.transAxes)
fp.save_fig(fig, plot_dir, f'fig_est2_dofs_{f}')
plt.close()

# Plot posterior scaling factors
title = f'Posterior emission scale factors' # ({f}\%)'
fig, ax, c = ip.plot_state(xhat, clusters, title=title,
                           **xhat_kwargs)
# for i in range(len(idx_gt[0])):
#     ax.scatter(clusters.lon[idx_gt[1][i]], clusters.lat[idx_gt[0][i]],
#                marker='x', s=10, color='red')
# for i in range(len(idx_rel_gt[0])):
#     ax.scatter(clusters.lon[idx_rel_gt[1][i]], clusters.lat[idx_rel_gt[0][i]],
#                marker='.', s=10, color='red')
# for i in range(len(idx_lt[0])):
#     ax.scatter(clusters.lon[idx_lt[1][i]], clusters.lat[idx_lt[0][i]],
#                marker='x', s=10, color='blue')
# for i in range(len(idx_rel_lt[0])):
#     ax.scatter(clusters.lon[idx_rel_lt[1][i]], clusters.lat[idx_rel_lt[0][i]],
#                marker='.', s=10, color='blue')
fp.save_fig(fig, plot_dir, f'fig_est2_xhat_{f}')
plt.close()

# Reset cbar kwargs
avker_kwargs['cbar_kwargs'] = {'title' : r'$\partial\hat{x}_i/\partial x_i$'}
xhat_kwargs['cbar_kwargs'] = {'title' : r'Scale factor'}

# Plot sensitivity results
# title = f'{title2_str}\nposterior emission scale factors' # ({f}\%)'
# fig, ax, c = ip.plot_state(xhat2, clusters, title=title,
#                            **xhat_kwargs)
# fp.save_fig(fig, plot_dir, f'fig_est2_xhat_{save2_str}_{f}')
# plt.close()

# title = f'{title2_str}\naveraging kernel sensitivities' # ({f}\%)'
# fig, ax, c = ip.plot_state(dofs2, clusters, title=title,
#                            **avker_kwargs)
# ax.text(0.025, 0.05, 'DOFS = %d' % round(dofs2.sum()),
#         fontsize=config.LABEL_FONTSIZE*config.SCALE,
#         transform=ax.transAxes)
# fp.save_fig(fig, plot_dir, f'fig_est2_dofs_{save2_str}_{f}')
# plt.close()

# title = f'{title_str} - standard\nposterior emission scale factors'
# fig, ax, c = ip.plot_state(xhat - xhat2, clusters, title=title,
#                            cmap='RdBu_r', vmin=-0.5, vmax=0.5, default_value=0,
#                            cbar_kwargs=d_xhat_cbar_kwargs,
#                            map_kwargs=small_map_kwargs)
# fp.save_fig(fig, plot_dir, f'fig_est2_xhat_diff_{f}')
# plt.close()

# title = f'{title_str} - standard\naveraging kernel sensitivities'
# fig, ax, c = ip.plot_state(dofs - dofs2, clusters, title=title,
#                            cmap='RdBu_r', vmin=-0.5, vmax=0.5, default_value=0,
#                            cbar_kwargs=d_avker_cbar_kwargs,
#                            map_kwargs=small_map_kwargs)
# ax.text(0.025, 0.05, r'$\Delta$DOFS = %d' % round(dofs.sum() - dofs2.sum()),
#         fontsize=config.LABEL_FONTSIZE*config.SCALE,
#         transform=ax.transAxes)
# fp.save_fig(fig, plot_dir, f'fig_est2_dofs_diff_{f}')
# plt.close()

# ## ------------------------------------------------------------------------ ##
# ## Sectoral attribution maps
# ## ------------------------------------------------------------------------ ##
# # Get sectoral xhat difference from prior (in units Mg/km2/yr)
# xhat_diff_sect =  w*(xhat - 1)

# ul = 10
# ncategory = len(emis)
# fig, ax = fp.get_figax(rows=2, cols=3, maps=True,
#                        lats=clusters.lat, lons=clusters.lon)
# plt.subplots_adjust(hspace=0.3)

# # fig, axis, c = ip.plot_state(xhat_diff_sect.sum(axis=0), clusters,
# #                              fig_kwargs={'figax' : [fig, ax[0, 0]]},
# #                              title='Total', vmin=-ul, vmax=ul,
# #                              cmap='bwr', cbar=False)
# # # Calculate annual difference
# # diff_sec_tot = (xhat_diff_sect.sum(axis=0)*area).sum()*1e-6
# # axis.text(0.05, 0.05, f'{diff_sec_tot:.2f} Tg', ha='left', va='bottom',
# #           fontsize=config.LABEL_FONTSIZE*config.SCALE,
# #           transform=axis.transAxes)

# d_xhat_cbar_kwargs = {'title' : r'$\Delta$ Emissions (Mg km$^{-2}$ a$^{-1}$)'}
# d_xhat_kwargs = {'cmap' : 'RdBu_r', 'vmin' : -ul, 'vmax' : ul,
#             # 'cmap' : sf_cmap, 'norm' : div_norm,
#                # 'cmap' : 'RdBu_r', 'vmin' : 0, 'vmax' : 6,
#                'default_value' : 0,
#                'cbar_kwargs' : d_xhat_cbar_kwargs,
#                'map_kwargs' : small_map_kwargs}


# for axis, (title, emis_label) in zip(ax.flatten(), emis.items()):
#     # Get sectoral values
#     if title == 'Total':
#         xhat_diff_sect_i = xhat_diff_sect.sum(axis=1).values
#     else:
#         xhat_diff_sect_i = xhat_diff_sect[emis_label].values

#     # # Calculate anomalous values
#     # xhat_g = ip.match_data_to_clusters(xhat_diff_sect.iloc[:, i]*area*1e-6,
#     #                                    clusters, default_value=0)
#     # idx_gt = np.where(xhat_g > xhat_mu + 2*xhat_std)
#     # idx_lt = np.where(xhat_g < xhat_mu - 2*xhat_std)

#     fig, axis, c = ip.plot_state(xhat_diff_sect_i, clusters,
#                                  fig_kwargs={'figax' : [fig, axis]},
#                                  title=title, default_value=0,
#                                  vmin=-ul, vmax=ul,
#                                  cmap='RdBu_r', cbar=False)

#     fig2, ax2, c2 = ip.plot_state(xhat_diff_sect_i, clusters,
#                                   title=title, **d_xhat_kwargs)


#     # for j in range(len(idx_gt[0])):
#     #     ax2.scatter(clusters.lon[idx_gt[1][j]], clusters.lat[idx_gt[0][j]],
#     #                 marker='x', s=10, color='red')
#     # for j in range(len(idx_lt[0])):
#     #     ax2.scatter(clusters.lon[idx_lt[1][j]], clusters.lat[idx_lt[0][j]],
#     #                 marker='x', s=10, color='blue')

#     d_xhat_kwargs['cbar_kwargs'] = {'title' : r'$\Delta$ Emissions (Mg km$^{-2}$ a$^{-1}$)'}
#     # if emis[i] == 'livestock':
#     #     ax2.set_xlim(-100, -86)
#     #     ax2.set_ylim(40, 50)

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

#     # Plot regions of interst
#     if (title != 'Total') & (emis_label in interest.keys()):
#         for j, reg in enumerate(interest[emis_label]):
#             fig3, ax3 = fp.get_figax(rows=2, cols=2, maps=True,
#                                      lats=reg[1][:2], lons=reg[1][2:])
#             fig3.suptitle(f'{reg[0]} ({title})', y=1.1,# x=0.5, y=1.2,
#                           fontsize=config.TITLE_FONTSIZE*config.SCALE)
#             plt.subplots_adjust(hspace=0.1, wspace=0.5)
#             fig3, ax3[0, 0], c32 = ip.plot_state(w.loc[:, emis_label], clusters,
#                                           fig_kwargs={'figax' : [fig3, ax3[0, 0]]},
#                                           title='Prior', cbar=False,
#                                           cmap=viridis_trans, 
#                                           vmin=reg[2][0], vmax=reg[2][1],
#                                           default_value=0,
#                                           map_kwargs=small_map_kwargs)
#             fig3, ax3[0, 1], _ = ip.plot_state(xhat_diff_sect_i + w.loc[:, emis_label], 
#                                         clusters, 
#                                        fig_kwargs={'figax' : [fig3, ax3[0, 1]]},
#                                        title='Posterior', cbar=False, 
#                                        cmap=viridis_trans, 
#                                        vmin=reg[2][0], vmax=reg[2][1],
#                                        default_value=0, 
#                                        map_kwargs=small_map_kwargs)
#             fig3, ax3[1, 0], c30 = ip.plot_state(xhat, clusters,
#                                           fig_kwargs={'figax' : [fig3, ax3[1, 0]]},
#                                           title=f'Scale factors',
#                                            cbar=False, cmap='PuOr_r',
#                                            vmin=0, vmax=2, default_value=1,
#                                            map_kwargs=small_map_kwargs)
#             d_xhat_kwargs['vmin'] = -reg[2][1]/4
#             d_xhat_kwargs['vmax'] = reg[2][1]/4
#             fig3, ax3[1, 1], c31 = ip.plot_state(xhat_diff_sect_i, clusters,
#                                           fig_kwargs={'figax' : [fig3, ax3[1, 1]]},
#                                           title='Emissions change',
#                                           cbar=False,
#                                           **d_xhat_kwargs)

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
#             cbar_str = r'$\Delta$ Emissions' '\n(Mg km$^{-2}$ a$^{-1}$)'
#             cb31 = fp.format_cbar(cb31,
#                                  cbar_title=cbar_str, x=4)
#             cax32 = fp.add_cax(fig3, ax3[0, 1], cbar_pad_inches=0.1)
#             cb32 = fig.colorbar(c32, cax=cax32,
#                                 ticks=np.arange(reg[2][0], reg[2][1]+5, 5))
#             cb32 = fp.format_cbar(cb32,
#                                  cbar_title='Emissions\n' r'(Mg km$^{-2}$ a$^{-1}$)', x=4)

#             fp.save_fig(fig3, plot_dir, 
#                         f'fig_est2_xhat_{emis_label}_reg{j}_{f}')
#             # d_xhat_kwargs['vmin'] = -ul
#             # d_xhat_kwargs['vmax'] = ul

# cax = fp.add_cax(fig, ax, cbar_pad_inches=0.3, horizontal=True)
# cb = fig.colorbar(c, cax=cax, ticks=np.arange(-ul, ul+1, ul/2),
#                   orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title=r'$\Delta$ Emissions (Mg km$^{-2}$ a$^{-1}$)',
#                     horizontal=True)
# # axis = fp.add_title(axis, titles[i])

# fp.save_fig(fig, plot_dir, f'fig_est2_xhat_sectoral_{f}')

## ------------------------------------------------------------------------ ##
## Sectoral attribution bar chart
## ------------------------------------------------------------------------ ##
# w = w*area.values*1e-6 # Convert to Tg/yr

# Open masks and create a total_mask array as well as a mask dictionary
mex_mask = np.load(f'{data_dir}Mexico_mask.npy').reshape((-1, 1))
can_mask = np.load(f'{data_dir}Canada_mask.npy').reshape((-1, 1))
conus_mask = np.load(f'{data_dir}CONUS_mask.npy').reshape((-1, 1))
total_mask = mex_mask + can_mask + conus_mask
masks = {'Canada' : can_mask, 'CONUS' : conus_mask, 'Mexico' : mex_mask}

xhat_abs = (xhat*xa_abs)
print(f'Total prior emissions (Tg/yr)    : {(xa_abs*area*1e-6).sum()}')
print(f'Total posterior emissions        : {(xhat_abs*area*1e-6).sum()}')
print(f'Difference                       : {((xhat_abs - xa_abs)*area*1e-6).sum()}')
print(f'Maximum scale factor             : {xhat.max()}')
print(f'Minimum scale factor             : {xhat.min()}')
print(f'Number of negative scale factors : {(xhat < 0).sum()}')

# fig, ax = fp.get_figax(aspect=2)
# xhat_sub = xhat[dofs >= DOFS_filter]
# ax.hist(xhat_sub, bins=500, density=True, color=fp.color(7))
# ax.axvline(xhat_sub.mean(), color='grey', ls='-')
# ax.axvline(xhat_sub.mean() + xhat_sub.std(), color='grey', ls='--')
# ax.axvline(xhat_sub.mean() - xhat_sub.std(), color='grey', ls='--')
# ax.set_xlim(0, 2)
# fp.save_fig(fig, plot_dir, 'fig_xhat_dist')

country_emis = {}
for country, mask in masks.items():
    sect_emis = pd.DataFrame(columns=['prior', 'post', 'diff',
                                      'prior_sub', 'post_sub', 'diff_sub'])
                                      #'diff_sub_pos', 'diff_sub_neg'])
    w_c = copy.deepcopy(w)*mask # convert to Mg/km2/yr in the country
    w_c *= area*1e-6 # Convert to Tg/yr

    # Get prior and posterior absolute emissions
    sect_emis['prior'] = w_c.sum(axis=0)
    sect_emis['post'] = (w_c*xhat).sum(axis=0)
    sect_emis['diff'] = sect_emis['post'] - sect_emis['prior']

    # Get prior and posterior absolute emissions only where DOFS > DOFS_filter
    sect_emis['prior_sub'] = w_c.loc[dofs >= DOFS_filter, :].sum(axis=0)
    sect_emis['post_sub'] = (w_c*xhat).loc[dofs >= DOFS_filter, :].sum(axis=0)
    sect_emis['diff_sub'] = sect_emis['post_sub'] - sect_emis['prior_sub']

    # Get prior and posterior absolute emissions only where DOFS > DOFS_filter2
    # sect_emis['prior_sub2'] = w_c.loc[:, dofs >= 0.5].sum(axis=1)
    # sect_emis['post_sub2'] = (w_c*xhat).loc[:, dofs >= 0.5].sum(axis=1)
    # sect_emis['diff_sub2'] = sect_emis['post_sub2'] - sect_emis['prior_sub2']

    # Get positive and negative adjustments

    # Reorder sect_emis
    sect_emis = sect_emis.loc[list(emis.values())[1:]]

    country_emis[country] = sect_emis

print(country_emis['CONUS'])

# Plot histogram (at least take #1)
xs = np.arange(0, len(emis.keys()) - 1)
fig, axis = fp.get_figax(aspect=2, cols=3)
j = 0
lims = [[0, 3], [0, 12], [0, 3]]
country_emis_sub = {key : country_emis[key]
                    for key in ['Canada', 'CONUS', 'Mexico']}
for country, sect_emis in country_emis_sub.items():
    ax = axis[j]
    diff_summ = sect_emis['diff'].sum()
    ax.text(0.05, 0.95, country, ha='left', va='top',
            fontsize=config.LABEL_FONTSIZE*config.SCALE,
            transform=ax.transAxes)
    ax.text(0.95, 0.95, f'({diff_summ:0.2f} Tg)', ha='right', va='top',
            fontsize=config.LABEL_FONTSIZE*config.SCALE/1.5,
            transform=ax.transAxes)
    ax.bar(xs - 0.16, sect_emis['prior'], width=0.3,
           color='white', edgecolor=fp.color(2*j+2), label='Prior (all)')
    ax.bar(xs - 0.16, sect_emis['prior_sub'], width=0.3, color=fp.color(2*j+2),
           label='Prior (optimized)')
    # ax.bar(xs - 0.16, sect_emis['prior_sub2'], width=0.3, color=fp.color(2*j+2),
    #        label='Prior (more optimized)')
    ax.bar(xs + 0.16, sect_emis['post'], width=0.3,
           color='white', edgecolor=fp.color(2*j+2), alpha=0.5,
           label='Posterior (all)')
    ax.bar(xs + 0.16, sect_emis['post_sub'], width=0.3,
           color=fp.color(2*j+2), alpha=0.5, 
           label='Posterior (optimized)')
    # ax.bar(xs + 0.16, sect_emis['post_sub2'], width=0.3,
    #        color=fp.color(2*j+2), alpha=0.5, 
    #        label='Posterior (more optimized)')
    # for i, e in enumerate(emis):
    #     ax.arrow(x=i + 0.13, y=sect_emis.at[e, 'prior_sub'],
    #              dx=0, dy=sect_emis.at[e, 'diff_sub_pos'],
    #              color=fp.color(2*j+2), width=0.002, head_width=0.0025)
    #     ax.arrow(x=i + 0.19,
    #              y=sect_emis.at[e, 'prior_sub'] + sect_emis.at[e, 'diff_sub_pos'],
    #              dx=0, dy=sect_emis.at[e, 'diff_sub_neg'],
    #              color=fp.color(2*j+2), width=0.002, head_width=0.0025)
    ax.set_ylim(lims[j])
    # ax.bar(xs + 0.2, sect_emis['post_sub_pos'], width=0.1, color=fp.color(7))
    # ax = fp.add_legend(ax)

    j += 1

axis[0] = fp.add_labels(axis[0], '', 'Emissions\n' + r'[Tg a$^{-1}$]',
                        fontsize=config.TICK_FONTSIZE*config.SCALE/1.5,
                        labelsize=config.TICK_FONTSIZE*config.SCALE/1.5)
for i in range(3):
    if i > 0:
        axis[i] = fp.add_labels(axis[i], '', '',
                                labelsize=config.TICK_FONTSIZE*config.SCALE/1.5)
    axis[i].set_xticks(xs)
    axis[i].set_xticklabels(list(emis.keys())[1:], rotation=90, ha='center',
                            fontsize=config.TICK_FONTSIZE*config.SCALE/1.5)

fp.save_fig(fig, plot_dir, f'fig_sectoral_bar_{f}')

# ## ------------------------------------------------------------------------ ##
# ## Cities analysis
# ## ------------------------------------------------------------------------ ##
# cities = pd.read_csv(f'{data_dir}/uscities.csv')
# ncities = 250
# # print(cities.columns)

# # Order by population and select the top 100
# cities = cities.sort_values(by='population', ascending=False).iloc[:ncities, :]
# cities = cities[['city', 'state_id', 'state_name', 'lat', 'lng',
#                  'population', 'density']]
# cities = cities.rename(columns={'lat' : 'lat_hr', 'lng' : 'lon_hr'})
# cities = cities.reset_index(drop=True)

# # Add in lat centers/lon centers
# lats, lons = gc.create_gc_grid(*s.lats, s.lat_delta, *s.lons, s.lon_delta,
#                                centers=False, return_xarray=False)
# cities['lat'] = lats[gc.nearest_loc(cities['lat_hr'].values, lats)]
# cities['lon'] = lons[gc.nearest_loc(cities['lon_hr'].values, lons)]
# cities['area'] = cities['population']/cities['density']

# # Append posterior
# xa_f = ip.match_data_to_clusters(xa_abs, clusters).rename('xa_abs')
# xa_f = xa_f.to_dataframe().reset_index()
# xhat_f = ip.match_data_to_clusters(xhat, clusters).rename('xhat')
# xhat_f = xhat_f.to_dataframe().reset_index()

# # Join
# cities = cities.merge(xa_f, on=['lat', 'lon'], how='left')
# cities = cities.merge(xhat_f, on=['lat', 'lon'], how='left')

# # Remove areaas with 1 correction (no information content)
# cities = cities[cities['xhat'] != 1]
# print(cities[cities['xhat'] == 0])
# cities = cities[cities['xhat'] != 0] # WHAT CITY IS THIS

# # Plot
# fig, ax = fp.get_figax(rows=1, cols=3, aspect=1.5, sharey=True)
# # ax[0] = fp.add_title(ax[0], 'Population')
# # ax[1] = fp.add_title(ax[1], 'Density')
# # ax[2] = fp.add_title(ax[2], 'Area')
# quantities = ['population', 'density', 'area']
# for i, q in enumerate(quantities):
#     ax[i] = fp.add_title(ax[i], q.capitalize())
#     ax[i].scatter(cities[q], (cities['xhat'] - 1), s=1)
#     ax[i].set_xscale('log')
#     # for j, city in cities.iterrows():
#     #     ax[i].scatter(city[q], (city['xhat'] - 1), s=1)
#     #     ax[i].annotate(city['city'], (city[q], (city['xhat'] - 1)),
#     #                    textcoords='offset points', xytext=(0, 2),
#     #                    ha='center', fontsize=7)
#     # ax.set_xscale('log')
# fp.save_fig(fig, plot_dir, f'cities_test_{f}')


# # print(cities.groupby(['lat_center', 'lon_center']).count().shape)
# # print(cities)
# # for c in range(100):

# ------------------------------------------------------------------------ ##
# Permian comparison
# ------------------------------------------------------------------------ ##
# f = fs[5]
# dofs = np.load(f'{data_dir}posterior/dofs2_{f}.npy')
# xhat = np.load(f'{data_dir}posterior/xhat2_{f}.npy')
# shat = np.load(f'{data_dir}shat_permian.npy') # rg4rt_rf1.0_sax1.0_poi80
# dofs_l = 1 - shat

# Filter
# xhat[dofs < DOFS_filter] = 1
# dofs[dofs < DOFS_filter] = 0

# # Plot averaging kernel
# fig, ax = fp.get_figax(aspect=2)
# print(dofs_l.shape)
# for i in range(20):
#     shat_row = shat[i, :]
#     ax.plot(shat_row, c=fp.color(i, lut=20), lw=0.1)
# fp.save_fig(fig, plot_dir, f'fig_permian_dofs_rg4rt_rf1.0_sax1.0_poi80')

# Combine the Permian clusters with the NA clusters
permian = xr.open_dataset(f'{data_dir}clusters_permian.nc')['Clusters']
c = clusters.squeeze(drop=True).to_dataset()
c['Permian'] = permian

# Get the Permian basin indices (discard the buffer cells)
cell_idx, cell_cnt  = np.unique(c['Permian'], return_counts=True)
cell_idx = cell_idx[cell_cnt == 1]
cell_idx = cell_idx[~np.isnan(cell_idx)]
permian = permian.where(permian.isin(cell_idx), 0)

# # Subset over the Permian
# c = c.where(c['Permian'].isin(cell_idx))['Clusters']
# c = c.sel(lat=permian.lat, lon=permian.lon)

# # Flatten and create boolean
# permian_idx = (ip.clusters_2d_to_1d(permian, c) - 1).astype(int)
permian_idx = np.load(f'{data_dir}permian_idx.npy')
# print(c)
# c[c > 0] = 1

nstate_permian = len(permian_idx)

# for dofs_t in [0.01, 0.05, 0.1, 0.25]:
#     xhat_sub = xhat[dofs >= dofs_t]
#     ja = ((xhat_sub - 1)**2/4).sum()/(len(xhat_sub))
#     print(f'{dofs_t:<5}{xhat_sub.min():.2f}  {xhat_sub.max():.2f}  {ja:.2f}')

# Subset the posterior
# nscale = 1
xhat_permian = xhat[permian_idx, :]
# xhat_permian = ((xhat_permian - 1)*nscale + 1)
xa_abs_permian = xa_abs[permian_idx, :]
# xhat_abs_permian = xhat_permian*xa_abs_permian
xhat_abs_permian = xhat_abs[permian_idx, :]
# soil_permian = soil[permian_idx]
dofs_permian = dofs[permian_idx, :]
area_permian = area[permian_idx, :]
# a_permian = np.load(f'{data_dir}posterior/a2_rg4rt_edf_permian.npy')
# shat_permian = shat[:, permian_idx]

# fig, ax = fp.get_figax(aspect=1)
# cax = fp.add_cax(fig, ax)
# c = ax.matshow(shat_permian, cmap='RdBu_r', vmin=-0.001, vmax=0.001)
# cb = fig.colorbar(c, cax=cax)#, ticks=np.arange(0, 6, 1))
# cb = fp.format_cbar(cb, cbar_title=r'$\hat{S}$')
# fp.save_fig(fig, plot_dir, f'fig_permian_shat_rg4rt_rf1.0_sax1.0_poi80')

# # Subset the posterior errors (this needs to be done remotely)
# shat = np.load(f'/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results/iteration2/shat/shat2_rf1.0_sax1.0_poi80.npy')
# permian_idx = np.load(f'/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results/permian_idx.npy')
# xa_abs = xr.open_dataarray(f'/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results/xa_abs.nc')
# xa_abs_permian = xa_abs[permian_idx]
# nstate_permian = len(xa_abs_permian)
# shat_permian = shat[permian_idx, permian_idx]
# shat_abs_permian = shat_permian*xa_abs_permian.values.reshape((-1, 1))*xa_abs_permian.values.reshape((1, -1))
# shat_abs_permian = xr.DataArray(shat_abs_permian, dims=('nstate1', 'nstate2'),
#                                 coords={'nstate1' : np.arange(1, nstate_permian+1),
#                                         'nstate2' : np.arange(1, nstate_permian+1)})
# shat_abs_permian.to_netcdf(f'/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/inversion_data/shat_abs_permian.nc')


# Save out
# permian.to_netcdf(f'{data_dir}clusters_permian.nc')

# xa_abs_permian = xr.DataArray(xa_abs_permian.reshape(-1,), 
#                               dims=('nstate'),
#                               coords={'nstate' : np.arange(1, nstate_permian+1)})
# # xa_abs_permian.to_netcdf(f'{data_dir}xa_abs_edf_permian.nc')

# xhat_abs_permian = xr.DataArray(xhat_abs_permian.reshape(-1,), 
#                                 dims=('nstate'),
#                                 coords={'nstate' : np.arange(1, nstate_permian+1)})
# # xhat_abs_permian.to_netcdf(f'{data_dir}xhat_abs_edf_permian.nc')

# Adjust units to Tg/yr
xa_abs_permian *= area_permian*1e-6 # Tg/yr
xhat_abs_permian *= area_permian*1e-6

# Calculate emissions
tot_prior_permian = xa_abs_permian.sum()
tot_post_permian = xhat_abs_permian.sum()
print(f'Minimum correction               : {xhat_permian.min()}')
print(f'Maximum correction               : {xhat_permian.max()}')
print(f'Median correction                : {np.median(xhat_permian)}')
print(f'Mean correction                  : {xhat_permian.mean()}')

print(f'Total prior emissions            : {tot_prior_permian}')
print(f'Total posterior emissions        : {tot_post_permian}')
print(f'Difference                       : {(tot_post_permian - tot_prior_permian)}')

# Adjust back to kg/km2/hr
xa_abs_permian = xa_abs_permian/area_permian/1e-9/(365*24)
xhat_abs_permian = xhat_abs_permian/area_permian/1e-9/(365*24)
# print(xa_abs_permian)
# print(xhat_permian)

fig, axis = fp.get_figax(rows=2, cols=2, maps=True,
                         lats=permian.lat, lons=permian.lon)
plt.subplots_adjust(hspace=0.1, wspace=1)

# Plot prior
fig_kwargs = {'figax' : [fig, axis[0, 0]]}
xhat_kwargs = {'cmap' : yor_trans, 'vmin' : 0, 'vmax' : 13,
               'default_value' : 0,
               'map_kwargs' : small_map_kwargs,
               'fig_kwargs' : fig_kwargs}
title = f'Prior emissions' # ({f}\%)'
fig, axis[0, 0], c = ip.plot_state(xa_abs_permian, permian, title=title, 
                                   **xhat_kwargs)
axis[0, 0].text(0.05, 0.05, f'{tot_prior_permian:.1f} Tg/yr',
                fontsize=config.LABEL_FONTSIZE*config.SCALE*0.75,
                transform=axis[0, 0].transAxes)


# Plot posterior emissions
fig_kwargs = {'figax' : [fig, axis[0, 1]]}
xhat_cbar_kwargs = {'title' : r'Emissions\\(kg km$^{-2}$ h$^{-1}$)'}
xhat_kwargs['fig_kwargs'] = fig_kwargs
xhat_kwargs['cbar_kwargs'] = xhat_cbar_kwargs
title = f'Posterior emissions' # ({f}\%)'
fig, axis[0, 0], c = ip.plot_state(xhat_abs_permian, permian, title=title, 
                                   **xhat_kwargs)
axis[0, 1].text(0.05, 0.05, f'{tot_post_permian:.1f} Tg/yr',
                fontsize=config.LABEL_FONTSIZE*config.SCALE*0.75,
                transform=axis[0, 1].transAxes)

# Plot posterior scaling factors
sf_cmap_1 = plt.cm.Oranges(np.linspace(0, 1, 256))
sf_cmap_2 = plt.cm.Purples(np.linspace(1, 0, 256))
sf_cmap = np.vstack((sf_cmap_2, sf_cmap_1))
sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
div_norm = colors.TwoSlopeNorm(vmin=0.5, vcenter=1, vmax=2)

xhat_cbar_kwargs = {'title' : r'Scale factor'}
fig_kwargs = {'figax' : [fig, axis[1, 0]]}
xhat_kwargs = {'cmap' : sf_cmap, 'norm' : div_norm,
               'default_value' : 1,
               'cbar_kwargs' : xhat_cbar_kwargs,
               'map_kwargs' : small_map_kwargs,
               'fig_kwargs' : fig_kwargs}
title = f'Posterior\nscale factors' # ({f}\%)'
fig, axis[1, 0], c = ip.plot_state(xhat_permian, permian, title=title,
                                 **xhat_kwargs)
axis[1, 0].text(0.05, 0.05,
                f'({(xhat_permian.min()):.1f}, {(xhat_permian.max()):.1f})',
                fontsize=config.LABEL_FONTSIZE*config.SCALE*0.75,
                transform=axis[1, 0].transAxes)

# Plot DOFS
fig_kwargs = {'figax' : [fig, axis[1, 1]]}
avker_cbar_kwargs = {'title' : r'$\partial\hat{x}_i/\partial x_i$'}
avker_kwargs = {'cmap' : plasma_trans, 'vmin' : 0, 'vmax' : 1,
                'cbar_kwargs' : avker_cbar_kwargs,
                'map_kwargs' : small_map_kwargs,
                'fig_kwargs' : fig_kwargs}
title = f'Averaging kernel\nsensitivities' # ({f}\%)'
fig, axis[1, 1], c = ip.plot_state(dofs_permian, permian, title=title,
                                 **avker_kwargs)
axis[1, 1].text(0.05, 0.05,
                f'DOFS = {dofs_permian.sum():.1f}',
                fontsize=config.LABEL_FONTSIZE*config.SCALE*0.75,
                transform=axis[1, 1].transAxes)


fp.save_fig(fig, plot_dir, f'fig_est2_xhat_permian_{f}')
plt.close()

fig, ax = fp.get_figax(aspect=1)
ax.hist(xa_abs_permian, bins=50, alpha=0.5)
ax.hist(xhat_abs_permian, bins=50, alpha=0.5)

# ax.scatter(xa_abs_permian, xhat_permian)
fp.save_fig(fig, plot_dir, f'fig_est2_xhat_permian_scatter_{f}')


# # Plot rows of the averaging kernel
# fig, ax = fp.get_figax(aspect=2)
# i = 0
# # for row in a_permian:
#     # ax.plot(row, c=fp.color(i, lut=nstate_permian), lw=0.1)
#     # i += 1
# ax.plot(a_permian[100, :], c=fp.color(100, lut=nstate_permian), lw=1)
# ax.set_xlim(1e4, 23691)
# fp.save_fig(fig, plot_dir, f'fig_est2_a_permian_{f}')

# print(i)

