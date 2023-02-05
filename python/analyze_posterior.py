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
# config.SCALE = config.PRES_SCALE
# config.BASE_WIDTH = config.PRES_WIDTH
# config.BASE_HEIGHT = config.PRES_HEIGHT
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
rdbu_trans = fp.cmap_trans_center('RdBu_r', nalpha=70)

sf_cmap_1 = plt.cm.PuOr_r(np.linspace(0.2, 0.5, 256))
sf_cmap_2 = plt.cm.PuOr_r(np.linspace(0.5, 1, 256))
sf_cmap = np.vstack((sf_cmap_1, sf_cmap_2))
sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
div_norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=1, vmax=3)

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
emis = {'Total' : 'total', 'Livestock' : 'livestock', 
        'Oil and gas' : 'ong', 'Coal' : 'coal', 
        'Landfills' : 'landfills', 'Wastewater' : 'wastewater', 
        'Other anthropogenic' : 'other_anth',
        'Wetlands' : 'wetlands',
        'Other biogenic' : 'other_bio'}

# Compare to other studies
lu2022 = {'livestock' : [10.6, 9.2, 11.8], 
          'ong' : [14.5, 12.1, 15.1], 
          'coal' : [2.8, 2.4, 3.5], 
          'landfills' : [7.2, 6.0, 7.6], 
          'wastewater' : [0.63, 0.56, 0.74],
          'wetlands' : [8.4, 6.4, 10.6], 
          'other_anth' : [0.45, 0.44, 0.54]}
lu2022 = pd.DataFrame(data=lu2022, index=['mean', 'min', 'max'])

shen2022 = pd.DataFrame(data={'ong' : [12.6, 2.1, 2.1]}, 
                        index=['mean', 'min', 'max'])

# MMT CO2 eq
epa2019 = {'livestock' : [176.1 + 58.7,
                          176*0.1 + 58.7*0.18, 
                          176*0.1 + 58.7*0.2],
           'ong' : [172.2 + 7.0 + 40.4, 
                    172.2*0.18 + 7.0*0.83 + 40.4*0.28,
                    172.2*0.18 + 7.0*1.97 + 40.4*0.32],
           'coal' : [47.4 + 5.9,
                     47.4*0.09 + 5.9*0.22,
                     47.4*0.17 + 5.9*0.2],
           'landfills' : [113.6, 113.6*0.23, 113.6*0.22],
           'wastewater' : [18.1, 18.1*0.35, 18.1*0.23],
           'other_anth' : [15.1 + 8.8 + 2.5 + 0.4 + 0.3 + 0.2,
                           15.1*0.75 + 8.8*0.34 + 2.5*0.08 + 0.4*0.18 + 0.3*0.57 + 0.2*0.54 + 2.3*0.58,
                           15.1*0.75 + 8.8*1.25 + 2.5*0.24 + 0.4*0.18 + 0.3*0.46 + 0.2*0.54 + 2.3*0.58]}
epa2019 = pd.DataFrame(data=epa2019, index=['mean', 'min', 'max'])/25
# x Enteric fermentation 176.1
# x Manure management 58.7
# x Natural gas systems 172.2
# x Abandoned wells 7.0
# x Petroleum systems 40.4
# x landfills 113.6
# x coal mining 47.4
# x Abandoned coal 5.9
# x wastewater 18.1
# x rice 15.1
# x stationary combustion 8.8
# x mobile combustion 2.5
# x field burning 0.4
# x petrochemical production 0.3 <- should this go in ong?
# x anaroebic digestion 0.2
# x composting 2.3

# Get ensemble values
ensemble = glob.glob(f'{data_dir}ensemble/xhat_fr2*')
ensemble.sort()
ensemble = [f.split('/')[-1][9:-12] for f in ensemble
            if f.split('10t_')[-1].split('_')[0] == 'w37']

# ID two priors and boundary condition elements
w37_cols = [s for s in ensemble if 'w37' in s]
# w404_cols = [s for s in ensemble if 'w404' in s]
bc_cols = [s for s in ensemble if s[:2] == 'bc']
anth_cols = ['livestock', 'ong', 'coal', 'landfills', 
             'wastewater', 'other_anth']
bio_cols = ['wetlands', 'other_bio']
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
# xa_abs_dict = {'w404_edf' : xa_w404, 'w37_edf' : xa_w37}
area = xr.open_dataarray(f'{data_dir}area.nc').values
soil = xr.open_dataarray(f'{data_dir}soil_abs_correct.nc').values
nstate = area.shape[0]

# Load weighting matrix (Mg/yr)
w_w404 = pd.read_csv(f'{data_dir}w_w404_edf.csv')
w_w37 = pd.read_csv(f'{data_dir}w_w37_edf.csv')
w = {'w404_edf' : w_w404, 'w37_edf' : w_w37}
w_mask = {}
for wkey, ww in w.items():
    w[wkey]['total'] = w[wkey].sum(axis=1)
    w[wkey]['total_anth'] = w[wkey][anth_cols].sum(axis=1)
    w[wkey]['total_bio'] = w[wkey][bio_cols].sum(axis=1)
    # w[wkey]['net'] = xa_abs_dict[wkey]*area
    # w[wkey]['other'] = w[wkey]['other_bio'] + w[wkey]['other_anth']
    # w[wkey] = w[wkey].drop(columns=['other_bio'])#, 'other_anth'])
    w[wkey] = w[wkey].T

    # Sectoral masks
    w_mask[wkey] = dc(w[wkey])
    w_mask[wkey] = w_mask[wkey].where(w[wkey]/1e3 > 0.5, 0)
    w_mask[wkey] = w_mask[wkey].where(w[wkey]/1e3 <= 0.5, 1)

# Create dataframes for the ensemble data
dofs = pd.DataFrame(columns=ensemble)
xa_abs = pd.DataFrame(columns=ensemble) # Total emissions, no soil absorption
xa_abs_anth = pd.DataFrame(columns=ensemble) # Anthropogenic emissions
xa_abs_bio = pd.DataFrame(columns=ensemble) # Bipogenic emissions
xhat = pd.DataFrame(columns=ensemble)
shat_abs = pd.DataFrame(columns=ensemble)

# And for the BC data
dofs_bc = pd.DataFrame(columns=bc_cols, index=['N', 'S', 'E', 'W'])
xhat_bc = pd.DataFrame(columns=bc_cols, index=['N', 'S', 'E', 'W'])
shat_bc = pd.DataFrame(columns=bc_cols, index=['N', 'S', 'E', 'W'])

# Iterate throuugh the ensemble to load the data
for suff in ensemble:
    # Get string information about the ensemble member
    short_suff = suff.split('rg2rt_10t_')[-1].split('_bc0')[0]
    sa_scale = float(suff.split('_sax')[-1].split('_')[0])

    # Load the files
    dofs_s = np.load(f'{data_dir}ensemble/dofs2_{suff}_poi80.0.npy')
    xhat_s = np.load(f'{data_dir}ensemble/xhat_fr2_{suff}_poi80.0.npy')
    shat_s = np.load(f'{data_dir}ensemble/shat_kpi2_{suff}_poi80.0.npy')

    # Filter on the DOFS filter
    xhat_s[dofs_s < DOFS_filter] = 1
    dofs_s[dofs_s < DOFS_filter] = 0
    shat_s[dofs_s < DOFS_filter] = sa_scale**2

    # If the ensemble member optimizes the boundary conditions, save
    # out the boundary condition and grid cell elements separately
    if suff[:2] == 'bc':
        optimize_bc = True

        # Add BC elements
        dofs_bc[suff] = dofs_s[-4:]
        xhat_bc[suff] = xhat_s[-4:]
        shat_bc[suff] = shat_s[-4:]

        # Shorten results
        xhat_s = xhat_s[:-4]
        dofs_s = dofs_s[:-4]
        shat_s = shat_s[:-4]

    # Save out the resulting values to the dataframe
    dofs[suff] = dofs_s
    xa_abs[suff] = w[short_suff].loc['total']
    xa_abs_anth[suff] = w[short_suff].loc['total_anth']
    xa_abs_bio[suff] = w[short_suff].loc['total_bio']
    xhat[suff] = xhat_s
    # shat_abs[suff] = shat_s*(xa_abs_dict[short_suff]**2)

# Calculate xhat_abs
xhat_abs = xa_abs*xhat
xhat_abs_anth = xa_abs_anth*xhat
xhat_abs_bio = xa_abs_bio*xhat

# Calculate the statistics of the posterior solution
dofs_mean = dofs.mean(axis=1)
xa_abs_mean = xa_abs.mean(axis=1)
xhat_abs_mean = xhat_abs.mean(axis=1)
xhat_mean = xhat_abs_mean/xa_abs_mean

# Save out the results
dofs.to_csv(f'{data_dir}ensemble/dofs.csv')
xhat.to_csv(f'{data_dir}ensemble/xhat.csv')

# # Calculate the posterior error covariance from the ensemble
# shat_e = (xhat_abs - xhat_abs_mean) @ (xhat_abs - xhat_abs_mean).T
# print(shat_e.shape)

# Get BC statistics
bc_stats = ip.get_ensemble_stats(xhat_bc)
bc_dofs_states = ip.get_ensemble_stats(dofs_bc)

# BC alteration
if optimize_bc:
    print('-'*75)
    print('Boundary condition optimization')
    print(bc_stats.round(2))
    print(bc_dofs_states.round(4))
    print('-'*75)

# Calculate statistics and print results
xa_abs_opt_frac = xa_abs[dofs > DOFS_filter].sum(axis=0)/xa_abs.sum(axis=0)
xa_abs_anth_opt_frac = xa_abs_anth[dofs > DOFS_filter].sum(axis=0)/xa_abs_anth.sum(axis=0)
xhat_abs_tot = (xhat_abs*1e-6).sum(axis=0)
xhat_abs_anth_tot = (xhat_abs_anth*1e-6).sum(axis=0)
n_opt = (dofs > DOFS_filter).sum(axis=0)
negs = (xhat_abs < 0).sum(axis=0)

print(f'We optimize {n_opt.mean():.0f} ({n_opt.min():d}, {n_opt.max():d}) grid cells if we analyze each ensemble member\nindividually. If we consider those grid cells that are included in the\nensemble mean, we optimize {(dofs_mean > DOFS_filter).sum():d} ({(dofs_mean > 0).sum():d}) grid cells.')
print('')
print(f'Across the ensemble, we optimize {(xa_abs_opt_frac.mean()*100):.1f} ({(xa_abs_opt_frac.min()*100):.1f} - {(xa_abs_opt_frac.max()*100):.1f})% of prior emissions\nand {(xa_abs_anth_opt_frac.mean()*100):.1f} ({(xa_abs_anth_opt_frac.min()*100):.1f} - {(xa_abs_anth_opt_frac.max()*100):.1f})% of prior anthropogenic emissions.')
print('')
print(f'This produces a mean of of {dofs.sum(axis=0).mean():.1f} ({dofs.sum(axis=0).min():.1f}, {dofs.sum(axis=0).max():.1f}) DOFS.')
print('')
print(f'There are {negs.mean():.0f} ({negs.min():d}, {negs.max():d}) new negative values. If we consider those grid cells\nthat are included in the ensemble mean, there are {(xhat_abs_mean[xa_abs_mean >= 0] < 0).sum():d} new negative values.')#\nwith a median of {xhat_abs[xhat_abs < 0].median(axis=0).mean():.2f} ({xhat_abs[xhat_abs < 0].median(axis=0).min():.2f} - {xhat_abs[xhat_abs < 0].median(axis=0).max():.2f}) Mg/yr. (The prior has\nminimum negative values of {(xa_w404*area).min():.2f} and {(xa_w37*area).min():.2f}.) (The median positive\nposterior value is {xhat_abs[xhat_abs >= 0].median(axis=0).mean():.2f} ({xhat_abs[xhat_abs >= 0].median(axis=0).min():.2f} - {xhat_abs[xhat_abs >= 0].median(axis=0).max():.2f}) Mg/yr.)')
print('')
# print(f'Total prior emissions (Tg/yr)    : {(xa_abs_mean*area*1e-6).sum():.2f}')
print(f'Total posterior emissions               : {xhat_abs_tot.mean():.2f} ({xhat_abs_tot.min():.2f}, {xhat_abs_tot.max():.2f}) Tg/yr')
print(f'Total anthropogenic posterior emissions : {xhat_abs_anth_tot.mean():.2f} ({xhat_abs_anth_tot.min():.2f}, {xhat_abs_anth_tot.max():.2f}) Tg/yr')
print('-'*75)

# Get county outlines for high resolution results
reader = shpreader.Reader(f'{data_dir}counties/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

## ------------------------------------------------------------------------ ##
## Plot posterior
## ------------------------------------------------------------------------ ##
fig, ax = ip.plot_posterior(xhat.mean(axis=1), dofs.mean(axis=1), clusters)
fp.save_fig(fig, plot_dir, f'posterior_ensemble')

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

## ------------------------------------------------------------------------ ##
## Negative values histogram
## ------------------------------------------------------------------------ ##
fig, ax = fp.get_figax(aspect=2)
for i, member in enumerate(xhat_abs.columns):
    # print((xhat_abs[member] < 0).sum())
    ax.hist(xhat_abs[member]/area, bins=np.arange(-25, 55, 1), density=True,
            color=fp.color(i*2, lut=30), alpha=0.5, histtype='step')
ax.axvline(soil.mean(), color='black', ls='--', lw=0.5, 
           label='Mean soil absorption')
ax.set_xlim(-25, 50)
ax = fp.add_labels(ax, r'Posterior emissions (Mg km$^{-2}$ a$^{-1}$)', 
                  'Density')
ax = fp.add_legend(ax)
fp.save_fig(fig, plot_dir, 'negative_values_ensemble')

## ------------------------------------------------------------------------ ##
## Sectoral attribution bar chart
## ------------------------------------------------------------------------ ##
# Convert the W matrices to Tg/yr
w = {wkey : ww*1e-6 for wkey, ww in w.items()}

# Open masks and create a total_mask array as well as a mask dictionary
mex_mask = np.load(f'{data_dir}Mexico_mask.npy').reshape((-1,))
can_mask = np.load(f'{data_dir}Canada_mask.npy').reshape((-1,))
conus_mask = np.load(f'{data_dir}CONUS_mask.npy').reshape((-1,))
other_mask = np.load(f'{data_dir}Other_mask.npy').reshape((-1,))
# total_mask = mex_mask + can_mask + conus_mask
masks = {'Canada' : can_mask, 'CONUS' : conus_mask, 'Mexico' : mex_mask,
         'Other' : other_mask}

print('-'*75)
total = emis.pop('Total')
summ_c = {}
for country, mask in masks.items():
    # Posterior terms
    post_c = ((dc(w['w37_edf'])*mask) @ xhat[w37_cols]).T
    # post_w404_c = (dc(w['w404_edf'])*mask) @ xhat[w404_cols]
    # post_c = pd.concat([post_w37_c, post_w404_c], axis=1).T
    post_tot_anth = post_c[anth_cols].sum(axis=1) # Calculate total
    post_tot_bio = post_c[bio_cols].sum(axis=1)
    post_tot = post_tot_anth + post_tot_bio
    post_c = post_c[list(emis.values())].add_prefix('post_')
    # if country == 'Mexico':
    #     print(post_c['post_ong'])

    # Only optimized terms
    xhat_sub = dc(xhat)
    xhat_sub[dofs < DOFS_filter] = 0
    post_sub_c = ((dc(w['w37_edf'])*mask) @ xhat_sub[w37_cols]).T
    # post_sub_w404_c = (dc(w['w404_edf'])*mask) @ xhat_sub[w404_cols]
    # post_sub_c = pd.concat([post_sub_w37_c, post_sub_w404_c], axis=1).T
    post_sub_c = post_sub_c[list(emis.values())].add_prefix('post_sub_')

    # Prior terms
    prior_c = (dc(w['w37_edf'])*mask) @ np.ones(xhat[w37_cols].shape)
    # prior_w404_c = (dc(w['w404_edf'])*mask) @ np.ones(xhat[w404_cols].shape)
    # prior_c = pd.concat([prior_w37_c, prior_w404_c], axis=1)
    prior_c = prior_c.T.reset_index(drop=True)
    prior_c = prior_c.rename({i : post_c.index[i] 
                              for i in range(prior_c.shape[0])})
    prior_tot = prior_c[anth_cols].sum(axis=1) # Calculate total
    prior_c = prior_c[list(emis.values())].add_prefix('prior_')

    # Only optimized terms
    xa_sub = pd.DataFrame(data=np.ones(xa_abs.shape), columns=xhat.columns)
    xa_sub[dofs < DOFS_filter] = 0
    prior_sub_c = ((dc(w['w37_edf'])*mask) @ xa_sub[w37_cols]).T
    # prior_sub_w404_c = (dc(w['w404_edf'])*mask) @ xa_sub[w404_cols]
    # prior_sub_c = pd.concat([prior_sub_w37_c, prior_sub_w404_c], axis=1).T
    prior_sub_c = prior_sub_c[list(emis.values())].add_prefix('prior_sub_')

    # Get statistics
    post_c = ip.get_ensemble_stats(post_c.T)
    post_sub_c = ip.get_ensemble_stats(post_sub_c.T)
    prior_c = ip.get_ensemble_stats(prior_c.T)
    prior_sub_c = ip.get_ensemble_stats(prior_sub_c.T)

    # Calculate error and save out
    summ = pd.concat([prior_c, prior_sub_c, post_c, post_sub_c], axis=0)
    summ['min'] = summ['mean'] - summ['min']
    summ['max'] = summ['max'] - summ['mean']
    summ.to_csv(f'{base_dir}paper/{country}.csv')

    # Remove bio other
    summ = summ.drop(index=['prior_other_bio', 'prior_sub_other_bio',
                            'post_other_bio', 'post_sub_other_bio'])

    # Save to dictionary
    summ_c[country] = summ

    # DOFS
    dofs_c = (dofs*mask.reshape((-1, 1))).sum(axis=0)

    # Reduced DOFS
    if country == 'CONUS':
        a_files = glob.glob(f'{data_dir}ensemble/a2_*_{country.lower()}.csv')
        a_files = [f for f in a_files if 'w37' in f]
        a_files.sort()
        print(a_files)
        dofs_r = pd.DataFrame(index=list(emis.values()), columns=xhat.columns)
        for f in a_files:
            short_f = f.split('/')[-1][3:-19]
            a_r = pd.read_csv(f, header=0, index_col=0)
            a_r = a_r.rename(columns={i : a_r.index[int(i)] 
                                      for i in a_r.columns})
            a_r = a_r.loc[list(emis.values()), list(emis.values())]
            dofs_r[short_f] = np.diag(a_r)
        dofs_r = ip.get_ensemble_stats(dofs_r).add_prefix('dofs_')

    # Print information 
    xa_abs_opt_frac = (xa_abs*mask.reshape((-1, 1)))[dofs > DOFS_filter].sum(axis=0)/(xa_abs*mask.reshape((-1, 1))).sum(axis=0)
    xa_abs_anth_opt_frac = (xa_abs_anth*mask.reshape((-1, 1)))[dofs > DOFS_filter].sum(axis=0)/(xa_abs_anth*mask.reshape((-1, 1))).sum(axis=0)
    xa_abs_bio_opt_frac = (xa_abs_bio*mask.reshape((-1, 1)))[dofs > DOFS_filter].sum(axis=0)/(xa_abs_bio*mask.reshape((-1, 1))).sum(axis=0)
    print(f'In {country}, we achieve {dofs_c.mean():.1f} ({dofs_c.min():.1f}, {dofs_c.max():.1f}) DOFS and optimize {(xa_abs_opt_frac.mean()*100):.1f} ({(xa_abs_opt_frac.min()*100):.1f} - {(xa_abs_opt_frac.max()*100):.1f})%\nof prior emissions and {(xa_abs_anth_opt_frac.mean()*100):.1f} ({(xa_abs_anth_opt_frac.min()*100):.1f} - {(xa_abs_anth_opt_frac.max()*100):.1f})% of prior anthropogenic emissions\nand {(xa_abs_bio_opt_frac.mean()*100):.1f} ({(xa_abs_bio_opt_frac.min()*100):.1f} - {(xa_abs_bio_opt_frac.max()*100):.1f})% of prior biogenic emissions. Anthropogenic emissions\nchange by on average {(post_tot_anth - prior_tot).mean():.2f} ({(post_tot_anth - prior_tot).min():.2f}, {(post_tot_anth - prior_tot).max():.2f}) Tg/yr from {prior_tot.mean():.2f} Tg/yr to\n{post_tot_anth.mean():.2f} ({post_tot_anth.min():.2f}, {post_tot_anth.max():.2f}) Tg/yr. Biogenic emissions are {post_tot_bio.mean():.2f} ({post_tot_bio.min():.2f}, {post_tot_bio.max():.2f}). Total\nemissions are {post_tot.mean():.2f} ({post_tot.min():.2f}, {post_tot.max():.2f}).') #{prior_c.sum(axis=1)}')
    if country == 'CONUS':
        print(dofs_r)
    print('-'*75)

# Plot histogram (at least take #1)
other_c = summ_c.pop('Other')
other_bio = emis.pop('Other biogenic')

fig, ax = fp.get_figax(aspect=1.5, max_width=config.BASE_WIDTH/2)
ys = np.arange(1, len(emis) + 1)
cc = [fp.color(i, lut=2*len(emis)) for i in [0, 6, 10, 2, 8, 12, 4]]
post_rows = [f'post_{e}' for e in emis.values()]
post_sub_rows = [f'post_sub_{e}' for e in emis.values()]
prior_rows = [f'prior_{e}' for e in emis.values()]
prior_sub_rows = [f'prior_sub_{e}' for e in emis.values()]

# Plot CONUS
c = 'CONUS'
e = summ_c[c]

# Title
ax = fp.add_title(ax, f'{c} sectoral emissions', 
                  fontsize=config.TITLE_FONTSIZE)

# Plot the total bar (all emissions, unoptimized and optimized)
ax.barh(ys - 0.175, e['mean'][prior_rows], height=0.3, color=cc, alpha=0.3)
ax.barh(ys + 0.175, e['mean'][post_rows], height=0.3, color=cc, alpha=0.3)

# Plot the optimized emissions
# Prior
ax.barh(ys - 0.175, e['mean'][prior_sub_rows], 
        left=e['mean'][prior_rows].values - e['mean'][prior_sub_rows].values,
        height=0.3, color=cc)

# If the optimized emissions are negative, force it to 0 (this is
# incorrect but will make the plot look correct)
e['mean'][post_sub_rows] = e['mean'][post_sub_rows].clip(lower=0)
ax.barh(ys + 0.175, e['mean'][post_sub_rows],
        left=e['mean'][post_rows].values - e['mean'][post_sub_rows].values,
        xerr=np.array(e[['min', 'max']].loc[post_rows]).T,
        error_kw={'ecolor' : '0.6', 'lw' : 0.5, 'capsize' : 2, 
                  'capthick' : 0.5},
        height=0.3, color=cc)

# # Add prior wetlands error bar
# ax.errorbar(e['mean']['prior_wetlands'], ys[-1] - 0.175, 
#             xerr=np.array(e[['min', 'max']].loc['prior_wetlands'])[:,None],
#             ecolor='0.6', lw=0.5, capsize=1, capthick=0.5)

# Plot EPA 2019
epa2019 = epa2019[list(emis.values())[:-1]]
ax.errorbar(epa2019.loc['mean'], ys[:-1] - 0.175,
            xerr=np.array(epa2019.loc[['min', 'max']]),
            fmt='o', markersize=2, zorder=20,
            markerfacecolor='0.6', markeredgecolor='white', 
            ecolor='0.6', lw=0.5, capsize=1, capthick=0.5,
            label='2019 EPA GHGI (2022)')

# Reorder Lu data
lu2022 = lu2022[emis.values()]
lu2022.loc['min'] = lu2022.loc['mean'] - lu2022.loc['min']
lu2022.loc['max'] = lu2022.loc['max'] - lu2022.loc['mean']
ax.errorbar(lu2022.loc['mean'], ys + 0.175, 
            xerr=np.array(lu2022.loc[['min', 'max']]), fmt='s', 
            markersize=2, markerfacecolor='0.6', markeredgecolor='white', 
            ecolor='0.6', lw=0.5, capsize=1, capthick=0.5, 
            label='Lu et al. (2022)')

# Plot Shen data
vs = np.array(list(emis.values()))
y = np.argwhere('ong' == vs)[0][0] + 1
ax.errorbar(shen2022['ong']['mean'], y + 0.175 + 0.05, 
            xerr=np.array(shen2022['ong'][['min', 'max']])[:, None], fmt='^', 
            markersize=2, markerfacecolor='0.6', markeredgecolor='white', 
            ecolor='0.6', lw=0.5, capsize=1, capthick=0.5, 
            label='Shen et al. (2022)')


ax.set_xlim(0, 15.2)
ax.set_ylim(0.5, 7.5)

# Add labels
ax.set_yticks(ys)
ax.set_yticklabels('', ha='right',
                        fontsize=config.TICK_FONTSIZE)
ax.invert_yaxis()
ax = fp.add_labels(ax, r'Emissions (Tg a$^{-1}$)', '',
                        fontsize=config.TICK_FONTSIZE,
                        labelsize=config.TICK_FONTSIZE,
                        labelpad=10)
for j in range(6):
    ax.axhline((j + 1) + 0.5, color='0.75', lw=0.5, zorder=-10)

ax.set_yticklabels(list(emis.keys()), ha='right',
                        fontsize=config.TICK_FONTSIZE)

# Add prior/posterior labels 
left = summ_c['CONUS']['mean'] + summ_c['CONUS']['max']
ax.text(0.1, ys[0] - 0.15, # summ_c['CONUS']['mean']['prior_livestock'] + 0.25
       'Prior', ha='left', va='center', fontsize=config.TICK_FONTSIZE - 2)
ax.text(0.1, ys[0] + 0.2, 'Posterior',  #left['post_livestock'] + 0.25
        ha='left', va='center', fontsize=config.TICK_FONTSIZE - 2)

# Add grid lines
for j in range(2):
    ax.axvline(5*(j + 1), color='0.75', lw=0.5, zorder=-10)

# Add legend
custom_patches = [patch(color='0.5', alpha=0.3),
                  patch(color='0.5', alpha=1)]
                  # patch(color=fp.color(0), alpha=0.3)]
custom_labels = [r'Not optimized (A$_{ii}$ $<$'f' {DOFS_filter})',
                 r'Optimized (A$_{ii}$ $\ge$'f' {DOFS_filter})']
                # #'Lu et al. (2022)']
patches, labels = ax.get_legend_handles_labels()
custom_patches.extend(patches)
custom_labels.extend(labels)
fp.add_legend(ax, handles=custom_patches, labels=custom_labels,
              bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=2,
              fontsize=config.TICK_FONTSIZE)

fp.save_fig(fig, plot_dir, f'sectors_bar_ensemble')

plt.close()

## ------------------------------------------------------------------------ ##
## Sectoral attribution maps
## ------------------------------------------------------------------------ ##
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
#     post_w37_c = dc(w['w37_edf']).loc[emis_label].values[:, None]*(xhat[w37_cols] - 1)
#     post_w404_c = dc(w['w404_edf']).loc[emis_label].values[:, None]*(xhat[w404_cols] - 1)
#     post_c = pd.concat([post_w37_c, post_w404_c], axis=1).T
#     xhat_diff_sect_i = post_c.mean(axis=0).values/area*1e6 # convert from Tg/yr to Mg/km2/yr

#     # if title != 'Total':
#     #     xhat_sect_i = (xhat - 1)*w_mask[emis_label].values.reshape(-1, 1) + 1
#     #     fig2b, ax2b, c2b = ip.plot_state(xhat_sect_i, clusters, 
#     #                                      title=f'{title} scaling factors',
#     #                                      **xhat_kwargs)
#     #     fp.save_fig(fig2b, plot_dir, f'xhat_sf_{emis_label}_{f}')

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
#     # diff_sec_tot = (xhat_diff_sect_i.reshape((-1, 1))*area)
#     # diff_sec_tot_pos = diff_sec_tot[diff_sec_tot > 0].sum()*1e-6
#     # diff_sec_tot_neg = diff_sec_tot[diff_sec_tot <= 0].sum()*1e-6
#     # diff_sec_tot = diff_sec_tot.sum()*1e-6
#     # axis.text(0.05, 0.05, f'{diff_sec_tot:.2f} Tg', ha='left', va='bottom',
#     #           fontsize=config.LABEL_FONTSIZE*config.SCALE,
#     #           transform=axis.transAxes)
#     # ax2.text(0.05, 0.05, f'{diff_sec_tot:.2f} Tg ({diff_sec_tot_neg:.2f} Tg, {diff_sec_tot_pos:.2f} Tg)', ha='left', va='bottom',
#     #          fontsize=config.LABEL_FONTSIZE*config.SCALE,
#     #          transform=ax2.transAxes)

#     # Save out individual plot
#     fp.save_fig(fig2, plot_dir, f'xhat_ensemble_{emis_label}')
#     plt.close(fig2)

#     # # Plot regions of interst
#     # if (emis_label in interest.keys()):
#     #     for j, reg in enumerate(interest[emis_label]):
#     #         fig3, ax3 = fp.get_figax(rows=2, cols=2, maps=True,
#     #                                  lats=reg[1][:2], lons=reg[1][2:],
#     #                                  max_height=config.BASE_WIDTH*config.SCALE)
#     #         figw, figh = fig3.get_size_inches()
#     #         plt.subplots_adjust(hspace=0.5/figh, wspace=6/figw)
#     #         # fig3.set_figheight(figh + 0.5)
#     #         # fig3.set_figwidth(figw + 7)

#     #         c = clusters.where((clusters.lat > reg[1][0]) &
#     #                            (clusters.lat < reg[1][1]) &
#     #                            (clusters.lon > reg[1][2]) &
#     #                            (clusters.lon < reg[1][3]), drop=True)
#     #         c_idx = (c.values[c.values > 0] - 1).astype(int)

#     #         fig3, ax3[0, 0], c32 = ip.plot_state(
#     #             w.loc[:, emis_label]/area.reshape(-1,), clusters, 
#     #             title='Sector prior', cbar=False, cmap=viridis_trans, 
#     #             vmin=reg[2][0], vmax=reg[2][1], default_value=0, 
#     #             fig_kwargs={'figax' : [fig3, ax3[0, 0]]}, 
#     #             map_kwargs=small_map_kwargs)
#     #         fig3, ax3[0, 1], _ = ip.plot_state(
#     #             xhat_diff_sect_i + w.loc[:, emis_label]/area.reshape(-1,),
#     #             clusters, title='Sector posterior', cbar=False, 
#     #             cmap=viridis_trans, vmin=reg[2][0], vmax=reg[2][1],
#     #             default_value=0, fig_kwargs={'figax' : [fig3, ax3[0, 1]]},
#     #             map_kwargs=small_map_kwargs)
#     #         fig3, ax3[1, 0], c30 = ip.plot_state(
#     #             xhat, clusters, title=f'Scale factors', 
#     #             cbar=False, cmap=sf_cmap, norm=div_norm, default_value=1,
#     #             fig_kwargs={'figax' : [fig3, ax3[1, 0]]},
#     #             map_kwargs=small_map_kwargs)
#     #         fig3, ax3[1, 1], c31 = ip.plot_state(
#     #             xhat_diff_sect_i, clusters, title='Emissions change', 
#     #             cbar=False, vmin=-reg[2][1]/4, vmax=reg[2][1]/4,
#     #             fig_kwargs={'figax' : [fig3, ax3[1, 1]]}, **d_xhat_kwargs)

#     #         tt = (xhat_diff_sect_i.reshape(-1,)*area.reshape(-1,))[c_idx].sum()*1e-6
#     #         ax3[1, 1].text(0.05, 0.05, f'{tt:.1f} Tg/yr',
#     #                         fontsize=config.LABEL_FONTSIZE*config.SCALE,
#     #                         transform=ax3[1, 1].transAxes)

#     #         if reg[3] is not None:
#     #             for label, point in reg[3].items():
#     #                 for axis3 in ax3.flatten():
#     #                     axis3.scatter(point[1], point[0], s=10, c='black')
#     #                     axis3.text(point[1], point[0], r'$~~$'f'{label}',
#     #                                fontsize=config.LABEL_FONTSIZE*config.SCALE/2)

#     #         for k, axis3 in enumerate(ax3.flatten()):
#     #             axis3.set_ylim(reg[1][:2])
#     #             axis3.set_xlim(reg[1][2:])
#     #             axis3.add_feature(COUNTIES, facecolor='none', 
#     #                                edgecolor='0.1', linewidth=0.1)
#     #         cax30 = fp.add_cax(fig3, ax3[1, 0], cbar_pad_inches=0.1)
#     #         cb30 = fig.colorbar(c30, cax=cax30,
#     #                             ticks=np.arange(0, 3, 1))
#     #         cb30 = fp.format_cbar(cb30,
#     #                              cbar_title='Scale factor', x=4)

#     #         cax31 = fp.add_cax(fig3, ax3[1, 1], cbar_pad_inches=0.1)
#     #         cb31 = fig.colorbar(c31, cax=cax31,
#     #                             ticks=np.arange(-reg[2][1]/4, reg[2][1]/4+1, 
#     #                                             reg[2][1]/8))
#     #         cbar_str = r'$\Delta$ Emissions\\(Mg km$^{-2}$ a$^{-1}$)'
#     #         cb31 = fp.format_cbar(cb31,
#     #                              cbar_title=cbar_str, x=4)
#     #         if emis_label == 'total':
#     #             step = 10
#     #         else:
#     #             step = 5
#     #         cax32 = fp.add_cax(fig3, ax3[0, 1], cbar_pad_inches=0.1)
#     #         cb32 = fig.colorbar(
#     #             c32, cax=cax32, 
#     #             ticks=np.arange(reg[2][0], reg[2][1]+step, step))
#     #         cb32 = fp.format_cbar(
#     #             cb32, cbar_title=r'Emissions\\(Mg km$^{-2}$ a$^{-1}$)', x=4)

#     #         fp.save_fig(fig3, plot_dir, 
#     #                     f'xhat_{emis_label}_reg{j}_{f}')
#     #         plt.close(fig3)

# cax = fp.add_cax(fig, ax, cbar_pad_inches=0.3, horizontal=True)
# cb = fig.colorbar(c, cax=cax, ticks=np.arange(-ul, ul+1, ul/2),
#                   orientation='horizontal')
# cb = fp.format_cbar(cb, cbar_title=r'$\Delta$ Emissions (Mg km$^{-2}$ a$^{-1}$)',
#                     horizontal=True)
# # axis = fp.add_title(axis, titles[i])

# fp.save_fig(fig, plot_dir, f'xhat_ensemble_sectoral')
# plt.close(fig)

# ------------------------------------------------------------------------ ##
# Plot sectoral error correlation 
# ------------------------------------------------------------------------ ##
rfile = f'{data_dir}ensemble/r2_ensemble_conus.csv'
r = pd.read_csv(rfile, index_col=0, header=0)
print(r.round(2))
r = r.loc[anth_cols, anth_cols]
labels = [list(emis.keys())[list(emis.values()).index(l)] for l in anth_cols]

fig, ax = fp.get_figax(max_width=config.BASE_WIDTH/3,
                       max_height=config.BASE_HEIGHT/3)
c = ax.matshow(r, vmin=-1, vmax=1, cmap='RdBu_r')
ax.set_xticks(np.arange(0, len(labels)))
ax.set_xticklabels(labels, ha='center', rotation=90)
ax.xaxis.set_ticks_position('bottom')
ax.set_yticks(np.arange(0, len(labels)))
ax.set_yticklabels(labels, ha='right')

cax = fp.add_cax(fig, ax)
cb = fig.colorbar(c, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
cb = fp.format_cbar(cb, cbar_title='Pearson correlation coefficient')
ax = fp.add_title(ax, 'CONUS')
fp.save_fig(fig, plot_dir, 'r_ensemble_conus')
