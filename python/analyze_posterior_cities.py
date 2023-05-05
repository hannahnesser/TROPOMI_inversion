import sys
from copy import deepcopy as dc
import glob
import math
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
from matplotlib import gridspec
from matplotlib.lines import Line2D
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import shapefile
from shapely.geometry import Polygon, MultiPolygon
from collections import OrderedDict
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

# Define colormaps
sf_cmap_1 = plt.cm.PuOr_r(np.linspace(0.2, 0.5, 256))
sf_cmap_2 = plt.cm.PuOr_r(np.linspace(0.5, 1, 256))
sf_cmap = np.vstack((sf_cmap_1, sf_cmap_2))
sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
div_norm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=1, vmax=3)

## ------------------------------------------------------------------------ ##
## Directories
## ------------------------------------------------------------------------ ##
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'
paper_dir = base_dir + 'paper/figures'

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# DOFS_filter
DOFS_filter = 0.2

# Number of cities
nc = 10

# Define emission categories
# sectors = ['landfills', 'wastewater', 'ong', 'livestock', 'other_anth']
sectors = ['landfills', 'wastewater', 'ong', 'livestock', 'coal', 'other_anth']
# sectors_urban = ['landfills', 'wastewater', 'ong', 'other_anth']

# define short name for major cities
cities = {'NYC' : 'New York--Newark, NY--NJ--CT',
          'LA'  : 'Los Angeles--Long Beach--Anaheim, CA',
          'CHI' : 'Chicago, IL--IN',
          'DFW' : 'Dallas--Fort Worth--Arlington, TX',
          'HOU' : 'Houston, TX',
          'DC'  : 'Washington, DC--VA--MD',
          'MIA' : 'Miami, FL',
          'PHI' : 'Philadelphia, PA--NJ--DE--MD',
          'ATL' : 'Atlanta, GA',
          'PHX' : 'Phoenix--Mesa, AZ',
          'BOS' : 'Boston, MA--NH--RI',
          'SFO' : 'San Francisco--Oakland, CA',
          'SDO' : 'San Diego, CA',
          'DET' : 'Detroit, MI',
          'SEA' : 'Seattle, WA'}

# All other cities do not have a specific CH4 estimate
city_inventories = {
    'New York' : 106, # Still not sure what GWP they use, but we use 2019 from their most recent inventory https://nyc-ghg-inventory.cusp.nyu.edu/ https://online.ucpress.edu/elementa/article/10/1/00082/119571/New-York-City-greenhouse-gas-emissions-estimated
    'Philadelphia' : 28.7, # 2019 https://www.phila.gov/media/20220418144709/2019-greenhouse-gas-inventory.pdf
                    }

# Load other studies
city_studies = pd.read_csv(f'{data_dir}cities/other_studies.csv')
city_studies = city_studies.sort_values(by=['Publication year', 'Label'], 
                                        ascending=False).reset_index(drop=True)

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load the cities and CONUS mask
w_city = pd.read_csv(f'{data_dir}cities/urban_areas_mask.csv', header=0).T
conus_mask = np.load(f'{data_dir}countries/CONUS_mask.npy').reshape((-1,))

# Try changing the definition of W
w_city = w_city/w_city.sum(axis=0)
w_city = w_city.fillna(0)

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()

# Load area (km2)
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load ensemble members (relative posterior and DOFS), all of which are
# previously filtered on DOFS and have the BC elements removed as needed
dofs = pd.read_csv(f'{data_dir}ensemble/dofs.csv', index_col=0)
xhat = pd.read_csv(f'{data_dir}ensemble/xhat.csv', index_col=0)
ensemble = xhat.columns

# Load reduced DOFS
a_files = glob.glob(f'{data_dir}ensemble/a2_*_urban.csv')
a_files.sort()
dofs_c = pd.DataFrame(index=w_city.index, columns=xhat.columns)
for f in a_files:
    short_f = f.split('/')[-1][3:-19]
    a_r = pd.read_csv(f, header=0, index_col=0)
    dofs_c[short_f] = np.diag(a_r)
dofs_stats_c = ip.get_ensemble_stats(dofs_c).add_prefix('dofs_')

# Load weighting matrices in units Gg/yr (we don't consider wetlands
# here, so it doesn't matter which W we use)
w = pd.read_csv(f'{data_dir}sectors/w.csv')*conus_mask.reshape(-1, 1)
w = dc(w[sectors])
w['total'] = w.sum(axis=1)
w = w.T*1e-3

w_hr = pd.read_csv(f'{data_dir}sectors/w_hr.csv')
w_hr = w_hr[['gas_distribution', 'landfills', 'wastewater']]
w_hr['total'] = w_hr.sum(axis=1)
w_hr = w_hr.T*1e-3

# Also get the W matrix for the hackish 2023 EPA GHGI
epa = pd.read_csv(f'{data_dir}sectors/epa_ghgi_2023.csv')

## Define various ONG contributions
epa_postmeter_2019 = 457 # Gg
epa_gasdist_2019 = 554 # Gg

## Group by sector
epa = epa.groupby('sector').agg({'mean' : 'sum', 
                                 'minus' : gc.add_quad, 'plus' : gc.add_quad})
epa = epa.loc[sectors]['mean']*1e3 # Convert to Gg from Tg

## Remove postmeter emissions before scaling (this probably removes
## Hawaii and Alaska postmeter emissions twice, but we're assuming 
## that's negligibly small)
epa['ong'] -= epa_postmeter_2019

## Calculate the total without post meter since we only add that once
## we get to the individual city calculations
epa.loc['total'] = epa.sum(axis=0)

## Create the W matrix
w2019 = dc(w)
w2019 = w2019.multiply(epa/w2019.sum(axis=1), axis='rows')

# Add mass to wastewater per recent literature
# w2019.loc['wastewater'] *= 1.27 # Song
# w2019.loc['wastewater'] *= 1.30 # Moore

## Recalculate grid cell totals since we've altered the sectors
## differently
w2019.loc['total'] = w2019.loc[sectors].sum(axis=0)

## Add in gas distribution using the HR sectoral information
## and scale it to match 2019
w2019.loc['gasdist'] = w_hr.loc['gas_distribution']*epa_gasdist_2019/w_hr.loc['gas_distribution'].sum()

## And remove it from ONG since now we're double counting
w2019.loc['ong'] -= w2019.loc['gasdist']

## And account for the (VERY small) negative values in ONG 
## created by that substraction
w2019_neg = dc(w2019.loc['ong'])
w2019_neg[w2019_neg > 0] = 0
w2019.loc['gasdist'] += w2019_neg
w2019.loc['ong'][w2019.loc['ong'] < 0] = 0

# Finally, get the posterior xhat_abs (this is n x 15)
xhat_abs = (w.loc['total'].values[:, None]*xhat)
xhat_abs_hr = (w_hr.loc['total'].values[:, None]*xhat)

## We will add post meter emissions later!

## ------------------------------------------------------------------------ ##
## Get list of metropolitan statistical area population
## ------------------------------------------------------------------------ ##
pop = pd.read_csv(f'{data_dir}cities/urban_areas_pop.csv', header=0,
                  dtype={'Name' : str, '2010' : int, '2000' : int})
pop = pop.rename(columns={'Name' : 'name'})
pop = pop.sort_values(by='2010', ascending=False, ignore_index=True)
pop = pop.reset_index(drop=True)
pop = pop.set_index('name')
pop_conus = 306675006

## ------------------------------------------------------------------------ ##
## Get gridded population
## ------------------------------------------------------------------------ ##
# Load the 2010 regridded data and convert from population density
# to population
pop_g = xr.open_dataarray(f'{data_dir}cities/pop_gridded.nc')
pop_g = ip.clusters_2d_to_1d(clusters, pop_g)[:, None]*area*1e6 # (km2->m2)

# Now get the intersections with cities and figure out how much
# each city needs to be scaled by in order to match 2019 population
pop_g_scale = pop.loc[w_city.index]/(w_city @ pop_g).values 

## ------------------------------------------------------------------------ ##
## Calculate metropolitan statistical areas sectoral prior and posterior
## ------------------------------------------------------------------------ ##
# Calculate the area, prior emissions, posterior emissions, and xhat
# for each column, then concatenate it into one dataframe
## Area
area_c = (w_city @ area.reshape(-1,)).rename('area')

## Prior (the same for all ensemble members)
# prior_c = (w_city @ w.T).add_prefix('prior_')
prior_c = (w_city @ w2019.T).add_prefix('prior_')

## Posterior (We want a matrix that is ncities x 15)
post_c = (w_city @ xhat_abs)
post_hr_c = (w_city @ xhat_abs_hr)

## Difference
diff_c = post_c - prior_c['prior_total'].values[:, None]

## Get statistics
post_stats_c = ip.get_ensemble_stats(post_c).add_prefix('post_')
post_hr_stats_c = ip.get_ensemble_stats(post_hr_c).add_prefix('post_hr_')
diff_stats_c = ip.get_ensemble_stats(diff_c).add_prefix('diff_')

## Aggregate
summ_c = pd.concat([pd.DataFrame(area_c), prior_c, post_stats_c,
                    post_hr_stats_c, diff_stats_c], axis=1)

# Merge DOFS into summ_c
summ_c = pd.concat([summ_c, dofs_stats_c.loc[summ_c.index]], axis=1)

# Merge in population
# summ_c['pop_2010'] = pop.loc[summ_c.index]['2019']
# summ_c['pop_2000'] = pop.loc[summ_c.index]['2012']
summ_c['pop_2010'] = pop.loc[summ_c.index]['2010']
summ_c['pop_2000'] = pop.loc[summ_c.index]['2000']

# Add in prior post meter emissions
summ_c['prior_postmeter'] = (summ_c['pop_2010']/pop_conus)*epa_postmeter_2019
summ_c['prior_total'] += summ_c['prior_postmeter']
## We do not substract it from the ONG total becaause it was not originally
## included.

## Posterior ratios
xhat_c = post_c/summ_c['prior_total'].values[:, None]

## Get statistics
xhat_stats_c = ip.get_ensemble_stats(xhat_c).add_prefix('xhat_')
summ_c = pd.concat([summ_c, xhat_stats_c], axis=1)

## Calculate the fraction of emissions from urban areas in CONUS
city_emis_frac_prior = summ_c['prior_total'].sum()/epa.loc['total']
# city_emis_landfills = summ_c[]
area_frac = summ_c['area'].sum()/(conus_mask.reshape((-1, 1))*area).sum()

## Print information about these cities
print('-'*75)
print(f'Across all {summ_c.shape[0]} cities:')
print(f'These cities are responsible for {(100*city_emis_frac_prior):.2f}% of prior anthropogenic emissions.\nThese cities take up {(100*area_frac):.2f}% of surface area.')
print('Minimum emission : %.2f' % (summ_c['prior_total'].min()))
print('Maximum emission : %.2f' % (summ_c['prior_total'].max()))
print('Mean emission : %.2f ' % (summ_c['prior_total'].mean()))
print('Mean emissions profile:')
mean_prof = summ_c[[f'prior_{i}' for i in sectors + ['postmeter', 'gasdist']]]
mean_prof = mean_prof/summ_c['prior_total'].values.reshape(-1, 1)
std_prof = mean_prof.std(axis=0)
mean_prof = mean_prof.mean(axis=0)
mean_prof = pd.concat([mean_prof, std_prof], axis=1)
print(mean_prof.round(3)*100)
print(mean_prof.sum()*100)
# Subset to remove cities that aren't optimized in any of the
# inversions
print('-'*75)
print('Non-optimized cities')
print(summ_c[summ_c['dofs_mean'] < DOFS_filter].index.values)
nc_old = summ_c.shape[0]
xhat_c = xhat_c[summ_c['dofs_mean'] >= DOFS_filter]
dofs_c = dofs_c.loc[summ_c.index][summ_c['dofs_mean'] > DOFS_filter]
post_c = post_c[summ_c['dofs_mean'] >= DOFS_filter]
summ_c = summ_c[summ_c['dofs_mean'] >= DOFS_filter]

# Print top-line statistics
## Calculate means in a way that treats each ensemble member first
xhat_mean = xhat_c.mean(axis=0) # Same as xhat mean
xhat_std = xhat_c.std(axis=0)
dofs_mean = dofs_c.mean(axis=0)

## Calculate the fraction of emissions from urban areas in CONUS
city_emis_frac_prior = summ_c['prior_total'].sum()/28.7e3
city_emis_frac_post = post_c.sum()/30.9e3
area_frac = summ_c['area'].sum()/(conus_mask.reshape((-1, 1))*area).sum()
prior_tot = summ_c['prior_total'].sum()

## Print
print('-'*75)
print(f'Analyzed {summ_c.shape[0]}/{nc_old} cities:')
print(f'These cities are responsible for {(100*city_emis_frac_prior):.2f}% of prior anthropogenic emissions\nand {(100*city_emis_frac_post.mean()):.2f} ({(100*city_emis_frac_post.min()):.2f} - {100*city_emis_frac_post.max():.2f})% of posterior anthropogenic emissions in CONUS.\nThese cities take up {(100*area_frac):.2f}% of surface area.')
print(f'We find prior emissions of {prior_tot*1e-3:.2f} Tg/yr.')
print(f'We find a net adjustment of {diff_c.sum(axis=0).mean():.2f} ({diff_c.sum(axis=0).min():.2f}, {diff_c.sum(axis=0).max():.2f}) Gg/yr.')
print(f'We find emissions of {post_c.sum(axis=0).mean()*1e-3:.2f} ({post_c.sum(axis=0).min()*1e-3:.2f}, {post_c.sum(axis=0).max()*1e-3:.2f}) Tg/yr.')
print(f'We find a total correction of {post_c.sum(axis=0).mean()/prior_tot:.2f} ({post_c.sum(axis=0).min()/prior_tot:.2f}, {post_c.sum(axis=0).max()/prior_tot:.2f}) Tg/yr.')
print(f'  xhat mean                      {xhat_mean.mean():.2f} ({xhat_mean.min():.2f}, {xhat_mean.max():.2f})')
print(f'  xhat std                       {xhat_std.mean():.2f}')
print(f'  dofs mean                      {dofs_mean.mean():.2f} ({dofs_mean.min():.2f}, {dofs_mean.max():.2f})')
print('  Ensemble means:')
print(xhat_c.mean(axis=0).round(2))
# print(f'  mean delta emissions (Gg/yr)   {}', delta_mean)

# Per capita methane emissions (Gg/person) (doesn't matter if
# we take the mean first or not because pop is constant pre city)
summ_c['post_mean_pc'] = summ_c['post_mean']/summ_c['pop_2010']

# Sort by per capita methane emissions and print out
summ_c = summ_c.sort_values(by='post_mean_pc', ascending=False)
print('-'*75)
print('Largest per capita methane emissions')
print(summ_c.index.values[:nc])

# Sort by population
summ_c = summ_c.sort_values(by='pop_2010', ascending=False)
print('-'*75)
print('Largest cities')
print(summ_c.index.values[:nc])

# Print information on the largest nc cities
city_emis_frac_prior = summ_c['prior_total'][:nc].sum()/summ_c['prior_total'].sum()
city_emis_frac_post = post_c.loc[summ_c.index[:nc]].sum()/post_c.loc[summ_c.index].sum()
print(f'The largest {nc} cities by population are responsible for {(100*city_emis_frac_prior):.2f}% of urban\nprior emissions and {(100*city_emis_frac_post.mean()):.2f} ({(100*city_emis_frac_post.min()):.2f} - {(100*city_emis_frac_post.max()):.2f})% of urban posterior emissions.')

# Sort by per km2 urban methane emissions
summ_c = summ_c.sort_values(by='post_hr_mean', ascending=False)
print('-'*75)
print('Largest urban methane emissions')
print(summ_c.index.values[:nc])

# Print information on the largest nc cities
city_emis_frac_prior = summ_c['prior_total'][:nc].sum()/summ_c['prior_total'].sum()
city_emis_frac_post = post_c.loc[summ_c.index[:nc]].sum()/post_c.loc[summ_c.index].sum()
print(f'The largest {nc} cities by methane emissions are responsible for {(100*city_emis_frac_prior):.2f}% of urban\nprior emissions and {(100*city_emis_frac_post.mean()):.2f} ({(100*city_emis_frac_post.min()):.2f} - {(100*city_emis_frac_post.max()):.2f})% of urban posterior emissions.')
print(summ_c['xhat_mean'][:nc])
top10 = xhat_c.loc[summ_c.index.values[:nc]].mean(axis=0)
top10_mean = summ_c['xhat_mean'][:nc].mean(axis=0) - 1
print(f'We find a mean adjustment of {100*top10_mean:.2f}% in the top {nc} cities.')
print(f'We find a mean adjustment of {100*(top10.mean() - 1):.2f} ({100*(top10.min() - 1):.2f} - {100*(top10.max() - 1):.2f})% in the top {nc} cities.')
print('-'*75)

# Save out csv
summ_c.to_csv(f'{data_dir}/cities/summ_cities.csv', header=True, index=True)

## ------------------------------------------------------------------------ ##
## Plot distribution
## ------------------------------------------------------------------------ ##
fig, ax = fp.get_figax(cols=2, rows=1, aspect=1.5, sharey=True,
                       max_height=config.BASE_HEIGHT*config.SCALE)
plt.subplots_adjust(wspace=0.1)
counts0, bins0 = np.histogram(summ_c['xhat_mean'], bins=20)
ax[0].hist(bins0[:-1], bins0, weights=counts0, color=fp.color(2))
ax[0].axvline(1, color='0.6', ls='-', lw=0.75)
ax[0].axvline(summ_c['xhat_mean'].mean(), color='0.6', ls='--', lw=0.75)
ax[0] = fp.add_labels(ax[0], 'Urban scale factor', 'Count', 
                      fontsize=config.TICK_FONTSIZE,
                      labelsize=config.TICK_FONTSIZE)
mu = summ_c['xhat_mean'].mean()

# DOFS
counts, bins = np.histogram(summ_c['dofs_mean'], bins=20)
ax[1].hist(bins[:-1], bins, weights=counts, color=fp.color(5))
ax[1].axvline(summ_c['dofs_mean'].mean(), color='0.6', ls='--', lw=0.75)
ax[1] = fp.add_labels(ax[1], 'Sensitivity', '', 
                      fontsize=config.TICK_FONTSIZE,
                      labelsize=config.TICK_FONTSIZE)
mu = summ_c['dofs_mean'].mean()

# Text
_, ymax = ax[0].get_ylim()
ax[0].text(summ_c['xhat_mean'].mean()*1.1, ymax*0.9, f'Mean : {mu:.2f}',
           fontsize=config.TICK_FONTSIZE)
ax[1].text(summ_c['dofs_mean'].mean()*1.1, ymax*0.9, f'Mean : {mu:.2f}',
           fontsize=config.TICK_FONTSIZE)
fp.save_fig(fig, plot_dir, f'cities_distribution')

## ------------------------------------------------------------------------ ##
## Plot maps
## ------------------------------------------------------------------------ ##
city = shapefile.Reader(f'{data_dir}cities/2019_tl_urban_areas/tl_2019_us_uac10_buffered.shp')

fig, ax = fp.get_figax(maps=True, lats=clusters.lat, lons=clusters.lon)
for shape in city.shapeRecords():
    # if shape.record[2] == 'Atlanta, GA':
    if shape.record[2] in summ_c.index.values:
        # Get edges of city
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        c_poly = Polygon(np.column_stack((x, y)))
        c_poly = c_poly.buffer(0)
        color = sf_cmap(div_norm(summ_c.loc[shape.record[2]]['xhat_mean']))
        ax.fill(x, y, facecolor=color, edgecolor='black', linewidth=0.1)
lats, lons = gc.create_gc_grid(*s.lats, s.lat_delta,*s.lons, s.lon_delta,
                               centers=False, return_xarray=False)
# for lat in lats:
#     if lat > np.min(y) and lat < np.max(y):
#         ax.axhline(lat, color='0.3')
# for lon in lons:
#     if lon > np.min(x) and lon < np.max(x):
#         ax.axvline(lon, color='0.3')

# ax = fp.add_title(ax, 'Atlanta, Georgia Urban Area')
# ax = fp.format_map(ax, lats=y, lons=x)
ax = fp.add_title(ax, 'CONUS urban areas')
ax = fp.format_map(ax, lats=clusters.lat, lons=clusters.lon)
cmap = plt.cm.ScalarMappable(cmap=sf_cmap, norm=div_norm)
cax = fp.add_cax(fig, ax)
cb = fig.colorbar(cmap, ax=ax, cax=cax)
cb = fp.format_cbar(cb, 'Scale factor')
fp.save_fig(fig, plot_dir, f'cities_map')

## ------------------------------------------------------------------------ ##
## Plot correlations with area and population
## ------------------------------------------------------------------------ ##
# # With respect to cities
# corr_c = dc(summ_c)
# corr_c = corr_c[corr_c['dofs_mean'] >= 0.19]
# # corr_c = corr_c.iloc[:15, :]

# # Plot
# ylabel = ['Posterior - prior\nemissions 'r'(Gg a$^{-1}$)', 'Scale factor']
# for i, q in enumerate(['diff', 'xhat']):
#     # Change _min and _max to errorbar terms
#     corr_c[f'{q}_max'] = corr_c[f'{q}_max'] - corr_c[f'{q}_mean']
#     corr_c[f'{q}_min'] = corr_c[f'{q}_mean'] - corr_c[f'{q}_min']

#     fig, ax = fp.get_figax(cols=2, rows=2, aspect=1.5, sharey=True,
#                            max_width=config.BASE_WIDTH*0.95)
#     plt.subplots_adjust(hspace=0.6, wspace=0.1)
#     lb = '\n'

#     ## Population
#     # ax[0, 0].errorbar(corr_c['pop_2000'], corr_c['diff_mean'], ms=1,
#     #                   yerr=np.array(corr_c[['diff_min', 'diff_max']]).T, alpha=0.1,
#     #                   color=fp.color(2), fmt='o', elinewidth=0.5, label='2012')
#     m, b, r, _, _ = gc.comparison_stats(np.log10(corr_c['pop_2010'].values), 
#                                         corr_c[f'{q}_mean'].values)
#     ax[0, 0].errorbar(corr_c['pop_2010'], corr_c[f'{q}_mean'], ms=1,
#                       yerr=np.array(corr_c[[f'{q}_min', f'{q}_max']]).T,
#                       color=fp.color(2), fmt='o', elinewidth=0.5, label='2019')
#     ax[0, 0] = fp.add_labels(ax[0, 0], 'Population', ylabel[i],
#                              fontsize=config.TICK_FONTSIZE, 
#                              labelsize=config.TICK_FONTSIZE, labelpad=10)
#     ax[0, 0].text(0.05, 0.95, 
#                   f'y = {m:.2f}x + {b:.2f}{lb}(R'r'$^2$'f' = {r**2:.2f})',
#                   fontsize=config.TICK_FONTSIZE, va='top', ha='left',
#                   transform=ax[0, 0].transAxes)

#     ## Area
#     m, b, r, _, _ = gc.comparison_stats(np.log10(corr_c['area'].values), 
#                                         corr_c[f'{q}_mean'].values)
#     ax[0, 1].errorbar(corr_c['area'], corr_c[f'{q}_mean'], ms=1,
#                       yerr=np.array(corr_c[[f'{q}_min', f'{q}_max']]).T,
#                       color=fp.color(4), fmt='o', elinewidth=0.5)
#     ax[0, 1] = fp.add_labels(ax[0, 1], r'Area (km$^2$)', '',
#                              fontsize=config.TICK_FONTSIZE, 
#                              labelsize=config.TICK_FONTSIZE, labelpad=10)
#     ax[0, 1].text(0.05, 0.95, 
#                   f'y = {m:.2f}x + {b:.2f}{lb}(R'r'$^2$'f' = {r**2:.2f})',
#                   fontsize=config.TICK_FONTSIZE, va='top', ha='left',
#                   transform=ax[0, 1].transAxes)

#     ## Population change
#     m, b, r, _, _ = gc.comparison_stats(
#                         (corr_c['pop_2010']/corr_c['pop_2000'] - 1).values, 
#                         corr_c[f'{q}_mean'].values)
#     ax[1, 0].errorbar(corr_c['pop_2010']/corr_c['pop_2000'] - 1, 
#                       corr_c[f'{q}_mean'], ms=1, color=fp.color(6), fmt='o',
#                       yerr=np.array(corr_c[[f'{q}_min', f'{q}_max']]).T,
#                       elinewidth=0.5)
#     ax[1, 0] = fp.add_labels(ax[1, 0], f'Relative population change{lb}2000-2010',
#                              ylabel[i], fontsize=config.TICK_FONTSIZE, 
#                              labelsize=config.TICK_FONTSIZE, labelpad=10)
#     ax[1, 0].text(0.05, 0.95, 
#                   f'y = {m:.2f}x + {b:.2f}{lb}(R'r'$^2$'f' = {r**2:.2f})',
#                   fontsize=config.TICK_FONTSIZE, va='top', ha='left',
#                   transform=ax[1, 0].transAxes)

#     ## Density
#     # ax[1, 1].errorbar(corr_c['pop_2000']/corr_c['area'], corr_c[f'{q}_mean'], ms=1,
#     #                   yerr=np.array(corr_c[[f'{q}_min', f'{q}_max']]).T, alpha=0.1,
#     #                   color=fp.color(8), fmt='o', elinewidth=0.5, label='2012')
#     m, b, r, _, _ = gc.comparison_stats(
#                         np.log((corr_c['pop_2010']/corr_c['area']).values), 
#                         corr_c[f'{q}_mean'].values)
#     ax[1, 1].errorbar(corr_c['pop_2010']/corr_c['area'], corr_c[f'{q}_mean'],
#                       ms=1, yerr=np.array(corr_c[[f'{q}_min', f'{q}_max']]).T,
#                       color=fp.color(8), fmt='o', elinewidth=0.5, label='2019')
#     ax[1, 1] = fp.add_labels(ax[1, 1], 'Population density\n'r'(km$^{-2}$)', 
#                              '', fontsize=config.TICK_FONTSIZE, 
#                              labelsize=config.TICK_FONTSIZE, labelpad=10)
#     ax[1, 1].text(0.05, 0.95, 
#                   f'y = {m:.2f}x + {b:.2f}{lb}(R'r'$^2$'f' = {r**2:.2f})',
#                   fontsize=config.TICK_FONTSIZE, va='top', ha='left',
#                   transform=ax[1, 1].transAxes)

#     for j in range(2):
#         for k in range(2):
#             ax[j, k].set_xscale('log')
#             ax[j, k].axhline(i, color='0.3', lw=0.5, ls='--')
#     ax[1, 0].set_xscale('linear')

#     fp.save_fig(fig, plot_dir, f'cities_correlations_{q}')

# # With respect to grid cells
# ## We want the number of people in each urban grid cell
# w_tot = w_city/w_city.sum(axis=0) # Don't double count people
# w_tot = w_city*pop_g.T*pop_g_scale['2010'].values[:, None] # Get population
# w_tot = w_tot.sum(axis=0) # Sum over cities

# # ## Get the distribution of population densities
# # fig, ax = fp.get_figax(aspect=1.5)
# # ax.hist(w_tot.values/area.flatten(), color=fp.color(4), bins=500)
# # ax.set_xlim(0, 1e3)
# # ax = fp.add_labels(ax, r'Population density (km$^{-2}$)', 'Count',
# #                    fontsize=config.TICK_FONTSIZE, 
# #                    labelsize=config.TICK_FONTSIZE, labelpad=10)
# # fp.save_fig(fig, plot_dir, f'cities_population_density_distribution')

# # Create a mask
# city_mask = (dofs.mean(axis=1) >= 0.25) & (w_tot > 100*area.flatten()).values 
# print('-'*75)
# print(f'We consider {city_mask.sum():d} grid cells in our urban analysis.')
# print('-'*75)
# w_tot = w_tot[city_mask]

# ## Now get the gridded difference between posterior and prior in
# ## urban areas
# prior_g = w.loc['total'].values[city_mask]
# post_g = xhat_abs.values[city_mask, :]
# diff_g = post_g - prior_g[:, None]
# diff_g = ip.get_ensemble_stats(diff_g)

# xhat_g = xhat.values[city_mask, :]
# xhat_g = ip.get_ensemble_stats(xhat_g)

# data_g = {'diff' : diff_g, 'xhat' : xhat_g}
# lims = [(-150, 150), (-1, 5)]
# for i, (q, data) in enumerate(data_g.items()):
#     ## Correct for error bar terms
#     data['max'] = data['max'] - data['mean']
#     data['min'] = data['mean'] - data['min']

#     # Plot
#     fig, ax = fp.get_figax(cols=2, aspect=1.5, sharey=True,
#                             max_width=config.BASE_WIDTH*0.95)
#     plt.subplots_adjust(wspace=0.1)

#     ## Population
#     m, b, r, _, _ = gc.comparison_stats(w_tot.values, data['mean'].values)
#     ax[0].errorbar(w_tot, data['mean'], yerr=np.array(data[['min', 'max']]).T,
#                    ms=1, color=fp.color(2), fmt='o', elinewidth=0.5)
#     ax[0] = fp.add_labels(ax[0], 'Population', ylabel[i],
#                           fontsize=config.TICK_FONTSIZE, 
#                           labelsize=config.TICK_FONTSIZE, labelpad=10)
#     ax[0].text(0.05, 0.95, 
#                f'y = {m:.2f}x + {b:.2f}{lb}'r'(R$^2$'f' = {r**2:.2f})',
#                fontsize=config.TICK_FONTSIZE, va='top', ha='left',
#                transform=ax[0].transAxes)

#     ## Population density
#     m, b, r, _, _ = gc.comparison_stats(w_tot.values/area.flatten()[city_mask],
#                                         data['mean'].values)
#     ax[1].errorbar(w_tot/area.flatten()[city_mask], data['mean'], 
#                    yerr=np.array(data[['min', 'max']]).T,
#                    ms=1, color=fp.color(4), fmt='o', elinewidth=0.5)
#     ax[1] = fp.add_labels(ax[1], r'Population density (km$^{-2}$)', '',
#                              fontsize=config.TICK_FONTSIZE, 
#                              labelsize=config.TICK_FONTSIZE, labelpad=10)
#     ax[1].text(0.05, 0.95, 
#                f'y = {m:.2f}x + {b:.2f}{lb}(R'r'$^2$'f' = {r**2:.2f})',
#                fontsize=config.TICK_FONTSIZE, va='top', ha='left',
#                transform=ax[1].transAxes)

#     for j in range(2):
#         ax[j].set_xscale('log')
#         ax[j].set_ylim(lims[i])
#         ax[j].axhline(i, color='0.3', lw=0.5, ls='--')
#         # ax[j].axhline(data['mean'].mean(), color=fp.color(2*j + 2), lw=0.5)

#     fp.save_fig(fig, plot_dir, f'cities_correlations_grid_{q}')

## ------------------------------------------------------------------------ ##
## Plot sectoral error correlation 
## ------------------------------------------------------------------------ ##
# # rfiles = glob.glob(f'{data_dir}cities/r2_urban_areas_*.csv')
# # rfiles.sort()
# # for ff in rfiles:
# #     short_name = ff.split('_')[-1].split('.')[0]
# #     short_name = '%s (%s)' % (cities[short_name].split('-')[0], 
# #                               cities[short_name].split(', ')[-1])
# #     r = pd.read_csv(ff, index_col=0, header=0)
# #     labels = [list(s.sectors.keys())[list(s.sectors.values()).index(l)] for l in r.columns]
# #     fig, ax = fp.get_figax()
# #     c = ax.matshow(r, vmin=-1, vmax=1, cmap='RdBu_r')
# #     ax.set_xticks(np.arange(0, len(labels)))
# #     ax.set_xticklabels(labels, ha='center', rotation=90)
# #     ax.xaxis.set_ticks_position('bottom')
# #     ax.set_yticks(np.arange(0, len(labels)))
# #     ax.set_yticklabels(labels, ha='right')

# #     cax = fp.add_cax(fig, ax)
# #     cb = fig.colorbar(c, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
# #     cb = fp.format_cbar(cb, cbar_title='Pearson correlation coefficient')
# #     ax = fp.add_title(ax, short_name)
# #     fp.save_fig(fig, plot_dir, ff.split('/')[-1].split('.csv')[0])

## ------------------------------------------------------------------------ ##
## Plot results
## ------------------------------------------------------------------------ ##
# Subset summary
summ_c = summ_c.iloc[:nc, :]
# print(summ_c)

# Adjust min/max definitions for error bars
summ_c['post_max'] = summ_c['post_max'] - summ_c['post_mean']
summ_c['post_min'] = summ_c['post_mean'] - summ_c['post_min']
summ_c['dofs_max'] = summ_c['dofs_max'] - summ_c['dofs_mean']
summ_c['dofs_min'] = summ_c['dofs_mean'] - summ_c['dofs_min']

print(summ_c.iloc[0, :])

# Define ys
ys = np.arange(1, nc + 1)

figsize = fp.get_figsize(aspect=1.5*7/nc*2.5, 
                         max_width=config.BASE_WIDTH*config.SCALE*0.7)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.5], wspace=0.1)
# gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharey=ax0)
ax2 = plt.subplot(gs[2], sharey=ax0)
ax = [ax0, ax1, ax2]

# Get labels
labels = summ_c.index.values
labels = ['%s (%s)' % (l.split(',')[0].split('--')[0], l.split(', ')[-1]) 
          for l in labels]
city_names = np.array(['%s' % (l.split(',')[0].split('--')[0]) 
                       for l in summ_c.index.values])

# Plot stacked bar
ax[0] = fp.add_title(ax[0], 'Urban area methane emissions', 
                     fontsize=config.TITLE_FONTSIZE)

# Formatters
formats = ['o', 's', '^', 'D']
sizes = [4, 4, 4.5, 4]

# # Prior
# left_prior = np.zeros(nc)
# for i, e in enumerate(sectors):
#     l = list(s.sectors.keys())[list(s.sectors.values()).index(e)]
#     ax[0].barh(ys - 0.2625, summ_c[f'prior_{e}'], left=left_prior, 
#                height=0.125, color=s.sector_colors[e], label=f'{l}')
#     left_prior += summ_c[f'prior_{e}']

# Prior #2
left_prior = np.zeros(nc)
for i, e in enumerate(sectors):
    l = list(s.sectors.keys())[list(s.sectors.values()).index(e)]
    if e == 'ong':
        ax[0].barh(ys - 0.175, summ_c[f'prior_postmeter'],
                   left=left_prior, height=0.3, color=s.sector_colors[e], 
                   alpha=0.3, label='Post-meter gas')
        left_prior += summ_c[f'prior_postmeter']

        ax[0].barh(ys - 0.175, summ_c[f'prior_gasdist'],
                   left=left_prior, height=0.3, color=s.sector_colors[e], 
                   alpha=0.6, label='Gas distribution')
        left_prior += summ_c[f'prior_gasdist']

    if e == 'ong':
        l = 'Upstream oil and gas'
    ax[0].barh(ys - 0.175, summ_c[f'prior_{e}'], left=left_prior, 
               height=0.3, color=s.sector_colors[e], label=l)
    left_prior += summ_c[f'prior_{e}']

# Posterior
ax[0].barh(ys + 0.175, summ_c['post_mean'],
           xerr=np.array(summ_c[['post_min', 'post_max']]).T,
           error_kw={'ecolor' : '0.65', 'lw' : 0.75, 'capsize' : 2,
                     'capthick' : 0.75},
           height=0.3, color='0.7', alpha=0.4, label='Total')

# Other studies
i = 0
names, counts = np.unique(city_studies['City'].values, return_counts=True)
city_count = {name : 0 for name in names}
for study in city_studies['Label'].unique():
    studies = city_studies[city_studies['Label'] == study]

    # Subset for only cities in the top nc and iterate through those
    studies = studies[studies['City'].isin(city_names)]
    if studies.shape[0] > 0:
        for city in studies['City'].unique():
            city_count[city] += 1
            result = studies[studies['City'] == city]
            y = np.argwhere(city == city_names)[0][0]
            count = counts[np.where(names == city)][0]
            # I want 
            # 1 --> 0
            # 2 --> -0.05 and +0.05
            # 3 --> -0.1, 0, +0.1
            # 4 --> -0.15, -0.05, 0.05, 0.15
            # 8 --> -0.35, -0.25, -0.15, -0.05 and positive
            ax[0].errorbar(
                result['Mean'].values, 
                y + 1 - 0.05*(count - 1) + 0.1*(city_count[city] - 1),
                xerr=result[['Min', 'Max']].values.T, fmt=formats[i % 4], 
                markersize=sizes[i % 4], markeredgecolor='black', 
                markerfacecolor=fp.color(math.ceil((i + 1)/4), 
                                         cmap='viridis', lut=4),
                ecolor='black', elinewidth=0.5, capsize=1, capthick=0.5, 
                zorder=10, label=study)
        i += 1

# Inventories
for i, (ci_name, ci) in enumerate(city_inventories.items()):
    if i == 0:
        label = 'City inventory'
    else:
        label = None
    if ci_name in city_names:
        y = np.argwhere(ci_name == city_names)[0][0]
        ax[0].scatter(ci, y + 1, marker='o', s=16, facecolor='white', 
                      edgecolor='black', zorder=10, label=label)

# Add labels
ax[0].set_yticks(ys)
ax[0].set_ylim(0.5, nc + 0.5)
ax[0].invert_yaxis()
ax[0].tick_params(axis='both', labelsize=config.TICK_FONTSIZE)

# Deal with scales
ax[0].set_xlim(0, 600)

# Final aesthetics
for j in range(1, 3):
    plt.setp(ax[j].get_yticklabels(), visible=False)
ax[0].set_yticklabels(labels, ha='right', fontsize=config.TICK_FONTSIZE)
ax[0] = fp.add_labels(ax[0], r'Emissions (Gg a$^{-1}$)', '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)

for i in range(20):
    ax[0].axvline((i + 1)*100, color='0.75', lw=0.5, zorder=-10)

# Plot emissions per capita
ax[1] = fp.add_title(ax[1], 
                     'Per capita emissions', 
                     fontsize=config.TITLE_FONTSIZE)
left_prior = np.zeros(nc)
left_post = np.zeros(nc)
for i, e in enumerate(sectors):
    l = list(s.sectors.keys())[list(s.sectors.values()).index(e)]
    if e == 'ong':
        ax[1].barh(ys - 0.175, 
                   summ_c[f'prior_postmeter']/summ_c['pop_2010']*1e6,
                   left=left_prior, height=0.3, color=s.sector_colors[e], 
                   alpha=0.3, label='Post-meter gas')
        left_prior += summ_c[f'prior_postmeter']/summ_c['pop_2010']*1e6

        ax[1].barh(ys - 0.175, summ_c[f'prior_gasdist']/summ_c['pop_2010']*1e6,
                   left=left_prior, height=0.3, color=s.sector_colors[e], 
                   alpha=0.6, label='Gas distribution')
        left_prior += summ_c[f'prior_gasdist']/summ_c['pop_2010']*1e6

    ax[1].barh(ys - 0.175, summ_c[f'prior_{e}']/summ_c['pop_2010']*1e6, 
               left=left_prior, height=0.3, color=s.sector_colors[e], 
               label=l)
    left_prior += summ_c[f'prior_{e}']/summ_c['pop_2010']*1e6

ax[1].barh(ys + 0.175, summ_c[f'post_mean']/summ_c['pop_2010']*1e6, 
           xerr=np.array(summ_c[['post_min', 'post_max']]/summ_c['pop_2010'].values[:, None]*1e6).T,
           error_kw={'ecolor' : '0.65', 'lw' : 0.75, 'capsize' : 2,
                     'capthick' : 0.75},
           height=0.3, color='0.7', alpha=0.4, zorder=10)

ax[1].set_yticks(ys)
ax[1].set_ylim(0.5, nc + 0.5)
ax[1].invert_yaxis()
ax[1] = fp.add_labels(ax[1], 
                      r'Emissions per capita (kg person$^{-1}$ a$^{-1}$)', 
                      '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)

ax[1].tick_params(axis='both', labelsize=config.TICK_FONTSIZE)
ax[1].axvline(0, ls=':', lw=0.5, color='0.5')
ax[1].set_xlim(0, 80)
for i in range(3):
    ax[1].axvline((i + 1)*20, color='0.75', lw=0.5, zorder=-10)

# Add labels
# ax[1].text((summ_c['prior_total']/summ_c['pop_2010']*1e6)[0] + 1, 
#            ys[0] + 0.05, 'EPA GHGI', ha='left',
#            va='bottom', fontsize=config.TICK_FONTSIZE - 2)
# ax[1].text(((summ_c['post_mean'] + summ_c['post_max'])/summ_c['pop_2010']*1e6)[0] + 1, 
#            ys[0] + 0.075, 'Posterior',
#            ha='left', va='top', 
#            fontsize=config.TICK_FONTSIZE - 2)

# Plot DOFS
ax[-1] = fp.add_title(ax[-1], 'Information content', 
                     fontsize=config.TITLE_FONTSIZE)
ax[-1].errorbar(summ_c['dofs_mean'], ys, #fmt='none',
               xerr=np.array(summ_c[['dofs_min', 'dofs_max']]).T,
               fmt='D', markersize=4, markerfacecolor='white', 
               markeredgecolor='0.65', ecolor='0.65', 
               elinewidth=0.75, capsize=2, capthick=0.75)
ax[-1].set_yticks(ys)
ax[-1].set_ylim(0.5, nc + 0.5)
ax[-1].set_xlim(0, 1)
ax[-1].invert_yaxis()
ax[-1] = fp.add_labels(ax[-1], 'Sensitivities', '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)
for i in range(4):
    ax[-1].axvline((i + 1)*0.25, color='0.75', lw=0.5, zorder=-10)
ax[-1].set_xticks(np.arange(0, 3)/2)

# Horizontal grid lines
for i in range(nc + 1):
    for k in range(3):
        if i % 5 == 0:
            ls = '-'
        else:
            ls = ':'
        ax[k].axhline(i + 0.5, color='0.75', lw=0.5, ls=ls, zorder=-10)


# Legend for summary plot
# m_handles, m_labels = ax0.get_legend_handles_labels()
handles, labels = ax[0].get_legend_handles_labels()

# Remove coal
idx = labels.index('Coal')
handles.pop(idx)
labels.pop(idx)

# Remove duplicates
labels = OrderedDict(zip(labels, handles))
handles = list(labels.values())
labels = list(labels.keys())

# Add a blank handle and label 
blank_handle = [Line2D([0], [0], markersize=0, lw=0)]
blank_label = ['']
handles.extend(blank_handle)
labels.extend(blank_label)

# Reorder
reorder = [-1, -1, -1, -1, -1, -1, -1, -1, -1,
            1,  4,  7,  8,  9, 10, 11, 12, 0,
            2,  5, -1, -1, 13, 14, 15, 16, -1,
            3,  6, -1, -1, 17, 18, 19, 20, -1]
# reorder = [-1, -1, -1, -1, -1, -1, -1, 
#            1, 5, 8, 9, 13, 17, 0, 
#            2, 6, -1, 10, 14, 18, -1,
#            3, 7, -1, 11, 15, 19, -1,
#            4, -1, -1, 12, 16, 20, -1]
# reorder = np.arange(1, 22).reshape((3, 7)).T.flatten()
# reorder[-1] = 0
# reorder = list(reorder)
handles = [handles[i] for i in reorder]
labels = [labels[i] for i in reorder]

labels[0] = 'Gridded 2023 EPA'
labels[1] = 'GHGI for 2019 : '
labels[3] = 'Posterior : '
labels[4] = 'Other studies : '
labels[8] = 'Other inventories : '

# Add legend
ax[2].legend(handles=handles, labels=labels, ncol=4, frameon=False,
             fontsize=config.TICK_FONTSIZE, loc='upper right', 
             bbox_to_anchor=(1, -0.35), bbox_transform=ax[2].transAxes)

fp.save_fig(fig, plot_dir, f'cities_ensemble')
fp.save_fig(fig, paper_dir, 'fig07', for_acp=True)
