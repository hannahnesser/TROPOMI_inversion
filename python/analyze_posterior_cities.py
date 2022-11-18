import sys
from copy import deepcopy as dc
import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import shapefile
from shapely.geometry import Polygon, MultiPolygon
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

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# DOFS_filter
DOFS_filter = 0.05

# Number of cities
nc = 15

# Define file names
f = 'rg2rt_10t_w404_edf_bc0_rf0.25_sax0.75_poi80.0'
xa_abs_file = 'xa_abs_w404_edf_bc0.nc'
w_file = 'w_w404_edf.csv'
optimize_BC = False

# Define emission categories
emis = {'Landfills' : 'landfills', 'Wastewater' : 'wastewater',
        'Oil and natural gas' : 'ong', 
        'Coal' : 'coal', 'Livestock' : 'livestock', 'Other' : 'other_anth'}

city_names = {'NYC' : 'New York-Newark-Jersey City, NY-NJ-PA',
              'LA'  : 'Los Angeles-Long Beach-Anaheim, CA',
              'CHI' : 'Chicago-Naperville-Elgin, IL-IN-WI',
              'DFW' : 'Dallas-Fort Worth-Arlington, TX',
              'HOU' : 'Houston-The Woodlands-Sugar Land, TX',
              'DC'  : 'Washington-Arlington-Alexandria, DC-VA-MD-WV',
              'MIA' : 'Miami-Fort Lauderdale-Pompano Beach, FL',
              'PHI' : 'Philadelphia-Camden-Wilmington, PA-NJ-DE-MD',
              'ATL' : 'Atlanta-Sandy Springs-Alpharetta, GA',
              'PHX' : 'Phoenix-Mesa-Chandler, AZ',
              'BOS' : 'Boston-Cambridge-Newton, MA-NH',
              'SFO' : 'San Francisco-Oakland-Berkeley, CA',
              'RIV' : 'Riverside-San Bernardino-Ontario, CA',
              'DET' : 'Detroit-Warren-Dearborn, MI',
              'SEA' : 'Seattle-Tacoma-Bellevue, WA'}

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load the cities mask
w_city = pd.read_csv(f'{data_dir}cities/cities_mask.csv', header=0).T

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()

# Load area (km2)
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load ensemble members (relative posterior and DOFS), all of which are
# previously filtered on DOFS and have the BC elements removed as needed
dofs = pd.read_csv(f'{data_dir}ensemble/dofs.csv', index_col=0)
xhat = pd.read_csv(f'{data_dir}ensemble/xhat.csv', index_col=0)
ensemble = xhat.columns

# ID two priors
w37_cols = [s for s in ensemble if 'w37' in s]
w404_cols = [s for s in ensemble if 'w404' in s]

# Load reduced DOFS
a_files = glob.glob(f'{data_dir}ensemble/a2_*_cities.csv')
a_files.sort()
dofs_c = pd.DataFrame(index=w_city.index, columns=xhat.columns)
for f in a_files:
    short_f = f.split('/')[-1][3:-19]
    a_r = pd.read_csv(f, header=0, index_col=0)
    dofs_c[short_f] = np.diag(a_r)
dofs_c = ip.get_ensemble_stats(dofs_c).add_prefix('dofs_')

# Load weighting matrices in units Gg/yr 
w_w404 = pd.read_csv(f'{data_dir}w_w404_edf.csv')
w_w37 = pd.read_csv(f'{data_dir}w_w37_edf.csv')
w = {'w404_edf' : w_w404, 'w37_edf' : w_w37}
for wkey, ww in w.items():
    ww = dc(ww[list(emis.values())])
    ww['total'] = ww.sum(axis=1)
    # w[wkey]['net'] = xa_abs_dict[wkey]*area
    w[wkey] = ww.T*1e-3

# Get the posterior xhat_abs (we can use 'w404_edf' because we
# aren't including wetlands in the total)
xhat_abs = (w['w404_edf'].loc['total'].values[:, None]*xhat)

## ------------------------------------------------------------------------ ##
## Get list of metropolitan statistical area population
## ------------------------------------------------------------------------ ##
pop = pd.read_csv(f'{data_dir}cities/cbsa_pop_est.csv', header=0,
                  dtype={'City' : str, '2012' : int, '2018' : int, '2019' : int})
pop = pop.rename(columns={'City' : 'name'})
pop = pop.sort_values(by='2019', ascending=False, ignore_index=True)
pop = pop.iloc[2:, :].reset_index(drop=True)
pop['name'] = pop['name'].str[:-11]

## ------------------------------------------------------------------------ ##
## Calculate metropolitan statistical areas sectoral prior and posterior
## ------------------------------------------------------------------------ ##
# Calculate the area, prior emissions, posterior emissions, and xhat
# for each column, then concatenate it into one dataframe
## Area
area_c = (w_city @ area.reshape(-1,)).rename('area')

## Prior (Because we don't include wetlands, we can just use w404)
## (which is to say that is the same for all ensemble members)
prior_c = (w_city @ w['w404_edf'].T).add_prefix('prior_')

## Posterior (We want a matrix that is ncities x 15)
post_c = (w_city @ xhat_abs)

## Posterior ratios
xhat_c = post_c/prior_c['prior_total'].values[:, None]

## Get statistics
xhat_c = ip.get_ensemble_stats(xhat_c).add_prefix('xhat_')
post_c = ip.get_ensemble_stats(post_c).add_prefix('post_')

## Aggregate
summ_c = pd.concat([pd.DataFrame(area_c), prior_c, post_c, xhat_c], axis=1)

# Calculate xhat and difference
summ_c['xhat'] = (summ_c['post_mean']/summ_c['prior_total'])
summ_c['diff'] = summ_c['post_mean'] - summ_c['prior_total']

# Merge DOFS into summ_c
summ_c = pd.concat([summ_c, dofs_c.loc[summ_c.index]], axis=1)

# Check whether the using a DOFS threshold matters
for i in [0, 0.05, 0.1, 0.15, 0.2, 0.25]:
    summ_c_dofs = summ_c[summ_c['dofs_mean'] > i]
    xhat_mean_2b = summ_c_dofs['post_mean'].sum()/summ_c_dofs['prior_total'].sum()
    print(summ_c_dofs['xhat'].mean())
    print(summ_c['post_mean'].sum()/summ_c['prior_total'].sum())
    print(f'  xhat for cities (DOFS >= {i:.2f}) ', xhat_mean_2b)

# Subset based on DOFS
print('-'*70)
print('Non-optimized cities')
print(summ_c[summ_c['dofs'] < DOFS_filter].index.values)
nc_old = summ_c.shape[0]
summ_c = summ_c[summ_c['dofs'] >= DOFS_filter]

# Calculate means
xhat_mean_1 = summ_c['xhat'].mean()
xhat_mean_2 = summ_c['post_net'].sum()/summ_c['prior_net'].sum()
delta_mean = (summ_c['post_total'] - summ_c['prior_total']).mean()

# Calculate the fraction of emissions from urban areas in CONUS
conus_mask = np.load(f'{data_dir}CONUS_mask.npy')
city_emis_frac_prior = summ_c['prior_total'].sum()/(w.loc['total']*conus_mask).sum()
city_emis_frac_post = summ_c['post_total'].sum()/((w.loc['total']*xhat.reshape(-1,))*conus_mask).sum()
area_frac = summ_c['area'].sum()/(conus_mask.reshape((-1, 1))*area).sum()
print(summ_c['area'].sum())
print((conus_mask.reshape((-1, 1))*area).sum())
print((conus_mask.reshape((-1, 1))*area).shape)

print('-'*70)
print(f'Analyzed {summ_c.shape[0]}/{nc_old} cities:')
print(f'These cities are responsible for {(100*city_emis_frac_prior):.2f}% of prior anthropogenic emissions and {(100*city_emis_frac_post):.2f}% of posterior anthropogenic emissions in CONUS.')
print(f'These cities take up {(100*area_frac):.2f}% of surface area.')
print('  xhat mean                      ', xhat_mean_1)
print('  xhat for cities                ', xhat_mean_2)
print('  mean delta emissions (Gg/yr)   ', delta_mean)

# Merge in population
summ_c['pop'] = pop.set_index('name').loc[summ_c.index]['2019']

# Per capita methane emissions (Gg/person)
summ_c['post_total_pc'] = summ_c['post_total']/summ_c['pop']

# Sort by per capita methane emissions and print out
summ_c = summ_c.sort_values(by='post_total_pc', ascending=False)
print('-'*70)
print('Largest per capita methane emissions')
print(summ_c.index.values[:10])
print('These cities ')

# Sort by per km2 methane emissions
summ_c = summ_c.sort_values(by='post_total', ascending=False)
print('-'*70)
print('Largest per km methane emissions')
print(summ_c.index.values[:10])
print('-'*70)

# Sort by population
summ_c = summ_c.sort_values(by='pop', ascending=False)
# summ_c = summ_c.iloc[:nc, :]

## ------------------------------------------------------------------------ ##
## Plot maps
## ------------------------------------------------------------------------ ##
city = shapefile.Reader(f'{data_dir}cities/2019_tl_us_cbsa/tl_2019_us_cbsa.shp')

fig, ax = fp.get_figax(maps=True, lats=clusters.lat, lons=clusters.lon)
for shape in city.shapeRecords():
    if shape.record[3] in summ_c.index.values:
        # Get edges of city
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        c_poly = Polygon(np.column_stack((x, y)))
        color = sf_cmap(div_norm(summ_c.loc[shape.record[3]]['xhat']))
        ax.fill(x, y, facecolor=color, edgecolor='black')
ax = fp.format_map(ax, lats=clusters.lat, lons=clusters.lon)
cmap = plt.cm.ScalarMappable(cmap=sf_cmap, norm=div_norm)
cax = fp.add_cax(fig, ax)
cb = fig.colorbar(cmap, ax=ax, cax=cax)
cb = fp.format_cbar(cb, 'Scale factor')
fp.save_fig(fig, plot_dir, f'cities_map')

## ------------------------------------------------------------------------ ##
## Plot correlations with area and population
## ------------------------------------------------------------------------ ##
# With respect to cities
fig, ax = fp.get_figax(cols=3, aspect=1.5, sharey=True)
ax[0].scatter(summ_c['pop'], summ_c['diff'], s=1, 
              color=fp.color(3))
ax[0] = fp.add_labels(ax[0], 'Population', 
                      'Posterior - prior\nemissions 'r'(Gg a$^{-1}$)',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE)
ax[1].scatter(summ_c['area'], summ_c['diff'], s=1, 
              color=fp.color(5))
ax[1] = fp.add_labels(ax[1], r'Area (km$^2$)', '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE)
ax[2].scatter(summ_c['pop']/summ_c['area'], summ_c['diff'], s=1, 
              color=fp.color(7))
ax[2] = fp.add_labels(ax[2], 'Population density\n'r'(km$^{-2}$)', '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE)
fp.save_fig(fig, plot_dir, f'cities_correlations_{f}')

# With respect to grid cells
w_tot = w_city.sum(axis=0)
w_tot[w_tot > 1] = 1

pop_g = xr.open_dataset(f'{data_dir}cities/census2010_population_c.nc')

## ------------------------------------------------------------------------ ##
## Plot sectoral error correlation 
## ------------------------------------------------------------------------ ##
# rfiles = glob.glob(f'{data_dir}cities/r2_{f}_cities_*.csv')
# rfiles.sort()
# for ff in rfiles:
#     short_name = ff.split('_')[-1].split('.')[0]
#     short_name = '%s (%s)' % (city_names[short_name].split('-')[0], 
#                               city_names[short_name].split(', ')[-1])
#     r = pd.read_csv(ff, index_col=0, header=0)
#     labels = [list(emis.keys())[list(emis.values()).index(l)] for l in r.columns]
#     fig, ax = fp.get_figax()
#     c = ax.matshow(r, vmin=-1, vmax=1, cmap='RdBu_r')
#     ax.set_xticks(np.arange(0, len(labels)))
#     ax.set_xticklabels(labels, ha='center')
#     ax.xaxis.set_ticks_position('bottom')
#     ax.set_yticks(np.arange(0, len(labels)))
#     ax.set_yticklabels(labels, ha='right')

#     cax = fp.add_cax(fig, ax)
#     cb = fig.colorbar(c, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
#     cb = fp.format_cbar(cb, cbar_title='Pearson correlation coefficient')
#     ax = fp.add_title(ax, short_name)
#     fp.save_fig(fig, plot_dir, ff.split('/')[-1].split('.csv')[0])

## ------------------------------------------------------------------------ ##
## Plot results
## ------------------------------------------------------------------------ ##
summ_c = summ_c.iloc[:nc, :]
ys = np.arange(1, nc + 1)

fig, ax = fp.get_figax(cols=3, aspect=1, sharey=True) 
                       # max_height=config.BASE_HEIGHT*config.SCALE)
plt.subplots_adjust(wspace=0.1)

# Get labels
labels = summ_c.index.values
labels = ['%s (%s)' % (l.split('-')[0], l.split(', ')[-1]) for l in labels]

# Plot stacked bar
ax[0] = fp.add_title(ax[0], 'Urban emissions\nin largest CONUS cities', 
                     fontsize=config.TITLE_FONTSIZE)
# cc = [0, 6, 10, 3, 8, 12, 0] # old
cc = [2, 8, 6, 10, 0, 12, 0]
left_prior = np.zeros(nc)
left_post = np.zeros(nc)
for i, (l, e) in enumerate(emis.items()):
    ax[0].barh(ys - 0.175, summ_c[f'prior_{e}'], left=left_prior, height=0.3,
               color=fp.color(cc[i], lut=2*len(cc)), label=f'{l}')
    left_prior += summ_c[f'prior_{e}']
ax[0].barh(ys + 0.175, summ_c[f'post_total'], height=0.3,
           color=fp.color(0), alpha=0.5, label='Posterior total')

# Add labels
ax[0].set_yticks(ys)
ax[0].set_yticklabels(labels, ha='right', fontsize=config.TICK_FONTSIZE)
ax[0].set_ylim(0, nc + 1)
ax[0].invert_yaxis()

# Final aesthetics
ax[0] = fp.add_labels(ax[0], r'Emissions (Gg a$^{-1}$)', '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)

ax[0].tick_params(axis='both', labelsize=config.TICK_FONTSIZE)
ax[0].axvline(0, ls=':', lw=0.5, color='grey')
# ax[0].axvline(delta_mean, ls='--', lw=1, color='grey',
#               label='Mean emissions correction')
ax[0].set_xlim(0, 600)
for i in range(2):
    ax[0].axvline((i + 1)*200, color='0.75', lw=0.5, zorder=-10)

# Add labels
ax[0].text(summ_c['prior_total'][0] + 10, ys[0] + 0.05, 'Prior', ha='left',
           va='bottom', fontsize=config.TICK_FONTSIZE)
ax[0].text(summ_c['post_total'][0] + 10, ys[0] + 0.075, 'Posterior',
           ha='left', va='top', 
           fontsize=config.TICK_FONTSIZE)


# Plot emissions per capita
ax[1] = fp.add_title(ax[1], 
                     'Per capita urban emissions\nin largest CONUS cities', 
                     fontsize=config.TITLE_FONTSIZE)
left_prior = np.zeros(nc)
left_post = np.zeros(nc)
for i, (l, e) in enumerate(emis.items()):
    ax[1].barh(ys - 0.175, summ_c[f'prior_{e}']/summ_c['pop']*1e6, 
              left=left_prior, height=0.3, 
              color=fp.color(cc[i], lut=2*len(cc)), label=l)
    left_prior += summ_c[f'prior_{e}']/summ_c['pop']*1e6
ax[1].barh(ys + 0.175, summ_c[f'post_total']/summ_c['pop']*1e6, 
           height=0.3, color=fp.color(0), alpha=0.5)

ax[1].set_yticks(ys)
ax[1].set_ylim(0, nc + 1)
ax[1].invert_yaxis()
ax[1] = fp.add_labels(ax[1], 
                      r'Emissions per capita''\n(kg person'r'$^{-1}$ a$^{-1}$'')', 
                      '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)

ax[1].tick_params(axis='both', labelsize=config.TICK_FONTSIZE)
ax[1].axvline(0, ls=':', lw=0.5, color='0.5')
ax[1].set_xlim(0, 75)
for i in range(3):
    ax[1].axvline((i + 1)*20, color='0.75', lw=0.5, zorder=-10)


# Plot DOFS
ax[2] = fp.add_title(ax[2], 'Information content', 
                     fontsize=config.TITLE_FONTSIZE)
ax[2].barh(ys, summ_c['dofs'], color='0.5', height=0.6,
           label='Averaging kernel sensitivities')
ax[2].set_yticks(ys)
ax[2].set_ylim(0, nc + 1)
ax[2].set_xlim(0, 1)
ax[2].invert_yaxis()
ax[2] = fp.add_labels(ax[2], 'Averaging kernel\nsensitivities', '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)
for i in range(4):
    ax[2].axvline((i + 1)*0.2, color='0.75', lw=0.5, zorder=-10)

# Horizontal grid lines
for i in range(2):
    for k in range(3):
        ax[k].axhline((i + 1)*5 + 0.5, color='0.75', lw=0.5, zorder=-10)

# Legend for summary plot
# m_handles, m_labels = ax0.get_legend_handles_labels()
handles, labels = ax[0].get_legend_handles_labels()
# handles.extend(m_handles)
# labels.extend(m_labels)
ax[0] = fp.add_legend(ax[1], handles=handles, labels=labels, ncol=7,
                      fontsize=config.TICK_FONTSIZE, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.3))

fp.save_fig(fig, plot_dir, f'cities_{f}')