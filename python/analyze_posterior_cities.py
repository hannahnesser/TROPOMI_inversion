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
        'Oil and\nnatural gas' : 'ong', 
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
# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()

# Load prior (Mg/km2/yr)
xa_abs = xr.open_dataarray(f'{data_dir}{xa_abs_file}').values.reshape((-1, 1))
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load posterior and DOFS
dofs = np.load(f'{data_dir}ensemble/dofs2_{f}.npy').reshape((-1, 1))
xhat = np.load(f'{data_dir}ensemble/xhat_fr2_{f}.npy').reshape((-1, 1))

# Load reduced DOFS
a_r = pd.read_csv(f'{data_dir}cities/a2_{f}_cities.csv', header=0,
                  index_col=0)
dofs_r = pd.DataFrame({'name' : a_r.index, 'dofs' : np.diag(a_r)})

# BC alteration
if optimize_BC:
    print('-'*30)
    print('Boundary condition optimization')
    print(' N E S W')
    print('xhat : ', xhat[-4:])
    print('dofs : ', dofs[-4:])
    print('-'*30)
    xhat = xhat[:-4]
    dofs = dofs[:-4]

# # Filter on DOFS filter
# xhat[dofs < DOFS_filter] = 1
# dofs[dofs < DOFS_filter] = 0

# Calculate xhat abs
xhat_abs = (xhat*xa_abs)

# Load weighting matrix in units Gg/yr
w = pd.read_csv(f'{data_dir}{w_file}') # Mg/yr
w = w[list(emis.values())]
w['total'] = w.sum(axis=1)
w['net'] = xa_abs*area
w = w.T*1e-3 # Mg/yr --> Gg/yr

# Load the cities mask
w_city = pd.read_csv(f'{data_dir}cities/cities_mask.csv', header=0).T

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
city_area = (w_city @ area.reshape(-1,)).rename('area')
city_prior = (w_city @ w.T).add_prefix('prior_')
city_post = (w_city @ (w.T*xhat)).add_prefix('post_')
city_summ = pd.concat([pd.DataFrame(city_area), city_prior, city_post], axis=1)

# Calculate xhat and difference
city_summ['xhat'] = (city_summ['post_net']/city_summ['prior_net'])
city_summ['diff'] = city_summ['post_total'] - city_summ['prior_total']

# Merge DOFS into city_summ
city_summ['dofs'] = dofs_r.set_index('name').loc[city_summ.index]
for i in [0.05, 0.1, 0.15, 0.2, 0.25]:
    city_summ_dofs = city_summ[city_summ['dofs'] >= i]
    # city_summ_dofs = city_summ_dofs[city_summ_dofs['prior_net'] > 0]
    xhat_mean_2b = city_summ_dofs['post_net'].sum()/city_summ_dofs['prior_net'].sum()
    print(f'  xhat for cities (DOFS >= {i:.2f}) ', xhat_mean_2b)

# Subset based on DOFS
print('-'*70)
print('Non-optimized cities')
print(city_summ[city_summ['dofs'] < DOFS_filter].index.values)
nc_old = city_summ.shape[0]
city_summ = city_summ[city_summ['dofs'] >= DOFS_filter]

# Calculate means
xhat_mean_1 = city_summ['xhat'].mean()
xhat_mean_2 = city_summ['post_net'].sum()/city_summ['prior_net'].sum()
delta_mean = (city_summ['post_total'] - city_summ['prior_total']).mean()

# Calculate the fraction of emissions from urban areas in CONUS
conus_mask = np.load(f'{data_dir}CONUS_mask.npy')
city_emis_frac_prior = city_summ['prior_total'].sum()/(w.loc['total']*conus_mask).sum()
city_emis_frac_post = city_summ['post_total'].sum()/((w.loc['total']*xhat.reshape(-1,))*conus_mask).sum()
area_frac = city_summ['area'].sum()/(conus_mask.reshape((-1, 1))*area).sum()
print(city_summ['area'].sum())
print((conus_mask.reshape((-1, 1))*area).sum())
print((conus_mask.reshape((-1, 1))*area).shape)

print('-'*70)
print(f'Analyzed {city_summ.shape[0]}/{nc_old} cities:')
print(f'These cities are responsible for {(100*city_emis_frac_prior):.2f}% of prior anthropogenic emissions and {(100*city_emis_frac_post):.2f}% of posterior anthropogenic emissions in CONUS.')
print(f'These cities take up {(100*area_frac):.2f}% of surface area.')
print('  xhat mean                      ', xhat_mean_1)
print('  xhat for cities                ', xhat_mean_2)
print('  mean delta emissions (Gg/yr)   ', delta_mean)

# Merge in population
city_summ['pop'] = pop.set_index('name').loc[city_summ.index]['2019']

# Per capita methane emissions (Gg/person)
city_summ['post_total_pc'] = city_summ['post_total']/city_summ['pop']

# Sort by per capita methane emissions and print out
city_summ = city_summ.sort_values(by='post_total_pc', ascending=False)
print('-'*70)
print('Largest per capita methane emissions')
print(city_summ.index.values[:10])
print('These cities ')

# Sort by per km2 methane emissions
city_summ = city_summ.sort_values(by='post_total', ascending=False)
print('-'*70)
print('Largest per km methane emissions')
print(city_summ.index.values[:10])
print('-'*70)

# Sort by population
city_summ = city_summ.sort_values(by='pop', ascending=False)
# city_summ = city_summ.iloc[:nc, :]

## ------------------------------------------------------------------------ ##
## Plot maps
## ------------------------------------------------------------------------ ##
city = shapefile.Reader(f'{data_dir}cities/2019_tl_us_cbsa/tl_2019_us_cbsa.shp')

fig, ax = fp.get_figax(maps=True, lats=clusters.lat, lons=clusters.lon)
for shape in city.shapeRecords():
    if shape.record[3] in city_summ.index.values:
        # Get edges of city
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        c_poly = Polygon(np.column_stack((x, y)))
        color = sf_cmap(div_norm(city_summ.loc[shape.record[3]]['xhat']))
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
ax[0].scatter(city_summ['pop'], city_summ['diff'], s=1, 
              color=fp.color(3))
ax[0] = fp.add_labels(ax[0], 'Population', 
                      'Posterior - prior\nemissions 'r'(Gg a$^{-1}$)',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE)
ax[1].scatter(city_summ['area'], city_summ['diff'], s=1, 
              color=fp.color(5))
ax[1] = fp.add_labels(ax[1], r'Area (km$^2$)', '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE)
ax[2].scatter(city_summ['pop']/city_summ['area'], city_summ['diff'], s=1, 
              color=fp.color(7))
ax[2] = fp.add_labels(ax[2], 'Population density\n'r'(km$^{-2}$)', '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE)
fp.save_fig(fig, plot_dir, f'cities_correlations_{f}')

# With respect to grid cells
w_tot = w_city.sum(axis=0)
w_tot[w_tot > 1] = 1

pop_g = xr.open_dataset(f'{data_dir}cities/census2010_population_c.nc')

# ## ------------------------------------------------------------------------ ##
# ## Plot sectoral error correlation 
# ## ------------------------------------------------------------------------ ##
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
city_summ = city_summ.iloc[:nc, :]
ys = np.arange(1, nc + 1)

fig, ax = fp.get_figax(cols=3, aspect=1, sharey=True) 
                       # max_height=config.BASE_HEIGHT*config.SCALE)
plt.subplots_adjust(wspace=0.2)

# Get labels
labels = city_summ.index.values
labels = ['%s (%s)' % (l.split('-')[0], l.split(', ')[-1]) for l in labels]

# Plot stacked bar
ax[0] = fp.add_title(ax[0], 'Urban emissions\nin largest CONUS cities', 
                     fontsize=config.TITLE_FONTSIZE)
cc = [1, 8, 3, 10, 5, 12, 7]
left_prior = np.zeros(nc)
left_post = np.zeros(nc)
for i, (l, e) in enumerate(emis.items()):
    ax[0].barh(ys - 0.175, city_summ[f'prior_{e}'], left=left_prior, height=0.3,
               color=fp.color(cc[i], lut=2*len(cc)), label=l)
    left_prior += city_summ[f'prior_{e}']
ax[0].barh(ys + 0.175, city_summ[f'post_total'], height=0.3,
           color=fp.color(1), alpha=0.6)

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

# Plot emissions per capita
ax[1] = fp.add_title(ax[1], 
                     'Per capita urban emissions\nin largest CONUS cities', 
                     fontsize=config.TITLE_FONTSIZE)
left_prior = np.zeros(nc)
left_post = np.zeros(nc)
for i, (l, e) in enumerate(emis.items()):
    ax[1].barh(ys - 0.175, city_summ[f'prior_{e}']/city_summ['pop']*1e6, 
              left=left_prior, height=0.3, 
              color=fp.color(cc[i], lut=2*len(cc)), label=l)
    left_prior += city_summ[f'prior_{e}']/city_summ['pop']*1e6
ax[1].barh(ys + 0.175, city_summ[f'post_total']/city_summ['pop']*1e6, 
           height=0.3, color=fp.color(1), alpha=0.6)

ax[1].set_yticks(ys)
ax[1].set_ylim(0, nc + 1)
ax[1].invert_yaxis()
ax[1] = fp.add_labels(ax[1], 
                      r'Emissions per capita''\n(kg person'r'$^{-1}$ a$^{-1}$'')', 
                      '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)

ax[1].tick_params(axis='both', labelsize=config.TICK_FONTSIZE)
ax[1].axvline(0, ls=':', lw=0.5, color='grey')
ax[1].set_xlim(0, 75)

# Plot DOFS
ax[2] = fp.add_title(ax[2], 'Information content', 
                     fontsize=config.TITLE_FONTSIZE)
ax[2].barh(ys, city_summ['dofs'], color='grey', height=0.6,
           label='Averaging kernel sensitivities')
ax[2].set_yticks(ys)
ax[2].set_ylim(0, nc + 1)
ax[2].set_xlim(0, 1)
ax[2].invert_yaxis()
ax[2] = fp.add_labels(ax[2], 'Averaging kernel\nsensitivities', '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)

# Legend for summary plot
# m_handles, m_labels = ax0.get_legend_handles_labels()
handles, labels = ax[0].get_legend_handles_labels()
# handles.extend(m_handles)
# labels.extend(m_labels)
ax[0] = fp.add_legend(ax[1], handles=handles, labels=labels, ncol=3,
                      fontsize=config.TICK_FONTSIZE, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.3))

fp.save_fig(fig, plot_dir, f'cities_{f}')