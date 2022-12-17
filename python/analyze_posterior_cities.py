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

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# DOFS_filter
DOFS_filter = 0.05

# Number of cities
nc = 10

# Define emission categories
emis = {'Landfills' : 'landfills', 'Wastewater' : 'wastewater',
        'Oil and natural gas' : 'ong', 
        'Coal' : 'coal', 'Livestock' : 'livestock',
        'Other anthropogenic' : 'other_anth'}

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
    'New York' : 104.737, # Still not sure what GWP they use, but we use 2019 from their most recent inventory https://nyc-ghg-inventory.cusp.nyu.edu/
    'Philadelphia' : 28.7, # 2019 https://www.phila.gov/media/20220418144709/2019-greenhouse-gas-inventory.pdf
                    }

city_studies = {
    'Lamb et al. (2016)' : {'Indianapolis' : [29, 14, 25],
                            'Indianapolis' : [41, 12, 12],
                            'Indianapolis' : [81, 11, 11]}, # https://pubs.acs.org/doi/10.1021/acs.est.6b01198
    'Jones et al. (2021)' : {'Indianapolis' : [37, 11, 11]}, # https://acp.copernicus.org/articles/21/13131/2021/acp-21-13131-2021.html
    'Sargent et al. (2021)' : {'Boston' : [198, 47, 47]}, # 2012 to 2020 https://www.pnas.org/doi/full/10.1073/pnas.2105804118
    'Ren et al. (2018)' : {'Washington' : [288, 142, 142]}, # 2016 values for winter only, DC Baltimore https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2018JD028851
    'Lopez-Coto et al. (2020)' : {'Washington' : [212, 61, 61]}, # February 2016 aircraft using Bayesian inversion for DC Baltimore https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7261234/
    'Huang et al. (2019)' : {'Washington' : [468, 108, 108]}, # 2016 observations from a tower site using a Lagrangian particle dispersion with geostat inversion https://pubs-acs-org.ezp-prod1.hul.harvard.edu/doi/pdf/10.1021/acs.est.9b02782 
    'Yadav et al. (2019)' : {'Los Angeles' : [333, 89, 89]}, # 2015-2016 mean, excluding Aliso Canyon leak https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JD030062
    'Cui et al. (2015)' : {'Los Angeles' : [406, 81, 81]}, # mean of 6 flights flown over 2010 https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/2014JD023002
    'Wunch et al. (2016)' : {'Los Angeles' : [413, 86, 86]}, # 2007 - 2016 average, relatively steady methane emissions https://acp.copernicus.org/articles/16/14091/2016/acp-16-14091-2016.pdf
    'Cusworth et al. (2020)' : {'Los Angeles' : [274, 72, 72]}, # Multi-tiered inversion with CLARS-FTS and TROPOMI aand AVIRIS-NG for Jan 2017- Sept 2018 https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2020GL087869
    'Jeong et al. (2016)' : {'Los Angeles' : [380, 79, 110],
                             'San Francisco' : [245, 86, 95]}, # values are medians June 2013 - May 2014 https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/2016JD025404
    'Jeong et al. (2017)' : {'San Francisco' : [226, 60, 63]}, # Median for Sept - Dec 2015, much of underestimation from landfills https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/2016GL071794
    'Guha et al. (2020)' : {'San Francisco' : [222.3, 40, 40]}, # 88% increase over inventory, 2015 - 2019 measurements https://pubs-acs-org.ezp-prod1.hul.harvard.edu/doi/pdf/10.1021/acs.est.0c01212
    'Pitt et al. (2022)' : {'New York' : [313, 96, 96]}, # Nine research flights during non-growing season of 2018 - 2020 https://online.ucpress.edu/elementa/article/10/1/00082/119571
    'Fairley and Fischer (2015)' : {'San Francisco' : [240, 60, 60]}, # 2009 - 2012 https://www.sciencedirect.com/science/article/pii/S1352231015000886
    # 'Karion et al. (2015)' : {'Dallas' : [660, 110, 110]}, # DFW mass baalance from 8 different flight days in March and October 2013 https://pubs.acs.org/doi/full/10.1021/acs.est.5b00217
    r'Plant et al. (2019) (CO$_2$)' : {'Washington' : [125, 42, 51],
                                       'Philadelphia' : [143, 39, 43],
                                       'New York' : [433, 105, 126],
                                       'Boston' : [77, 16, 19]}, # Aircraft campaign measurement of CH4/CO and CH4/CO2 ratios
    r'Plant et al. (2019) (CO)' : {'Washington' : [360, 175, 316],
                                   'Philadelphia' : [425, 164, 272],
                                   'New York' : [1116, 454, 857],
                                   'Boston' : [266, 122, 132]}, # Aircraft campaign measurement of CH4/CO and CH4/CO2 ratios https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019GL082635
    'Plant et al. (2022)' : {'Atlanta' : [221, 136, 259],
                             'Boston' : [161, 101, 189],
                             'Washington' : [139, 88, 170],
                             'Philadelphia' : [180, 110, 205],
                             'New York' : [574, 353, 662]} # with TROPOMI https://www.sciencedirect.com/science/article/pii/S0034425721004764
               }

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load the cities mask
w_city = pd.read_csv(f'{data_dir}cities/urban_areas_mask.csv', header=0).T

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
w = pd.read_csv(f'{data_dir}w_w404_edf.csv')
w = dc(w[list(emis.values())])
w['total'] = w.sum(axis=1)
w = w.T*1e-3

# Get the posterior xhat_abs (this is n x 15)
xhat_abs = (w.loc['total'].values[:, None]*xhat)

## ------------------------------------------------------------------------ ##
## Get list of metropolitan statistical area population
## ------------------------------------------------------------------------ ##
pop = pd.read_csv(f'{data_dir}cities/urban_areas_pop.csv', header=0,
                  dtype={'Name' : str, '2010' : int, '2000' : int})
# pop = pd.read_csv(f'{data_dir}cities/cbsa_pop_est.csv', header=0,
#                   dtype={'City' : str, '2012' : int, '2018' : int, '2019' : int})
pop = pop.rename(columns={'Name' : 'name'})
pop = pop.sort_values(by='2010', ascending=False, ignore_index=True)
# pop = pop.iloc[2:, :].reset_index(drop=True)
pop = pop.reset_index(drop=True)
# pop['name'] = pop['name'].str[:-11]
pop = pop.set_index('name')

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

# # Regrid the data ## This needs to happen before the three lines of
# ## code above can be run!
# ## Load 2010 data, which we will scale on a city level to 
# ## match 2019 population
# pop_g = xr.open_dataset(f'{data_dir}cities/census2010_population_c.nc')
# pop_g = pop_g['pop_density']

# ## Subset it for CONUS
# conus_mask = np.load(f'{data_dir}CONUS_mask.npy')
# conus_g = ip.match_data_to_clusters(conus_mask, clusters)
# conus_g = conus_g.where(conus_g > 0, drop=True)
# pop_g = pop_g.sel(lat=slice(conus_g.lat.min(), conus_g.lat.max()), 
#                   lon=slice(conus_g.lon.min(), conus_g.lon.max()))

# ## Get population grid
# delta_lon_pop = 0.01
# delta_lat_pop = 0.01
# Re = 6375e3 # Radius of the earth in m
# lon_e_pop = np.round(np.append(pop_g.lon.values - delta_lon_pop/2,
#                               pop_g.lon[-1].values + delta_lon_pop/2), 3)
# lat_e_pop = np.round(np.append(pop_g.lat.values - delta_lat_pop/2,
#                               pop_g.lat[-1].values + delta_lat_pop/2), 3)
# area_pop = Re**2*(np.sin(lat_e_pop[1:]/180*np.pi) - 
#                  np.sin(lat_e_pop[:-1]/180*np.pi))*delta_lon_pop/180*np.pi
# grid_pop = {'lat' : pop_g.lat, 'lon' : pop_g.lon,
#            'lat_b' : lat_e_pop, 'lon_b' : lon_e_pop}

# ## Get GEOS-Chem grid
# lon_e_gc = np.append(clusters.lon.values - s.lon_delta/2,
#                      clusters.lon[-1].values + s.lon_delta/2)
# lat_e_gc = np.append(clusters.lat.values - s.lat_delta/2,
#                      clusters.lat[-1].values + s.lat_delta/2)
# area_gc = Re**2*(np.sin(lat_e_gc[1:]/180*np.pi) - 
#                  np.sin(lat_e_gc[:-1]/180*np.pi))*s.lon_delta/180*np.pi
# grid_gc = {'lat' : clusters.lat, 'lon' : clusters.lon,
#            'lat_b' : lat_e_gc, 'lon_b' : lon_e_gc}

# ## Total emissions as check
# total = (pop_g*area_pop[:, None]).sum(['lat', 'lon']) # Mg/m2/yr -> Mg/yr
# print('Total 2010 population 0.01x0.01          : ', total.values)

# ## Get the regridder
# # regridder = xe.Regridder(grid_pop, grid_gc, 'conservative')
# # regridder.to_netcdf(f'{data_dir}cities/regridder_0.01x0.01_0.25x0.3125.nc')
# regridder = xe.Regridder(grid_pop, grid_gc, 'conservative', 
#                          weights=f'{data_dir}cities/regridder_0.01x0.01_0.25x0.3125.nc')

# ## Regrid the data
# pop_rg = regridder(pop_g)
# total_rg = (pop_rg*area_gc[:, None]).sum(['lat', 'lon'])
# print('Total 2010 population 0.25x0.3125        : ', total_rg.values)

# ## Scale the regridded population by the difference lost in the 
# ## regridding (we don't need this to be perfect--it's just for
# ## our urban area analysis)
# pop_rg *= total/total_rg
# total_rg_scaled = (pop_rg*area_gc[:, None]).sum(['lat', 'lon'])
# print('Total 2010 scaled population 0.25x0.3125 : ', total_rg_scaled.values)

# ## Save out
# pop_rg.to_netcdf(f'{data_dir}cities/pop_gridded.nc')

## ------------------------------------------------------------------------ ##
## Calculate metropolitan statistical areas sectoral prior and posterior
## ------------------------------------------------------------------------ ##
# Calculate the area, prior emissions, posterior emissions, and xhat
# for each column, then concatenate it into one dataframe
## Area
area_c = (w_city @ area.reshape(-1,)).rename('area')

## Prior (the same for all ensemble members)
prior_c = (w_city @ w.T).add_prefix('prior_')

## Posterior (We want a matrix that is ncities x 15)
post_c = (w_city @ xhat_abs)

## Difference
diff_c = post_c - prior_c['prior_total'].values[:, None]

## Posterior ratios
xhat_c = post_c/prior_c['prior_total'].values[:, None]

## Get statistics
xhat_stats_c = ip.get_ensemble_stats(xhat_c).add_prefix('xhat_')
post_stats_c = ip.get_ensemble_stats(post_c).add_prefix('post_')
diff_stats_c = ip.get_ensemble_stats(diff_c).add_prefix('diff_')

## Aggregate
summ_c = pd.concat([pd.DataFrame(area_c), prior_c, 
                    post_stats_c, xhat_stats_c, diff_stats_c], axis=1)

# Merge DOFS into summ_c
summ_c = pd.concat([summ_c, dofs_stats_c.loc[summ_c.index]], axis=1)

# Subset to remove cities that aren't optimized in any of the
# inversions
print('-'*75)
print('Non-optimized cities')
print(summ_c[summ_c['dofs_mean'] == 0].index.values)
nc_old = summ_c.shape[0]
xhat_c = xhat_c[summ_c['dofs_mean'] > 0]
dofs_c = dofs_c.loc[summ_c.index][summ_c['dofs_mean'] > 0]
post_c = post_c[summ_c['dofs_mean'] > 0]
summ_c = summ_c[summ_c['dofs_mean'] > 0]

# Print top-line statistics
## Calculate means in a way that treats each ensemble member first
xhat_mean = xhat_c.mean(axis=0) # Same as xhat mean
xhat_std = xhat_c.std(axis=0)
dofs_mean = dofs_c.mean(axis=0)

## Calculate the fraction of emissions from urban areas in CONUS
conus_mask = np.load(f'{data_dir}CONUS_mask.npy')
city_emis_frac_prior = summ_c['prior_total'].sum()/(w.loc['total']*conus_mask).sum()
city_emis_frac_post = post_c.sum()/(xhat_abs*conus_mask.reshape((-1, 1))).sum()
area_frac = summ_c['area'].sum()/(conus_mask.reshape((-1, 1))*area).sum()

## Print
print('-'*75)
print(f'Analyzed {summ_c.shape[0]}/{nc_old} cities:')
print(f'These cities are responsible for {(100*city_emis_frac_prior):.2f}% of prior anthropogenic emissions\nand {(100*city_emis_frac_post.mean()):.2f} ({(100*city_emis_frac_post.min()):.2f} - {100*city_emis_frac_post.max():.2f})% of posterior anthropogenic emissions in CONUS.\nThese cities take up {(100*area_frac):.2f}% of surface area.')
print(f'We find a net adjustment of {diff_c.sum(axis=0).mean():.2f} ({diff_c.sum(axis=0).min():.2f}, {diff_c.sum(axis=0).max():.2f}) Gg/yr.')
print(f'  xhat mean                      {xhat_mean.mean():.2f} ({xhat_mean.min():.2f}, {xhat_mean.max():.2f})')
print(f'  xhat std                       {xhat_std.mean():.2f}')
print(f'  dofs mean                      {dofs_mean.mean():.2f} ({dofs_mean.min():.2f}, {dofs_mean.max():.2f})')
print('  Ensemble means:')
print(xhat_c.mean(axis=0).round(2))
# print(f'  mean delta emissions (Gg/yr)   {}', delta_mean)

# Merge in population
# summ_c['pop_2010'] = pop.loc[summ_c.index]['2019']
# summ_c['pop_2000'] = pop.loc[summ_c.index]['2012']
summ_c['pop_2010'] = pop.loc[summ_c.index]['2010']
summ_c['pop_2000'] = pop.loc[summ_c.index]['2000']

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

# Sort by per km2 methane emissions
summ_c = summ_c.sort_values(by='post_mean', ascending=False)
print('-'*75)
print('Largest methane emissions')
print(summ_c.index.values[:nc])

# Print information on the largest nc cities
city_emis_frac_prior = summ_c['prior_total'][:nc].sum()/summ_c['prior_total'].sum()
city_emis_frac_post = post_c.loc[summ_c.index[:nc]].sum()/post_c.loc[summ_c.index].sum()
print(f'The largest {nc} cities by methane emissions are responsible for {(100*city_emis_frac_prior):.2f}% of urban\nprior emissions and {(100*city_emis_frac_post.mean()):.2f} ({(100*city_emis_frac_post.min()):.2f} - {(100*city_emis_frac_post.max()):.2f})% of urban posterior emissions.')
print('-'*75)

# Save out csv
summ_c.to_csv(f'{data_dir}/cities/summ_cities.csv', header=True, index=True)

## ------------------------------------------------------------------------ ##
## Plot distribution
## ------------------------------------------------------------------------ ##
fig, ax = fp.get_figax(cols=2, rows=1, aspect=1.5, sharey=True,
                       max_height=config.BASE_HEIGHT*config.SCALE)
plt.subplots_adjust(wspace=0.1)
counts0, bins0 = np.histogram(summ_c['xhat_mean'], bins=30)
ax[0].hist(bins0[:-1], bins0, weights=counts0, color=fp.color(2))
ax[0].axvline(summ_c['xhat_mean'].mean(), color='0.6', ls='--', lw=0.75)
ax[0] = fp.add_labels(ax[0], 'Urban scale factor', 'Count', 
                      fontsize=config.TICK_FONTSIZE,
                      labelsize=config.TICK_FONTSIZE)
mu = summ_c['xhat_mean'].mean()
ax[0].text(summ_c['xhat_mean'].mean()*1.1, counts0.max(), f'Mean : {mu:.2f}',
           fontsize=config.TICK_FONTSIZE)

# DOFS
counts, bins = np.histogram(summ_c['dofs_mean'], bins=30)
ax[1].hist(bins[:-1], bins, weights=counts, color=fp.color(5))
ax[1].axvline(summ_c['dofs_mean'].mean(), color='0.6', ls='--', lw=0.75)
ax[1] = fp.add_labels(ax[1], 'Sensitivity', '', 
                      fontsize=config.TICK_FONTSIZE,
                      labelsize=config.TICK_FONTSIZE)
mu = summ_c['dofs_mean'].mean()
ax[1].text(summ_c['dofs_mean'].mean()*1.1, counts0.max(), f'Mean : {mu:.2f}',
           fontsize=config.TICK_FONTSIZE)
fp.save_fig(fig, plot_dir, f'cities_distribution')

## ------------------------------------------------------------------------ ##
## Plot maps
## ------------------------------------------------------------------------ ##
city = shapefile.Reader(f'{data_dir}cities/2019_tl_urban_areas/tl_2019_us_uac10_buffered.shp')

fig, ax = fp.get_figax(maps=True, lats=clusters.lat, lons=clusters.lon)
for shape in city.shapeRecords():
    if shape.record[2] in summ_c.index.values:
        # Get edges of city
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        c_poly = Polygon(np.column_stack((x, y)))
        color = sf_cmap(div_norm(summ_c.loc[shape.record[2]]['xhat_mean']))
        ax.fill(x, y, facecolor=color, edgecolor='black', linewidth=0.1)
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
corr_c = dc(summ_c)
corr_c = corr_c[corr_c['dofs_mean'] >= 0.19]
# corr_c = corr_c.iloc[:15, :]

# Plot
ylabel = ['Posterior - prior\nemissions 'r'(Gg a$^{-1}$)', 'Scale factor']
for i, q in enumerate(['diff', 'xhat']):
    # Change _min and _max to errorbar terms
    corr_c[f'{q}_max'] = corr_c[f'{q}_max'] - corr_c[f'{q}_mean']
    corr_c[f'{q}_min'] = corr_c[f'{q}_mean'] - corr_c[f'{q}_min']

    fig, ax = fp.get_figax(cols=2, rows=2, aspect=1.5, sharey=True)
    plt.subplots_adjust(hspace=0.4, wspace=0.1)

    ## Population
    # ax[0, 0].errorbar(corr_c['pop_2000'], corr_c['diff_mean'], ms=1,
    #                   yerr=np.array(corr_c[['diff_min', 'diff_max']]).T, alpha=0.1,
    #                   color=fp.color(2), fmt='o', elinewidth=0.5, label='2012')
    m, b, r, _, _ = gc.comparison_stats(np.log10(corr_c['pop_2010'].values), 
                                        corr_c[f'{q}_mean'].values)
    ax[0, 0].errorbar(corr_c['pop_2010'], corr_c[f'{q}_mean'], ms=1,
                      yerr=np.array(corr_c[[f'{q}_min', f'{q}_max']]).T,
                      color=fp.color(2), fmt='o', elinewidth=0.5, label='2019')
    ax[0, 0] = fp.add_labels(ax[0, 0], 'Population', ylabel[i],
                             fontsize=config.TICK_FONTSIZE, 
                             labelsize=config.TICK_FONTSIZE, labelpad=10)
    ax[0, 0].text(0.05, 0.95, 
                  f'y = {m:.2f}x + {b:.2f} (R'r'$^2$'f' = {r**2:.2f})',
                  fontsize=config.TICK_FONTSIZE, va='top', ha='left',
                  transform=ax[0, 0].transAxes)

    ## Area
    m, b, r, _, _ = gc.comparison_stats(np.log10(corr_c['area'].values), 
                                        corr_c[f'{q}_mean'].values)
    ax[0, 1].errorbar(corr_c['area'], corr_c[f'{q}_mean'], ms=1,
                      yerr=np.array(corr_c[[f'{q}_min', f'{q}_max']]).T,
                      color=fp.color(4), fmt='o', elinewidth=0.5)
    ax[0, 1] = fp.add_labels(ax[0, 1], r'Area (km$^2$)', '',
                             fontsize=config.TICK_FONTSIZE, 
                             labelsize=config.TICK_FONTSIZE, labelpad=10)
    ax[0, 1].text(0.05, 0.95, 
                  f'y = {m:.2f}x + {b:.2f} (R'r'$^2$'f' = {r**2:.2f})',
                  fontsize=config.TICK_FONTSIZE, va='top', ha='left',
                  transform=ax[0, 1].transAxes)

    ## Population change
    m, b, r, _, _ = gc.comparison_stats(
                        (corr_c['pop_2010']/corr_c['pop_2000'] - 1).values, 
                        corr_c[f'{q}_mean'].values)
    ax[1, 0].errorbar(corr_c['pop_2010']/corr_c['pop_2000'] - 1, 
                      corr_c[f'{q}_mean'], ms=1, color=fp.color(6), fmt='o',
                      yerr=np.array(corr_c[[f'{q}_min', f'{q}_max']]).T,
                      elinewidth=0.5)
    ax[1, 0] = fp.add_labels(ax[1, 0], r'Relative population change 2000-2010',
                             ylabel[i], fontsize=config.TICK_FONTSIZE, 
                             labelsize=config.TICK_FONTSIZE, labelpad=10)
    ax[1, 0].text(0.05, 0.95, 
                  f'y = {m:.2f}x + {b:.2f} (R'r'$^2$'f' = {r**2:.2f})',
                  fontsize=config.TICK_FONTSIZE, va='top', ha='left',
                  transform=ax[1, 0].transAxes)


    ## Density
    # ax[1, 1].errorbar(corr_c['pop_2000']/corr_c['area'], corr_c[f'{q}_mean'], ms=1,
    #                   yerr=np.array(corr_c[[f'{q}_min', f'{q}_max']]).T, alpha=0.1,
    #                   color=fp.color(8), fmt='o', elinewidth=0.5, label='2012')
    m, b, r, _, _ = gc.comparison_stats(
                        np.log((corr_c['pop_2010']/corr_c['area']).values), 
                        corr_c[f'{q}_mean'].values)
    ax[1, 1].errorbar(corr_c['pop_2010']/corr_c['area'], corr_c[f'{q}_mean'],
                      ms=1, yerr=np.array(corr_c[[f'{q}_min', f'{q}_max']]).T,
                      color=fp.color(8), fmt='o', elinewidth=0.5, label='2019')
    ax[1, 1] = fp.add_labels(ax[1, 1], 'Population density\n'r'(km$^{-2}$)', 
                             '', fontsize=config.TICK_FONTSIZE, 
                             labelsize=config.TICK_FONTSIZE, labelpad=10)
    ax[1, 1].text(0.05, 0.95, 
                  f'y = {m:.2f}x + {b:.2f} (R'r'$^2$'f' = {r**2:.2f})',
                  fontsize=config.TICK_FONTSIZE, va='top', ha='left',
                  transform=ax[1, 1].transAxes)

    for j in range(2):
        for k in range(2):
            ax[j, k].set_xscale('log')
            ax[j, k].axhline(i, color='0.3', lw=0.5, ls='--')
    ax[1, 0].set_xscale('linear')

    fp.save_fig(fig, plot_dir, f'cities_correlations_{q}')

# With respect to grid cells
## We want the number of people in each urban grid cell
w_tot = w_city/w_city.sum(axis=0) # Don't double count people
w_tot = w_city*pop_g.T*pop_g_scale['2010'].values[:, None] # Get population
w_tot = w_tot.sum(axis=0) # Sum over cities

# ## Get the distribution of population densities
# fig, ax = fp.get_figax(aspect=1.5)
# ax.hist(w_tot.values/area.flatten(), color=fp.color(4), bins=500)
# ax.set_xlim(0, 1e3)
# ax = fp.add_labels(ax, r'Population density (km$^{-2}$)', 'Count',
#                    fontsize=config.TICK_FONTSIZE, 
#                    labelsize=config.TICK_FONTSIZE, labelpad=10)
# fp.save_fig(fig, plot_dir, f'cities_population_density_distribution')

# Create a mask
city_mask = (dofs.mean(axis=1) >= 0.25) & (w_tot > 100*area.flatten()).values 
print('-'*75)
print(f'We consider {city_mask.sum():d} grid cells in our urban analysis.')
print('-'*75)
w_tot = w_tot[city_mask]

## Now get the gridded difference between posterior and prior in
## urban areas
prior_g = w.loc['total'].values[city_mask]
post_g = xhat_abs.values[city_mask, :]
diff_g = post_g - prior_g[:, None]
diff_g = ip.get_ensemble_stats(diff_g)

xhat_g = xhat.values[city_mask, :]
xhat_g = ip.get_ensemble_stats(xhat_g)

data_g = {'diff' : diff_g, 'xhat' : xhat_g}
lims = [(-150, 150), (-1, 5)]
for i, (q, data) in enumerate(data_g.items()):
    ## Correct for error bar terms
    data['max'] = data['max'] - data['mean']
    data['min'] = data['mean'] - data['min']

    # Plot
    fig, ax = fp.get_figax(cols=2, aspect=1.5, sharey=True)
    plt.subplots_adjust(wspace=0.1)

    ## Population
    m, b, r, _, _ = gc.comparison_stats(w_tot.values, data['mean'].values)
    ax[0].errorbar(w_tot, data['mean'], yerr=np.array(data[['min', 'max']]).T,
                   ms=1, color=fp.color(2), fmt='o', elinewidth=0.5)
    ax[0] = fp.add_labels(ax[0], 'Population', ylabel[i],
                          fontsize=config.TICK_FONTSIZE, 
                          labelsize=config.TICK_FONTSIZE, labelpad=10)
    ax[0].text(0.05, 0.95, 
               f'y = {m:.2f}x + {b:.2f} (R'r'$^2$'f' = {r**2:.2f})',
               fontsize=config.TICK_FONTSIZE, va='top', ha='left',
               transform=ax[0].transAxes)

    ## Population density
    m, b, r, _, _ = gc.comparison_stats(w_tot.values/area.flatten()[city_mask],
                                        data['mean'].values)
    ax[1].errorbar(w_tot/area.flatten()[city_mask], data['mean'], 
                   yerr=np.array(data[['min', 'max']]).T,
                   ms=1, color=fp.color(4), fmt='o', elinewidth=0.5)
    ax[1] = fp.add_labels(ax[1], r'Population density (km$^{-2}$)', '',
                             fontsize=config.TICK_FONTSIZE, 
                             labelsize=config.TICK_FONTSIZE, labelpad=10)
    ax[1].text(0.05, 0.95, 
               f'y = {m:.2f}x + {b:.2f} (R'r'$^2$'f' = {r**2:.2f})',
               fontsize=config.TICK_FONTSIZE, va='top', ha='left',
               transform=ax[1].transAxes)

    for j in range(2):
        ax[j].set_xscale('log')
        ax[j].set_ylim(lims[i])
        ax[j].axhline(i, color='0.3', lw=0.5, ls='--')
        # ax[j].axhline(data['mean'].mean(), color=fp.color(2*j + 2), lw=0.5)

    fp.save_fig(fig, plot_dir, f'cities_correlations_grid_{q}')

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
# Subset summary
summ_c = summ_c.iloc[:nc, :]

# Adjust min/max definitions for error bars
summ_c['post_max'] = summ_c['post_max'] - summ_c['post_mean']
summ_c['post_min'] = summ_c['post_mean'] - summ_c['post_min']
summ_c['dofs_max'] = summ_c['dofs_max'] - summ_c['dofs_mean']
summ_c['dofs_min'] = summ_c['dofs_mean'] - summ_c['dofs_min']

# Define ys
ys = np.arange(1, nc + 1)

# And begin the plot!
fig, ax = fp.get_figax(cols=3, aspect=1.1, sharey=True) 
                       # max_height=config.BASE_HEIGHT*config.SCALE)
plt.subplots_adjust(wspace=0.1)

# figsize = fp.get_figsize(aspect=1.1*3)
# fig = plt.figure(figsize=figsize)
# gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.1)
# gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[0.75, 0.25],
#                                        subplot_spec=gs[0], hspace=0.1)
# ax0 = plt.subplot(gs2[0])
# ax1 = plt.subplot(gs2[1], sharey=ax0)
# ax2 = plt.subplot(gs[1], sharey=ax0)
# ax3 = plt.subplot(gs[2], sharey=ax0)
# ax = [ax0, ax1, ax2, ax3]

# Get labels
labels = summ_c.index.values
labels = ['%s (%s)' % (l.split(',')[0].split('--')[0], l.split(', ')[-1]) 
          for l in labels]
city_names = np.array(['%s' % (l.split(',')[0].split('--')[0]) 
                       for l in summ_c.index.values])

# Plot stacked bar
ax[0] = fp.add_title(ax[0], 'Largest urban methane\nemissions in CONUS', 
                     fontsize=config.TITLE_FONTSIZE)

# Formatters
cc = [fp.color(i, lut=2*7) for i in [2, 8, 6, 10, 0, 12, 0]]
# formats = ['o', 'v', 's', '^', 'P', '*', 'X', '<', 'p', 'd']
# sizes   = [3, 3, 3, 3, 5, 5, 5, 3, 5, 3]
formats = ['o', 's', '^']
sizes = [4, 4, 4]
# colors = ['white', 'black']

# Prior
left_prior = np.zeros(nc)
for i, (l, e) in enumerate(emis.items()):
    ax[0].barh(ys - 0.175, summ_c[f'prior_{e}'], left=left_prior, 
               height=0.3, color=cc[i], label=f'{l}')
    left_prior += summ_c[f'prior_{e}']

# Posterior
ax[0].barh(ys + 0.175, summ_c['post_mean'],
           xerr=np.array(summ_c[['post_min', 'post_max']]).T,
           error_kw={'ecolor' : '0.6', 'lw' : 0.5, 'capsize' : 1,
                     'capthick' : 0.5},
           height=0.3, color=fp.color(3), alpha=0.3, 
           label='Posterior total')

# Other studies
i = 0
for cs_name, cs in city_studies.items():
    study_used = False
    for c_name, cd in cs.items():
        if c_name in city_names:
            if ~study_used:
                label = cs_name
            else:
                label = None
            study_used = True
            # print(cs_name, i, formats[i])
            y = np.argwhere(c_name == city_names)[0][0]
            ax[0].errorbar(cd[0], y + 1, xerr=np.array(cd[1:])[:, None], 
                           fmt=formats[i % 3], markersize=sizes[i % 3], 
                           markerfacecolor=fp.color(math.ceil((i + 1)/3), 
                                                    cmap='viridis', lut=5),
                           markeredgecolor='black',
                           ecolor='black', elinewidth=0.25, 
                           capsize=1, capthick=0.5, zorder=10,
                           label=label)
    if study_used:
        i += 1

# Inventories
for i, (ci_name, ci) in enumerate(city_inventories.items()):
    if i == 0:
        label = 'City inventory'
    else:
        label = None
    y = np.argwhere(ci_name == city_names)[0][0]
    ax[0].scatter(ci, y + 1, marker='o', s=10, facecolor='white', 
                  edgecolor='black', zorder=10, label=label)

# Add labels
ax[0].set_yticks(ys)
ax[0].set_ylim(0, nc + 1)
ax[0].invert_yaxis()
ax[0].tick_params(axis='both', labelsize=config.TICK_FONTSIZE)

# Deal with scales
# ax[0].set_xlim(0, 2e3)
# ax[0].set_xscale('log')
# ax[0].set_xlim(10, 2.1e3)
ax[0].set_xlim(0, 600)

# Final aesthetics
ax[0].set_yticklabels(labels, ha='right', fontsize=config.TICK_FONTSIZE)
ax[0] = fp.add_labels(ax[0], r'Emissions (Gg a$^{-1}$)', '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)

for i in range(20):
    ax[0].axvline((i + 1)*100, color='0.75', lw=0.5, zorder=-10)

# Plot emissions per capita
ax[1] = fp.add_title(ax[1], 
                     'Per capita urban emissions', 
                     fontsize=config.TITLE_FONTSIZE)
left_prior = np.zeros(nc)
left_post = np.zeros(nc)
for i, (l, e) in enumerate(emis.items()):
    ax[1].barh(ys - 0.175, summ_c[f'prior_{e}']/summ_c['pop_2010']*1e6, 
              left=left_prior, height=0.3, color=cc[i], label=l)
    left_prior += summ_c[f'prior_{e}']/summ_c['pop_2010']*1e6
ax[1].barh(ys + 0.175, summ_c[f'post_mean']/summ_c['pop_2010']*1e6, 
           xerr=np.array(summ_c[['post_min', 'post_max']]/summ_c['pop_2010'].values[:, None]*1e6).T,
           error_kw={'ecolor' : '0.6', 'lw' : 0.5, 'capsize' : 1,
                     'capthick' : 0.5},
           height=0.3, color=fp.color(3), alpha=0.3, zorder=10)

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
ax[1].set_xlim(0, 60)
for i in range(6):
    ax[1].axvline((i + 1)*10, color='0.75', lw=0.5, zorder=-10)

# Add labels
ax[1].text((summ_c['prior_total']/summ_c['pop_2010']*1e6)[0] + 1, 
           ys[0] + 0.05, 'Prior', ha='left',
           va='bottom', fontsize=config.TICK_FONTSIZE)
ax[1].text(((summ_c['post_mean'] + summ_c['post_max'])/summ_c['pop_2010']*1e6)[0] + 1, 
           ys[0] + 0.075, 'Posterior',
           ha='left', va='top', 
           fontsize=config.TICK_FONTSIZE)

# Plot DOFS
ax[-1] = fp.add_title(ax[-1], 'Information content', 
                     fontsize=config.TITLE_FONTSIZE)
# ax[-1].barh(ys, summ_c['dofs_mean'], color='0.3', height=0.5,
#            xerr=np.array(summ_c[['dofs_min', 'dofs_max']]).T,
#            error_kw={'ecolor' : '0.6', 'lw' : 0.5, 'capsize' : 1, 
#                      'capthick' : 0.5},
#            label='Averaging kernel sensitivities')
ax[-1].errorbar(summ_c['dofs_mean'], ys, #fmt='none',
               xerr=np.array(summ_c[['dofs_min', 'dofs_max']]).T,
               fmt='D', markersize=2.5, markerfacecolor='white', 
               markeredgecolor='black', 
               ecolor='0.6', elinewidth=0.5, capsize=1, capthick=0.5)
ax[-1].set_yticks(ys)
ax[-1].set_ylim(0, nc + 1)
ax[-1].set_xlim(0, 1)
ax[-1].invert_yaxis()
ax[-1] = fp.add_labels(ax[-1], 'Averaging kernel\nsensitivities', '',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)
for i in range(4):
    ax[-1].axvline((i + 1)*0.2, color='0.75', lw=0.5, zorder=-10)

# Horizontal grid lines
for i in range(nc - 1):
    for k in range(3):
        if i % 5 == 0:
            ls = '-'
        else:
            ls = ':'
        ax[k].axhline((i + 1) + 0.5, color='0.75', lw=0.5, ls=ls, zorder=-10)


# Legend for summary plot
# m_handles, m_labels = ax0.get_legend_handles_labels()
handles, labels = ax[0].get_legend_handles_labels()
reorder = list(np.arange(1, len(handles))) + [0]
handles = [handles[i] for i in reorder]
labels = [labels[i] for i in reorder]
# print(len(labels))
# Remove duplicates
# labels = OrderedDict(zip(labels, handles))
# handles = list(labels.values())
# labels = list(labels.keys())

# # Sort
# reorder = [0, 5, 10, 15, 1, 6, 11, 16, 2, 7, 12, 3, 8, 13, 4, 9, 14]
# handles = [handles[idx] for idx in reorder]
# labels = [labels[idx] for idx in reorder]

# Add legend
ax[1] = fp.add_legend(ax[1], handles=handles, labels=labels, ncol=3,
                      fontsize=config.TICK_FONTSIZE, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.3))

fp.save_fig(fig, plot_dir, f'cities_ensemble')