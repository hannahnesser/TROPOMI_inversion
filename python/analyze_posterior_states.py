import sys
from copy import deepcopy as dc
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapefile
from shapely.geometry import Polygon, MultiPolygon
pd.set_option('display.max_columns', 20)

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
## Set user preferences
## ------------------------------------------------------------------------ ##
# DOFS_filter
DOFS_filter = 0.05

# Number of states
ns = 48

# Define file names
f = 'rg2rt_10t_w404_edf_bc0_rf0.25_sax0.75_poi80.0'
xa_abs_file = 'xa_abs_w404_edf_bc0.nc'
w_file = 'w_w404_edf.csv'
optimize_BC = False

# Define emission categories
emis = {'Livestock' : 'livestock', 'Coal' : 'coal', 
        'Oil and natural gas' : 'ong', 'Landfills' : 'landfills', 
        'Wastewater' : 'wastewater', 'Other' : 'other_anth'}

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc').squeeze()

# Load prior (Mg/km2/yr)
xa_abs = xr.open_dataarray(f'{data_dir}{xa_abs_file}').values.reshape((-1, 1))
area = xr.open_dataarray(f'{data_dir}area.nc').values.reshape((-1, 1))

# Load weighting matrix
w = pd.read_csv(f'{data_dir}{w_file}')
w = w[list(emis.values())]
w['total'] = w.sum(axis=1)
w['net'] = xa_abs

# Load posterior and DOFS
dofs = np.load(f'{data_dir}ensemble/dofs2_{f}.npy').reshape((-1, 1))
xhat = np.load(f'{data_dir}ensemble/xhat_fr2_{f}.npy').reshape((-1, 1))

# Load reduced DOFS
a_r = pd.read_csv(f'{data_dir}states/a2_{f}_states.csv', header=0,
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

## ------------------------------------------------------------------------ ##
## Get list of population
## ------------------------------------------------------------------------ ##
pop = pd.read_csv(f'{data_dir}states/states_pop.csv', header=0,
                  dtype={'State' : str, '2019' : int})
pop = pop.rename(columns={'State' : 'name'})
pop = pop.iloc[5:, :].reset_index(drop=True)
pop = pop.sort_values(by='2019', ascending=False, ignore_index=True)

## ------------------------------------------------------------------------ ##
## Get overlap between states and posterior
## ------------------------------------------------------------------------ ##
# Open the 2019 state boundaries
state = shapefile.Reader(f'{data_dir}states/2019_tl_us_state/tl_2019_us_state.shp')

# Load a numpy array for the information content analysis
w_state = pd.DataFrame(columns=np.arange(len(xhat)))


        # Having iterated through all the GC grid cells, append the state
        # information to the dataframe
        basic_info = {'name' : shape.record[6], 'area' : c_area[0], 
                      'xhat' : (c_emis_post['net']/c_emis_prior['net'])[0]}
        c_emis_prior = {f'prior_{c_emis_prior.columns[i]}' : c_emis_prior.values[0][i]
                        for i in range(len(c_emis_prior.columns))}
        c_emis_post = {f'post_{c_emis_post.columns[i]}' : c_emis_post.values[0][i] 
                       for i in range(len(c_emis_post.columns))}
        state_summ =  state_summ.append({**basic_info, **c_emis_prior, 
                                       **c_emis_post},
                                      ignore_index=True)

# Merge DOFS into state summary
dofs_r = dofs_r.set_index('name')
dofs_r = dofs_r.loc[state_summ['name']].reset_index(drop=True)
state_summ['dofs'] = dofs_r

# Merge in population
state_summ = state_summ.merge(pop[['name', '2019']], how='left')
state_summ = state_summ.rename(columns={'2019' : 'pop'})

# Per capita methane emissions (Gg/person)
state_summ['post_total_pc'] = state_summ['post_total']/state_summ['pop']

# Calculate difference and sort by it
state_summ['diff'] = state_summ['post_total'] - state_summ['prior_total']


state_summ = state_summ.sort_values(by='diff', 
                                    ascending=False).reset_index(drop=True)

## ------------------------------------------------------------------------ ##
## Plot results : absolute emissions
## ------------------------------------------------------------------------ ##
xs = np.arange(1, ns + 1)

fig, ax = fp.get_figax(rows=2, aspect=4, sharex=True,
                       max_height=config.BASE_HEIGHT*config.SCALE)

## Largest relative adjustments
# Sort the array
# state_summ = state_summ.sort_values(by=['xhat'], 
#                                   ascending=False).reset_index(drop=True)
# print(state_summ[state_summ['name'] == 'Pennsylvania'])
# print(state_summ[state_summ['name'] == 'New Mexico'])
# print(state_summ[state_summ['name'] == 'Illinois'])

# Get labels
labels = state_summ['name'].values
# labels = ['%s (%s)' % (l.split(',')[0].split('-')[0], l.split(', ')[-1]) 
#           for l in labels[:ns]]

# Plot stacked bar
cc = [1, 8, 3, 10, 5, 12, 7]
bottom_prior = np.zeros(ns)
bottom_post = np.zeros(ns)
for i, (l, e) in enumerate(emis.items()):
    ax[0].bar(xs - 0.15, state_summ[f'prior_{e}'], 
              bottom=bottom_prior, 
              width=0.3, color=fp.color(cc[i], lut=2*len(cc)), label=l)
    bottom_prior += state_summ[f'prior_{e}']
    ax[0].bar(xs + 0.15, state_summ[f'post_{e}'], 
              bottom=bottom_post, width=0.3,
              color=fp.color(cc[i], lut=2*len(cc)), alpha=0.6)
    bottom_post += state_summ[f'post_{e}']

# Add labels
ax[0].set_xticks(xs)
# ax[0].set_xticklabels(labels, ha='center', fontsize=config.TICK_FONTSIZE,
#                       rotation=90)
ax[0].set_xlim(0, ns+1)

# Final aesthetics
ax[0] = fp.add_labels(ax[0], '', 'Emissions\n'r'(Gg a$^{-1}$)', 
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)

ax[0].tick_params(axis='both', labelsize=config.TICK_FONTSIZE)
ax[0].set_ylim(0, 2.5e3)

# Plot DOFS
ax[1].bar(xs, state_summ['dofs'], color='grey', width=0.6,
          label='Averaging kernel sensitivities')
ax[1].set_xticks(xs)
ax[1].set_xticklabels(labels, ha='center', fontsize=config.TICK_FONTSIZE,
                      rotation=90)
# ax[1].set_yticklabels(dofs_r.in, ha='right', fontsize=config.TICK_FONTSIZE)
ax[1].set_xlim(0, ns + 1)
ax[1].set_ylim(0, 1)
ax[1] = fp.add_labels(ax[1], '', 'Averaging kernel\nsensitivities',
                      fontsize=config.TICK_FONTSIZE, 
                      labelsize=config.TICK_FONTSIZE, labelpad=10)


# Legend for summary plot
handles, labels = ax[0].get_legend_handles_labels()
# handles.extend(m_handles)
# labels.extend(m_labels)
ax[1] = fp.add_legend(ax[1], handles=handles, labels=labels, ncol=3,
                      fontsize=config.TICK_FONTSIZE, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.6))


fp.save_fig(fig, plot_dir, f'states_{f}')