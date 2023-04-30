from os.path import join
from os import listdir
import sys
import glob
import copy
import math
import xarray as xr
import dask.array as da
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 10)

## ------------------------------------------------------------------------ ##
## Directories
## ------------------------------------------------------------------------ ##
base_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'inversion_results/'

# Custom packages
sys.path.append(code_dir)
import gcpy as gc
import troppy as tp
import invpy as ip
import inversion_settings as s

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# DOFS_filter
DOFS_filter = 0.05

# Get ensemble values
ensemble = glob.glob(f'{data_dir}iteration2/ensemble/xhat_fr2*')
ensemble.sort()
ensemble = [f.split('/')[-1][9:] for f in ensemble]

## -------------------------------------------------------------------- ##
## Set up a dask client
## -------------------------------------------------------------------- ##
from dask.distributed import Client, LocalCluster, progress
from dask.diagnostics import ProgressBar
import dask.config
dask.config.set({'distributed.comm.timeouts.connect' : 90,
                 'distributed.comm.timeouts.tcp' : 150,
                 'distributed.adaptive.wait-count' : 90,
                 'array.slicing.split_large_chunks' : False,
                 'temporary_directory' : f'{data_dir}/ensemble_dask_worker'})

# Open cluster and client
n_workers = 4
threads_per_worker = 2

cluster = LocalCluster(n_workers=n_workers,
                       threads_per_worker=threads_per_worker)
client = Client(cluster)

# Set chunk size.
nstate_chunk = 1e3 # int(np.sqrt(max_chunk_size)/5)
nobs_chunk = 5e4 # int(max_chunk_size/nstate_chunk/5)
chunks = {'nstate' : nstate_chunk, 'nobs' : nobs_chunk}

## ------------------------------------------------------------------------ ##
## Load files
## ------------------------------------------------------------------------ ##
# Default
optimize_BC = False

# Load prior (Mg/km2/yr)
xa_w404 = xr.open_dataarray(f'{data_dir}xa_abs_w404_edf.nc').values
xa_w37 = xr.open_dataarray(f'{data_dir}xa_abs_w37_edf.nc').values
xa_abs_dict = {'w404_edf' : xa_w404, 'w37_edf' : xa_w37}
area = xr.open_dataarray(f'{data_dir}area.nc').values
nstate = area.shape[0]

# Create dataframes for the ensemble data
dofs = pd.DataFrame(columns=[s[:-12] for s in ensemble])
xa_abs = pd.DataFrame(columns=[s[:-12] for s in ensemble])
# xhat = pd.DataFrame(columns=[s[:-12] for s in ensemble])
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
    dofs_s = np.load(f'{data_dir}iteration2/ensemble/dofs2_{suff}')
    xhat_s = np.load(f'{data_dir}iteration2/ensemble/xhat_fr2_{suff}')
    shat_s = np.load(f'{data_dir}iteration2/ensemble/shat_kpi2_{suff}')

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
    # xhat[suff[:-12]] = xhat_s
    xhat_abs[suff[:-12]] = xhat_s*xa_abs_dict[short_suff]
    shat_abs[suff[:-12]] = shat_s*(xa_abs_dict[short_suff]**2)

# Calculate the statistics of the posterior solution
dofs_mean = dofs.mean(axis=1)
xa_abs_mean = xa_abs.mean(axis=1)
xhat_abs_mean = xhat_abs.mean(axis=1)
xhat_mean = xhat_abs_mean/xa_mean

# Calculate the posterior error covariance from the ensemble
x = (xhat_abs - xhat_abs_mean).values
shat_e = da.tensordot(x, x.T, axes=(1, 0))
shat_e = shat_e/xa_abs_dict.reshape((-1, 1))/xa_abs_dict.reshape((1, -1))
print(shat_e.shape)

# BC alteration
if optimize_bc:
    print('-'*75)
    print('Boundary condition optimization')
    print(xhat_bc.T.round(2))
    print('Mean correction:')
    print(xhat_bc.mean(axis=1))
    print('Standard deviation')
    print(xhat_bc.std(axis=1))
    print('-'*75)

# Print information
# print('-'*75)
# print(f'We optimize {(dofs_mean >= DOFS_filter).sum():d} grid cells, including {xa_abs[dofs >= DOFS_filter].sum():.2E}/{xa_abs.sum():.2E} = {(xa_abs[dofs >= DOFS_filter].sum()/xa_abs.sum()*100):.2f}% of prior\nemissions. This produces {dofs[dofs >= DOFS_filter].sum():.2f} ({dofs.sum():.2f}) DOFS with an xhat range of {xhat.min():.2f}\nto {xhat.max():.2f}. There are {len(xhat[xhat < 0]):d} negative values.')
# print('-'*75)