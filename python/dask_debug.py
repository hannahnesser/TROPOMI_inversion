'''
A script created for debugging dask issues in RC office hours and/or
RC ticket submission.

'''
# Define the location where this work will occur.
data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion/'

# Next import dask and generate an instance of Client(). We set the local
# directory, which seems to allow dask to save temporary files at it
# performs operations. This solves one of the error messages I've received.
# try turning off adaptive scaling?
# change wait_count to 10?

from dask.distributed import Client, LocalCluster, progress
import dask.config
dask.config.set({'distributed.comm.timeouts.connect' : 90,
                 'distributed.comm.timeouts.tcp' : 150,
                 'distributed.adaptive.wait-count' : 90})

# Open cluster and client
cluster = LocalCluster(local_directory=data_dir,
                       n_workers=3, threads_per_worker=2)
client = Client(cluster)

# did 2 and 3 last time, which required 25 minutes for m = 227721
# 3 and 2 should take 11 minutes for m = 126571 (55%)...doesn't seem to make
# much difference
# 4 and 2 (m = 177,642, or 78% of January) took 13 minutes--this is an improvement
# 5 and 2 (m = 254,473) --> failed
# 6 and 1 --> failed
#all optimized for tensordort
# 1 and 2 always works

# How about einsum
# 4 and 3 --> 6% in 6.5 minutes
# 6 and 2 --> 8% in 10.5 minutes

# Import the remaining packages
import numpy as np
import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar

# Import custom packages
import sys
sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python/')
import gcpy as gc

##############################################################################
##### Minimally reproducible example                                     #####
##############################################################################
# Define dimensions
# My actual problem has nobs ~ 3e6 and nstate ~ 3e4. I have gotten this
# toy example working well for nobs = 1e5 and nstate = 1e4, but errors persist
# when I scale nobs up.
# nobs = 1e6 # number of observations
# nstate = 1e4 # number of state vector elements

# Define chunk size
# Dask suggests that we save space for 10 chunks per core. We have
# 45 GB across 12 cores, which means each chunk should be 0.375 GB
# or 9.375e7 float32 values. To leave plenty of wiggle room, I'll use 1e7
# values in each chunk.

# The square root of 1e7 is about 1e3. Since our final array is nstate
# x nstate, we'll set nstate_chunk = 1000 and define nobs_chunk accordingly.
# (I'll note here that I tested many configurations of these chunks on smaller
# toy examples and found that this was optimal.)
nstate_chunk = 2e3
nobs_chunk = 1e5
# 1e3 and 1e4 (m = 177,642) took 13 minutes
# 5e3 and 1e4 (m = 254,473 or 143%)

# # Define random matrices x (nobs x nstate) and y (nobs x 1).
# x = da.random.random((nobs, nstate),
#                      chunks=(nobs_chunk, nstate_chunk)).astype('float32')
# y = da.random.random((nobs,), chunks=(nobs_chunk)).astype('float32')

data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion/'
k_file   = [f'{data_dir}k0_m{i:02d}.nc' for i in range(1, 13)]
xa_file  = f'{data_dir}xa.nc'
sa_file  = f'{data_dir}sa.nc'
y_file   = f'{data_dir}y.nc'
ya_file  = f'{data_dir}ya.nc'
so_file  = f'{data_dir}so.nc'
c_file   = f'{data_dir}c.nc'

state_dim = 'nstate'
obs_dim = 'nobs'
dims = {'xa' : [state_dim], 'sa' : [state_dim, state_dim],
        'y' : [obs_dim], 'ya' : [obs_dim], 'so' : [obs_dim, obs_dim],
        'c' : [obs_dim], 'k' : [obs_dim, state_dim],
        'xhat' : [state_dim],
        'yhat' : [obs_dim],   'shat' : [state_dim, state_dim],
        'dofs' : [state_dim], 'a'    : [state_dim, state_dim],
        'rf' : None}

chunks = {'nstate' : nstate_chunk, 'nobs' : nobs_chunk}
k_chunks = {i : chunks[i] for i in dims['k']}
so_chunks = {i : chunks[i] for i in dims['so']}

# Read So in its entirety (not hard)
so = gc.read_file(so_file, chunks=so_chunks)
so = so.compute()

# Initialize indexing
# i0 = 0
# i0 = 227721
# i0 = 354292
# i0 = 531934
# i0 = 786407
# i0 = 1061898
# i0 = 1291612
# i0 = 1520433
# i0 = 1764612
# i0 = 2024161
# i0 = 2432958
i0 = 2694825

# Repeatable
month = 12

# Read the monthly Jacobian in chunks
k_m_file = k_file[month-1]
print('\n', k_m_file)
k_m = gc.read_file(k_m_file, chunks=k_chunks)

# Get the indices for So
i1 = i0 + k_m.shape[0]
print(f'{k_m.shape[0]} observations ({i0}, {i1})')

# Subset So
so_m = so[i0:i1]

# Calculate KT So inv K
test = da.tensordot(k_m.T/so_m, k_m, axes=(1, 0))
test = client.persist(test)
progress(test)

# Update i0
i0 = i1

# Save out
test = test.compute()
np.savetxt(f'{data_dir}ktsoinvk_{month:02d}.csv', test, delimiter=',')

# Next step
try:
    ktsoinvk = np.loadtxt(f'{data_dir}ktsoinvk_01.csv')
except:
    ktsoinvk = np.loadtxt(f'{data_dir}ktsoinvk_01.csv', delimiter=',')

for i in range(2, 13):
    print(i)
    try:
        temp = np.loadtxt(f'{data_dir}ktsoinvk_{i:02d}.csv')
    except:
        temp = np.loadtxt(f'{data_dir}ktsoinvk_{i:02d}.csv', delimiter=',')
    ktsoinvk = ktsoinvk + temp

# OLD OLD OLD
import matplotlib.pyplot as plt

fig, ax = fp.get_figax(maps=True, lats=settings.lats, lons=settings.lons)
c = ax.scatter(obs['LON'], obs['LAT'], c=k_m[:, 100], cmap=fp.cmap_trans('plasma_r'), vmin=0, vmax=0.05)
ax = fp.format_map(ax, settings.lats, settings.lons)
plt.show()

# Read in Jacobian
kwargs = {}
kwargs['combine'] = 'nested'
kwargs['concat_dim'] = dims['k']
kwargs['chunks'] = k_chunks
k = gc.read_file(*k_file, **kwargs)

kwargs.pop('combine')
kwargs.pop('concat_dim')
kwargs['chunks'] = {i : chunks[i] for i in dims['so']}

# Read in observational error covariance matrix and compute
so = gc.read_file(*[so], **kwargs)
so = so.compute()

# # Compute y (this improves performance).
# y = y.compute()

# Calculate the matrix product x.T @ diag(y^-1) @ x. We calculate this in two
# ways.
# z1 = da.einsum('ij,jk', x.T/y, x)
# z2 = da.tensordot(x.T/y, x, axes=(1, 0))
test = da.tensordot(k.T/so, k, axes=(1, 0))
# test = (k.T/so).dot(k)
# test = da.einsum('ij,jk', k.T/so, k)

# Persist the resulting nstate x nstate matrices and track progress
# z1 = client.persist(z1)
# progress(z1)

test = client.persist(test)
progress(test)

# If things go south...
client.cancel(test)
client.shutdown()
del(test)
del(so_m)
del(k_m)

'''
Record of tests
nstate_chunk         nobs_chunk           result
9375                 1e4                  stalled at 5% after ~30 seconds
1e4                  9e3                  gets to 8% in one minute and stalls
1e5                  9e2                  this hits the memory issue
1e4                  1e4                  1% in one minute sigh
5e4                  1e3                  too big
2e4                  9e3                  too big

The whole thing works well when nobs = 1e5 and nstate = 1e3, so we'll
do some optimzing here.
nstate_chunk      nobs_chunk      other        result
2e4               1e3             no y         3.7 s
"                 "               y            11.6 s
"                 "               y, computed  10.4 s/16.2 s
2e4               2e3             y, computed  11.6 s/12.1 s
2e4               3e3             y, computed  9.7 s.         0.24 GB
2e4               4e3             y, comp.     12.2 s
1e4               4e3             y, comp.     12.5 s
1e4               3e3             y, comp.     10 s
3e4               1e3             y, computed  13.4 s/

Then I increased nobs = 1e5 and nstate = 1e4. It takes 15 minutes. We're going
to try again and change the number of threads. It takes 15 minutes with
OMP_NUM_THREADS = 6.

With OMP_NUM_THREADS = 6 (though this doesn't seem to matter), I manually
changed the cluster settings. Before the default was processes=4, threads=12
(presumably n_workers=4, threads_per_worker=3). Summary:
n_workers       threads_per_worker          time
4               3                           14 min 49 sec
4               6                           14 min 52 sec
6               3                           17 min 2 sec
6               2                           15 min 49 sec
6               6                           16 min 10 sec

ugh okay so doesn't matter too much// 4 and 3 works best

Also need to try tensordot. This is orders of magnitude better! Now trying
many of the same tests
nstate_chunk         nobs_chunk           result
1e3                  1e3                  42.8 s
1e3                  1e4                  41.3 s

Then further try optimizing
# cluster = SLURMCluster(queue='huce_intel', cores=12, memory='45GB')
# cluster.scale(jobs=2)
'''
