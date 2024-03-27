import xarray as xr
import numpy as np
import pandas as pd
import glob
import sys
sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python/')
# sys.path.append('/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/python/')
import gcpy as gc
import inversion_settings as s

data_dir = '/n/jacob_lab/Users/hnesser/TROPOMI_inversion'
# data_dir = '../inversion_data'

# Idea 2
# What if we look at the mean (v14 - v19) difference for each 
# reverse footprint (i.e., look at the mean (v14 - v19) difference
# for all observations for which dy/dx_i > some threshold)

# Step 1: Create K in a gridded space (observations)
# We want this to be pre-whitened, so unfortunately we'll have
# to do this many times...

# Start by getting the accompanying data (lats/lons)
obs_filter = pd.read_csv(f'{data_dir}/inversion_data/obs_filter.csv', header=0)['FILTER'].values
data = gc.load_obj(f'{data_dir}/inversion_data/2019_full.pkl')
data = data[obs_filter[:, None]]

# Separate out lats and lons and also round to the nearest grid cell center
lat = data['LAT'].values
lat = np.round(lat/s.lat_delta)*s.lat_delta

lon = data['LON'].values
lon = np.round(lon/s.lon_delta)*s.lon_delta

# Then get the observing system errors
so_mb = gc.read_file(f'{data_dir}/inversion_data/so_mb.nc')
so_lb = gc.read_file(f'{data_dir}/inversion_data/so_lb.nc')

# Load the average of the DOFS (technically, it would be 
# better to match the DOFS to the gamma/sa pair, but that
# would complicate thigns when we are iterating through
# the columns)
dofs_files = glob.glob(f'{data_dir}/posterior/dofs2*')
dofs = np.load(dofs_files[0])[:-4]
for i, f in enumerate(dofs_files[1:]):
    if f.split('dofs2_')[1][:2] == 'bc':
        dofs += np.load(f)[:-4]
    else:
        dofs += np.load(f)
dofs /= len(dofs_files)
dofs[dofs < 0.05] = 0

# Then, take the mean of the pre-whitened Jacobian by grid box
# j = 0
j = (11 - 1)*150000
for i in range(11, 21):
    print(f'Processing {i}')
    # Load K
    k = xr.open_dataarray(f'{data_dir}/inversion_data/iteration2/k/k2_c{i:02d}.nc', chunks={'nobs' : 15000})

    # Subset for DOFS
    k = k[:, dofs > 0]

    # Multiply by So
    ## To do: later, update to also be so_lb
    so_i = so_lb[j:(j + k.shape[0])]
    k = k*so_i

    # And, group by lats and lons
    k['lat'] = xr.DataArray(lat[j:(j + k.shape[0])], dims=('nobs'))
    k['lon'] = xr.DataArray(lon[j:(j + k.shape[0])], dims=('nobs'))
    k = k.set_index(nobs=['lat', 'lon'])
    k_ct = k.groupby('nobs').count()
    k = k.groupby('nobs').mean()

    # Combine
    k = xr.Dataset({'count' : k_ct, 'mean' : k})

    # Unstack and rename
    k = k.unstack() 
    k = k.rename({'nobs_level_0' : 'lat', 'nobs_level_1' : 'lon'})

    # Save out and close
    k.to_netcdf(f'{data_dir}/inversion_data/k_agg_lb_{i:02d}.nc')
    k.close()

    # Ramp j up
    j += len(so_i)


# Add them together

clusters = xr.open_dataset(f'{data_dir}/inversion_data/clusters.nc')
clusters = clusters.squeeze(drop=True)
clusters = clusters['Clusters']
clusters = clusters.where(clusters == 0, 0)

# Initialize with 01
def process_k(idx, clusters=clusters):
    k = xr.open_dataset(f'{data_dir}/inversion_data/k_agg_lb_{idx:02d}.nc')
    k = xr.merge([k, clusters]).drop('Clusters')
    k = k.fillna(0)
    k['mean'] = k['mean']*k['count']
    k['nstate'] = k['nstate']
    return k

k = process_k(1)

# Iterate through the remaining ones
for i in range(2, 21):
    print(i)
    knew = process_k(i)
    k['mean'] = k['mean'] + knew['mean']
    k['count'] = k['count'] + knew['count']
    knew.close()

k.to_netcdf(f'{data_dir}/inversion_data/k_lb_agg.nc')


# Load and combine
k_lb = xr.open_dataset(f'{data_dir}/inversion_data/k_lb_agg.nc')
k_mb = xr.open_dataset(f'{data_dir}/inversion_data/k_mb_agg.nc')
k = (0.625*k_lb + 0.75*k_mb)/2
k = k['mean']/k['count']



# dofs_files = glob.glob(f'{data_dir}/posterior/dofs2*')
# dofs = np.load(dofs_files[0])[:-4]
# for i, f in enumerate(dofs_files[1:]):
#     if f.split('dofs2_')[1][:2] == 'bc':
#         dofs += np.load(f)[:-4]
#     else:
#         dofs += np.load(f)
# dofs /= len(dofs_files)
# dofs[dofs < 0.05] = 0

# # Calculate or load k_sum
# k_sum = np.array([])
# for i in range(1, 21):
#     print(i)

#     # Load K
#     k = xr.open_dataarray(f'{data_dir}/inversion_data/iteration2/k/k2_c{i:02d}.nc')

#     # Take the absolute value of the pre-whitened Jacobian
#     k = np.abs(k)
#     k = k[:, dofs > 0]
#     k = k.sum(dim='nstate')
#     k_sum = np.append(k_sum, k)
# np.save(f'{data_dir}/inversion_data/ksum_dofs.npy', k_sum)
k_sum = np.load(f'{data_dir}/reduced_rank/ksum_dofs.npy')

# Observation mask
obs_filter = pd.read_csv(f'{data_dir}/observations/obs_filter.csv', header=0)['FILTER'].values

# Prior error
# sa = gc.read_file(f'{data_dir}/prior/sa.nc', cache=True)
# sa = sa**0.5

# Open So
so_mb = gc.read_file(f'{data_dir}/observations/so_mb.nc')
so_lb = gc.read_file(f'{data_dir}/observations/so_lb.nc')
# so = 1/so**0.5

# Open data
data = gc.load_obj(f'{data_dir}/observations/2019_full.pkl')
data = data[obs_filter[:, None]]

# Get latitudes and longitudes for grouping
lat = data['LAT'].values
lat = np.round(lat/s.lat_delta)*s.lat_delta

lon = data['LON'].values
lon = np.round(lon/s.lon_delta)*s.lon_delta

kw = pd.DataFrame({'lat' : lat, 'lon' : lon})
ensemble = {0 : (so_lb, 0.2, 0.5),
            1 : (so_lb, 0.45, 0.75),
            2 : (so_lb, 0.175, 0.5),
            3 : (so_lb, 0.35, 0.75),
            4 : (so_mb, 0.175, 0.5),
            5 : (so_mb, 0.3, 0.75),
            6 : (so_mb, 0.5, 1),
            7 : (so_mb, 0.175, 0.75)}

for key, tup in ensemble.items():
    so_sqrt_inv = (tup[1] / tup[0]**0.5)
    sa_sqrt = tup[2]
    kw[key] = so_sqrt_inv * k_sum * sa_sqrt

kw = kw.set_index(['lat', 'lon'])
kw = kw.groupby(['lat', 'lon']).sum()
kw = kw.mean(axis=1).rename('kw').to_frame()

kw = kw['kw']
# Account for clusters
clusters = xr.open_dataarray(f'{data_dir}/clusters.nc')
clusters = clusters.squeeze(drop=True)

# Create a dataframe
cdf = clusters.to_dataframe()
cdf = cdf[cdf['Clusters'] > 0]

# Join
kw = cdf.join(kw)
kw = kw.fillna(0)
kw = kw.sort_values('Clusters')

# Select kw, convert to xarray, and zero out NaNs
kw = kw['kw'].to_xarray()
kw.to_netcdf(f'{data_dir}/reduced_rank/kw_sum.nc')

# import matplotlib.pyplot as plt
# import format_plots as fp
# kw.plot(vmin=0, cmap=fp.cmap_trans('viridis'))
# plt.show()


# np.save(f'{data_dir}/reduced_rank/kw_mean.npy', kw['kw'].values)