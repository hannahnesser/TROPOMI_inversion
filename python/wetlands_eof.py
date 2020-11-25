import xarray as xr
import numpy as np
import pandas as pd
import math
from numpy.linalg import eigh
import datetime
import imageio

# Plotting
import matplotlib.pyplot as plt

# Import other modules
from os import listdir
from os.path import join
import sys
sys.path.append('.')
import plots as p
import format_plots as fp
import config

## -------------------------------------------------------------------------##
## Open wetlands file
## -------------------------------------------------------------------------##
w_loc = '../prior/wetlands/WetCHARTs_Highest_performance_Ensemble_v1.3.1_2010_2019.nc'
w = xr.open_dataset(w_loc, decode_times=False)

print('Wetlands file opened.')
print('There are %d ensemble members.\n' % len(w.model))

## -------------------------------------------------------------------------##
## Adjust emissions units
## -------------------------------------------------------------------------##
# Change from mg CH4 m-2 day-1 to molec cm-2 s-1
w['wetland_CH4_emissions'] *= 1e-3*6.02214129e23/16.04/1e4/(3600*24)

## -------------------------------------------------------------------------##
## Narrow to North American domain
## -------------------------------------------------------------------------##
#  Set latitude and longitude as coordinates
w = w.assign_coords({'lon' : w.longitude, 'lat' : w.latitude})
w = w.drop(['longitude', 'latitude'])

# Grid resolution   : 0.25 x 0.3125 (lat x lon)
# Longitdue min//max: -130 -60
# Latitude min/max  : 9.75 60.0
res = [0.25, 0.3125]
lon_lim = [-130, -60]
lat_lim = [9.75, 60]

# Adjust lon_lim and lat_lim for the fact that they are grid cell centers
lon_lim = [lon_lim[0] - res[1]/2, lon_lim[1] + res[1]/2]
lat_lim = [lat_lim[0] - res[0]/2, lat_lim[1] + res[0]/2]

# Limit the data to that domain
w = w.where((w.lon > lon_lim[0]) &
            (w.lon < lon_lim[1]) &
            (w.lat > lat_lim[0]) &
            (w.lat < lat_lim[1]),
            drop=True)

print('Defined latitude and longitude as coordinates and subsetted the data')
print('to the North American domain.\n')

## -------------------------------------------------------------------------##
## Adjust the time dimension
## -------------------------------------------------------------------------##
# Convert the time variable from months since Jan 1st 2010?
# to year-month

# We use the first of each month as the day since pandas requires it
days = np.ones(12*10)
months = ((w.time.values - 1) % 12) + 1
years = np.repeat(np.arange(2010, 2020, 1), 12, axis=0)

# Convert to pandas datetime
dates = pd.DataFrame({'year' : years, 'month' : months, 'day' : days})
dates = pd.to_datetime(dates)

# Convert to dataframe
dates = xr.DataArray(dates.values, dims='time')

# Save as variable
w = w.assign(time=dates)

# Limit it to year 2019
w = w.where(w.time.dt.year == 2019, drop=True)

print('Redefined time dimension and subsetted to 2019.\n')

## -------------------------------------------------------------------------##
## Create mapping from lat/lon grid cell to number
## -------------------------------------------------------------------------##
# Stack latitude and longitude dimension to create an n x p (number of
# grid cells by number of ensemble members) matrix for each month.
w = w.stack(z=['lat', 'lon'])

# Create xarray to serve as the basis for our index map
# This loop checks whether all models save values for the same
# grid cells across all months. It only needs to be run once.
count = w['wetland_CH4_emissions'].where((w.time == w.time[0]) &
                                         (w.model == w.model[0]),
                                         drop=True).drop(['time',
                                                          'model']).count()
print('There are %d grid cells at time t = %s and model m = %s.'
      % (count, str(w.time[0].values), str(w.model[0].values)))
for t in w.time:
    for m in w.model:
        clusters = w['wetland_CH4_emissions'].where((w.time == t) &
                                                    (w.model == m),
                                                    drop=True).squeeze()
        clusters = clusters.drop(['time', 'model'])
        if clusters.count() != count:
            print('!!! WARNING !!!')
            print('There are %d grid cells at time t = %s and model m = %s.'
                  % (clusters.count, str(t), str(m)))
            print('!!!!!!!!!!!!!!!')

print('\n')

# Create array containing possible index values
idx = np.zeros(clusters.shape)
idx[~xr.ufuncs.isnan(clusters)] = np.arange(count) + 1

# Replace non-nan values in clusters and fill with 0s  where nan
clusters = clusters.where(xr.ufuncs.isnan(clusters), idx)
clusters = clusters.fillna(0)

# Unstack
clusters = clusters.unstack()

# Save
clusters.to_netcdf('../Prior/Wetlands/wetland_clusters.nc')

## -------------------------------------------------------------------------##
## Calculate the data covariance matrix
## -------------------------------------------------------------------------##

# Drop nas
w = w.dropna(dim='z')

# Select wetlands
w = w['wetland_CH4_emissions']

# Remove ensemble mean
w -= w.mean(dim='model')

# Define variable order for matrix multiplication
w = w.transpose('time', 'model', 'z')
wT = w.transpose('time', 'z', 'model')

# Multiply wTw to find the covariance matrix
cov = np.einsum('...ij,...jk', wT, w)

print('Calculated the covariance matrix.\n')

## -------------------------------------------------------------------------##
## Eigendecomposition
## -------------------------------------------------------------------------##
# # Do the eignedecomposition
# for i in range(cov.shape[0]):
#     # Calculate eigenvalues and eigenvectors
#     evals, evecs = eigh(cov[i, :, :])

#     # Sort by the eigenvalues
#     idx = np.argsort(evals)[::-1]
#     evals = evals[idx]
#     evecs = evecs[:, idx]

#     # Check for imaginary eigenvector components
#     if np.any(np.iscomplex(evecs)):
#         imag_idx = np.where(np.iscomplex(evecs))
#         print('Imaginary eigenvectors exist at index %d of %d.'
#               % (imag_idx[1][0], len(evecs)))
#         print('The maximum absolute imaginary component is %.2e'
#               % np.max(np.abs(np.imag(evecs[imag_idx]))))
#         print('and occurs at ',
#               np.where(np.imag(evecs) ==
#                        np.max(np.abs(np.imag(evecs[imag_idx])))), '.')
#         print('Forcing eigenvectors to real component alone.')

#         # Force real
#         evecs = np.real(evecs)

#     # Save out evecs
#     np.savetxt('../Prior/Wetlands/EOFs_m%02d.csv' % int(i+1), evecs,
#                delimiter=',')
#     np.savetxt('../Prior/Wetlands/EOFs_eval_m%02d.csv' % int(i+1), evals,
#                delimiter=',')
#     print('Eigendecomposition of month %d complete.' % (i+1))

## -------------------------------------------------------------------------##
## Plot the first few eigenvectors
## -------------------------------------------------------------------------##

# for i in range(1, 13):
#     month = datetime.date(1900, i, 1).strftime('%B')
#     evecs = np.loadtxt('../prior/wetlands/EOFS_m%02d.csv' % i, delimiter=',')
#     fig, ax, c = p.plot_state_grid(evecs[:, :3], 1, 3,
#                                    clusters, cmap='RdBu_r',
#                                    vmin=-0.1, vmax=0.1,
#                                    titles=['1', '2', '3'])
#     fig.suptitle(month, y=1.2, fontsize=config.TITLE_FONTSIZE*config.SCALE)
#     fp.save_fig(fig, '../plots', 'EOFs_m%02d' % i)
# # plt.show()

# Create a gif
images = []
filenames = listdir('../plots/')
filenames = [f for f in filenames if f[:3] == 'EOF']
filenames.sort()
for f in filenames:
    images.append(imageio.imread(join('../plots', f)))
imageio.mimsave('../plots/EOFs.gif', images, duration=2)


## -------------------------------------------------------------------------##
## Scratch
## -------------------------------------------------------------------------##

