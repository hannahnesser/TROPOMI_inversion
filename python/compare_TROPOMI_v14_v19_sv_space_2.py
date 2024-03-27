import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import glob
import sys
# sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python/')
sys.path.append('/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/python/')
import gcpy as gc
import inversion_settings as s
import format_plots as fp

# data_dir = '/n/jacob_lab/Users/hnesser/TROPOMI_inversion'
data_dir = '../inversion_data'

# Idea 3
clusters = xr.open_dataarray(f'{data_dir}/clusters.nc')
k = xr.open_dataarray(f'{data_dir}/observations/k_agg.nc')
diff = xr.open_dataarray(f'{data_dir}/observations/v14_v19_diff.nc')
k = k.where(k['lon'].isin(diff['lon']), drop=True)
fig, ax = fp.get_figax()#maps=True, lats=clusters.lat, lons=clusters.lon,
                        #rows=1, cols=1)
mean_bias = []
for i in k['nstate']:
    kk = k.sel(nstate=i)
    # ax.hist(kk, bins=50)
    # diff = diff.where(np.abs(kk) > 1)
    dd = diff.where(np.abs(kk) > 1)
    mean_bias.append(float(dd.mean().values))
    # if dd.mean().isnull():
        # kk = kk.where(np.abs(kk) > 10)
        # dd.plot(vmin=-10, vmax=10, cmap='RdBu_r', ax=ax)
        # diff.plot(vmin=-10, vmax=10, cmap='RdBu_r', ax=ax,
        #           add_colorbar=False)
        # break

# print(mean_bias)
# ax = fp.format_map(ax, lats=clusters.lat, lons=clusters.lon)

print(np.mean(mean_bias))
print(np.std(mean_bias))
ax.hist(mean_bias, bins=50)
plt.show()