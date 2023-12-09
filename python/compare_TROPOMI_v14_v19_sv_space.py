import xarray as xr
import numpy as np
import sys
sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python/')
import gcpy as gc

k_sum = np.array([])
#data = gc.load_obj(f'inversion_data/2019_full.pkl')
for i in range(1, 21):
    print(i)
    k = xr.open_dataarray(f'inversion_data/iteration2/k/k2_c{i:02d}.nc')
    k = np.abs(k)
    k = k.sum(dim='nstate')
    k_sum = np.append(k_sum, k)