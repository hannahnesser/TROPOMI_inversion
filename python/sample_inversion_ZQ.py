import numpy as np
import xarray as xr
import pickle
import os
import pandas as pd
import math
import numpy as np
from os.path import join


## -------------------------------------------------------------------------##
## Define save and load functions
## -------------------------------------------------------------------------##
def save_obj(obj, name ):
        with open(name , 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
        with open( name, 'rb') as f:
                return pickle.load(f)

## -------------------------------------------------------------------------##
## Read data
## -------------------------------------------------------------------------##

data_dir='/n/seasasfs02/hnesser/TROPOMI_inversion/evec_perturbations_ZQ'

rrsd = load_obj(join(data_dir, 'rrsd.pkl'))
gc = pd.read_csv(join(data_dir, 'gc_output'), delim_whitespace=True, header=0,
                 usecols=['I', 'J', 'GOSAT', 'model', 'S_OBS'])
k = load_obj(join(data_dir, 'kA.pkl'))

# Now we will filter
# Define filters
k_idx = np.isfinite(k).all(axis=1)
gc_idx = np.isfinite(gc).all(axis=1)

# Apply
k = k[k_idx & gc_idx]
gc = gc[k_idx & gc_idx]

# Reshape error by applying indices from gc
so_vec = (rrsd[gc['J'] - 1, gc['I'] - 1]*gc['GOSAT']*1e9)**2
so_vec[~(so_vec < 0)] = gc['model'][~(so_vec < 0)]*1e18
so_vec = so_vec.values

# Now save out some quantities
y_base = gc['model'].values
y = gc['GOSAT'].values
xa = np.ones(k.shape[1])
sa_vec = np.ones(k.shape[1])*0.5**2

# save out
save_obj(k, join(data_dir, 'k.pkl'))
save_obj(so_vec, join(data_dir, 'so_vec.pkl'))
save_obj(y_base, join(data_dir, 'y_base.pkl'))
save_obj(y, join(data_dir, 'y.pkl'))
save_obj(xa, join(data_dir, 'xa.pkl'))
save_obj(sa_vec, join(data_dir, 'sa_vec.pkl'))
