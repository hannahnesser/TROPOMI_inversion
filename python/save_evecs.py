# import copy
import os
from os.path import join
import math
# import itertools
import pickle

import sys
sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python')
import inversion as inv

import xarray as xr
import numpy as np
import pandas as pd

## -------------------------------------------------------------------------##
## Plotting defaults
## -------------------------------------------------------------------------##


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
## Load data
## -------------------------------------------------------------------------##

data_dir='/n/seasasfs02/hnesser/TROPOMI_inversion/evec_perturbations_ZQ'

# Jacobian
k = load_obj(join(data_dir, 'k.pkl'))

# Native-resolution prior and prior error
xa = load_obj(join(data_dir, 'xa.pkl'))
sa_vec = load_obj(join(data_dir, 'sa_vec.pkl'))

# Vectorized observations and error
y = load_obj(join(data_dir, 'y.pkl'))
y_base = load_obj(join(data_dir, 'y_base.pkl'))
so_vec = load_obj(join(data_dir, 'so_vec.pkl'))

## -------------------------------------------------------------------------##
## Set constants
## -------------------------------------------------------------------------##

RF = 0.001

## -------------------------------------------------------------------------##
## Create inversion object
## -------------------------------------------------------------------------##

# Create a true Reduced Rank Jacobian object
inv_zq = inv.ReducedRankJacobian(k, xa, sa_vec, y, y_base, so_vec)
inv_zq.rf = RF

# Save out the PPH
pph = inv_zq.pph()
save_obj(pph, join(data_dir, 'pph.pkl'))

# Complete an eigendecomposition of the prior pre-
# conditioned Hessian, filling in the eigenvalue
# and eigenvector attributes of true.
inv_zq.edecomp(number_of_evals=10)


