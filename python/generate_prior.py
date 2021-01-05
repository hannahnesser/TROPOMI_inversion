import pickle
import numpy as np
import xarray as xr

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
## Define functions to generate absolute and relative priors
## -------------------------------------------------------------------------##

def generate_xa_rel(nstate):
    return np.ones(nstate)

def generate_sa_rel(nstate):
    # We assume 50% errors everywhere
    return 0.5*np.ones(nstate)

def generate xa_abs():

def generate_sa_abs():


## -------------------------------------------------------------------------##
## Save out absolute and relative priors
## -------------------------------------------------------------------------##
if __name__ == "__main__":

