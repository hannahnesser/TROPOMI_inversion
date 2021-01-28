# import glob
# import numpy as np
# import xarray as xr
# import re
import pickle
from os.path import join
from os import listdir

import numpy as np

# import pandas as pd
# import datetime
# import copy

def save_obj(obj, name):
    with open(name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_pf_check/ProcessedDir'
years = [2018]
months = [5]
days = np.arange(1, 32, 1)
files = 'YYYYMMDD_GCtoTROPOMI.pkl'

## -------------------------------------------------------------------------##
## Read files
## -------------------------------------------------------------------------##
for y in years:
    for m in months:
        for d in days:
            file = files.replace('YYYY', '%04d' % y)
            file = file.replace('MM', '%02d' % m)
            file = file.replace('DD', '%02d' % d)
            if file in listdir(data_dir):
                data = load_obj(d)
