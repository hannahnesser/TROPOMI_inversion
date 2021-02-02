# ========================================================================== ## Diagnosing error in prior simulation

import xarray as xr
from os import listdir
from os.path import join

data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/prior/species_conc'

# List files
files = listdir(data_dir)
files.sort()

# Open files
# for f in files:
sc1 = xr.open_dataset(join(data_dir, files[4]))['SpeciesConc_CH4']*1e9
sc2 = xr.open_dataset(join(data_dir, files[7]))['SpeciesConc_CH4']*1e9

print((sc1.values == sc2.values).sum())
print(sc1.shape)
# print(sc1.values.shape)
# print(sc2.values.shape)

# # Check the first level
# d_lev = sc.where(sc.lev == sc.lev[0], drop=True).squeeze()
# for t in d_lev.time:
#     print(d_lev.where(d_lev.time == t, drop=True).max())








# ========================================================================== #
# Original settings for TROPOMI operator
# import sys

# sat_data_dir = sys.argv[1]
# GC_data_dir = sys.argv[2]
# output_dir = sys.argv[3]

# LON_MIN = sys.argv[4]
# LON_MAX = sys.argv[5]
# LON_DELTA = sys.argv[6]

# LAT_MIN = sys.argv[7]
# LAT_MAX = sys.argv[8]
# LAT_DELTA = sys.argv[9]

# BUFFER = sys.argv[10:14]

# YEAR = sys.argv[14]
# MONTH = sys.argv[15]

# print(sat_data_dir)
# print(GC_data_dir)
# print(output_dir)
# print(LON_MIN)
# print(LON_MAX)
# print(LON_DELTA)
# print(LAT_MIN)
# print(LAT_MAX)
# print(LAT_DELTA)
# print(BUFFER)
# print(YEAR)
# print(MONTH)
