## ========================================================================== ## Creating big mem inversion class
# ## ==========================================================================
from os.path import join
import os
from copy import deepcopy
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import xarray as xr
import pickle
import os
import pandas as pd
import math
import numpy as np
from scipy.linalg import eigh

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams

import sys
sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python')
import inversion as inv
import gcpy as gc
import invpy as ip
import format_plots as fp

def open_k(file_name):
    try:
        k = gc.load_obj(file_name)
        return k
    except FileNotFoundError:
        print(f'{file_name} not found.')


k = [f for f in os.listdir() if f[:2] == 'k0']
y = gc.load_obj('y.pkl')
y_base = gc.load_obj('kxa.pkl')
so_vec = gc.load_obj('so.pkl')
xa = gc.load_obj('xa.pkl')
sa_vec = gc.load_obj('sa.pkl')

nstate = xa.shape[0]
nobs = y.shape[0]

# kn = open_k(k[0])

def calculate_c():
    '''
    Calculate c for the forward model, defined as ybase = Kxa + c.
    Save c as an element of the object.
    '''
    c = np.zeros(nobs)
    i0 = 0
    i1 = 0
    for file_name in k:
        kn = open_k(file_name)
        i1 += kn.shape[0]
        print(i0, i1)
        c[i0:i1] = y_base[i0:i1] - np.matmul(kn, xa)
        print('hello')
        i0 = i1
    return c

c = calculate_c()

# # ========================================================================== ## Testing errors in eigenvector perturbations

# # ========================================================================== ## Diagnosing error in prior simulation part 2

from os.path import join
from os import listdir
import sys
import copy
import calendar as cal

import xarray as xr
# import xbpch as xb
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# data = xb.open_bpchdataset(filename='trac_avg.merra2_4x5_CH4.201501010000',
#                            tracerinfo_file='tracerinfo.dat',
#                            diaginfo_file='diaginfo.dat')

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# # Local preferences
# base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
# code_dir = base_dir + 'python'
# data_dir = base_dir + 'observations'
# output_dir = base_dir + 'inversion_data'
# plot_dir = base_dir + 'plots'

# # Cannon preferences
# base_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/'
# code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
# data_dir = f'{base_dir}ProcessedDir'
# output_dir = f'{base_dir}SummaryDir'
# plot_dir = None



# Information on the grid
lat_bins = np.arange(10, 65, 5)
lat_min = 9.75
lat_max = 60
lat_delta = 0.25
lon_min = -130
lon_max = -60
lon_delta = 0.3125
buffers = [3, 3, 3, 3]

## ------------------------------------------------------------------------ ##
## Import custom packages
## ------------------------------------------------------------------------ ##
# Custom packages
sys.path.append('.')
import config
import gcpy as gc
import troppy as tp
import format_plots as fp

tmp = gc.load_obj('/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/inversion_data/y.pkl')
print(tmp)

# data
year = 2019
month = 12
days = np.arange(1, 32, 1)
files = [f'{year}{month}{dd:02d}_GCtoTROPOMI.pkl' for dd in days]
data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/observations'

# # data = np.array([]).reshape(0, 13)
# # for f in files:
# #     month = int(f[4:6])
# #     day = int(f[6:8])
# #     print(day)
# #     new_data = gc.load_obj(join(data_dir, f))['obs_GC']
# #     new_data = np.insert(new_data, 11, month, axis=1)
# #     new_data = np.insert(new_data, 12, day, axis=1)
# #     data = np.concatenate((data, new_data))

# # columns = ['OBS', 'MOD', 'LON', 'LAT', 'iGC', 'jGC', 'PREC',
# #            'ALBEDO_SWIR', 'ALBEDO_NIR', 'AOD', 'MOD_COL',
# #            'MONTH', 'DAY']
# # data = pd.DataFrame(data, columns=columns)
# # data['DIFF'] = data['MOD'] - data['OBS']

# # print(data)

# # c = plt.scatter(data['OBS'], data['MOD'], c=data['DAY'], s=10, alpha=0.5)
# # plt.colorbar(c)
# # plt.ylim(1600, 2000)
# # plt.xlim(1600, 2000)
# # plt.show()

# # Isolated early days--we'll use day 1 as a test case
# # is it the restart file?
# # rst12 = xr.open_dataset(join(data_dir, 'GEOSChem.Restart.20191201_0000z.nc4'))
# # rst11 = xr.open_dataset(join(data_dir, 'GEOSChem.Restart.20191101_0000z.nc4'))
# # # for lev in rst11.lev:
# # #     print(lev.values)
# # #     print('Dec: ', rst12['SpeciesRst_CH4'].where(rst12.lev == lev, drop=True).min().values*1e9)
# # #     print('Nov: ', rst11['SpeciesRst_CH4'].where(rst11.lev == lev, drop=True).min().values*1e9)
# # #     print('\n')
# # rst12.plot.scatter(x='SpeciesRst_CH4', y='lev')
# # rst11.plot.scatter(x='SpeciesRst_CH4', y='lev')
# # plt.show()


# # Probably not because the December restart file is actually larger than
# # the November restart file.

# # Where in the atmosphere does it originate?
# files = [f'GEOSChem.SpeciesConc.{year}{month}{dd:02d}_0000z.nc4' for dd in days]
# # for f in files[0]:
# f = files[15]
# data = xr.open_dataset(join(data_dir, f))
# data = data.isel(time=1)
# data.plot.scatter(x='SpeciesConc_CH4', y='lev')
# # c = plt.scatter(data.lev, data['SpeciesConc_CH4'], c=data.time)
# # plt.colorbar(c)
# plt.show()


# # ========================================================================== ## Diagnosing error in prior simulation

# import xarray as xr
# from os import listdir
# from os.path import join

# data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/prior/species_conc'

# # List files
# files = listdir(data_dir)
# files.sort()

# # Open files
# # for f in files:
# sc1 = xr.open_dataset(join(data_dir, files[4]))['SpeciesConc_CH4']*1e9
# sc2 = xr.open_dataset(join(data_dir, files[7]))['SpeciesConc_CH4']*1e9

# print((sc1.values == sc2.values).sum())
# print(sc1.shape)
# # print(sc1.values.shape)
# # print(sc2.values.shape)

# # # Check the first level
# # d_lev = sc.where(sc.lev == sc.lev[0], drop=True).squeeze()
# # for t in d_lev.time:
# #     print(d_lev.where(d_lev.time == t, drop=True).max())

# # ========================================================================== ## Adding stratospheric analysis to TROPOMI operator

# VCH4_apriori = TROPOMI['methane_profile_apriori']/TROPOMI
# TROPOMI['dry_air_subcolumns']


# # ========================================================================== ## Diagnosing error in TROPOMI operator

# import numpy as np
# import xarray as xr
# import re
# import pickle
# import os
# import sys
# import pandas as pd
# import datetime
# import copy
# import glob

# sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python')
# import GC_to_TROPOMI as oper

# sat_data_dir = "/n/seasasfs02/hnesser/TROPOMI/downloads_14_14/"
# GC_data_dir = "/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/OutputDir/"
# output_dir = "/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/ProcessedDir/"

# LON_MIN = -130
# LON_MAX = -60
# LON_DELTA = 0.3125
# LAT_MIN = 9.75
# LAT_MAX = 60
# LAT_DELTA = 0.25
# BUFFER = [3, 3, 3, 3] # [N S E W]

# YEAR = 2019
# MONTH = 1
# DAY = 1

# ## ---------------------------------------------------------------------##
# ## Remove buffer boxes
# ## ---------------------------------------------------------------------##
# LAT_MAX -= LAT_DELTA*BUFFER[0]
# LAT_MIN += LAT_DELTA*BUFFER[1]
# LON_MAX -= LON_DELTA*BUFFER[2]
# LON_MIN += LON_DELTA*BUFFER[3]

# ## ---------------------------------------------------------------------##
# ## List all satellite files for the year and date defined
# ## ---------------------------------------------------------------------##
# # List all raw netcdf TROPOMI files
# allfiles=glob.glob(sat_data_dir+'*.nc')
# allfiles.sort()

# # Create empty list
# Sat_files = {}

# # Iterate through the raw TROPOMI data
# for index in range(len(allfiles)):
#     filename = allfiles[index]

#     # Get the date (YYYY, MM, and DD) of the raw TROPOMI file
#     shortname = re.split('\/|\.', filename)[-2]
#     strdate = re.split('_+|T', shortname)
#     start_date = strdate[4]
#     end_date = strdate[6]

#     # start condition
#     start = ((int(start_date[:4]) == YEAR)
#              and (int(start_date[4:6]) == MONTH))
#     end = ((int(end_date[:4]) == YEAR)
#            and (int(end_date[4:6]) == MONTH))

#     # Skip observations not in range
#     # if not (year and month):
#     if not (start or end):
#         continue

#     # Add the file to the list of Sat_files
#     if start:
#         if start_date in Sat_files.keys():
#             Sat_files[start_date].append(filename)
#         else:
#             Sat_files[start_date] = [filename]
#     elif end:
#         if end_date in Sat_files.keys():
#             Sat_files[end_date].append(filename)
#         else:
#             Sat_files[end_date] = [filename]

# print('Number of dates: ', len(Sat_files))

# ## -------------------------------------------------------------------------##
# ## Take a closer look at the specified date
# ## -------------------------------------------------------------------------##
# date = '%04d%02d%02d' % (YEAR, MONTH, DAY)
# filenames = Sat_files[date]
# print('=========== %s ===========' % date)
# preprocess = lambda d: oper.filter_tropomi(d, date,
#                                            LON_MIN, LON_MAX, LON_DELTA,
#                                            LAT_MIN, LAT_MAX, LAT_DELTA)
# TROPOMI = xr.open_mfdataset(filenames, concat_dim='nobs',
#                             combine='nested',
#                             chunks=10000,
#                             preprocess=preprocess)
# TROPOMI = oper.process_tropomi(TROPOMI, date)

# if TROPOMI is None:
#     print('No observations remain.')
#     print('================================')

# # Get observation dimension (number of good observations in that single
# # observation file)
# NN = TROPOMI.nobs.shape[0]
# print('Processing %d Observations' % NN)
# print('================================')

# # create an empty matrix to store TROPOMI CH4, GC CH4,
# # lon, lat, II, and JJ (GC indices)
# temp_obs_GC=np.zeros([NN, 11],dtype=np.float32)

# #================================
# #--- now compute sensitivity ---
# #================================

# # Then, read in the GC data for these dates. This works by
# # reading the lon, lat, pressure edge, xch4, xch4_adjusted
# # (which I believe is the stratospheric corrected data), TROPP
# # (which is the planetary boundary layer info), and dry air.
# GC = oper.read_GC(GC_data_dir, date)

# # Find the grid box and time indices corresponding to TROPOMI obs
# iGC, jGC, tGC = oper.nearest_loc(GC, TROPOMI)

# # Then select GC accordingly
# GC_P = GC['PEDGE'].values[tGC, iGC, jGC, :]
# GC_DA = GC['DRYAIR'].values[tGC, iGC, jGC, :]
# GC_CH4 = GC['CH4'].values[tGC, iGC, jGC, :]
# GC_COL = GC['GCCOL'].values[tGC, iGC, jGC]

# # Create mapping between GC and TROPOMI pressure levels
# intmap = oper.get_intmap(TROPOMI['pressures'].values, GC_P)
# newmap = oper.get_newmap(intmap, TROPOMI['pressures'].values, GC_P,
#                          GC_CH4, GC_DA)

# # Finally, apply the averaging kernel
# GC_base_post2 = oper.apply_avker(TROPOMI['column_AK'].values,
#                                 TROPOMI['methane_profile_apriori'].values,
#                                 TROPOMI['dry_air_subcolumns'].values,
#                                 newmap['GC_CH4'], newmap['GC_WEIGHT'])
# # GC_base_pri = apply_avker(np.ones(TROPOMI['column_AK'].shape),
# #                           TROPOMI['methane_profile_apriori'].values,
# #                           TROPOMI['dry_air_subcolumns'].values,
# #                           newmap['GC_CH4'], newmap['GC_WEIGHT'])

# GC_base_strat = apply_avker(TROPOMI['column_AK'].values,
#                             TROPOMI['methane_profile_apriori'].values,
#                             TROPOMI['dry_air_subcolumns'].values,
#                             newmap['GC_CH4'], newmap['GC_WEIGHT'],
#                             filt=(TROPOMI['pressures_mid'] < 200))

# # Save out values
# # The columns are: OBS, MOD, LON, LAT, iGC, jGC, PRECISION,
# # ALBEDO_SWIR, ALBEDO_NIR, AOD, MOD_COL
# temp_obs_GC[:, 0] = TROPOMI['methane']
# temp_obs_GC[:, 1] = GC_base_post
# temp_obs_GC[:, 2] = TROPOMI['longitude']
# temp_obs_GC[:, 3] = TROPOMI['latitude']
# temp_obs_GC[:, 4] = iGC
# temp_obs_GC[:, 5] = jGC
# temp_obs_GC[:, 6] = TROPOMI['precision']
# temp_obs_GC[:, 7] = TROPOMI['albedo'][:,1]
# temp_obs_GC[:, 8] = TROPOMI['albedo'][:,0]
# temp_obs_GC[:, 9] = TROPOMI['aerosol_optical_depth'][:,1]
# temp_obs_GC[:, 10] = GC_COL

# result={}
# result['obs_GC'] = temp_obs_GC





# # ========================================================================== #
# # Original settings for TROPOMI operator
# # import sys

# # sat_data_dir = sys.argv[1]
# # GC_data_dir = sys.argv[2]
# # output_dir = sys.argv[3]

# # LON_MIN = sys.argv[4]
# # LON_MAX = sys.argv[5]
# # LON_DELTA = sys.argv[6]

# # LAT_MIN = sys.argv[7]
# # LAT_MAX = sys.argv[8]
# # LAT_DELTA = sys.argv[9]

# # BUFFER = sys.argv[10:14]

# # YEAR = sys.argv[14]
# # MONTH = sys.argv[15]

# # print(sat_data_dir)
# # print(GC_data_dir)
# # print(output_dir)
# # print(LON_MIN)
# # print(LON_MAX)
# # print(LON_DELTA)
# # print(LAT_MIN)
# # print(LAT_MAX)
# # print(LAT_DELTA)
# # print(BUFFER)
# # print(YEAR)
# # print(MONTH)
