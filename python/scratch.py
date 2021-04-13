import xarray as xr
import sys
sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python')
import gcpy as gc

regularization_factor = 1
reduced_memory = True
available_memory_GB = 45

chunks = {'nstate' : -1, 'nobs' : None}
# First, define a dictionary of tuples with dimensions
dims = {'xa' : ['nstate'], 'sa' : ['nstate'],
        'y' : ['nobs'], 'ya' : ['nobs'], 'so' : ['nobs'],
        'k' : ['nobs', 'nstate']}

data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion/'

k = [f'{data_dir}k0_m{i:02d}.nc' for i in range(1, 13)]
xa = f'{data_dir}xa.nc'
sa_vec = f'{data_dir}sa.nc'
y = f'{data_dir}y.nc'
ya = f'{data_dir}ya.nc'
so_vec = f'{data_dir}so.nc'
c = f'{data_dir}c.nc'

def read(item, dims=None, chunks={}, **kwargs):
    # If item is a string or a list, load the file
    if type(item) in [str, list]:
        item = _load(item, dims=dims, chunks=chunks, **kwargs)

    # Force the items to be dataarrays
    if type(item) != xr.core.dataarray.DataArray:
        item = _to_dataarray(item, dims=dims, chunks=chunks)

    return item

def _load(item, dims, chunks, **kwargs):
    # This function is only called by read()
    # Add chunks to kwargs.
    kwargs['chunks'] = {k : chunks[k] for k in dims}
    if type(item) == list:
        # If it's a list, set some specific settings for open_mfdataset
        # This currently assumes that the coords are not properly set,
        # which was an oopsy when making the K0 datasets.
        kwargs['combine'] = 'nested'
        kwargs['concat_dim'] = dims
    else:
        item = [item]

    item = gc.read_file(*item, **kwargs)
    return item

def _to_dataarray(item, dims, chunks):
    # If it's a dataset, require that there be only one variable
    if type(item) == xr.core.dataset.Dataset:
        variables = list(item.keys())
        assert len(variables) == 1, \
              'A dataset with multiple variables was provided. This \
               class requires that any datasets contain only one \
               variable.'
        item = item[variables[0]]
    # Else convert to a dataarray
    else:
        assert dims is not None, \
               'Creating an xarray dataset and dims is not provided.'
        chunks = {k : chunks[k] for k in dims}
        item = xr.DataArray(item, dims=tuple(dims))
        item = item.chunk(chunks)
    return item

xa = read(xa, dims=dims['xa'], chunks=chunks)
sa_vec = read(sa_vec, dims=dims['sa'], chunks=chunks)
nstate = xa.shape[0]

max_chunk_size = gc.calculate_chunk_size(available_memory_GB)
chunks['nobs'] = int(max_chunk_size/nstate)

# Load observational data
k = read(k, dims=dims['k'], chunks=chunks)
y = read(y, dims=dims['y'], chunks=chunks)
ya = read(ya, dims=dims['ya'], chunks=chunks)
so_vec = read(so_vec, dims=dims['so'], chunks=chunks)
c = read(c, dims=dims['c'], chunks=chunks)
nobs = y.shape[0]

so_vec /= regularization_factor

# Now create some holding spaces for values that may be filled
# in the course of solving the inversion.
# xhat = None
# shat = None
# a = None
# yhat = None

# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# sys.path.append('.')
# import config
# config.SCALE = config.PRES_SCALE
# config.BASE_WIDTH = config.PRES_WIDTH
# config.BASE_HEIGHT = config.PRES_HEIGHT
# import gcpy as gc
# import format_plots as fp

# import pandas as pd
# pd.set_option('display.max_columns', 100)


# data = gc.load_obj('../inversion_data/2019_corrected.pkl')
# data['STD'] = data['SO']**0.5

# # bins = np.arange(0, 41)
# # data['STD_BIN'] = pd.cut(data['STD'], bins)

# # labels = np.arange(0.5, 40.5, 1)
# # bottom = np.zeros(len(labels))
# # Local preferences
# base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
# code_dir = base_dir + 'python'
# data_dir = base_dir + 'observations'
# output_dir = base_dir + 'inversion_data'
# plot_dir = base_dir + 'plots'

# fig, ax = fp.get_figax(aspect=1.75)
# ax.set_xlim(0, 25)
# ax = fp.add_labels(ax, 'Observational Error (ppb)', 'Count')
# ax = fp.add_title(ax, 'Observational Error')
# for i, season in enumerate(np.unique(data['SEASON'])):
#     hist_data = data[data['SEASON'] == season]['STD']
#     ax.hist(hist_data, histtype='step', bins=275, label=season,
#             color=fp.color(2+2*i), lw=1)
#     ax.axvline(hist_data.mean(), color=fp.color(2+2*i), lw=1, ls=':')
# ax = fp.add_legend(ax)
# fp.save_fig(fig, plot_dir, 'observational_error_seasonal')

# # LATITUDE
# # Update lat bins
# lat_bins = np.arange(10, 65, 10)
# data['LAT_BIN'] = pd.cut(data['LAT'], lat_bins)

# fig, ax = fp.get_figax(aspect=1.75)
# ax.set_xlim(0, 25)
# ax = fp.add_labels(ax, 'Observational Error (ppb)', 'Count')
# ax = fp.add_title(ax, 'Observational Error')
# for i, lat_bin in enumerate(np.unique(data['LAT_BIN'])):
#     hist_data = data[data['LAT_BIN'] == lat_bin]['STD']
#     ax.hist(hist_data, histtype='step', bins=275, label=lat_bin,
#             color=fp.color(2*i), lw=1)
#     ax.axvline(hist_data.mean(), color=fp.color(2*i), lw=1, ls=':')

# ax = fp.add_legend(ax)
# fp.save_fig(fig, plot_dir, 'observational_error_latitude_hist')

# fig, ax = fp.get_figax(aspect=1.75)
# ax.scatter(data['LAT'], data['STD'], c=fp.color(4), s=2, alpha=0.1)
# ax = fp.add_labels(ax, 'Latitude', 'Observational Error (ppb)')
# ax = fp.add_title(ax, 'Observational Error')
# fp.save_fig(fig, plot_dir, 'observational_error_latitude_scatter')

# # ALBEDO
# # Update albedo bins
# albedo_bins = np.arange(0, 1, 0.1)
# data['ALBEDO_BIN'] = pd.cut(data['ALBEDO_SWIR'], albedo_bins)

# fig, ax = fp.get_figax(aspect=1.75)
# ax.set_xlim(0, 25)
# ax = fp.add_labels(ax, 'Observational Error (ppb)', 'Count')
# ax = fp.add_title(ax, 'Observational Error')
# for i, alb_bin in enumerate(np.unique(data['ALBEDO_BIN'])):
#     hist_data = data[data['ALBEDO_BIN'] == alb_bin]['STD']
#     ax.hist(hist_data, histtype='step', bins=275, label=alb_bin,
#             color=fp.color(2*i), lw=1)
#     ax.axvline(hist_data.mean(), color=fp.color(2*i), lw=1, ls=':')

# ax = fp.add_legend(ax)
# fp.save_fig(fig, plot_dir, 'observational_error_albedo_hist')

# fig, ax = fp.get_figax(aspect=1.75)
# ax.scatter(data['ALBEDO_SWIR'], data['STD'], c=fp.color(4), s=2, alpha=0.1)
# ax = fp.add_labels(ax, 'Albedo', 'Observational Error (ppb)')
# ax = fp.add_title(ax, 'Observational Error')
# fp.save_fig(fig, plot_dir, 'observational_error_albedo_scatter')

#     ax.bar(labels, hist_data.values, bottom=bottom)
#     bottom += hist_data.values
# plt.show()

# # kn = open_k(k[0])

# def calculate_c():
#     '''
#     Calculate c for the forward model, defined as ybase = Kxa + c.
#     Save c as an element of the object.
#     '''
#     c = np.zeros(nobs)
#     i0 = 0
#     i1 = 0
#     for file_name in k:
#         kn = open_k(file_name)
#         i1 += kn.shape[0]
#         print(i0, i1)
#         c[i0:i1] = y_base[i0:i1] - np.matmul(kn, xa)
#         print('hello')
#         i0 = i1
#     return c

# c = calculate_c()

# o = 1
# t = 2
# s = 3
# print(f'{o: <10}{t: <15}{s: <15}')
# import xarray as xr
# import pandas as pd

# file = '/Users/hannahnesser/Downloads/s5p_l2_ch4_0014_15511.nc'
# d = xr.open_dataset(file, group='diagnostics')
# mask_qa = (d['qa_value'] <= 1)

# tmp = xr.open_dataset(file,
#                       group='instrument')
# dates = pd.DataFrame(tmp['time'].values[mask_qa][:,:-1],
#                      columns=['year', 'month', 'day', 'hour', 'minute', 'second'])
# dates = pd.to_datetime(dates).dt.strftime('%Y%m%dT%H%M%S')
# start_date = dates.min()
# end_date = dates.max()

# print(start_date)
# print(end_date)

# import matplotlib.pyplot as plt
# import matplotlib as mpl

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


# # print(data)

# # c = plt.scatter(data['OBS'], data['MOD'], c=data['DAY'], s=10, alpha=0.5)
# # plt.colorbar(c)
# # plt.ylim(1600, 2000)
# # plt.xlim(1600, 2000)
# # plt.show()

# Isolated early days--we'll use day 1 as a test case
# is it the restart file?
# files = [f for f in listdir(join(data_dir, 'gc_inputs/restarts'))
#          if f[-3:] == 'nc4']
# files.sort()
# for i, f in enumerate(files):
#     data = xr.open_dataset(join(data_dir, 'gc_inputs/restarts', f))
#     fig, ax = fp.get_figax()
#     try:
#         data = data.rename({'SpeciesRst_CH4' : 'CH4'})
#     except ValueError:
#         data = data.rename({'SPC_CH4' : 'CH4'})

#     data['CH4'].attrs['long_name'] = 'CH4'
#     data.plot.scatter(x='CH4', y='lev',
#                       ax=ax, c=fp.color(4), s=4, alpha=0.5)
#     fp.save_fig(fig, join(data_dir, 'plots'), f'restart{i:02d}.png')
# rst12 = xr.open_dataset(join(data_dir, 'GEOSChem.Restart.20191201_0000z.nc4'))
# rst11 = xr.open_dataset(join(data_dir, 'GEOSChem.Restart.20191101_0000z.nc4'))
# # for lev in rst11.lev:
# #     print(lev.values)
# #     print('Dec: ', rst12['SpeciesRst_CH4'].where(rst12.lev == lev, drop=True).min().values*1e9)
# #     print('Nov: ', rst11['SpeciesRst_CH4'].where(rst11.lev == lev, drop=True).min().values*1e9)
# #     print('\n')
# rst12.plot.scatter(x='SpeciesRst_CH4', y='lev')
# rst11.plot.scatter(x='SpeciesRst_CH4', y='lev')
# plt.show()


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
# print(mask_qa.shape)
# # # ========================================================================== ## Testing errors in eigenvector perturbations

# # # ========================================================================== ## Diagnosing error in prior simulation part 2

# # from os.path import join
# # from os import listdir
# # import sys
# # import copy
# # import calendar as cal

# # import xarray as xr
# # # import xbpch as xb
# # import numpy as np
# # import pandas as pd

# # import matplotlib.pyplot as plt

# # # data = xb.open_bpchdataset(filename='trac_avg.merra2_4x5_CH4.201501010000',
# # #                            tracerinfo_file='tracerinfo.dat',
# # #                            diaginfo_file='diaginfo.dat')

# # ## ------------------------------------------------------------------------ ##
# # ## Set user preferences
# # ## ------------------------------------------------------------------------ ##
# # # # Local preferences
# # # base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
# # # code_dir = base_dir + 'python'
# # # data_dir = base_dir + 'observations'
# # # output_dir = base_dir + 'inversion_data'
# # # plot_dir = base_dir + 'plots'

# # # # Cannon preferences
# # # base_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/'
# # # code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
# # # data_dir = f'{base_dir}ProcessedDir'
# # # output_dir = f'{base_dir}SummaryDir'
# # # plot_dir = None



# # # Information on the grid
# # lat_bins = np.arange(10, 65, 5)
# # lat_min = 9.75
# # lat_max = 60
# # lat_delta = 0.25
# # lon_min = -130
# # lon_max = -60
# # lon_delta = 0.3125
# # buffers = [3, 3, 3, 3]

# # ## ------------------------------------------------------------------------ ##
# # ## Import custom packages
# # ## ------------------------------------------------------------------------ ##
# # # Custom packages
# # sys.path.append('.')
# # import config
# # import gcpy as gc
# # import troppy as tp
# # import format_plots as fp

# # # data
# # year = 2019
# # month = 12
# # days = np.arange(1, 32, 1)
# # files = [f'{year}{month}{dd:02d}_GCtoTROPOMI.pkl' for dd in days]
# # data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/observations'

# # # data = np.array([]).reshape(0, 13)
# # # for f in files:
# # #     month = int(f[4:6])
# # #     day = int(f[6:8])
# # #     print(day)
# # #     new_data = gc.load_obj(join(data_dir, f))['obs_GC']
# # #     new_data = np.insert(new_data, 11, month, axis=1)
# # #     new_data = np.insert(new_data, 12, day, axis=1)
# # #     data = np.concatenate((data, new_data))

# # # columns = ['OBS', 'MOD', 'LON', 'LAT', 'iGC', 'jGC', 'PREC',
# # #            'ALBEDO_SWIR', 'ALBEDO_NIR', 'AOD', 'MOD_COL',
# # #            'MONTH', 'DAY']
# # # data = pd.DataFrame(data, columns=columns)
# # # data['DIFF'] = data['MOD'] - data['OBS']

# # # print(data)

# # # c = plt.scatter(data['OBS'], data['MOD'], c=data['DAY'], s=10, alpha=0.5)
# # # plt.colorbar(c)
# # # plt.ylim(1600, 2000)
# # # plt.xlim(1600, 2000)
# # # plt.show()

# # # Isolated early days--we'll use day 1 as a test case
# # # is it the restart file?
# # # rst12 = xr.open_dataset(join(data_dir, 'GEOSChem.Restart.20191201_0000z.nc4'))
# # # rst11 = xr.open_dataset(join(data_dir, 'GEOSChem.Restart.20191101_0000z.nc4'))
# # # # for lev in rst11.lev:
# # # #     print(lev.values)
# # # #     print('Dec: ', rst12['SpeciesRst_CH4'].where(rst12.lev == lev, drop=True).min().values*1e9)
# # # #     print('Nov: ', rst11['SpeciesRst_CH4'].where(rst11.lev == lev, drop=True).min().values*1e9)
# # # #     print('\n')
# # # rst12.plot.scatter(x='SpeciesRst_CH4', y='lev')
# # # rst11.plot.scatter(x='SpeciesRst_CH4', y='lev')
# # # plt.show()


# # # Probably not because the December restart file is actually larger than
# # # the November restart file.

# # # Where in the atmosphere does it originate?
# # files = [f'GEOSChem.SpeciesConc.{year}{month}{dd:02d}_0000z.nc4' for dd in days]
# # # for f in files[0]:
# # f = files[15]
# # data = xr.open_dataset(join(data_dir, f))
# # data = data.isel(time=1)
# # data.plot.scatter(x='SpeciesConc_CH4', y='lev')
# # # c = plt.scatter(data.lev, data['SpeciesConc_CH4'], c=data.time)
# # # plt.colorbar(c)
# # plt.show()


# # # ========================================================================== ## Diagnosing error in prior simulation

# # import xarray as xr
# # from os import listdir
# # from os.path import join

# # data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/prior/species_conc'

# # # List files
# # files = listdir(data_dir)
# # files.sort()

# # # Open files
# # # for f in files:
# # sc1 = xr.open_dataset(join(data_dir, files[4]))['SpeciesConc_CH4']*1e9
# # sc2 = xr.open_dataset(join(data_dir, files[7]))['SpeciesConc_CH4']*1e9

# # print((sc1.values == sc2.values).sum())
# # print(sc1.shape)
# # # print(sc1.values.shape)
# # # print(sc2.values.shape)

# # # # Check the first level
# # # d_lev = sc.where(sc.lev == sc.lev[0], drop=True).squeeze()
# # # for t in d_lev.time:
# # #     print(d_lev.where(d_lev.time == t, drop=True).max())

# # # ========================================================================== ## Adding stratospheric analysis to TROPOMI operator

# # VCH4_apriori = TROPOMI['methane_profile_apriori']/TROPOMI
# # TROPOMI['dry_air_subcolumns']


# # # ========================================================================== ## Diagnosing error in TROPOMI operator

# # import numpy as np
# # import xarray as xr
# # import re
# # import pickle
# # import os
# # import sys
# # import pandas as pd
# # import datetime
# # import copy
# # import glob

# # sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python')
# # import GC_to_TROPOMI as oper

# # sat_data_dir = "/n/seasasfs02/hnesser/TROPOMI/downloads_14_14/"
# # GC_data_dir = "/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/OutputDir/"
# # output_dir = "/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/ProcessedDir/"

# # LON_MIN = -130
# # LON_MAX = -60
# # LON_DELTA = 0.3125
# # LAT_MIN = 9.75
# # LAT_MAX = 60
# # LAT_DELTA = 0.25
# # BUFFER = [3, 3, 3, 3] # [N S E W]

# # YEAR = 2019
# # MONTH = 1
# # DAY = 1

# # ## ---------------------------------------------------------------------##
# # ## Remove buffer boxes
# # ## ---------------------------------------------------------------------##
# # LAT_MAX -= LAT_DELTA*BUFFER[0]
# # LAT_MIN += LAT_DELTA*BUFFER[1]
# # LON_MAX -= LON_DELTA*BUFFER[2]
# # LON_MIN += LON_DELTA*BUFFER[3]

# # ## ---------------------------------------------------------------------##
# # ## List all satellite files for the year and date defined
# # ## ---------------------------------------------------------------------##
# # # List all raw netcdf TROPOMI files
# # allfiles=glob.glob(sat_data_dir+'*.nc')
# # allfiles.sort()

# # # Create empty list
# # Sat_files = {}

# # # Iterate through the raw TROPOMI data
# # for index in range(len(allfiles)):
# #     filename = allfiles[index]

# #     # Get the date (YYYY, MM, and DD) of the raw TROPOMI file
# #     shortname = re.split('\/|\.', filename)[-2]
# #     strdate = re.split('_+|T', shortname)
# #     start_date = strdate[4]
# #     end_date = strdate[6]

# #     # start condition
# #     start = ((int(start_date[:4]) == YEAR)
# #              and (int(start_date[4:6]) == MONTH))
# #     end = ((int(end_date[:4]) == YEAR)
# #            and (int(end_date[4:6]) == MONTH))

# #     # Skip observations not in range
# #     # if not (year and month):
# #     if not (start or end):
# #         continue

# #     # Add the file to the list of Sat_files
# #     if start:
# #         if start_date in Sat_files.keys():
# #             Sat_files[start_date].append(filename)
# #         else:
# #             Sat_files[start_date] = [filename]
# #     elif end:
# #         if end_date in Sat_files.keys():
# #             Sat_files[end_date].append(filename)
# #         else:
# #             Sat_files[end_date] = [filename]

# # print('Number of dates: ', len(Sat_files))

# # ## -------------------------------------------------------------------------##
# # ## Take a closer look at the specified date
# # ## -------------------------------------------------------------------------##
# # date = '%04d%02d%02d' % (YEAR, MONTH, DAY)
# # filenames = Sat_files[date]
# # print('=========== %s ===========' % date)
# # preprocess = lambda d: oper.filter_tropomi(d, date,
# #                                            LON_MIN, LON_MAX, LON_DELTA,
# #                                            LAT_MIN, LAT_MAX, LAT_DELTA)
# # TROPOMI = xr.open_mfdataset(filenames, concat_dim='nobs',
# #                             combine='nested',
# #                             chunks=10000,
# #                             preprocess=preprocess)
# # TROPOMI = oper.process_tropomi(TROPOMI, date)

# # if TROPOMI is None:
# #     print('No observations remain.')
# #     print('================================')

# # # Get observation dimension (number of good observations in that single
# # # observation file)
# # NN = TROPOMI.nobs.shape[0]
# # print('Processing %d Observations' % NN)
# # print('================================')

# # # create an empty matrix to store TROPOMI CH4, GC CH4,
# # # lon, lat, II, and JJ (GC indices)
# # temp_obs_GC=np.zeros([NN, 11],dtype=np.float32)

# # #================================
# # #--- now compute sensitivity ---
# # #================================

# # # Then, read in the GC data for these dates. This works by
# # # reading the lon, lat, pressure edge, xch4, xch4_adjusted
# # # (which I believe is the stratospheric corrected data), TROPP
# # # (which is the planetary boundary layer info), and dry air.
# # GC = oper.read_GC(GC_data_dir, date)

# # # Find the grid box and time indices corresponding to TROPOMI obs
# # iGC, jGC, tGC = oper.nearest_loc(GC, TROPOMI)

# # # Then select GC accordingly
# # GC_P = GC['PEDGE'].values[tGC, iGC, jGC, :]
# # GC_DA = GC['DRYAIR'].values[tGC, iGC, jGC, :]
# # GC_CH4 = GC['CH4'].values[tGC, iGC, jGC, :]
# # GC_COL = GC['GCCOL'].values[tGC, iGC, jGC]

# # # Create mapping between GC and TROPOMI pressure levels
# # intmap = oper.get_intmap(TROPOMI['pressures'].values, GC_P)
# # newmap = oper.get_newmap(intmap, TROPOMI['pressures'].values, GC_P,
# #                          GC_CH4, GC_DA)

# # # Finally, apply the averaging kernel
# # GC_base_post2 = oper.apply_avker(TROPOMI['column_AK'].values,
# #                                 TROPOMI['methane_profile_apriori'].values,
# #                                 TROPOMI['dry_air_subcolumns'].values,
# #                                 newmap['GC_CH4'], newmap['GC_WEIGHT'])
# # # GC_base_pri = apply_avker(np.ones(TROPOMI['column_AK'].shape),
# # #                           TROPOMI['methane_profile_apriori'].values,
# # #                           TROPOMI['dry_air_subcolumns'].values,
# # #                           newmap['GC_CH4'], newmap['GC_WEIGHT'])

# # GC_base_strat = apply_avker(TROPOMI['column_AK'].values,
# #                             TROPOMI['methane_profile_apriori'].values,
# #                             TROPOMI['dry_air_subcolumns'].values,
# #                             newmap['GC_CH4'], newmap['GC_WEIGHT'],
# #                             filt=(TROPOMI['pressures_mid'] < 200))

# # # Save out values
# # # The columns are: OBS, MOD, LON, LAT, iGC, jGC, PRECISION,
# # # ALBEDO_SWIR, ALBEDO_NIR, AOD, MOD_COL
# # temp_obs_GC[:, 0] = TROPOMI['methane']
# # temp_obs_GC[:, 1] = GC_base_post
# # temp_obs_GC[:, 2] = TROPOMI['longitude']
# # temp_obs_GC[:, 3] = TROPOMI['latitude']
# # temp_obs_GC[:, 4] = iGC
# # temp_obs_GC[:, 5] = jGC
# # temp_obs_GC[:, 6] = TROPOMI['precision']
# # temp_obs_GC[:, 7] = TROPOMI['albedo'][:,1]
# # temp_obs_GC[:, 8] = TROPOMI['albedo'][:,0]
# # temp_obs_GC[:, 9] = TROPOMI['aerosol_optical_depth'][:,1]
# # temp_obs_GC[:, 10] = GC_COL

# # result={}
# # result['obs_GC'] = temp_obs_GC





# # # ========================================================================== #
# # # Original settings for TROPOMI operator
# # # import sys

# # # sat_data_dir = sys.argv[1]
# # # GC_data_dir = sys.argv[2]
# # # output_dir = sys.argv[3]

# # # LON_MIN = sys.argv[4]
# # # LON_MAX = sys.argv[5]
# # # LON_DELTA = sys.argv[6]

# # # LAT_MIN = sys.argv[7]
# # # LAT_MAX = sys.argv[8]
# # # LAT_DELTA = sys.argv[9]

# # # BUFFER = sys.argv[10:14]

# # # YEAR = sys.argv[14]
# # # MONTH = sys.argv[15]

# # # print(sat_data_dir)
# # # print(GC_data_dir)
# # # print(output_dir)
# # # print(LON_MIN)
# # # print(LON_MAX)
# # # print(LON_DELTA)
# # # print(LAT_MIN)
# # # print(LAT_MAX)
# # # print(LAT_DELTA)
# # # print(BUFFER)
# # # print(YEAR)
# # # print(MONTH)
