
##########################
# Set default file paths
PLOT_DIR = '../plots/'
##########################
import imageio
from os.path import join

files = [f'blended_albedo_filter_{m:02d}_BAF.png' for m in range(1, 13)]
files.sort()
print(files)

images = []
for f in files:
    images.append(imageio.imread(join(PLOT_DIR, f)))
imageio.mimsave(join(PLOT_DIR, 'BAF.gif'), images,
                **{'duration' : 1})


# # This is a package I wrote to deal with my specific problems. I don't
# # use it for anything other than reading in the files, and I don't
# # use it in the simplest example below.
# import sys
# sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python')
# import inversion as inv

# # Define the files that we need to read in.
# # Prior files
# xa_file  = f'{data_dir}xa.nc' # n x 1
# sa_file  = f'{data_dir}sa.nc' # n x 1

# # Observing system files
# y_file   = f'{data_dir}y.nc' # m x 1
# ya_file  = f'{data_dir}ya.nc' # m x 1
# so_file  = f'{data_dir}so.nc' # m x 1
# c_file   = f'{data_dir}c.nc' # m x 1
# k_file   = [f'{data_dir}k0_m{i:02d}.nc' for i in range(1, 13)] # m x n

# # We set the chunk size for nstate (equal to nstate or -1). Then we read
# # in the prior files, which gives us an integer value for nstate. Given the memory of our session, we can then infer a reasonable chunk size for nobs.
# xa = xr.open_dataarray(xa_file, chunks=-1)
# sa = xr.open_dataarray(sa_file, chunks=-1)
# nstate = xa.shape[0]

# # Now calculate the optimal chunk size in the other (nobs) dimension.
# # First we guess the number of chunks that will be held in memory per
# # the dask documentation.
# chunks_in_memory = 10*int(os.environ['OMP_NUM_THREADS'])

# # Then we calculate the number of GB per chunk
# GB_per_chunk = 45/chunks_in_memory

# # And then convert that to the number of float32 elements
# number_of_elements = int(GB_per_chunk*1e9/4)

# # Finally, calculate the chunk size for the nobs dimension and
# # save out chunk information.
# nstate_chunk = -1
# nobs_chunk = int(number_of_elements/nstate)
# chunks = {'nstate' : nstate_chunk, 'nobs' : nobs_chunk}

# # Now read in the observing system
# y = xr.open_dataarray(y_file, chunks=chunks['nobs'])
# ya = xr.open_dataarray(ya_file, chunks=chunks['nobs'])
# so = xr.open_dataarray(so_file, chunks=chunks['nobs'])
# c = xr.open_dataarray(c_file, chunks=chunks['nobs'])
# k = xr.open_mfdataset([[f] for f in k_file], chunks=chunks, combine='nested',
#                       concat_dim=['nobs', 'nstate'])

# nobs = y.shape[0]

# # Having read in the data, we aim to calculate shat, the posterior error
# # covariance matrix:
# # shat = (sa^-1 + k^T so^-1 k)^-1
# # where sa and so are diagonal matrices with diagonal values given by the
# # n x 1 and m x 1 vectors we loaded previouusly, respectively.

# ##############################################################################
# ##### TEST 1                                                             #####
# ##############################################################################
# # We begin by trying to calculate the n x n matrix (k^T so^-1 k) and we
# # take advantage of the fact that so is a diagonal matrix.
# shat = da.einsum('ij,jk', k.T, k/so)

# # This is a 23,691 x 23,691 matrix. It should require 2.24 GB. We attempt
# # to load it into memory.
# shat = client.persist(shat)

# # Now we check for progress
# progress(shat)

# # And this is when it fails.
# client.cancel(shat)
# client.restart()
# del(shat)

# ##############################################################################
# ##### TEST 2                                                             #####
# ##############################################################################
# # We begin by trying to save k/so
# ksoinv = k/so
# delayed = ksoinv.to_netcdf(f'{data_dir}ksoinv.nc', compute=False)
# with ProgressBar():
#     delayed.compute()

# # This fails in pretty much the same way.
# client.cancel(delayed)
# client.restart()
# del(ksoinv)


# # from dask_jobqueue import SLURMCluster
# # cluster = SLURMCluster(queue='huce_intel', cores=12, memory='45GB')
# # cluster.scale(jobs=2)

# # To do this run
# # xterm & // conda activate TROPOMI_inversion // dask-scheduler
# # xterm & // conda activate TROPOMI_inversion // dask-worker localhost:8786 --nprocs 6 --nthreads 2

# import os
# os.environ['OMP_NUM_THREADS'] = '6'

# from dask.distributed import Client, LocalCluster, progress
# cluster = LocalCluster(local_directory='.')
# client = Client(cluster)

# import xarray as xr
# import dask.array as da
# from dask.diagnostics import ProgressBar
# import sys
# sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python')
# import inversion as inv

data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion/'

k   = [f'{data_dir}k0_m{i:02d}.nc' for i in range(1, 13)]
xa  = f'{data_dir}xa.nc'
sa  = f'{data_dir}sa.nc'
y   = f'{data_dir}y.nc'
ya  = f'{data_dir}ya.nc'
so  = f'{data_dir}so.nc'
c   = f'{data_dir}c.nc'

data = inv.Inversion(k, xa, sa, y, ya, so, c,
                     regularization_factor=1, reduced_memory=True,
                     available_memory_GB=45, k_is_positive=True)

# # data.so = data.so.drop('nobs')
# data.sa = data.sa.rename('sa')

# sa = data.sa.compute()
# so = data.so.compute()

# # nstate_chunk = -1
# # nobs_chunk = 5.3e3

# sa_sqrt = data.sa**0.5
# # pph_temp = (data.sa**0.5)*data.k.T

# # k = data.k.chunk((1e4, -1))
# # pph_temp = pph_temp.chunk((nstate_chunk, nobs_chunk))
# # data.so = data.so.chunk(nobs_chunk)
# pph = da.einsum('ij,jk', sa_sqrt*k.T, k*sa_sqrt/so)
# # pph = pph.rechunk((1e4, 1e3))

# pph = client.persist(pph)
# progress(pph)

# client.cancel(pph)
# client.restart()
# del(pph)


# k01 = xr.open_dataarray(k[0], chunks={'nobs' : 7914, 'nstate' : -1})
# so01 = xr.open_dataarray(so)[:227721]
# sa01 = xr.open_dataarray(sa)
# pph_temp = (sa01**0.5)*k01.T
# # pph_temp = pph_temp.chunk((nstate_chunk, nobs_chunk))
# # so = so.chunk(nobs_chunk)
# pph = da.einsum('ij,jk', pph_temp, pph_temp.T/so01)


# # with ProgressBar():
# #     pph = pph.compute()

# pph = xr.DataArray(pph, dims=('nstate', 'nstate'), name='pph')


# # from os.path import join


#     # pph.to_netcdf(join(data_dir, 'pph0.nc'))


# # import xarray as xr
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import sys
# # sys.path.append('.')
# # import config
# # config.SCALE = config.PRES_SCALE
# # config.BASE_WIDTH = config.PRES_WIDTH
# # config.BASE_HEIGHT = config.PRES_HEIGHT
# # import gcpy as gc
# # import format_plots as fp

# # import pandas as pd
# # pd.set_option('display.max_columns', 100)


# # data = gc.load_obj('../inversion_data/2019_corrected.pkl')
# # data['STD'] = data['SO']**0.5

# # # bins = np.arange(0, 41)
# # # data['STD_BIN'] = pd.cut(data['STD'], bins)

# # # labels = np.arange(0.5, 40.5, 1)
# # # bottom = np.zeros(len(labels))
# # # Local preferences
# # base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
# # code_dir = base_dir + 'python'
# # data_dir = base_dir + 'observations'
# # output_dir = base_dir + 'inversion_data'
# # plot_dir = base_dir + 'plots'

# # fig, ax = fp.get_figax(aspect=1.75)
# # ax.set_xlim(0, 25)
# # ax = fp.add_labels(ax, 'Observational Error (ppb)', 'Count')
# # ax = fp.add_title(ax, 'Observational Error')
# # for i, season in enumerate(np.unique(data['SEASON'])):
# #     hist_data = data[data['SEASON'] == season]['STD']
# #     ax.hist(hist_data, histtype='step', bins=275, label=season,
# #             color=fp.color(2+2*i), lw=1)
# #     ax.axvline(hist_data.mean(), color=fp.color(2+2*i), lw=1, ls=':')
# # ax = fp.add_legend(ax)
# # fp.save_fig(fig, plot_dir, 'observational_error_seasonal')

# # # LATITUDE
# # # Update lat bins
# # lat_bins = np.arange(10, 65, 10)
# # data['LAT_BIN'] = pd.cut(data['LAT'], lat_bins)

# # fig, ax = fp.get_figax(aspect=1.75)
# # ax.set_xlim(0, 25)
# # ax = fp.add_labels(ax, 'Observational Error (ppb)', 'Count')
# # ax = fp.add_title(ax, 'Observational Error')
# # for i, lat_bin in enumerate(np.unique(data['LAT_BIN'])):
# #     hist_data = data[data['LAT_BIN'] == lat_bin]['STD']
# #     ax.hist(hist_data, histtype='step', bins=275, label=lat_bin,
# #             color=fp.color(2*i), lw=1)
# #     ax.axvline(hist_data.mean(), color=fp.color(2*i), lw=1, ls=':')

# # ax = fp.add_legend(ax)
# # fp.save_fig(fig, plot_dir, 'observational_error_latitude_hist')

# # fig, ax = fp.get_figax(aspect=1.75)
# # ax.scatter(data['LAT'], data['STD'], c=fp.color(4), s=2, alpha=0.1)
# # ax = fp.add_labels(ax, 'Latitude', 'Observational Error (ppb)')
# # ax = fp.add_title(ax, 'Observational Error')
# # fp.save_fig(fig, plot_dir, 'observational_error_latitude_scatter')

# # # ALBEDO
# # # Update albedo bins
# # albedo_bins = np.arange(0, 1, 0.1)
# # data['ALBEDO_BIN'] = pd.cut(data['ALBEDO_SWIR'], albedo_bins)

# # fig, ax = fp.get_figax(aspect=1.75)
# # ax.set_xlim(0, 25)
# # ax = fp.add_labels(ax, 'Observational Error (ppb)', 'Count')
# # ax = fp.add_title(ax, 'Observational Error')
# # for i, alb_bin in enumerate(np.unique(data['ALBEDO_BIN'])):
# #     hist_data = data[data['ALBEDO_BIN'] == alb_bin]['STD']
# #     ax.hist(hist_data, histtype='step', bins=275, label=alb_bin,
# #             color=fp.color(2*i), lw=1)
# #     ax.axvline(hist_data.mean(), color=fp.color(2*i), lw=1, ls=':')

# # ax = fp.add_legend(ax)
# # fp.save_fig(fig, plot_dir, 'observational_error_albedo_hist')

# # fig, ax = fp.get_figax(aspect=1.75)
# # ax.scatter(data['ALBEDO_SWIR'], data['STD'], c=fp.color(4), s=2, alpha=0.1)
# # ax = fp.add_labels(ax, 'Albedo', 'Observational Error (ppb)')
# # ax = fp.add_title(ax, 'Observational Error')
# # fp.save_fig(fig, plot_dir, 'observational_error_albedo_scatter')

# #     ax.bar(labels, hist_data.values, bottom=bottom)
# #     bottom += hist_data.values
# # plt.show()

# # # kn = open_k(k[0])

# # def calculate_c():
# #     '''
# #     Calculate c for the forward model, defined as ybase = Kxa + c.
# #     Save c as an element of the object.
# #     '''
# #     c = np.zeros(nobs)
# #     i0 = 0
# #     i1 = 0
# #     for file_name in k:
# #         kn = open_k(file_name)
# #         i1 += kn.shape[0]
# #         print(i0, i1)
# #         c[i0:i1] = y_base[i0:i1] - np.matmul(kn, xa)
# #         print('hello')
# #         i0 = i1
# #     return c

# # c = calculate_c()

# # o = 1
# # t = 2
# # s = 3
# # print(f'{o: <10}{t: <15}{s: <15}')
# # import xarray as xr
# # import pandas as pd

# # file = '/Users/hannahnesser/Downloads/s5p_l2_ch4_0014_15511.nc'
# # d = xr.open_dataset(file, group='diagnostics')
# # mask_qa = (d['qa_value'] <= 1)

# # tmp = xr.open_dataset(file,
# #                       group='instrument')
# # dates = pd.DataFrame(tmp['time'].values[mask_qa][:,:-1],
# #                      columns=['year', 'month', 'day', 'hour', 'minute', 'second'])
# # dates = pd.to_datetime(dates).dt.strftime('%Y%m%dT%H%M%S')
# # start_date = dates.min()
# # end_date = dates.max()

# # print(start_date)
# # print(end_date)

# # import matplotlib.pyplot as plt
# # import matplotlib as mpl

# # data = xb.open_bpchdataset(filename='trac_avg.merra2_4x5_CH4.201501010000',
# #                            tracerinfo_file='tracerinfo.dat',
# #                            diaginfo_file='diaginfo.dat')

# ## ------------------------------------------------------------------------ ##
# ## Set user preferences
# ## ------------------------------------------------------------------------ ##
# # # Local preferences
# # base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
# # code_dir = base_dir + 'python'
# # data_dir = base_dir + 'observations'
# # output_dir = base_dir + 'inversion_data'
# # plot_dir = base_dir + 'plots'

# # # Cannon preferences
# # base_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/'
# # code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
# # data_dir = f'{base_dir}ProcessedDir'
# # output_dir = f'{base_dir}SummaryDir'
# # plot_dir = None


# # # print(data)

# # # c = plt.scatter(data['OBS'], data['MOD'], c=data['DAY'], s=10, alpha=0.5)
# # # plt.colorbar(c)
# # # plt.ylim(1600, 2000)
# # # plt.xlim(1600, 2000)
# # # plt.show()

# # Isolated early days--we'll use day 1 as a test case
# # is it the restart file?
# # files = [f for f in listdir(join(data_dir, 'gc_inputs/restarts'))
# #          if f[-3:] == 'nc4']
# # files.sort()
# # for i, f in enumerate(files):
# #     data = xr.open_dataset(join(data_dir, 'gc_inputs/restarts', f))
# #     fig, ax = fp.get_figax()
# #     try:
# #         data = data.rename({'SpeciesRst_CH4' : 'CH4'})
# #     except ValueError:
# #         data = data.rename({'SPC_CH4' : 'CH4'})

# #     data['CH4'].attrs['long_name'] = 'CH4'
# #     data.plot.scatter(x='CH4', y='lev',
# #                       ax=ax, c=fp.color(4), s=4, alpha=0.5)
# #     fp.save_fig(fig, join(data_dir, 'plots'), f'restart{i:02d}.png')
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
# # print(mask_qa.shape)
# # # # ========================================================================== ## Testing errors in eigenvector perturbations

# # # # ========================================================================== ## Diagnosing error in prior simulation part 2

# # # from os.path import join
# # # from os import listdir
# # # import sys
# # # import copy
# # # import calendar as cal

# # # import xarray as xr
# # # # import xbpch as xb
# # # import numpy as np
# # # import pandas as pd

# # # import matplotlib.pyplot as plt

# # # # data = xb.open_bpchdataset(filename='trac_avg.merra2_4x5_CH4.201501010000',
# # # #                            tracerinfo_file='tracerinfo.dat',
# # # #                            diaginfo_file='diaginfo.dat')

# # # ## ------------------------------------------------------------------------ ##
# # # ## Set user preferences
# # # ## ------------------------------------------------------------------------ ##
# # # # # Local preferences
# # # # base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
# # # # code_dir = base_dir + 'python'
# # # # data_dir = base_dir + 'observations'
# # # # output_dir = base_dir + 'inversion_data'
# # # # plot_dir = base_dir + 'plots'

# # # # # Cannon preferences
# # # # base_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/'
# # # # code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
# # # # data_dir = f'{base_dir}ProcessedDir'
# # # # output_dir = f'{base_dir}SummaryDir'
# # # # plot_dir = None



# # # # Information on the grid
# # # lat_bins = np.arange(10, 65, 5)
# # # lat_min = 9.75
# # # lat_max = 60
# # # lat_delta = 0.25
# # # lon_min = -130
# # # lon_max = -60
# # # lon_delta = 0.3125
# # # buffers = [3, 3, 3, 3]

# # # ## ------------------------------------------------------------------------ ##
# # # ## Import custom packages
# # # ## ------------------------------------------------------------------------ ##
# # # # Custom packages
# # # sys.path.append('.')
# # # import config
# # # import gcpy as gc
# # # import troppy as tp
# # # import format_plots as fp

# # # # data
# # # year = 2019
# # # month = 12
# # # days = np.arange(1, 32, 1)
# # # files = [f'{year}{month}{dd:02d}_GCtoTROPOMI.pkl' for dd in days]
# # # data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/observations'

# # # # data = np.array([]).reshape(0, 13)
# # # # for f in files:
# # # #     month = int(f[4:6])
# # # #     day = int(f[6:8])
# # # #     print(day)
# # # #     new_data = gc.load_obj(join(data_dir, f))['obs_GC']
# # # #     new_data = np.insert(new_data, 11, month, axis=1)
# # # #     new_data = np.insert(new_data, 12, day, axis=1)
# # # #     data = np.concatenate((data, new_data))

# # # # columns = ['OBS', 'MOD', 'LON', 'LAT', 'iGC', 'jGC', 'PREC',
# # # #            'ALBEDO_SWIR', 'ALBEDO_NIR', 'AOD', 'MOD_COL',
# # # #            'MONTH', 'DAY']
# # # # data = pd.DataFrame(data, columns=columns)
# # # # data['DIFF'] = data['MOD'] - data['OBS']

# # # # print(data)

# # # # c = plt.scatter(data['OBS'], data['MOD'], c=data['DAY'], s=10, alpha=0.5)
# # # # plt.colorbar(c)
# # # # plt.ylim(1600, 2000)
# # # # plt.xlim(1600, 2000)
# # # # plt.show()

# # # # Isolated early days--we'll use day 1 as a test case
# # # # is it the restart file?
# # # # rst12 = xr.open_dataset(join(data_dir, 'GEOSChem.Restart.20191201_0000z.nc4'))
# # # # rst11 = xr.open_dataset(join(data_dir, 'GEOSChem.Restart.20191101_0000z.nc4'))
# # # # # for lev in rst11.lev:
# # # # #     print(lev.values)
# # # # #     print('Dec: ', rst12['SpeciesRst_CH4'].where(rst12.lev == lev, drop=True).min().values*1e9)
# # # # #     print('Nov: ', rst11['SpeciesRst_CH4'].where(rst11.lev == lev, drop=True).min().values*1e9)
# # # # #     print('\n')
# # # # rst12.plot.scatter(x='SpeciesRst_CH4', y='lev')
# # # # rst11.plot.scatter(x='SpeciesRst_CH4', y='lev')
# # # # plt.show()


# # # # Probably not because the December restart file is actually larger than
# # # # the November restart file.

# # # # Where in the atmosphere does it originate?
# # # files = [f'GEOSChem.SpeciesConc.{year}{month}{dd:02d}_0000z.nc4' for dd in days]
# # # # for f in files[0]:
# # # f = files[15]
# # # data = xr.open_dataset(join(data_dir, f))
# # # data = data.isel(time=1)
# # # data.plot.scatter(x='SpeciesConc_CH4', y='lev')
# # # # c = plt.scatter(data.lev, data['SpeciesConc_CH4'], c=data.time)
# # # # plt.colorbar(c)
# # # plt.show()


# # # # ========================================================================== ## Diagnosing error in prior simulation

# # # import xarray as xr
# # # from os import listdir
# # # from os.path import join

# # # data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/prior/species_conc'

# # # # List files
# # # files = listdir(data_dir)
# # # files.sort()

# # # # Open files
# # # # for f in files:
# # # sc1 = xr.open_dataset(join(data_dir, files[4]))['SpeciesConc_CH4']*1e9
# # # sc2 = xr.open_dataset(join(data_dir, files[7]))['SpeciesConc_CH4']*1e9

# # # print((sc1.values == sc2.values).sum())
# # # print(sc1.shape)
# # # # print(sc1.values.shape)
# # # # print(sc2.values.shape)

# # # # # Check the first level
# # # # d_lev = sc.where(sc.lev == sc.lev[0], drop=True).squeeze()
# # # # for t in d_lev.time:
# # # #     print(d_lev.where(d_lev.time == t, drop=True).max())

# # # # ========================================================================== ## Adding stratospheric analysis to TROPOMI operator

# # # VCH4_apriori = TROPOMI['methane_profile_apriori']/TROPOMI
# # # TROPOMI['dry_air_subcolumns']


# # # # ========================================================================== ## Diagnosing error in TROPOMI operator

# # # import numpy as np
# # # import xarray as xr
# # # import re
# # # import pickle
# # # import os
# # # import sys
# # # import pandas as pd
# # # import datetime
# # # import copy
# # # import glob

# # # sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python')
# # # import GC_to_TROPOMI as oper

# # # sat_data_dir = "/n/seasasfs02/hnesser/TROPOMI/downloads_14_14/"
# # # GC_data_dir = "/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/OutputDir/"
# # # output_dir = "/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000/ProcessedDir/"

# # # LON_MIN = -130
# # # LON_MAX = -60
# # # LON_DELTA = 0.3125
# # # LAT_MIN = 9.75
# # # LAT_MAX = 60
# # # LAT_DELTA = 0.25
# # # BUFFER = [3, 3, 3, 3] # [N S E W]

# # # YEAR = 2019
# # # MONTH = 1
# # # DAY = 1

# # # ## ---------------------------------------------------------------------##
# # # ## Remove buffer boxes
# # # ## ---------------------------------------------------------------------##
# # # LAT_MAX -= LAT_DELTA*BUFFER[0]
# # # LAT_MIN += LAT_DELTA*BUFFER[1]
# # # LON_MAX -= LON_DELTA*BUFFER[2]
# # # LON_MIN += LON_DELTA*BUFFER[3]

# # # ## ---------------------------------------------------------------------##
# # # ## List all satellite files for the year and date defined
# # # ## ---------------------------------------------------------------------##
# # # # List all raw netcdf TROPOMI files
# # # allfiles=glob.glob(sat_data_dir+'*.nc')
# # # allfiles.sort()

# # # # Create empty list
# # # Sat_files = {}

# # # # Iterate through the raw TROPOMI data
# # # for index in range(len(allfiles)):
# # #     filename = allfiles[index]

# # #     # Get the date (YYYY, MM, and DD) of the raw TROPOMI file
# # #     shortname = re.split('\/|\.', filename)[-2]
# # #     strdate = re.split('_+|T', shortname)
# # #     start_date = strdate[4]
# # #     end_date = strdate[6]

# # #     # start condition
# # #     start = ((int(start_date[:4]) == YEAR)
# # #              and (int(start_date[4:6]) == MONTH))
# # #     end = ((int(end_date[:4]) == YEAR)
# # #            and (int(end_date[4:6]) == MONTH))

# # #     # Skip observations not in range
# # #     # if not (year and month):
# # #     if not (start or end):
# # #         continue

# # #     # Add the file to the list of Sat_files
# # #     if start:
# # #         if start_date in Sat_files.keys():
# # #             Sat_files[start_date].append(filename)
# # #         else:
# # #             Sat_files[start_date] = [filename]
# # #     elif end:
# # #         if end_date in Sat_files.keys():
# # #             Sat_files[end_date].append(filename)
# # #         else:
# # #             Sat_files[end_date] = [filename]

# # # print('Number of dates: ', len(Sat_files))

# # # ## -------------------------------------------------------------------------##
# # # ## Take a closer look at the specified date
# # # ## -------------------------------------------------------------------------##
# # # date = '%04d%02d%02d' % (YEAR, MONTH, DAY)
# # # filenames = Sat_files[date]
# # # print('=========== %s ===========' % date)
# # # preprocess = lambda d: oper.filter_tropomi(d, date,
# # #                                            LON_MIN, LON_MAX, LON_DELTA,
# # #                                            LAT_MIN, LAT_MAX, LAT_DELTA)
# # # TROPOMI = xr.open_mfdataset(filenames, concat_dim='nobs',
# # #                             combine='nested',
# # #                             chunks=10000,
# # #                             preprocess=preprocess)
# # # TROPOMI = oper.process_tropomi(TROPOMI, date)

# # # if TROPOMI is None:
# # #     print('No observations remain.')
# # #     print('================================')

# # # # Get observation dimension (number of good observations in that single
# # # # observation file)
# # # NN = TROPOMI.nobs.shape[0]
# # # print('Processing %d Observations' % NN)
# # # print('================================')

# # # # create an empty matrix to store TROPOMI CH4, GC CH4,
# # # # lon, lat, II, and JJ (GC indices)
# # # temp_obs_GC=np.zeros([NN, 11],dtype=np.float32)

# # # #================================
# # # #--- now compute sensitivity ---
# # # #================================

# # # # Then, read in the GC data for these dates. This works by
# # # # reading the lon, lat, pressure edge, xch4, xch4_adjusted
# # # # (which I believe is the stratospheric corrected data), TROPP
# # # # (which is the planetary boundary layer info), and dry air.
# # # GC = oper.read_GC(GC_data_dir, date)

# # # # Find the grid box and time indices corresponding to TROPOMI obs
# # # iGC, jGC, tGC = oper.nearest_loc(GC, TROPOMI)

# # # # Then select GC accordingly
# # # GC_P = GC['PEDGE'].values[tGC, iGC, jGC, :]
# # # GC_DA = GC['DRYAIR'].values[tGC, iGC, jGC, :]
# # # GC_CH4 = GC['CH4'].values[tGC, iGC, jGC, :]
# # # GC_COL = GC['GCCOL'].values[tGC, iGC, jGC]

# # # # Create mapping between GC and TROPOMI pressure levels
# # # intmap = oper.get_intmap(TROPOMI['pressures'].values, GC_P)
# # # newmap = oper.get_newmap(intmap, TROPOMI['pressures'].values, GC_P,
# # #                          GC_CH4, GC_DA)

# # # # Finally, apply the averaging kernel
# # # GC_base_post2 = oper.apply_avker(TROPOMI['column_AK'].values,
# # #                                 TROPOMI['methane_profile_apriori'].values,
# # #                                 TROPOMI['dry_air_subcolumns'].values,
# # #                                 newmap['GC_CH4'], newmap['GC_WEIGHT'])
# # # # GC_base_pri = apply_avker(np.ones(TROPOMI['column_AK'].shape),
# # # #                           TROPOMI['methane_profile_apriori'].values,
# # # #                           TROPOMI['dry_air_subcolumns'].values,
# # # #                           newmap['GC_CH4'], newmap['GC_WEIGHT'])

# # # GC_base_strat = apply_avker(TROPOMI['column_AK'].values,
# # #                             TROPOMI['methane_profile_apriori'].values,
# # #                             TROPOMI['dry_air_subcolumns'].values,
# # #                             newmap['GC_CH4'], newmap['GC_WEIGHT'],
# # #                             filt=(TROPOMI['pressures_mid'] < 200))

# # # # Save out values
# # # # The columns are: OBS, MOD, LON, LAT, iGC, jGC, PRECISION,
# # # # ALBEDO_SWIR, ALBEDO_NIR, AOD, MOD_COL
# # # temp_obs_GC[:, 0] = TROPOMI['methane']
# # # temp_obs_GC[:, 1] = GC_base_post
# # # temp_obs_GC[:, 2] = TROPOMI['longitude']
# # # temp_obs_GC[:, 3] = TROPOMI['latitude']
# # # temp_obs_GC[:, 4] = iGC
# # # temp_obs_GC[:, 5] = jGC
# # # temp_obs_GC[:, 6] = TROPOMI['precision']
# # # temp_obs_GC[:, 7] = TROPOMI['albedo'][:,1]
# # # temp_obs_GC[:, 8] = TROPOMI['albedo'][:,0]
# # # temp_obs_GC[:, 9] = TROPOMI['aerosol_optical_depth'][:,1]
# # # temp_obs_GC[:, 10] = GC_COL

# # # result={}
# # # result['obs_GC'] = temp_obs_GC





# # # # ========================================================================== #
# # # # Original settings for TROPOMI operator
# # # # import sys

# # # # sat_data_dir = sys.argv[1]
# # # # GC_data_dir = sys.argv[2]
# # # # output_dir = sys.argv[3]

# # # # LON_MIN = sys.argv[4]
# # # # LON_MAX = sys.argv[5]
# # # # LON_DELTA = sys.argv[6]

# # # # LAT_MIN = sys.argv[7]
# # # # LAT_MAX = sys.argv[8]
# # # # LAT_DELTA = sys.argv[9]

# # # # BUFFER = sys.argv[10:14]

# # # # YEAR = sys.argv[14]
# # # # MONTH = sys.argv[15]

# # # # print(sat_data_dir)
# # # # print(GC_data_dir)
# # # # print(output_dir)
# # # # print(LON_MIN)
# # # # print(LON_MAX)
# # # # print(LON_DELTA)
# # # # print(LAT_MIN)
# # # # print(LAT_MAX)
# # # # print(LAT_DELTA)
# # # # print(BUFFER)
# # # # print(YEAR)
# # # # print(MONTH)
