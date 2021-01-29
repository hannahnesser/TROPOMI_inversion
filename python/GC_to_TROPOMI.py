#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## -------------------------------------------------------------------------##
## Load packages and set environment defaults
## -------------------------------------------------------------------------##
import glob
import numpy as np
import xarray as xr
import re
import pickle
import os
import sys
import pandas as pd
import datetime
import copy

# ## -------------------------------------------------------------------------##
# ## Set user preferences
# ## -------------------------------------------------------------------------##
# sat_data_dir = sys.argv[1]
# GC_data_dir = sys.argv[2]
# output_dir = sys.argv[3]

# LON_MIN = float(sys.argv[4])
# LON_MAX = float(sys.argv[5])
# LON_DELTA = float(sys.argv[6])

# LAT_MIN = float(sys.argv[7])
# LAT_MAX = float(sys.argv[8])
# LAT_DELTA = float(sys.argv[9])

# BUFFER = sys.argv[10:14]
# BUFFER = [int(b) for b in BUFFER]

# YEAR = int(sys.argv[14])
# MONTH = int(sys.argv[15])

# ## -------------------------------------------------------------------------##
# ## Remove buffer boxes
# ## -------------------------------------------------------------------------##
# LAT_MAX -= LAT_DELTA*BUFFER[0]
# LAT_MIN += LAT_DELTA*BUFFER[1]
# LON_MAX -= LON_DELTA*BUFFER[2]
# LON_MIN += LON_DELTA*BUFFER[3]

## -------------------------------------------------------------------------##
## Define functions
## -------------------------------------------------------------------------##
def save_obj(obj, name):
    with open(name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)

def filter_tropomi(data, date, lon_min, lon_max, lon_delta,
                   lat_min, lat_max, lat_delta):
    # Filter on qa_value
    data = data.where(data['qa_value'] > 0.5, drop=True)

    # Filter on masked methane
    data = data.where(data['xch4_corrected'] != 9.96921e36, drop=True)

    # Filter on lat/lon domain
    data = data.where((data['longitude_center'] >= lon_min-lon_delta/2) &
                      (data['longitude_center'] <= lon_max+lon_delta/2),
                      drop=True)
    data = data.where((data['latitude_center'] >= lat_min-lat_delta/2) &
                      (data['latitude_center'] <= lat_max+lat_delta/2),
                      drop=True)

    # Filter on dates
    data = data.where(((data['time'][:, 0] == int(date[:4]))
                       & (data['time'][:, 1] == int(date[4:6]))
                       & (data['time'][:, 2] == int(date[6:]))), drop=True)

    return data

def process_tropomi(data, date):
    # Do other processing
    # Add date variable
    dates = pd.DataFrame(data['time'].values[:, :-1],
                         columns=['year', 'month', 'day',
                                  'hour', 'minute', 'second'])
    dates = xr.DataArray(pd.to_datetime(dates),
                         dims=['nobs']).reset_index('nobs', drop=True)
    data = data.assign(utctime=dates)

    # Albedo and AOD have two columns [NIR, SWIR]. We select SWIR.
    # Both of these are needed for the albedo filter
    #data = data.where(data.nwin == 1, drop=True).squeeze()

    # Correct units from molecules/cm2 to mol/m2
    data['ch4_profile_apriori'] *= 1e4/6.02214e23
    data['dry_air_subcolumns'] *= 1e4/6.02214e23

    # Flip vertical direction
    data = data.sortby('nlayer', ascending=False)

    # Pressure information (hPa)
    pressure_interval = data['dp'].values.reshape((-1, 1))
    surface_pressure = data['surface_pressure'].values.reshape((-1, 1))

    # Create a pressures array corresponding to vertical levels
    # HN 2020/09/09 - converted from for loop to numpy
    z = data.nlayer.shape[0]
    pressures = (surface_pressure
                 - np.arange(z + 1).reshape((1, -1))*pressure_interval)
    pressures = xr.DataArray(pressures, dims=['nobs', 'ilayer'])
    pressures_mid = (surface_pressure
                     - (np.arange(z).reshape((1, -1)) + 0.5)*pressure_interval)
    pressures_mid = xr.DataArray(pressures_mid, dims=['nobs', 'nlayer'])
    data = data.assign(pressures=pressures, pressures_mid=pressures_mid)

    # Remove irrelevant variables
    data = data.drop(labels=['latitude_corners', 'longitude_corners',
                             'glintflag', 'altitude_levels',
                             'surface_altitude', 'time'])

    # Rename variables
    data = data.rename({'xch4_corrected' : 'methane',
                        'longitude_center' : 'longitude',
                        'latitude_center' : 'latitude',
                        'xch4_precision' : 'precision',
                        'surface_albedo' : 'albedo',
                        'aerosol_optical_thickness' : 'aerosol_optical_depth',
                        'xch4_column_averaging_kernel' : 'column_AK',
                        'ch4_profile_apriori' : 'methane_profile_apriori'})

    # Transpose
    data = data.transpose('nobs', 'nwin', 'nlayer', 'ilayer')

    return data

def get_diagnostic(diag_name, date):
    short_date = date[:8]
    filename = os.path.join(GC_data_dir,
                            'GEOSChem.'+diag_name+'.'+short_date+'_0000z.nc4')
    data = xr.open_dataset(filename)
    return data

def read_GC(date):
    # Start by downloading methane data (ppb)
    data = get_diagnostic('SpeciesConc', date)[['SpeciesConc_CH4']]*1e9

    # Now get the other variables
    met = get_diagnostic('StateMet', date)
    met = met[['Met_PBLH', 'Met_AIRDEN', 'Met_BXHEIGHT', 'Met_AD']]
    met = met.assign(DRYAIR=met['Met_AIRDEN']*met['Met_BXHEIGHT'])
    data = xr.merge([data, met])
    met.close()

    # Calculate the GC column
    GCCOL = ((data['SpeciesConc_CH4']*data['Met_AD']).sum(dim='lev')
             /data['Met_AD'].sum(dim='lev'))
    data = data.assign(GCCOL=GCCOL)

    # Remove superfluous variables
    data = data.drop(['Met_AIRDEN', 'Met_BXHEIGHT', 'Met_AD'])

    # Get pressure information (hPa)
    pres = get_diagnostic('LevelEdgeDiags', date)[['Met_PEDGE']]
    data = xr.merge([data, pres])
    pres.close()

    # Rename variables
    data = data.rename({'SpeciesConc_CH4' : 'CH4',
                        'Met_PBLH' : 'TROPP',
                        'Met_PEDGE' : 'PEDGE'})

    # Flip order
    data = data.transpose('time', 'lon', 'lat', 'lev', 'ilev')

    return data

# quzhen 2020/2/13
def get_intmap(Sat_p, GC_p):
    '''I think this is equivalent to the cal_weights function, at least
    a little bit.'''
    nobs = Sat_p.shape[0]
    ngc = GC_p.shape[1] - 1
    ntrop = Sat_p.shape[1]

    intmap = np.zeros((nobs, ngc, ntrop))
    count = np.zeros(nobs)

    # Start by considering the cases where the satellite pressures
    # are out of the GC bounds.

    # Case 1: Satellite pressures are higher than GC surface pressure
    idx = np.greater(Sat_p, GC_p[:, 0].reshape(-1, 1))
    count = idx.sum(axis=1)

    # set the satelllite column corresponding to count - 1
    # equal to 1. (i.e. if count == 1, set the 0th ntrop dimension
    # equal to 1 for that observation)
    idx = ((count - 1).reshape(-1, 1) == np.arange(ntrop))[:, None, :]
    idx = np.tile(idx, (1, ngc, 1))
    intmap[idx] = 1

    # Now set the GC row up to and including the count-1 satellite
    # column equal to 1/count
    idx = np.zeros((nobs, ngc, ntrop), dtype=bool)
    idx[:, 0, :] = ((count - 1).reshape(-1, 1) >= np.arange(ntrop))
    count = np.tile(count[:, None, None], (1, ngc, ntrop))
    intmap[idx] = 1/count[idx]

    # Case 2: Satellite pressures are lower than top level GC pressure
    idx = np.zeros((nobs, ngc, ntrop), dtype=bool)
    idx[:, -1, :] = np.less(Sat_p, GC_p[:, -1].reshape(-1, 1))
    intmap[idx] = 1

    # Case 3: Satellite pressures are between the top and bottom GC pressure
    # Criteria 1:
    idx1 = (~np.greater_equal(Sat_p, GC_p[:, 0].reshape(-1, 1)) &
            ~np.less_equal(Sat_p, GC_p[:, -1].reshape(-1, 1)))[:, None, :]
    idx1 = np.tile(idx1, (1, ngc, 1))

    lo = GC_p[:, 1:]
    hi = GC_p[:, :-1]
    diff = hi - lo

    # Criteria 2: satellite pressure is between the geos-chem pressure levels
    idx2 = (np.less_equal(Sat_p[:, None, :], hi[:, :, None]) &
            np.greater(Sat_p[:, None, :], lo[:, :, None]))

    # Combine the criteria
    idx = idx1 & idx2

    # Create an array for GC indices
    idx_gc = np.tile(np.arange(1, ngc+1)[None, :, None], (nobs, 1, ntrop))

    # In all circumstances:
    idx_gc_count = copy.deepcopy(idx_gc)
    idx_gc_count[~idx] = nobs + ngc + ntrop
    idx_gc_count = np.min(idx_gc_count, axis=1)
    idx_gen_f = (idx_gc > idx_gc_count[:, None, :])
    intmap[idx_gen_f] = 1

    # If also the lowest satellite level, set all GC indices up to
    # the maximum true G index equal to 1
    idx_losat = np.zeros((nobs, ngc, ntrop), dtype=bool)
    idx_losat[:, :, 0] = True
    idx_gc_count = copy.deepcopy(idx_gc)
    idx_gc_count[~(idx & idx_losat)] = 0
    idx_gc_count = np.max(idx_gc_count, axis=1)
    idx_losat_f = (idx_gc <= idx_gc_count[:, None, :])
    intmap[idx_losat_f] = 1

    # If not the lowest satellite level
    # Change 1
    iobs, igc, itrop = np.where(idx & ~idx_losat)
    intmap[iobs, igc, itrop-1] = ((hi[iobs, igc] - Sat_p[iobs, itrop])
                                  /diff[iobs, igc])
    intmap[iobs, igc, itrop] = ((Sat_p[iobs, itrop] - lo[iobs, igc])
                                /diff[iobs, igc])

    # Change 2
    idx_nohigc = np.roll((idx & ~idx_losat), -1, axis=2)
    idx_gc_count = copy.deepcopy(idx_gc)
    idx_gc_count[~idx_nohigc] = nobs + ngc + ntrop # sufficiently big
    idx_gc_count = np.min(idx_gc_count, axis=1)
    idx_nohigc_f = (idx_gc > idx_gc_count[:, None, :])
    intmap[idx_nohigc_f] = 0

    return intmap

def get_newmap(intmap, Sat_p, GC_p, gc_ch4_native, dryair):
    nobs, ngc, ntrop = intmap.shape
    # gc_ch4 = np.zeros((nobs, ntrop - 1))
    # gc_weight = np.zeros((nobs, ntrop - 1))

    temp_gc = (intmap * gc_ch4_native[:, :, None] * dryair[:, :, None])
    temp_gc = temp_gc.sum(axis=1)[:, :-1]
    temp_dry = (intmap * dryair[:, :, None]).sum(axis=1)[:, :-1]

    gc_ch4 = temp_gc / temp_dry
    gc_weight = temp_dry / dryair.sum(axis=1)[:, None]

    met = {}
    met['GC_CH4'] = gc_ch4
    met['GC_WEIGHT'] = gc_weight

    return met

def apply_avker(avker, prior, dryair, sat_ch4, gc_weight):
    rat = prior / dryair * 1e9
    temp = (gc_weight * (rat + avker * (sat_ch4 - rat))).sum(axis=1)
    return temp

def nearest_loc(GC, TROPOMI):
    # Find the grid box and time indices corresponding to TROPOMI obs
    # i index
    iGC = np.abs(GC.lon.values.reshape((-1, 1))
                 - TROPOMI['longitude'].values.reshape((1, -1)))
    iGC = iGC.argmin(axis=0)

    # j index
    jGC = np.abs(GC.lat.values.reshape((-1, 1))
                 - TROPOMI['latitude'].values.reshape((1, -1)))
    jGC = jGC.argmin(axis=0)

    # Time index
    tGC = np.where(TROPOMI['utctime'].dt.hour == GC.time.dt.hour)[1]

    return iGC, jGC, tGC

## -------------------------------------------------------------------------##
## TROPOMI operator
## -------------------------------------------------------------------------##
if __name__ == '__main__':
    ## ---------------------------------------------------------------------##
    ## Remove buffer boxes
    ## ---------------------------------------------------------------------##
    # Decrease threads
    os.environ['OMP_NUM_THREADS'] = '2'

    ## ---------------------------------------------------------------------##
    ## Read in user preferences
    ## ---------------------------------------------------------------------##
    sat_data_dir = sys.argv[1]
    GC_data_dir = sys.argv[2]
    output_dir = sys.argv[3]

    LON_MIN = float(sys.argv[4])
    LON_MAX = float(sys.argv[5])
    LON_DELTA = float(sys.argv[6])

    LAT_MIN = float(sys.argv[7])
    LAT_MAX = float(sys.argv[8])
    LAT_DELTA = float(sys.argv[9])

    BUFFER = sys.argv[10:14]
    BUFFER = [int(b) for b in BUFFER]

    YEAR = int(sys.argv[14])
    MONTH = int(sys.argv[15])

    ## ---------------------------------------------------------------------##
    ## Remove buffer boxes
    ## ---------------------------------------------------------------------##
    LAT_MAX -= LAT_DELTA*BUFFER[0]
    LAT_MIN += LAT_DELTA*BUFFER[1]
    LON_MAX -= LON_DELTA*BUFFER[2]
    LON_MIN += LON_DELTA*BUFFER[3]

    ## ---------------------------------------------------------------------##
    ## List all satellite files for the year and date defined
    ## ---------------------------------------------------------------------##
    # List all raw netcdf TROPOMI files
    allfiles=glob.glob(sat_data_dir+'*.nc')
    allfiles.sort()

    # Create empty list
    Sat_files = {}

    # Iterate through the raw TROPOMI data
    for index in range(len(allfiles)):
        filename = allfiles[index]

        # Get the date (YYYY, MM, and DD) of the raw TROPOMI file
        shortname = re.split('\/|\.', filename)[-2]
        strdate = re.split('_+|T', shortname)
        start_date = strdate[4]
        end_date = strdate[6]

        # start condition
        start = ((int(start_date[:4]) == YEAR)
                 and (int(start_date[4:6]) == MONTH))
        end = ((int(end_date[:4]) == YEAR)
               and (int(end_date[4:6]) == MONTH))
        # year = (int(start_date[:4]) == YEAR)
        # month = ((int(start_date[4:6]) == MONTH)
        #           or (int(end_date[4:6]) == MONTH))

        # Skip observations not in range
        # if not (year and month):
        if not (start or end):
            continue

        # Add the file to the list of Sat_files
        if start_date in Sat_files.keys():
            Sat_files[start_date].append(filename)
        else:
            Sat_files[start_date] = [filename]

        if start_date != end_date:
            if end_date in Sat_files.keys():
                Sat_files[end_date].append(filename)
            else:
                Sat_files[end_date] = [filename]

    print('Number of dates: ', len(Sat_files))

    ## -------------------------------------------------------------------------##
    ## Iterate throught the Sat_files we created
    ## -------------------------------------------------------------------------##
    for date, filenames in Sat_files.items():
        print('=========== %s ===========' % date)
        preprocess = lambda d: filter_tropomi(d, date,
                                              LON_MIN, LON_MAX, LON_DELTA,
                                              LAT_MIN, LAT_MAX, LAT_DELTA)
        TROPOMI = xr.open_mfdataset(filenames, concat_dim='nobs',
                                    combine='nested',
                                    chunks=10000,
                                    preprocess=preprocess)
        TROPOMI = process_tropomi(TROPOMI, date)

        if TROPOMI is None:
            print('No observations remain.')
            print('================================')
            continue

        # Get observation dimension (number of good observations in that single
        # observation file)
        NN = TROPOMI.nobs.shape[0]
        print('Processing %d Observations' % NN)
        print('================================')

        # create an empty matrix to store TROPOMI CH4, GC CH4,
        # lon, lat, II, and JJ (GC indices)
        temp_obs_GC=np.zeros([NN, 11],dtype=np.float32)

        #================================
        #--- now compute sensitivity ---
        #================================

        # Then, read in the GC data for these dates. This works by
        # reading the lon, lat, pressure edge, xch4, xch4_adjusted
        # (which I believe is the stratospheric corrected data), TROPP
        # (which is the planetary boundary layer info), and dry air.
        GC = read_GC(date)

        # Find the grid box and time indices corresponding to TROPOMI obs
        iGC, jGC, tGC = nearest_loc(GC, TROPOMI)

        # Then select GC accordingly
        GC_P = GC['PEDGE'].values[tGC, iGC, jGC, :]
        GC_DA = GC['DRYAIR'].values[tGC, iGC, jGC, :]
        GC_CH4 = GC['CH4'].values[tGC, iGC, jGC, :]
        GC_COL = GC['GCCOL'].values[tGC, iGC, jGC]

        # Create mapping between GC and TROPOMI pressure levels
        intmap = get_intmap(TROPOMI['pressures'].values, GC_P)
        newmap = get_newmap(intmap, TROPOMI['pressures'].values, GC_P,
                            GC_CH4, GC_DA)

        # Finally, apply the averaging kernel
        GC_base_posteri = apply_avker(TROPOMI['column_AK'].values,
                                      TROPOMI['methane_profile_apriori'].values,
                                      TROPOMI['dry_air_subcolumns'].values,
                                      newmap['GC_CH4'], newmap['GC_WEIGHT'])
        GC_base_pri = apply_avker(np.ones(TROPOMI['column_AK'].shape),
                                  TROPOMI['methane_profile_apriori'].values,
                                  TROPOMI['dry_air_subcolumns'].values,
                                  newmap['GC_CH4'], newmap['GC_WEIGHT'])

        # Save out values
        temp_obs_GC[:, 0] = TROPOMI['methane']
        temp_obs_GC[:, 1] = GC_base_posteri
        temp_obs_GC[:, 2] = TROPOMI['longitude']
        temp_obs_GC[:, 3] = TROPOMI['latitude']
        temp_obs_GC[:, 4] = iGC
        temp_obs_GC[:, 5] = jGC
        temp_obs_GC[:, 6] = TROPOMI['precision']
        temp_obs_GC[:, 7] = TROPOMI['albedo'][:,1]
        temp_obs_GC[:, 8] = TROPOMI['albedo'][:,0]
        temp_obs_GC[:, 9] = TROPOMI['aerosol_optical_depth'][:,1]
        temp_obs_GC[:, 10] = GC_COL

        result={}
        result['obs_GC'] = temp_obs_GC

        save_obj(result, output_dir + date + '_GCtoTROPOMI.pkl')

    print('CODE FINISHED')
