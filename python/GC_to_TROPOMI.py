#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import numpy as np
import xarray as xr
import re
import pickle
import os
import pandas as pd
import datetime
import copy
#import scipy.integrate as integrate
#from scipy.integrate import quad

LON_MIN = -140
LON_MAX = -40
LON_DELTA = 0.625
LAT_MIN = 10
LAT_MAX = 70
LAT_DELTA = 0.5

#----- define function -------
def save_obj(obj, name):
    with open(name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)

def read_tropomi(data, date, lon_min, lon_max, lon_delta,
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

    # Do other processing
    # Add date variable
    dates = pd.DataFrame(data['time'].values[:, :-1],
                         columns=['year', 'month', 'day',
                                  'hour', 'minute', 'second'])
    dates = xr.DataArray(pd.to_datetime(dates),
                         dims=['nobs']).reset_index('nobs', drop=True)
    data = data.assign(utctime=dates)

    # Albedo and AOD have two columns [NIR, SWIR]. We select SWIR.
    data = data.where(data.nwin == 1, drop=True).squeeze()

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
    data = data.transpose('nobs', 'nlayer', 'ilayer')

    return data

def get_diagnostic(diag_name, date):
    short_date = date[:8]
    #hour = int(date[-2:])
    filename = os.path.join(GC_datadir,
                            'GEOSChem.'+diag_name+'.'+short_date+'.0000z.nc4')
    data = xr.open_dataset(filename)
    #data = data.where(data.time.dt.hour == hour, drop=True).squeeze()
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

# def newmap2(intmap,lgos, GC_p, Sat_p,gc_sens,dryair):
#     gc_ch4 = np.zeros((lgos-1,1009))
#     count = 0e0
#     for ll in range(lgos-1):
#       temp_gc = 0e0
#       temp_dry = 0e0
#       for l in range(len(GC_p)-1):
#         temp_gc += abs(intmap[l,ll]) * gc_sens[l,:] * dryair[l]
#         temp_dry += abs(intmap[l,ll]) * dryair[l]
#         count += abs(intmap[l,ll]) * dryair[l]
#       gc_ch4[ll,:] = temp_gc / temp_dry
#     met={}
#     met['Sens']=gc_ch4

#     return met

# def remap2(Sensi, data_type, Com_p, location, first_2):
#     MM=Sensi.shape[1]
#     conc=np.zeros((len(Com_p)-1,MM))
#     conc.fill(np.nan)
#     k=0
#     for i in range(first_2,len(Com_p)-1):
#         conc[i,:]=Sensi[k,:]
#         if data_type[i+1]==2:
#             k=k+1
#     if first_2>0:
#         conc[:first_2,:]=conc[first_2,:]

#     Sat_CH4=np.zeros((12,MM));Sat_CH4.fill(np.nan)

#     delta_p=Com_p[:-1]-Com_p[1:]
#     delta_ps=np.transpose(np.tile(delta_p,(MM,1)))
#     for i in range(len(location)-1):
#         start=location[i]
#         end=location[i+1]
#         fenzi=np.sum(conc[start:end,:]*delta_ps[start:end,:],0)
#         fenmu=np.sum(delta_p[start:end])
#         Sat_CH4[i,:]=fenzi/fenmu

#     return Sat_CH4

#==============================================================================
#===========================Define functions ==================================
#==============================================================================
#Sat_datadir="/n/seasasfs02/hnesser/TROPOMI/downloads_201910/"
#Sat_datadir="/n/holyscratch01/jacob_lab/mwinter/newTROPOMI/"
Sat_datadir="/n/seasasfs02/hnesser/TROPOMI/downloads_14_14/"
GC_datadir="/n/holyscratch01/jacob_lab/mwinter/Nested_NA/run_dirs/Hannah_NA_0000/OutputDir_2018/"
outputdir="/net/seasasfs02/srv/export/seasasfs02/share_root/mwinter/TROPOMI_processed/data_2018/"
biasdir="/net/seasasfs02/srv/export/seasasfs02/share_root/mwinter/TROPOMI_processed/bias/"
#Sensi_datadir="/n/holyscratch01/jacob_lab/zhenqu/aggregate/data/"

# #==== read lat_ratio ===
# df=pd.read_csv("./lat_ratio.csv",index_col=0)
# lat_mid=df.index
# lat_ratio=df.values

#==== read Satellite ===

# List all raw netcdf TROPOMI files
allfiles=glob.glob(Sat_datadir+'*.nc')
allfiles.sort()

# Create empty list
Sat_files = {}
#Sat_dates = []

# Iterate through the raw TROPOMI data
for index in range(len(allfiles)):
    filename = allfiles[index]

    # Get the date (YYYY, MM, and DD) of the raw TROPOMI file
    shortname = re.split('\/|\.', filename)[-2]
    strdate = re.split('_+|T', shortname)
    start_date = strdate[4]
    end_date = strdate[6]
    year = (int(start_date[:4]) == 2018)
    month = ((int(start_date[4:6]) == 5)
              or (int(end_date[4:6]) == 5))

    # Skip observations not in range
    if not (year and month):
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

# Create an array that corresponds to the state vector
b = np.zeros((46,72))
bcount = np.zeros((46,72))

# Iterate throught the Sat_files we created
for date, filenames in Sat_files.items():
    print('========================')
    process = lambda d: read_tropomi(d, date,
                                     LON_MIN, LON_MAX, LON_DELTA,
                                     LAT_MIN, LAT_MAX, LAT_DELTA)
    TROPOMI = xr.open_mfdataset(filenames, concat_dim='nobs',
                                combine='nested',
                                preprocess=process)

    # If already processed, skip the rest of the processing
    # within this loop
    # if os.path.isfile(outputdir+date+'_GCtoTROPOMI.pkl'):
    #     continue

    # Get observation dimension (number of good observations in that single
    # observation file)
    NN = TROPOMI.nobs.shape[0]
    print('========================')
    print('========================')
    print('Processing %d Observations' % NN)
    if NN == 0:
        continue

    # state vector dimension (we will need to change this)
    # MM=1009

    # create an empty matrix for the Jacobian
    # temp_KK=np.zeros([NN,MM],dtype=np.float32)#Store the K

    # create an empty matrix to store TROPOMI CH4, GC CH4,
    # lon, lat, II, and JJ (GC indices)
    temp_obs_GC=np.zeros([NN, 10],dtype=np.float32)

    #================================
    #--- now compute sensitivity ---
    #================================

    # Then, read in the GC data for these dates. This works by
    # reading the lon, lat, pressure edge, xch4, xch4_adjusted
    # (which I believe is the stratospheric corrected data), TROPP
    # (which is the planetary boundary layer info), and dry air.
    GC = read_GC(date)

    # Find the grid box and time indices corresponding to TROPOMI obs
    iGC = np.abs(GC.lon.values.reshape((-1, 1))
                 - TROPOMI['longitude'].values.reshape((1, -1)))
    iGC = iGC.argmin(axis=0)
    jGC = np.abs(GC.lat.values.reshape((-1, 1))
                 - TROPOMI['latitude'].values.reshape((1, -1)))
    jGC = jGC.argmin(axis=0)
    tGC = np.where(TROPOMI['utctime'].dt.hour == GC.time.dt.hour)[1]

    # # Then select GC accordingly
    GC_p = GC['PEDGE'].values[tGC, iGC, jGC, :]
    dryair = GC['DRYAIR'].values[tGC, iGC, jGC, :]
    GC_CH4 = GC['CH4'].values[tGC, iGC, jGC, :]
    GC_COL = GC['GCCOL'].values[tGC, iGC, jGC]

    # Create mapping between GC and TROPOMI pressure levels
    intmap = get_intmap(TROPOMI['pressures'].values, GC_p)
    newmap = get_newmap(intmap, TROPOMI['pressures'].values, GC_p,
                        GC_CH4, dryair)
    Sat_CH4 = newmap['GC_CH4']
    GC_WEIGHT = newmap['GC_WEIGHT']

    # Finally, apply the averaging kernel
    def apply_avker(avker, prior, dryair, sat_ch4, gc_weight):
        rat = prior / dryair * 1e9
        temp = (gc_weight * (rat + avker * (sat_ch4 - rat))).sum(axis=1)
        return temp

    GC_base_posteri = apply_avker(TROPOMI['column_AK'].values,
                                  TROPOMI['methane_profile_apriori'].values,
                                  TROPOMI['dry_air_subcolumns'].values,
                                  Sat_CH4, GC_WEIGHT)
    GC_base_pri = apply_avker(np.ones(TROPOMI['column_AK'].shape),
                              TROPOMI['methane_profile_apriori'].values,
                              TROPOMI['dry_air_subcolumns'].values,
                              Sat_CH4, GC_WEIGHT)

        #Sensi=GC['Sensi'][iGC,jGC,:,:]
        #temp = newmap2(intmap, len(Sat_p), GC_p, Sat_p, Sensi, dryair)
        #Sens = temp['Sens']
        #print(Sens.shape)
        #temp_gcsens = np.zeros(1009)
        #for ll in range(12):
        #    temp_gcsens[:] += GC_WEIGHT[ll]*AK[ll]*Sens[ll,:]

            # perturbation = temp_gcsens-temp_gc
            # pert[iGC,jGC,isens] += (temp_gcsens-temp_gc)/0.5 # for grid aggregate
        #pert[iNN,:] = temp_gcsens/0.5 # for observation individual

        #print('GC_pos', GC_base_posteri)
        #print('GC_pri', GC_base_pri)

    temp_obs_GC[:, 0] = TROPOMI['methane']
    temp_obs_GC[:, 1] = GC_base_posteri
    temp_obs_GC[:, 2] = TROPOMI['longitude']
    temp_obs_GC[:, 3] = TROPOMI['latitude']
    temp_obs_GC[:, 4] = iGC
    temp_obs_GC[:, 5] = jGC
    temp_obs_GC[:, 6] = TROPOMI['precision']
    temp_obs_GC[:, 7] = TROPOMI['albedo']
    temp_obs_GC[:, 8] = TROPOMI['aerosol_optical_depth']
    temp_obs_GC[:, 9] = GC_COL

        # Total error (unclear why not abs) in each grid cell
        #b[jGC, iGC] += GC_base_posteri - TROPOMI['methane'][iSat,jSat]

        # Number of observations in each grid cell
        #bcount[jGC, iGC] += 1

    result={}
    result['obs_GC'] = temp_obs_GC
    # result['KK']=pert


    save_obj(result, outputdir + date + '_GCtoTROPOMI.pkl')
#b[bcount>0] = b[bcount>0]/bcount[bcount>0]
#save_obj(b,biasdir+'1.pkl')
print('CODE FINISHED')
