'''
This is a package designed to deal with common GEOS-Chem
processing needs.
'''
import numpy as np
import pandas as pd
import xarray as xr

def adjust_grid_bounds(lat_min, lat_max, lat_delta,
                       lon_min, lon_max, lon_delta,
                       buffer=[0, 0, 0, 0]):
    '''
    This function adjusts the default GEOS-Chem grid bounds,
    which are given as grid box centers, to grid box edges.
    It also allows for the option to remove buffer grid cells,
    although the default is to assume 0 buffer grid cells.
    (Buffer grid cells should be expressed in the standard
    GEOS-Chem convention of [N S E W].)
    '''
    lat_min = lat_min + lat_delta*buffer[1] - lat_delta/2
    lat_max = lat_max - lat_delta*buffer[0] + lat_delta/2
    lon_min = lon_min + lon_delta*buffer[3] - lon_delta/2
    lon_max = lon_max - lon_delta*buffer[2] + lon_delta/2
    return [lat_min, lat_max], [lon_min, lon_max]

def create_gc_grid(lat_min, lat_max, lat_delta,
                   lon_min, lon_max, lon_delta,
                   centers=True, return_xarray=True):
    '''
    This function creates a grid with values corresponding to the
    centers of each grid cell. The latitude and longitude limits
    provided correspond to grid cell centers if centers=True and
    edges otherwise.
    '''
    if not centers:
        lat_min += lat_delta/2
        lat_max -= lat_delta/2
        lon_min += lon_delta/2
        lon_max -= lon_delta/2

    lats = np.arange(lat_min, lat_max + lat_delta, lat_delta)
    lons = np.arange(lon_min, lon_max + lon_delta, lon_delta)

    if return_xarray:
        data = xr.DataArray(np.zeros((len(lats), len(lons))),
                            coords=[lats, lons],
                            dims=['lats', 'lons'])
    else:
        data = [lats, lons]

    return data

def fill_GC_first_hour(data):
    data0 = data.where(data.time == data.time[0], drop=True)

    # Change the time of that data to the 0th hour
    t = pd.Timestamp(data0.time.values[0]).replace(hour=0).to_datetime64()
    data0.time.values[0] = t

    # Merge the datasets
    data = xr.concat([data0, data], dim='time')

    return data

def nearest_loc(data, compare_data):
    return np.abs(compare_data[:, None] - data[None, :]).argmin(axis=0)
