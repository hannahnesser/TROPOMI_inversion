import xarray as xr
import xesmf as xe
import numpy as np
import sys
sys.path.append('.')
import invpy as ip

# Regrid the data ## This needs to happen before the three lines of
## code above can be run!
## Load 2010 data, which we will scale on a city level to 
## match 2019 population
pop_g = xr.open_dataset(f'{data_dir}cities/census2010_population_c.nc')
pop_g = pop_g['pop_density']

## Subset it for CONUS
conus_mask = np.load(f'{data_dir}CONUS_mask.npy')
conus_g = ip.match_data_to_clusters(conus_mask, clusters)
conus_g = conus_g.where(conus_g > 0, drop=True)
pop_g = pop_g.sel(lat=slice(conus_g.lat.min(), conus_g.lat.max()), 
                  lon=slice(conus_g.lon.min(), conus_g.lon.max()))

## Get population grid
delta_lon_pop = 0.01
delta_lat_pop = 0.01
Re = 6375e3 # Radius of the earth in m
lon_e_pop = np.round(np.append(pop_g.lon.values - delta_lon_pop/2,
                              pop_g.lon[-1].values + delta_lon_pop/2), 3)
lat_e_pop = np.round(np.append(pop_g.lat.values - delta_lat_pop/2,
                              pop_g.lat[-1].values + delta_lat_pop/2), 3)
area_pop = Re**2*(np.sin(lat_e_pop[1:]/180*np.pi) - 
                 np.sin(lat_e_pop[:-1]/180*np.pi))*delta_lon_pop/180*np.pi
grid_pop = {'lat' : pop_g.lat, 'lon' : pop_g.lon,
           'lat_b' : lat_e_pop, 'lon_b' : lon_e_pop}

## Get GEOS-Chem grid
lon_e_gc = np.append(clusters.lon.values - s.lon_delta/2,
                     clusters.lon[-1].values + s.lon_delta/2)
lat_e_gc = np.append(clusters.lat.values - s.lat_delta/2,
                     clusters.lat[-1].values + s.lat_delta/2)
area_gc = Re**2*(np.sin(lat_e_gc[1:]/180*np.pi) - 
                 np.sin(lat_e_gc[:-1]/180*np.pi))*s.lon_delta/180*np.pi
grid_gc = {'lat' : clusters.lat, 'lon' : clusters.lon,
           'lat_b' : lat_e_gc, 'lon_b' : lon_e_gc}

## Total emissions as check
total = (pop_g*area_pop[:, None]).sum(['lat', 'lon']) # Mg/m2/yr -> Mg/yr
print('Total 2010 population 0.01x0.01          : ', total.values)

## Get the regridder
# regridder = xe.Regridder(grid_pop, grid_gc, 'conservative')
# regridder.to_netcdf(f'{data_dir}cities/regridder_0.01x0.01_0.25x0.3125.nc')
regridder = xe.Regridder(grid_pop, grid_gc, 'conservative', 
                         weights=f'{data_dir}cities/regridder_0.01x0.01_0.25x0.3125.nc')

## Regrid the data
pop_rg = regridder(pop_g)
total_rg = (pop_rg*area_gc[:, None]).sum(['lat', 'lon'])
print('Total 2010 population 0.25x0.3125        : ', total_rg.values)

## Scale the regridded population by the difference lost in the 
## regridding (we don't need this to be perfect--it's just for
## our urban area analysis)
pop_rg *= total/total_rg
total_rg_scaled = (pop_rg*area_gc[:, None]).sum(['lat', 'lon'])
print('Total 2010 scaled population 0.25x0.3125 : ', total_rg_scaled.values)

## Save out
pop_rg.to_netcdf(f'{data_dir}cities/pop_gridded.nc')
