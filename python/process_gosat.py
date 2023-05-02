from os.path import join
from os import listdir
import sys
import copy
import xarray as xr
import pandas as pd
pd.set_option('display.max_columns', 15)

# Import Custom packages
sys.path.append(code_dir)
import config
import gcpy as gc
import inversion_settings as settings

output_dir = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/inversion_data/'

# Compare to GOSAT?
gosat_dir = '/n/jacob_lab/Lab/seasasfs02/CH4_inversion/InputData/Obs/ULGOSAT_v9/2019'
gosat = [f'{gosat_dir}/UoL-GHG-L2-CH4-GOSAT-OCPR-{settings.year}{mm:02d}{dd:02d}-fv9.0.nc'
         for mm in settings.months for dd in settings.days]

gosat_data = pd.DataFrame(columns=['DATE', 'LAT', 'LON', 'OBS'])
for file in gosat:
    # Check if that file is in the data directory
    if file.split('/')[-1] not in listdir(gosat_dir):
        print(f'{file} is not in the data directory.')
        continue

    # Load the data.
    gosat_fields = ['latitude', 'longitude', 'xch4', 'xch4_quality_flag']
    new_data = gc.read_file(join(gosat_dir, file))[gosat_fields]
    new_data = new_data.where((new_data['xch4_quality_flag'] == 0) &
                              (new_data['latitude'] > settings.lat_min) &
                              (new_data['latitude'] < settings.lat_max) &
                              (new_data['longitude'] > settings.lon_min) &
                              (new_data['longitude'] < settings.lon_max),
                              drop=True)
    new_data = new_data.rename({'latitude' : 'LAT', 'longitude' : 'LON',
                                'xch4' : 'OBS'})
    new_data = new_data[['LAT', 'LON', 'OBS']].to_dataframe()
    new_data['DATE'] = int(file.split('-')[-2])

    # Concatenate
    gosat_data = pd.concat([gosat_data, new_data]).reset_index(drop=True)

# Save nearest latitude and longitude centers
gosat_data['LAT_CENTER'] = lats_l[gc.nearest_loc(gosat_data['LAT'].values,
                                               lats_l)]
gosat_data['LON_CENTER'] = lons_l[gc.nearest_loc(gosat_data['LON'].values,
                                               lons_l)]

# Add month and season
gosat_data.loc[:, 'MONTH'] = pd.to_datetime(gosat_data['DATE'].values,
                                   format='%Y%m%d').month
gosat_data.loc[:, 'SEASON'] = 'DJF'
gosat_data.loc[gosat_data['MONTH'].isin([3, 4, 5]), 'SEASON'] = 'MAM'
gosat_data.loc[gosat_data['MONTH'].isin([6, 7, 8]), 'SEASON'] = 'JJA'
gosat_data.loc[gosat_data['MONTH'].isin([9, 10, 11]), 'SEASON'] = 'SON'

# Save the data out
print(f'Saving data in {output_dir}/{settings.year}_gosat.pkl')
gc.save_obj(gosat_data, join(output_dir, f'{settings.year}_gosat.pkl'))

# Grid the GOSAT data
gosat_grid = gosat_data.groupby(['LAT_CENTER', 'LON_CENTER',
                                 'DATE']).mean()['OBS']
gosat_grid = gosat_grid.to_xarray().rename({'LAT_CENTER' : 'lats',
                                            'LON_CENTER' : 'lons',
                                            'DATE' : 'date'})

# Save out
gosat_grid.to_netcdf(join(output_dir, f'{settings.year}_gosat_gridded.nc'))