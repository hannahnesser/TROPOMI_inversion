import xarray as xr
import numpy as np
import sys
import copy

# Local preferences
base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
code_dir = base_dir + 'python/'
data_dir = base_dir + 'inversion_data/'
plot_dir = base_dir + 'plots/'

sys.path.append(code_dir)
import invpy as ip

# Define function to open masks
def open_mask(country, data_dir=data_dir):
    data = xr.open_dataset(f'{data_dir}{country}_Mask.001x001.nc')
    data = data.squeeze(drop=True)['MASK']
    return data

# Define function to regrid the masks to the inversion resolution
def regrid_mask(mask, clusters):
    # Subset mask to be as small as possible
    mask = mask.where(mask > 0, drop=True)
    mask = mask.fillna(0)

    # Regrid
    rg = mask.interp(lat=clusters.lat, lon=clusters.lon, method='linear')

    # Flatten and return
    flat = ip.clusters_2d_to_1d(clusters, rg)
    return flat

# Load clusters
clusters = xr.open_dataarray(f'{data_dir}clusters.nc')

# Open masks and create a total_mask array as well as a mask dictionary
mex_mask = regrid_mask(open_mask('Mexico'), clusters)
can_mask = regrid_mask(open_mask('Canada'), clusters)
conus_mask = regrid_mask(open_mask('CONUS'), clusters)
total_mask = mex_mask + can_mask + conus_mask
masks = {'Canada' : can_mask, 'CONUS' : conus_mask, 'Mexico' : mex_mask}

# Normalize with the total mask to deal with places that have bloopers
for c, m in masks.items():
    # fig, ax, _ = ip.plot_state(m, clusters)
    # plt.show()
    tmp = m/total_mask
    tmp = np.nan_to_num(tmp, 0)
    masks[c] = tmp

# Recalculate the total mask
total_mask = masks['Canada'] + masks['CONUS'] + masks['Mexico']

# Define a mask for Central American and Caribbean countries
other_mask = 1 - copy.deepcopy(total_mask)
other_mask = ip.match_data_to_clusters(other_mask, clusters)
other_countries_cond = (((other_mask.lon > -92)   & (other_mask.lat < 18.3)) |
                        ((other_mask.lon > -90)   & (other_mask.lat < 19.8)) |
                        ((other_mask.lon > -86)   & (other_mask.lat < 24))   |
                        ((other_mask.lon > -79.5) & (other_mask.lat < 27))   |
                        ((other_mask.lon > -66)   & (other_mask.lat < 36)))
other_mask = other_mask.where(other_countries_cond, 0)
other_mask = ip.clusters_2d_to_1d(clusters, other_mask)
masks['Other'] = other_mask

## Now deal with off shore emissions and spare grid cells
# Set up a mask that has identifying numbers for the country that
# occupies most of the grid cell
total_mask_id = np.zeros(total_mask.shape) # Default
total_mask_id[masks['Mexico'] > 0] = 1 # Mexico
total_mask_id[masks['CONUS'] > masks['Mexico']] = 2 # CONUS
total_mask_id[masks['Canada'] > masks['CONUS']] = 3 # Canada

# Match that to clusters and set areas where the mask == 0 to nan so
# that those values can be interpolated using neareswt neighbors
total_mask_id = ip.match_data_to_clusters(total_mask_id, clusters)
total_mask_id = total_mask_id.where(total_mask_id > 0)
total_mask_id = total_mask_id.interpolate_na(dim='lat', method='nearest')
total_mask_id = ip.clusters_2d_to_1d(clusters, total_mask_id)

# Replace values from "other" that were falsely filled
total_mask_id[masks['Other'] > 0] = 4 # Other

# Distribute into each country's mask
for i, country in enumerate(['Mexico', 'CONUS', 'Canada']):
    temp_mask = copy.deepcopy(total_mask_id)
    temp_mask[temp_mask != (i + 1)] = 0
    temp_mask[temp_mask > 0] = 1
    temp_bool = (masks[country] == 0) & (temp_mask > 0)
    masks[country][temp_bool] = temp_mask[temp_bool]
    np.save(f'{data_dir}{country}_mask.npy', masks[country])

# Recalculate the total mask
total_mask = (masks['Canada'] + masks['CONUS'] + masks['Mexico'] +
              masks['Other'])

# Still need to confirm that these add to proper values!
