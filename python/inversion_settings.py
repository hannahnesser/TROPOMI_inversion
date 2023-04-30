import sys
sys.path.append('.')
import gcpy as gc
import format_plots as fp

# Time settings
year = 2019
months = [i for i in range(1, 13)]
days = [i for i in range(1, 32, 1)]

# Information on the grid (centers)
lat_min = 9.75
lat_max = 60
lat_delta = 0.25

lon_min = -130
lon_max = -60
lon_delta = 0.3125

buffers = [3, 3, 3, 3] # N S E W

# Adjust for buffer cells (edges)
lat_min = lat_min + lat_delta*buffers[1] - lat_delta/2
lat_max = lat_max - lat_delta*buffers[0] + lat_delta/2
lon_min = lon_min + lon_delta*buffers[3] - lon_delta/2
lon_max = lon_max - lon_delta*buffers[2] + lon_delta/2

lats = [lat_min, lat_max]
lons = [lon_min, lon_max]

# Sectors
sector_groups = {'wetlands' : 'Wetlands',
                 'livestock' : 'Livestock',
                 'coal' : 'Coal',
                 'ong' : ['Oil', 'Gas'],
                 'landfills' : 'Landfills',
                 'wastewater' : 'Wastewater',
                 'other_anth' : ['Rice', 'OtherAnth'],
                 'other_bio' : ['Termites', 'Seeps', 'BiomassBurn', 'Lakes']}

# High resolution sectoral breakdown
sector_groups_hr = {'enteric_fermentation' : 'LIVESTOCK_4A',
                    'manure_management' : 'LIVESTOCK_4B',
                    'coal_abandoned' : 'COAL_ABANDONED',
                    'coal_surface' : 'COAL_SURFACE',
                    'coal_underground' : 'COAL_UNDERGROUND',
                    'ong_upstream' : ['OIL', 'GAS_PRODUCTION', 
                                      'GAS_PROCESSING', 'GAS_TRANSMISSION'],
                    'gas_distribution' : 'GAS_DISTRIBUTION',
                    'landfills' : ['LANDFILLS_IND', 'LANDFILLS_MUN'],
                    'wastewater' : ['WASTEWATER_IND', 'WASTEWATER_DOM'],
                    'other_anth' : ['OTHERANTH_1A_M', 'OTHERANTH_1A_S',
                                    'OTHERANTH_2B5', 'OTHERANTH_2C2', 
                                    'OTHERANTH_4F', 'OTHERANTH_5',
                                    'OTHERANTH_6D']}

# Sector naames
sectors = {'Total'               : 'total', 
           'Livestock'           : 'livestock', 
           'Oil and gas'         : 'ong',
           'Coal'                : 'coal', 
           'Landfills'           : 'landfills',
           'Wastewater'          : 'wastewater', 
           'Other anthropogenic' : 'other_anth',
           'Wetlands'            : 'wetlands',
           'Other biogenic'      : 'other_bio'}

# Sectoral colors
sector_colors = {'livestock'  : fp.color(4, cmap='plasma', lut=17),
                 'ong'        : fp.color(0, cmap='plasma', lut=17),
                 'coal'       : fp.color(11, cmap='plasma', lut=17),
                 'landfills'  : fp.color(8, cmap='plasma', lut=17),
                 'wastewater' : fp.color(15, cmap='plasma', lut=17),
                 'other_anth' : fp.color(13, cmap='viridis', lut=17),
                 'wetlands'   : '0.5'}