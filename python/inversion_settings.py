import sys
sys.path.append('.')
import gcpy as gc

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
lats, lons = gc.adjust_grid_bounds(lat_min, lat_max, lat_delta,
                                   lon_min, lon_max, lon_delta, buffers)
lat_min, lat_max = lats
lon_min, lon_max = lons
