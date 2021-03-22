from os.path import join
from os import listdir
import os
import sys
import copy
import xarray as xr
import numpy as np
import pandas as pd

## ------------------------------------------------------------------------ ##
## Set user preferences
## ------------------------------------------------------------------------ ##
# Local preferences
base_dir = sys.argv[1]
code_dir = sys.argv[2]
data_dir = f'{base_dir}ProcessedDir'
output_dir = f'{base_dir}SummaryDir'
plot_dir = None

# The prior_run can either be a list of files or a single file
# with all of the data for simulation
year = 2019
months = np.arange(1, 13, 1) # excluding December for now
days = np.arange(1, 32, 1)
# prior_run = f'{year}.pkl'
prior_run = [f'{year}{mm:02d}{dd:02d}_GCtoTROPOMI.pkl'
             for mm in months for dd in days]
prior_run.sort()

# Define the blended albedo threshold
filter_on_blended_albedo = True
blended_albedo_threshold = 1
albedo_bins = np.arange(0, 1.1, 0.1)

# Remove latitudinal bias
remove_latitudinal_bias = True

# Which analyses do you wish to perform?
analyze_biases = False
calculate_so = True

# Information on the grid
lat_bins = np.arange(10, 65, 5)
lat_min = 9.75
lat_max = 60
lat_delta = 0.25
lon_min = -130
lon_max = -60
lon_delta = 0.3125
buffers = [3, 3, 3, 3]

## ------------------------------------------------------------------------ ##
## Import custom packages
## ------------------------------------------------------------------------ ##
# Custom packages
sys.path.append(code_dir)
import config
# config.SCALE = config.PRES_SCALE
# config.BASE_WIDTH = config.PRES_WIDTH
# config.BASE_HEIGHT = config.PRES_HEIGHT
import gcpy as gc
import troppy as tp
import format_plots as fp

# Define an empty suffix for file names
suffix = ''
if filter_on_blended_albedo:
    suffix += '_BAF'
if remove_latitudinal_bias:
    suffix += '_BC'
