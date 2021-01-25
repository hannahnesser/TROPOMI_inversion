'''
'''

## -------------------------------------------------------------------------##
## Load packages and set environment defaults
## -------------------------------------------------------------------------##
import xarray as xr
import numpy as np
import pandas as pd
import datetime

## -------------------------------------------------------------------------##
## User preferences
## -------------------------------------------------------------------------##
old_bc_file = '../gc_inputs/GEOSChem.BoundaryConditions.20190101_0000z_old.nc4'
new_bc_file = '../gc_inputs/GEOSChem.BoundaryConditions.20190101_0000z.nc4'

## -------------------------------------------------------------------------##
## Load file
## -------------------------------------------------------------------------##
bc = xr.open_dataset(old_bc_file)

## -------------------------------------------------------------------------##
## Update the initial time
## -------------------------------------------------------------------------##
# Get the data for the 3rd hour, dropping all other data
bc0 = bc.where(bc.time == bc.time[0], drop=True)

# Change the time of that data to the 0th hour
t = pd.Timestamp(bc0.time.values[0]).replace(hour=0).to_datetime64()
bc0.time.values[0] = t

# Merge the datasets
bc_new = xr.merge([bc, bc0])

## -------------------------------------------------------------------------##
## Save the output
## -------------------------------------------------------------------------##
bc_new.to_netcdf(new_bc_file)
