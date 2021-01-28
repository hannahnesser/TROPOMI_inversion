## -------------------------------------------------------------------------##
## Load packages and set environment defaults
## -------------------------------------------------------------------------##

import xarray as xr
import matplotlib.pyplot as plt
import math
import numpy as np

from os.path import join
from os import listdir

import sys
sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python')
import inversion as inv
import plots as p
import format_plots as fp

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
data_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/prior/total_emissions'
emis_file = 'HEMCO_diagnostics.YYYYMM010000.nc'
years = [2019]
months = np.arange(1, 13, 1)

# Choose emissions from the following options:
# - SoilAbsorb          - Termites
# - Lakes               - Seeps
# - Wetlands            - BiomassBurn
# - OtherAnth           - Rice
# - Wastewater          - Landfills
# - Livestock           - Coal
# - Gas                 - Oil
# - Total
emissions = ['Wetlands', 'Livestock',
            ['Coal', 'Oil', 'Gas'],
            ['Wastewater', 'Landfills'],
            ['Termites', 'Seeps', 'BiomassBurn', 'Lakes'],
            ['Rice', 'OtherAnth']]
titles = ['Wetlands', 'Livestock', 'Coal, Oil, and\nNatural Gas',
          'Wastewater\nand Landfills', 'Other Biogenic\nSources',
          'Other Anthropogenic\nSources']

# Set colormap
colormap = fp.cmap_trans('viridis')

plot_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/plots'

## -------------------------------------------------------------------------##
## Load data
## -------------------------------------------------------------------------##
for y in years:
    for m in months:
        file = emis_file.replace('YYYY', '%d' % y)
        file = file.replace('MM', '%02d' % m)
        if file in listdir(data_dir):
            emis = xr.open_dataset(join(data_dir, file))
        else:
            print('%s is not in the data directory.' % file)
            continue

        emis = emis.drop(['hyam', 'hybm', 'P0', 'AREA'])
        print(emis)

        ## ---------------------------------------------------------------------##
        ## Plot data
        ## ---------------------------------------------------------------------##
        # ncategory = len(emissions)
        # fig, ax = fp.get_figax(rows=2, cols=math.ceil(ncategory/2),
        #                        maps=True, lats=emis.lat, lons=emis.lon)
        # plt.subplots_adjust(hspace=0.5)
        # cax = fp.add_cax(fig, ax, cbar_pad_inches=0.5)
        # for i, axis in enumerate(ax.flatten()):
        #     axis = fp.format_map(axis, lats=emis.lat, lons=emis.lon)
        #     if type(emissions[i]) == str:
        #         e = emis['EmisCH4_%s' % emissions[i]].squeeze()
        #     elif type(emissions[i] == list):
        #         e = sum(emis['EmisCH4_%s' % em].squeeze()
        #                 for em in emissions [i])

        #     e *= 0.001*60*60*24*365*1000*1000
        #     c = e.plot(ax=axis, cmap=colormap, vmin=0, vmax=5,
        #                add_colorbar=False)
        #     cb = fig.colorbar(c, cax=cax, ticks=np.arange(0, 6, 1))
        #     cb = fp.format_cbar(cb, cbar_title=r'Emissions (Mg km$^2$ a$^{-1}$)')
        #     axis = fp.add_title(axis, titles[i])

        # fp.save_fig(fig, plot_dir, 'prior_emissions_%d%02d' % (y, m))

print(emis['EmisCH4_Total'].max())
print(emis['EmisCH4_Total'].min())
